# Image engines — and how much of them belongs in `BaseEngine`

`watchmal/engine/images/` holds the engines for **image-like** data: events rendered as
a 2D PMT grid and fed to a CNN (ResNet and friends). They are the direct descendants of
the original WatChMaL core. The sibling `watchmal/engine/graph/` holds the engines for
PyG graphs, and `watchmal/engine/multiring/` the sparse-3D segmentation one.

The families differ for real reasons — a CNN step is a dense tensor forward with AMP and
an iteration-based loop; a graph step batches variable-size graphs and runs an
epoch-based loop with early stopping — so the split is not an accident, and the merge
plan deliberately kept the two train loops separate. **This note is about the rest**:
what the two families duplicate today, and how much of it could move to `BaseEngine`.

---

## The current division

| | `BaseEngine` | image | graph |
|---|---|---|---|
| construction, tracker, device, DDP wrap | ✅ | — | — |
| `configure_{loss,optimizers,scheduler,amp,early_stopping}` | ✅ | — | overrides `early_stopping` (identical) |
| collectives (`get_reduced`, `get_gathered`, `get_synchronized_*`) | ✅ | — | — |
| `save_state` / `restore_state` / `restore_best_state` | ✅ | **overrides all three** | — |
| `setup_data_loaders` | ✅ (`build_loader`) | **overrides** (`get_data_loader`) | — |
| train / validate / evaluate loops | — | ✅ | ✅ |
| per-batch mounting, metric accumulation | — | `process_data` / `step` | `_mount_batch` / `_accumulate_metrics` |

---

## What is redundant *today* — in rough order of how easily it collapses

### 1. `configure_early_stopping` — pure duplication (2 lines)

`graph/reconstruction.py:81` is character-for-character what `base_engine.py:212`
already does (`self.early_stopping = instantiate(...)`). Delete the override. No risk.

### 2. `save_state` / `restore_state` / `restore_best_state` — one behaviour, two formats

The image engine overrides all three. The difference is not the mechanism, it is **what
goes in the dict**:

| | base | image override |
|---|---|---|
| keys written | `global_step, epoch, seed, optimizer, scheduler, state_dict` | `global_step, optimizer, state_dict[, scaler]` |
| missing-key policy on restore | `.get()`, degrades quietly | `self.state_data['optimizer']`, raises |

So the image format loses `epoch`, `seed` and `scheduler`, and the base format has no
place for the AMP `scaler`. **A checkpoint written by one family cannot be trusted to
the other** — a real hazard now that both come from one class hierarchy.

Collapsing this is the highest-value item here: make `BaseEngine.save_state` write the
superset (including `scaler` when `self.scaler is not None`), keep `.get()` on restore,
and delete the override. The only genuinely image-specific part is the scaler, which the
base already knows about via `configure_amp`.

### 3. `setup_data_loaders` — two factories for one job

Image calls `get_data_loader` (`dataset/data_utils.py`), everyone else calls
`build_loader` (`base_engine.py`). They do the same three things — instantiate the
dataset, build a sampler from the split file, wrap for DDP — with different argument
names and slightly different defaults (notably the DDP tail policy and the per-rank
batch floor, which have already had to be fixed twice, once in each).

This is the one the merge plan explicitly left open, and it is the largest single block
of duplicated logic. Unifying it means agreeing on one signature; the behavioural
differences are small enough to be parameters.

### 4. Loop bookkeeping — same skeleton, different bodies

Both train loops independently implement: iteration/epoch counters, "is this the best
validation loss so far" tracking, `save_state("_BEST")` on improvement, periodic
validation, metric accumulation over an epoch, and the tracker calls. Only the *step*
inside differs.

This is where the temptation to over-unify lives. A template method — `BaseEngine.fit()`
owning the bookkeeping and calling an abstract `run_epoch()` / `run_validation()` — would
remove a lot of duplication, but it would also freeze the loop's shape for every future
family, and the multi-ring engine already needs a differently-shaped loop (cache
refresh, diagnostics hooks). **Recommendation: extract the *pieces* (a `BestTracker`
helper, a `log_epoch()` on the tracker) rather than the control flow.** The plan's
decision to keep the loops separate stands; it is the accounting around them that should
be shared.

### 5. Metric accumulation

`graph/_accumulate_metrics` (a static dict-accumulator) and the image loop's inline
`sum / count` do the same job. A tiny `MetricAccumulator` on the base would serve both
and would also make "which metrics did this epoch produce" a single definition.

---

## What should *not* move to the base

- **The train/validate/evaluate loops themselves** (see 4).
- **`process_data` / `_mount_batch`.** Both mount a batch onto the device, but one
  handles a dense tensor + transforms and the other a PyG `Batch` with an optional
  hierarchical PMT/mPMT structure. Any shared version would be a `dict` of special
  cases.
- **AMP.** Only the image loop uses it. `configure_amp` on the base is already the right
  amount of sharing: it is a no-op elsewhere.
- **Plot hooks** (`make_plots`, `to_disk_data_reformat`). These are per-family by
  definition, and abstract on the graph side already.

---

## Estimated size

| Item | Effort | Risk | Removes |
|---|---|---|---|
| 1. early-stopping override | minutes | none | 2 lines |
| 2. one checkpoint format | small | **needs a cross-family restore test first** | ~60 lines, one hazard |
| 5. metric accumulator | small | none | ~15 lines |
| 3. one loader factory | medium | behavioural (tail policy, batch floor) | ~80 lines |
| 4. loop bookkeeping pieces | medium | high if done as a template method | ~50 lines |

Items 1, 5 and 2 are worth doing next; 3 wants a decision on the signature; 4 should be
done as helpers, never as a shared loop.

The acceptance criterion for 2 is already written as a check in the CI plan (T3.3,
cross-family checkpoint round-trip) — land the test first, then the collapse.
