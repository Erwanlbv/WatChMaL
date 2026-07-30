# Your own workspace

The repository ships a reference config tree under `tutorial/`. **Do not edit it.** Make a
personal copy instead:

```bash
bash setup/make_dirs.sh
```

That creates two gitignored directories at the repo root:

| Directory | Copied from | What it is for |
|---|---|---|
| `config/` | `tutorial/config/` | your configs — dataset paths, hyper-parameters, new experiments |
| `launch/` | `tutorial/launch/` | your launch scripts — SLURM options, container binds, account names |

Both are in `.gitignore`, so your local paths never end up in a commit, and a `git pull`
never clobbers your edits.

!!! warning "The script refuses to overwrite"
    `make_dirs.sh` stops if `config/` or `launch/` already exists, because the copy is
    file-by-file: an existing workspace would silently lose every local edit, and since
    both directories are gitignored there is nothing to recover from. Pass `--force` once
    you are sure.

## Why the copy, rather than editing in place

Three reasons, in order of how often they bite:

1. **`git pull` conflicts.** The tutorial tree changes upstream. If your dataset paths
   live in it, every pull is a merge conflict in a file you did not mean to own.
2. **Accidental commits.** Absolute paths like `/sps/t2k/<you>/...` in a tracked config
   are noise for everyone else and a small information leak.
3. **A working reference.** When your config stops working, the shipped one still runs, so
   you can diff against something known-good.

## Running from your workspace

Point `--config-path` at your copy instead of the tutorial tree:

```bash
# shipped reference
python main.py --config-path tutorial/config/caverns/main --config-name gat_classification

# your copy
python main.py --config-path config/caverns/main --config-name gat_classification
```

The launch scripts find the repository root by searching upward from their own location,
so a copy under `launch/` works without editing any path. Override with `NEUNET_ROOT` if
you need to.

!!! note "`sbatch` scripts must be submitted from the repo root"
    They use `SLURM_SUBMIT_DIR` to locate the repository, and their relative `logs/`
    output requires it too — run `mkdir -p logs` first.

## What to change first

For a run on CC-IN2P3, the values that are genuinely yours are the dataset path, the split
index list, and your SLURM account. Everything else in the shipped configs points at
shared reference resources that are readable group-wide and can be left alone for a first
run — see [paths to change](../clusters/cc-in2p3-paths.md).

## A naming convention worth adopting

If you keep several parallel experiments in one workspace, prefix the files belonging to
each so they stay distinguishable from the stock configs you copied. A config named
`MYSTUDY_gat_lr_scan.yaml` next to the original `gat_classification.yaml` tells you at a
glance which one you are responsible for.
