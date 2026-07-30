# Working in this repository

## Documentation register — `docs/` and `landing/`

**The published site is read inside *and* outside the collaboration.** It is linked from
talks and shown to people who have never used the framework, so its text is a piece of
scientific writing, not internal notes or product copy.

Write it in an **academic register**:

- **Declarative and impersonal.** State what the framework does and what was measured.
  Prefer "the graph engines write both formats" to "we now also write". Avoid "you'll
  love", "simply", "just", "powerful", "seamless", "blazing".
- **Second person is allowed in procedures only** — install, quickstart, how-to steps read
  better as "run this, then check that". Everywhere else, drop it.
- **Quantify instead of characterising.** Not "much faster" but "23 s versus 4 min"; not
  "a small subset" but "200 events, ~30 MB".
- **Attribute every claim.** A statement about behaviour cites the file, the config or the
  test that establishes it. If nothing establishes it, say so — see the status vocabulary
  in `docs/model-zoo.md` (verified / ships / code-only / broken).
- **Define terms on first use**, including detector acronyms (IWCD, WCTE, mPMT, PID) and
  framework terms (engine, task, config group). An outside reader has neither vocabulary.
- **No hype, no emoji in prose.** Status glyphs in tables are fine and carry meaning.
- **Hedge honestly.** "has not been measured" and "unverified" are acceptable and
  preferable to silence. Do not imply a result that does not exist.
- **Name detectors only when confirmed.** Several shipped datasets do not record which
  detector they came from; do not label a figure or table "Hyper-K" on inference from a
  directory name.

The pages written in the first pass (2026-07-30) were drafted before this rule and read
too conversationally in places. They are to be revised to this register — that revision is
tracked in `claude_report/docs_site_plan_phase2.md`, not done ad hoc.

## Internal notes stay internal

`claude_report/` is gitignored and holds planning and audit notes. Nothing from it is
published, quoted or linked from `docs/`. The site cites code, configs and tests.

## Two config trees, for now

`tutorial/config/watchmal` and `tutorial/config/caverns` are both live, and `main.py`
defaults to the first. Documentation that enumerates config-group paths will churn when
they merge — prefer describing structure over listing files, and check
`claude_report/unified_core_status.md` §2.1 before writing anything path-exhaustive.
