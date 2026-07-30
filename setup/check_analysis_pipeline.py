#!/usr/bin/env python3
"""
Check that a finished run can actually be read and plotted by `analysis/`.

The training CSVs have been analysis-compatible for every family since P1, but the
evaluation *outputs* were not: `analysis/classification.py` reads `softmax`, and
`analysis/regression.py` reads `predicted_<quantity>`, while the graph engines wrote
only `preds`/`targets`. Both are now written, and this script is the end-to-end proof —
it loads a run through the real analysis classes and produces the real plots.

    python setup/check_analysis_pipeline.py RUN_DIR --kind classification
    python setup/check_analysis_pipeline.py RUN_DIR --kind position-regression

Plots go to `plots/<run-dir-name>/`. Exit code is non-zero if anything fails to load.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# run from anywhere: analysis/ and watchmal/ live at the repo root
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--kind", choices=("classification", "position-regression"),
                        required=True)
    parser.add_argument("--label", default=None, help="run label used in legends")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--signal-labels", type=int, nargs="*", default=None,
                        help="ROC signal classes, in the RAW label values the run's "
                             "engine.label_set uses (e.g. 11 for e-). Derived from the "
                             "run config when omitted.")
    parser.add_argument("--background-labels", type=int, nargs="*", default=None,
                        help="ROC background classes, same convention as --signal-labels")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    run_dir = args.run_dir.resolve()
    label = args.label or run_dir.name

    def index_sorted(truth):
        """Put a truth array into the same order as everything analysis/ returns.

        `WatChMaLOutput` returns `outputs[indices.argsort()]` (analysis/read.py), i.e.
        sorted by dataset index - but the engine writes the .npy files in *evaluation*
        order, which is the order the sampler walked the split. Those agree only when the
        split's test_idxs happen to be sorted; with a shuffled split they do not, and
        comparing a file-order truth against an index-sorted prediction silently produces
        a chance-level result rather than an error.
        """
        order = np.load(run_dir / "outputs" / "indices.npy").argsort()
        return np.asarray(truth)[order]

    out = args.out or (REPO_ROOT / "plots" / run_dir.name)
    out.mkdir(parents=True, exist_ok=True)

    checks: list[tuple[str, str]] = []

    if args.kind == "classification":
        from analysis.classification import WatChMaLClassification, plot_rocs

        # Same split convention as the regression branch below: the graph engine writes
        # `targets.npy`, the image classifier engine writes `labels.npy`. Accept either.
        truth_file = next((f for f in ("labels.npy", "targets.npy")
                           if (run_dir / "outputs" / f).exists()), None)
        if truth_file is None:
            raise SystemExit(f"no truth array in {run_dir / 'outputs'}: expected "
                             f"labels.npy (image engine) or targets.npy (graph engine)")
        targets = index_sorted(np.load(run_dir / "outputs" / truth_file))
        run = WatChMaLClassification(str(run_dir), run_label=label, true_labels=targets)

        # analysis/classification.py uses `signal_labels` in two different spaces:
        # plot_rocs compares them against true_labels directly (np.isin), while
        # combine_softmax pushes them through label_map to pick softmax columns. So
        # true_labels must hold the RAW label values. The image engine writes
        # already-mapped 0..N-1 labels to labels.npy, which makes the np.isin half match
        # nothing ("No positive samples in y_true") and the ROC meaningless. Invert
        # label_map to put the truth back in raw space.
        if run.label_map and set(np.unique(targets).tolist()) <= set(run.label_map.values()):
            inverse = {v: k for k, v in run.label_map.items()}
            run.true_labels = np.array([inverse[int(t)] for t in targets])
            checks.append(("labels un-mapped to raw values",
                           f"{sorted(set(targets.tolist()))} -> {sorted(set(run.true_labels.tolist()))}"))

        softmaxes = run.softmaxes
        checks.append(("softmax read by WatChMaLClassification",
                       f"{softmaxes.shape}, rows sum to 1: "
                       f"{np.allclose(softmaxes.sum(axis=1), 1.0)}"))

        fig = run.plot_training_progression()[0]
        fig.savefig(out / "training_progression.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        checks.append(("plot_training_progression", "ok"))

        # plot_rocs pushes its labels through run.label_map, which WatChMaLClassification
        # builds from the config's engine.label_set - so the labels here are the RAW values
        # (11/13 for HK, 1/2 for the mPMT toy), not the mapped 0..N-1 class indices. When
        # label_set is absent label_map is None and labels ARE column indices. Hardcoding
        # either convention breaks the other, so derive it from the run.
        classes = sorted(run.label_map) if run.label_map else sorted(set(np.asarray(targets).tolist()))
        if len(classes) != 2 and (args.signal_labels is None or args.background_labels is None):
            raise SystemExit(f"run has {len(classes)} classes {classes}; pass --signal-labels "
                             f"and --background-labels explicitly")
        signal = args.signal_labels if args.signal_labels else [classes[0]]
        background = args.background_labels if args.background_labels else classes[1:]
        checks.append(("ROC classes", f"signal {signal} vs background {background} "
                                      f"(label_map {run.label_map})"))

        fig, ax = plt.subplots(figsize=(5.5, 5))
        plot_rocs([run], signal_labels=signal, background_labels=background, ax=ax,
                  x_label="signal efficiency", y_label="background rejection")
        fig.savefig(out / "roc_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        checks.append(("plot_rocs (consumes softmax)", "ok"))

    else:
        from analysis.regression import WatChMaLPositionRegression, plot_histograms

        # The two families name the truth array differently: the graph engine writes
        # `targets.npy`, while the image regression engine writes one file per target key
        # (`positions.npy` alongside `predicted_positions.npy`). Accept either, so this
        # check covers both - it previously only ever ran against a graph run.
        truth_file = next((f for f in ("positions.npy", "targets.npy")
                           if (run_dir / "outputs" / f).exists()), None)
        if truth_file is None:
            raise SystemExit(f"no truth array in {run_dir / 'outputs'}: expected "
                             f"positions.npy (image engine) or targets.npy (graph engine)")
        truths = index_sorted(np.load(run_dir / "outputs" / truth_file))
        run = WatChMaLPositionRegression(str(run_dir), run_label=label,
                                         true_positions=truths)

        predictions = run.position_prediction
        checks.append(("predicted_positions read by WatChMaLPositionRegression",
                       f"{predictions.shape}"))
        checks.append(("residuals computed",
                       f"mean |Δ| = {np.abs(run.position_residuals).mean():.4f}"))

        fig = run.plot_training_progression()[0]
        fig.savefig(out / "training_progression.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        checks.append(("plot_training_progression", "ok"))

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        for axis, component in zip(axes, ("x_residuals", "y_residuals", "z_residuals")):
            plot_histograms([run], quantity=component, ax=axis, bins=25,
                            x_label=component.replace("_", " "), y_label="events")
        fig.tight_layout()
        fig.savefig(out / "position_residuals.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        checks.append(("plot_histograms of residuals", "ok"))

    width = max(len(name) for name, _ in checks)
    print(f"\n{label}  ->  {out}")
    for name, detail in checks:
        print(f"  PASS  {name:<{width}}  {detail}")


if __name__ == "__main__":
    main()
