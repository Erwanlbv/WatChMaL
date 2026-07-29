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
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    run_dir = args.run_dir.resolve()
    label = args.label or run_dir.name
    out = args.out or (REPO_ROOT / "plots" / run_dir.name)
    out.mkdir(parents=True, exist_ok=True)

    checks: list[tuple[str, str]] = []

    if args.kind == "classification":
        from analysis.classification import WatChMaLClassification, plot_rocs

        targets = np.load(run_dir / "outputs" / "targets.npy")
        run = WatChMaLClassification(str(run_dir), run_label=label, true_labels=targets)

        softmaxes = run.softmaxes
        checks.append(("softmax read by WatChMaLClassification",
                       f"{softmaxes.shape}, rows sum to 1: "
                       f"{np.allclose(softmaxes.sum(axis=1), 1.0)}"))

        fig = run.plot_training_progression()[0]
        fig.savefig(out / "training_progression.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        checks.append(("plot_training_progression", "ok"))

        # class 1 is the signal (e-), class 0 the background (mu-), per MapLabels
        fig, ax = plt.subplots(figsize=(5.5, 5))
        plot_rocs([run], signal_labels=[1], background_labels=[0], ax=ax,
                  x_label="signal efficiency", y_label="background rejection")
        fig.savefig(out / "roc_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        checks.append(("plot_rocs (consumes softmax)", "ok"))

    else:
        from analysis.regression import WatChMaLPositionRegression, plot_histograms

        truths = np.load(run_dir / "outputs" / "targets.npy")
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
