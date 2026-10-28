"""Draw the two figures of the report, sized for one column of the IEEE template.

- report/fig_learning_curve.pdf: error of the logistic regression on the training
  data and on held-out data as the training set grows
- report/fig_ablation.pdf: F1 score of each step towards the final model, in 5-fold
  cross-validation and on the test set

Usage: python make_figures.py (about 10 minutes, for the learning curve)
"""

import os

import matplotlib

matplotlib.use("Agg")  # no window, just the files
import matplotlib.pyplot as plt
import numpy as np

import cross_validation
import run

OUT_DIR = "report"
FIGSIZE = (3.4, 1.5)  # inches, one column of the IEEE template
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#e3e3e3"

# Same style for both figures: small serif text, light grid, no top/right border
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",  # the 10^4 of the log axis, in a Times-like font
        "font.size": 8,
        "axes.labelcolor": INK,
        "axes.edgecolor": GRID,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "legend.fontsize": 6.5,
        "savefig.bbox": "tight",  # no white margin around the plot
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,  # embed TrueType fonts
    }
)

# Each step towards the final model: 5-fold CV F1 (mean, std) and test F1 (nan if
# never submitted). The CV numbers are printed by the experiments in
# cross_validation.py, the test numbers come from AIcrowd.
ABLATION = [
    ("logistic, original inputs", 0.4253, 0.0022, 0.439),
    ("logistic, yes/no indicators", 0.4278, 0.0020, 0.439),
    ("logistic, squares + 10 factors", 0.4295, 0.0027, 0.442),
    ("logistic, Newton's method", 0.4312, 0.0020, 0.443),
    ("logistic, 20 risk factors", 0.4326, 0.0014, 0.444),
    ("boosted trees", 0.4347, 0.0021, np.nan),
    ("final: avg of trees and best log reg", 0.4364, 0.0010, 0.447),
]


def fig_learning_curve():
    """Error on the training data (dashed) and on held-out data (solid)."""
    x, _, y, _, names = run.load_data()
    folds = run.kfold_indices(len(y), run.K_FOLDS, run.SEED)
    # rows of (variant, training size, training loss, held-out loss, held-out F1)
    rows = cross_validation.learning_curve(x, (y + 1) / 2, names, folds)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for variant, label, color, marker in [
        ("baseline (198 features)", "original inputs", BLUE, "o"),
        ("squares + interactions (360 features)", "with extra inputs", ORANGE, "s"),
    ]:
        size = [r[1] for r in rows if r[0] == variant]
        train_loss = [r[2] for r in rows if r[0] == variant]
        held_out_loss = [r[3] for r in rows if r[0] == variant]
        style = dict(
            color=color,
            marker=marker,
            markersize=4,
            markeredgecolor="white",
            markeredgewidth=0.8,
            linewidth=1.5,
        )
        ax.plot(size, held_out_loss, "-", label=f"{label}, held-out", **style)
        ax.plot(size, train_loss, "--", label=f"{label}, training", **style)

    ax.set_xscale("log")
    ax.set_xlabel("number of training respondents")
    ax.set_ylabel("error (logistic loss)")
    # legend above the plot, in two columns
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        handlelength=3.2,
        columnspacing=1.0,
        borderaxespad=0.2,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_learning_curve.pdf"))


def fig_ablation():
    """CV F1 with its std (blue dots) and test F1 (orange diamonds) of each step."""
    steps, cv_f1, cv_std, test_f1 = zip(*ABLATION)  # one list per column
    pos = range(len(steps))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.errorbar(
        cv_f1,
        pos,
        xerr=cv_std,
        fmt="o",
        color=BLUE,
        markersize=5,
        markeredgecolor="white",
        markeredgewidth=0.8,
        elinewidth=1.2,
        capsize=2,
        label="cross-validation (mean ± std)",
    )
    ax.plot(
        test_f1,
        pos,
        "D",
        markersize=4.5,
        markerfacecolor="white",
        markeredgecolor=ORANGE,
        markeredgewidth=1.3,
        label="test set",
    )

    ax.set_yticks(pos, steps, fontsize=6.5, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()  # first step at the top
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("F1 score")
    ax.set_xlim(0.420, 0.450)
    # legend above the plot
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.4, 1.0),
        ncol=2,
        handletextpad=0.3,
        borderaxespad=0.2,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_ablation.pdf"))


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    fig_ablation()
    fig_learning_curve()
    print(f"figures saved in {OUT_DIR}/")
