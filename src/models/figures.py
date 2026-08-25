"""Charts for the card-rail results.

Run: python src/models/figures.py
Writes PNGs to reports/figures/.

Colours come from a validated categorical palette (blue/orange pass CVD and
contrast checks against a light surface). Text is always ink, never the series
colour, so identity is never carried by colour alone.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg
from models.metrics import operating_points
from policy.bands import ALLOW, BLOCK, STEP_UP, evaluate_policy, fit_policy

FIGURES = cfg.REPORTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_SOFT,
    "text.color": INK,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "grid.color": GRID,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "figure.dpi": 130,
})


def _clean(ax, ygrid=True):
    """Recessive axes: no top/right spine, grid behind the marks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if ygrid:
        ax.yaxis.grid(True, linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)


def chart_tradeoff(y_test, scores):
    """The business chart: fraud caught against genuine customers disturbed."""
    fine = operating_points(y_test, scores, list(np.linspace(0.0005, 0.15, 200)))
    marks = operating_points(y_test, scores, [0.005, 0.01, 0.02, 0.03, 0.05, 0.10])

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(fine.genuine_disturbed_pct * 100, fine.recall * 100,
            color=BLUE, linewidth=2, zorder=3)
    ax.scatter(marks.genuine_disturbed_pct * 100, marks.recall * 100,
               s=64, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=4)

    for _, r in marks.iterrows():
        chosen = abs(r.alert_rate - 0.02) < 1e-9
        rate = f"{r.alert_rate:.1%}" if r.alert_rate < 0.01 else f"{r.alert_rate:.0%}"
        ax.annotate(
            f"{rate} flagged" + ("  <- chosen" if chosen else ""),
            (r.genuine_disturbed_pct * 100, r.recall * 100),
            textcoords="offset points", xytext=(9, -4),
            fontsize=9, color=INK if chosen else INK_SOFT,
            fontweight="bold" if chosen else "normal",
        )

    ax.set_xlabel("Genuine customers disturbed (%)")
    ax.set_ylabel("Fraud caught (%)")
    ax.set_title("The trade-off you are choosing on")
    ax.text(0.5, -0.16, "Each point is one flag-rate setting. Up is better, left is better.",
            transform=ax.transAxes, ha="center", fontsize=9, color=INK_SOFT)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "01_tradeoff_curve.png", bbox_inches="tight")
    plt.close(fig)


def chart_seed_spread():
    """Why the V-column experiment needed three seeds instead of one."""
    path = cfg.REPORTS / "step2_v_column_multiseed.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    variants = list(df.variant.unique())
    colours = {variants[0]: BLUE, variants[1]: ORANGE}

    for i, variant in enumerate(variants):
        part = df[df.variant == variant]
        x = np.full(len(part), i) + np.linspace(-0.06, 0.06, len(part))
        ax.scatter(x, part.pr_auc, s=90, color=colours[variant],
                   edgecolor=SURFACE, linewidth=2, zorder=4,
                   label=f"{variant} ({part.features.iloc[0]} features)")
        mean = part.pr_auc.mean()
        ax.hlines(mean, i - 0.19, i + 0.19, color=colours[variant],
                  linewidth=2.5, zorder=3)
        ax.annotate(f"mean {mean:.4f}", (i + 0.22, mean), fontsize=9,
                    color=INK, va="center", fontweight="bold")

    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants)
    ax.set_ylabel("PR-AUC")
    ax.set_title("One seed was not enough to answer this")
    ax.text(0.5, -0.17,
            "Three seeds each. The 339-column version loses every time - the earlier\n"
            "single-seed result that favoured it was noise.",
            transform=ax.transAxes, ha="center", fontsize=9, color=INK_SOFT)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "02_seed_spread.png", bbox_inches="tight")
    plt.close(fig)


def chart_policy_bands(y_test, scores):
    """Where the 2% alert budget goes, and who it touches."""
    policy = fit_policy(scores, 0.005, 0.02)
    table = evaluate_policy(y_test, scores, policy)
    order = [BLOCK, STEP_UP, ALLOW]
    table = table.set_index("band").loc[order].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    labels = ["Block", "Send OTP", "Allow"]

    ax = axes[0]
    ax.barh(labels, table.fraud_recall * 100, height=0.55, color=BLUE)
    for i, v in enumerate(table.fraud_recall * 100):
        ax.annotate(f"{v:.1f}%", (v, i), xytext=(6, 0),
                    textcoords="offset points", va="center",
                    fontsize=10, color=INK, fontweight="bold")
    ax.set_xlabel("Share of all fraud (%)  -  Allow = missed")
    ax.set_title("Where fraud lands")
    ax.invert_yaxis()
    ax.set_xlim(0, max(table.fraud_recall * 100) * 1.28)
    _clean(ax, ygrid=False)
    ax.xaxis.grid(True, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)

    ax = axes[1]
    genuine = table.genuine_pct * 100
    ax.barh(labels, genuine, height=0.55, color=ORANGE)
    for i, v in enumerate(genuine):
        ax.annotate(f"{v:.3f}%" if v < 1 else f"{v:.1f}%", (v, i), xytext=(6, 0),
                    textcoords="offset points", va="center",
                    fontsize=10, color=INK, fontweight="bold")
    ax.set_xlabel("Share of genuine customers (%)")
    ax.set_title("Genuine customers affected")
    ax.invert_yaxis()
    ax.set_xscale("symlog", linthresh=0.01)
    _clean(ax, ygrid=False)
    ax.xaxis.grid(True, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)

    fig.suptitle("Only 0.025% of genuine customers are blocked outright",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "03_policy_bands.png", bbox_inches="tight")
    plt.close(fig)


def chart_feature_importance():
    """What the model actually leans on."""
    path = cfg.REPORTS / "step2_card_feature_importance.csv"
    if not path.exists():
        return
    df = pd.read_csv(path).head(20).iloc[::-1]

    # Sequential encoding: one hue, light to dark by magnitude.
    ranks = df.gain / df.gain.max()
    colours = plt.cm.Blues(0.35 + 0.55 * ranks)

    fig, ax = plt.subplots(figsize=(8, 6.6))
    ax.barh(df.feature, df.gain / 1000, height=0.68, color=colours)
    ax.set_xlabel("Importance (gain, thousands)")
    ax.set_title("Top 20 features")
    ax.text(0.5, -0.10,
            "V258 and V257 lead - both from the V block the previous project discarded.",
            transform=ax.transAxes, ha="center", fontsize=9, color=INK_SOFT)
    _clean(ax, ygrid=False)
    ax.xaxis.grid(True, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=8.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "04_feature_importance.png", bbox_inches="tight")
    plt.close(fig)


def chart_fraud_over_time():
    """Does the fraud rate move? And where does the test period begin?"""
    from data import loaders
    from validation.splits import chronological_split

    df = loaders.load_card()[["TransactionDT", "isFraud"]]
    split = chronological_split(df, test_fraction=0.20)
    day = df.TransactionDT / 86400
    weekly = df.assign(week=(day // 7).astype(int)).groupby("week").isFraud.agg(["mean", "size"])
    weekly = weekly[weekly["size"] > 500]
    cut_week = split.cut_time / 86400 / 7

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(weekly.index, weekly["mean"] * 100, color=BLUE, linewidth=2, zorder=3)
    ax.axvline(cut_week, color=ORANGE, linewidth=2, linestyle="--", zorder=2)
    ax.annotate("train  |  test", (cut_week, ax.get_ylim()[1] * 0.94),
                xytext=(6, 0), textcoords="offset points",
                fontsize=9, color=ORANGE, fontweight="bold")
    ax.set_xlabel("Week")
    ax.set_ylabel("Fraud rate (%)")
    ax.set_title("Fraud rate over time")
    ax.text(0.5, -0.19,
            "The rate is not stable, which is why the split is chronological:\n"
            "a random split would let the model train on the period it is scored on.",
            transform=ax.transAxes, ha="center", fontsize=9, color=INK_SOFT)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "05_fraud_over_time.png", bbox_inches="tight")
    plt.close(fig)


def chart_progress():
    """This project against the previous attempt, on the same source."""
    labels = ["Previous project\n(autoencoder + fusion)", "This project\n(LightGBM)"]
    values = [0.0730, 0.5865]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    bars = ax.bar(labels, values, width=0.5, color=[ORANGE, BLUE])
    for bar, v in zip(bars, values):
        ax.annotate(f"{v:.4f}", (bar.get_x() + bar.get_width() / 2, v),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=12, fontweight="bold", color=INK)
    ax.set_ylabel("PR-AUC")
    ax.set_title("IEEE-CIS, same data, same held-out period")
    ax.text(0.5, -0.16, "8.0x better. The difference is the V columns, not the model family.",
            transform=ax.transAxes, ha="center", fontsize=9, color=INK_SOFT)
    ax.set_ylim(0, 0.68)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "06_progress.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    scores = np.load(cfg.INTERIM / "card_scores_lightgbm.npy")
    y_test = np.load(cfg.INTERIM / "card_y_test.npy")

    chart_tradeoff(y_test, scores)
    chart_seed_spread()
    chart_policy_bands(y_test, scores)
    chart_feature_importance()
    chart_progress()
    chart_fraud_over_time()

    for path in sorted(FIGURES.glob("*.png")):
        print(f"  {path.name:32s} {path.stat().st_size / 1024:>7.0f} KB")
    print(f"\nwritten to {FIGURES}")


if __name__ == "__main__":
    main()
