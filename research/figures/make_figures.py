"""Generate the figures for the TAF-from-NWP review (research/).

Encodes the measured results from the model-development + autonomous-research sessions
(frozen-2025 test; matched 9 standard leads unless noted) and renders each figure as both
PNG (for the Markdown review) and PDF (for the LaTeX/paper build).

    uv run python research/figures/make_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# --- Fig 1: model comparison (BSS, matched 9 leads, same eval population) -------
def fig_model_comparison():
    models = ["official TAF", "LSTM", "gbm", "TFT", "MLP", "linear\nblend",
              "{MLP,gbm}", "stacked\n{MLP,gbm,TFT}"]
    bss = [-0.108, 0.248, 0.273, 0.287, 0.316, 0.334, 0.337, 0.348]
    colors = ["#b03030"] + ["#6f9fd8"] * 4 + ["#d8a23a"] * 2 + ["#2e8b57"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(models, bss, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Brier Skill Score (vs climatology)")
    ax.set_title("Adverse-event forecast skill — frozen 2025, 9 standard leads")
    for b, v in zip(bars, bss):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.012 if v >= 0 else -0.03),
                f"{v:+.3f}", ha="center", fontsize=8.5)
    ax.set_ylim(-0.18, 0.40)
    save(fig, "fig1_model_comparison")


# --- Fig 2: reliability / calibration of the stacked ensemble -------------------
def fig_reliability():
    mean_pred = [0.005, 0.067, 0.132, 0.244, 0.424, 0.608, 0.859]
    obs_freq = [0.005, 0.066, 0.145, 0.236, 0.426, 0.613, 0.864]
    n = [91285, 7224, 2991, 3195, 1067, 1261, 583]
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax.plot(mean_pred, obs_freq, "o-", color="#2e8b57", label="stacked ensemble")
    for x, y, c in zip(mean_pred, obs_freq, n):
        ax.annotate(f"n={c:,}", (x, y), textcoords="offset points", xytext=(6, -10), fontsize=7)
    ax.set_xlabel("Mean forecast probability"); ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability — stacked {MLP,gbm,TFT}")
    ax.legend(loc="upper left"); ax.set_xlim(0, 0.95); ax.set_ylim(0, 0.95)
    save(fig, "fig2_reliability")


# --- Fig 3: skill vs lead time --------------------------------------------------
def fig_skill_vs_lead():
    leads = [1, 2, 3, 6, 9, 12, 18, 24, 30]
    bss = [0.503, 0.419, 0.382, 0.323, 0.318, 0.269, 0.312, 0.276, 0.295]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(leads, bss, "o-", color="#2e8b57", label="stacked ensemble")
    ax.axhline(-0.108, color="#b03030", ls="--", lw=1, label="official TAF (pooled)")
    ax.axhline(0, color="k", lw=0.8, label="climatology")
    ax.set_xlabel("Lead time (h)"); ax.set_ylabel("Brier Skill Score")
    ax.set_title("Skill vs lead time"); ax.legend(); ax.set_ylim(-0.18, 0.56)
    save(fig, "fig3_skill_vs_lead")


# --- Fig 4: data-scaling curve (TFT, all horizons) ------------------------------
def fig_scaling():
    pct = [25, 40, 60, 100]
    bss = [0.189, 0.229, 0.271, 0.271]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(pct, bss, "o-", color="#d8a23a")
    ax.annotate("plateau", (80, 0.272), fontsize=9, color="#7a5c12")
    ax.set_xlabel("Training sample (% of issue-hours)")
    ax.set_ylabel("BSS (all 30 horizons)")
    ax.set_title("TFT data-scaling — improves then saturates by ~60%")
    ax.set_ylim(0.16, 0.30)
    save(fig, "fig4_scaling")


# --- Fig 5: per-region skill ----------------------------------------------------
def fig_region():
    reg = ["Canaries", "Peninsula", "Balearics", "Melilla"]
    bss = [0.451, 0.331, 0.213, 0.127]
    rate = [0.034, 0.040, 0.018, 0.036]
    fig, ax = plt.subplots(figsize=(6.8, 4))
    bars = ax.bar(reg, bss, color="#2e8b57")
    for b, v, r in zip(bars, bss, rate):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:+.3f}\n(rate {r:.1%})",
                ha="center", fontsize=8)
    ax.set_ylabel("BSS (9 leads)"); ax.set_title("Skill by region (stacked ensemble)")
    ax.set_ylim(0, 0.52)
    save(fig, "fig5_region")


# --- Fig 6: stack ablation (subset stacks) --------------------------------------
def fig_ablation():
    subsets = ["LSTM", "gbm", "TFT", "MLP", "MLP+gbm", "linear blend\nMLP+TFT",
               "MLP+gbm+TFT", "MLP+gbm+TFT+LSTM"]
    bss = [0.248, 0.273, 0.287, 0.316, 0.337, 0.334, 0.348, 0.348]
    colors = ["#6f9fd8"] * 4 + ["#d8a23a"] * 2 + ["#2e8b57"] * 2
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.barh(subsets, bss, color=colors)
    for b, v in zip(bars, bss):
        ax.text(v + 0.004, b.get_y() + b.get_height() / 2, f"{v:+.3f}", va="center", fontsize=8.5)
    ax.set_xlabel("BSS (9 leads)"); ax.set_title("Ensemble ablation — LSTM is redundant")
    ax.set_xlim(0, 0.40)
    save(fig, "fig6_ablation")


if __name__ == "__main__":
    fig_model_comparison(); fig_reliability(); fig_skill_vs_lead()
    fig_scaling(); fig_region(); fig_ablation()
    print("wrote figures to", OUT)
