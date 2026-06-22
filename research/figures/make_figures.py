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


# --- Fig 7: parsing & normalization pipeline -----------------------------------
def fig_pipeline():
    import matplotlib.patches as mp
    fig, ax = plt.subplots(figsize=(10, 3.4)); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 3)
    boxes = [
        (0.1, "Raw METAR / TAF\ntext", "#eef2f7",
         "METAR LEMD 0600Z\n24008KT 0500 FG\nOVC002 ..."),
        (2.6, "Parse\n(metar-taf-parser)", "#dfeaf5",
         "wind, visibility,\nclouds, weather,\ntemp/dew"),
        (5.1, "Normalize → SI\n(NormalizedConditions)", "#dff0e6",
         "m/s→kt; 9999→10 km;\nCAVOK; ceiling = lowest\nBKN/OVC/OVX (ft)"),
        (7.6, "Derive flight\ncategory", "#e9f7df",
         "worst-of(ceiling, vis)\nLIFR/IFR/MVFR/VFR"),
    ]
    for x, title, col, sub in boxes:
        ax.add_patch(mp.FancyBboxPatch((x, 1.15), 2.0, 1.2, boxstyle="round,pad=0.04",
                     fc=col, ec="#333"))
        ax.text(x + 1.0, 2.05, title, ha="center", va="center", fontsize=9.5, weight="bold")
        ax.text(x + 1.0, 1.45, sub, ha="center", va="center", fontsize=7.2, family="monospace")
        if x > 0.2:
            ax.annotate("", (x - 0.05, 1.75), (x - 0.5, 1.75),
                        arrowprops=dict(arrowstyle="-|>", color="#333"))
    ax.text(5, 0.5, "Identical normalization for METAR observations and TAF trend groups "
            "→ a common canonical state", ha="center", fontsize=8.5, style="italic")
    save(fig, "fig7_pipeline")


# --- Fig 8: flight-category = worst-of(ceiling, visibility) grid ----------------
def fig_category_grid():
    import numpy as np
    order = ["LIFR", "IFR", "MVFR", "VFR"]
    rank = {c: i for i, c in enumerate(order)}
    ceil_lab = ["<500", "500–1000", "1000–3000", "≥3000"]      # ft, low→high
    vis_lab = ["<1600", "1600–4800", "4800–8000", "≥8000"]     # m
    by_ceil = ["LIFR", "IFR", "MVFR", "VFR"]
    by_vis = ["LIFR", "IFR", "MVFR", "VFR"]
    grid = np.zeros((4, 4))
    labels = [[None] * 4 for _ in range(4)]
    for i in range(4):           # ceiling band (row, top = highest ceiling)
        for j in range(4):       # vis band (col)
            worst = min(by_ceil[i], by_vis[j], key=rank.get)
            grid[3 - i, j] = rank[worst]
            labels[3 - i][j] = worst
    cmap = matplotlib.colors.ListedColormap(["#7b3294", "#c2549d", "#fdae61", "#a6d96a"])
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=3, aspect="auto")
    for r in range(4):
        for c in range(4):
            ax.text(c, r, labels[r][c], ha="center", va="center", fontsize=10,
                    weight="bold", color="white" if grid[r, c] < 2 else "black")
    ax.set_xticks(range(4), vis_lab); ax.set_yticks(range(4), ceil_lab[::-1])
    ax.set_xlabel("Visibility band (m)"); ax.set_ylabel("Ceiling band (ft, BKN/OVC)")
    ax.set_title("Flight category = worst-of(ceiling, visibility)")
    save(fig, "fig8_category_grid")


# --- Fig 9: TAF expansion to an hourly timeline + obs alignment -----------------
def fig_taf_timeline():
    import matplotlib.patches as mp
    fig, ax = plt.subplots(figsize=(10, 4.2)); ax.axis("off")
    ax.set_xlim(0, 12); ax.set_ylim(0, 4.2)
    # TAF groups row
    ax.text(-0.0, 3.9, "TAF groups", fontsize=9, weight="bold")
    ax.add_patch(mp.Rectangle((0.5, 3.3), 11, 0.5, fc="#dfeaf5", ec="#333"))
    ax.text(1.2, 3.55, "BASE", fontsize=8)
    ax.annotate("FM 09Z", (4.5, 3.55), fontsize=7.5, ha="center")
    ax.plot([4.5, 4.5], [3.3, 3.8], color="#b03030", lw=1.5)
    ax.add_patch(mp.Rectangle((6.0, 3.3), 2.5, 0.5, fc="#f5d6d6", ec="#b03030", hatch="//"))
    ax.text(7.25, 3.55, "PROB30 TEMPO  0500 FG", fontsize=7, ha="center")
    # hourly expansion row
    ax.text(-0.0, 2.55, "Hourly\nExpectedHour", fontsize=9, weight="bold")
    cats = ["VFR", "VFR", "VFR", "MVFR", "MVFR", "IFR", "IFR", "VFR", "VFR", "VFR", "VFR"]
    cmap = {"VFR": "#a6d96a", "MVFR": "#fdae61", "IFR": "#c2549d", "LIFR": "#7b3294"}
    for k, c in enumerate(cats):
        ax.add_patch(mp.Rectangle((1.0 + k, 2.1), 0.9, 0.6, fc=cmap[c], ec="#333"))
        ax.text(1.45 + k, 2.4, c, ha="center", va="center", fontsize=6.5)
        if 5 <= k <= 7:   # tempo/prob overlay marker
            ax.add_patch(mp.Rectangle((1.0 + k, 2.1), 0.9, 0.6, fill=False, ec="#b03030",
                         lw=1.4, hatch="//"))
    # obs row
    ax.text(-0.0, 1.15, "Nearest METAR\n(±40 min)", fontsize=9, weight="bold")
    obs = ["VFR", "VFR", "VFR", "VFR", "MVFR", "IFR", "MVFR", "VFR", "VFR", "VFR", "VFR"]
    for k, c in enumerate(obs):
        ax.add_patch(mp.Circle((1.45 + k, 1.0), 0.28, fc=cmap[c], ec="#333"))
        ax.text(1.45 + k, 1.0, c, ha="center", va="center", fontsize=5.6)
        ax.annotate("", (1.45 + k, 1.32), (1.45 + k, 2.08),
                    arrowprops=dict(arrowstyle="-", color="#999", lw=0.6))
    ax.text(6, 0.35, "Each forecast hour is scored against the aligned observation "
            "(hit/miss/false-alarm/correct-neg + element errors)", ha="center",
            fontsize=8.3, style="italic")
    save(fig, "fig9_taf_timeline")


# --- Fig 10: per-hour scoring -> combined metrics -------------------------------
def fig_scoring():
    import matplotlib.patches as mp
    fig, ax = plt.subplots(figsize=(10, 4.6)); ax.axis("off")
    ax.set_xlim(0, 12); ax.set_ylim(0, 5)
    # contingency 2x2 (left)
    ax.text(2.4, 4.7, "Categorical: IFR-or-worse event", fontsize=9.5, weight="bold", ha="center")
    cells = {(0, 1): ("Hit", "#a6d96a"), (1, 1): ("Miss", "#fdae61"),
             (0, 0): ("False\nalarm", "#fdae61"), (1, 0): ("Correct\nneg", "#a6d96a")}
    for (cx, cy), (lab, col) in cells.items():
        ax.add_patch(mp.Rectangle((0.8 + cx * 1.6, 2.4 + cy * 1.4), 1.5, 1.3, fc=col, ec="#333"))
        ax.text(0.8 + cx * 1.6 + 0.75, 2.4 + cy * 1.4 + 0.65, lab, ha="center", va="center", fontsize=8.5)
    ax.text(0.4, 3.7, "Obs\nadverse", fontsize=8, ha="center", rotation=90, va="center")
    ax.text(2.4, 1.9, "Forecast adverse →", fontsize=8, ha="center")
    ax.text(2.4, 1.4, "→ HSS, POD, FAR, CSI", fontsize=9, ha="center", weight="bold")
    # probabilistic (middle)
    ax.text(6.4, 4.7, "Probabilistic", fontsize=9.5, weight="bold", ha="center")
    ax.text(6.4, 3.9, "adverse_probability(hour):", fontsize=8, ha="center", family="monospace")
    ax.text(6.4, 3.4, "prevailing adverse → 1.0\nPROB30 → 0.3 · PROB40/TEMPO → 0.4\nelse 0.0",
            fontsize=7.6, ha="center", family="monospace")
    ax.text(6.4, 2.3, "Brier = mean (p − event)²", fontsize=8.5, ha="center")
    ax.text(6.4, 1.4, "→ BSS = 1 − BS/BS_clim", fontsize=9, ha="center", weight="bold")
    # weighted score (right)
    ax.text(10.2, 4.7, "Weighted per-hour score", fontsize=9.5, weight="bold", ha="center")
    rows = ["prevailing exact  → 3.0", "TEMPO captures obs → 2.0",
            "PROB≥40 captures  → 1.5", "PROB/off-by-one  → 1.0", "else → 0.0"]
    for i, r in enumerate(rows):
        ax.text(10.2, 3.9 - i * 0.42, r, fontsize=7.8, ha="center", family="monospace")
    ax.text(10.2, 1.4, "→ mean weighted score", fontsize=9, ha="center", weight="bold")
    # promotion note
    ax.add_patch(mp.FancyBboxPatch((1.0, 0.15), 10, 0.7, boxstyle="round,pad=0.05",
                 fc="#eef2f7", ec="#333"))
    ax.text(6, 0.5, "Promotion target: HSS (categorical, primary) with BSS (probabilistic); "
            "element MAE + weighted score are diagnostics", ha="center", fontsize=8.3, style="italic")
    save(fig, "fig10_scoring")


if __name__ == "__main__":
    fig_model_comparison(); fig_reliability(); fig_skill_vs_lead()
    fig_scaling(); fig_region(); fig_ablation()
    fig_pipeline(); fig_category_grid(); fig_taf_timeline(); fig_scoring()
    print("wrote figures to", OUT)
