"""
Dual-bottleneck / error-decomposition summary figure for Section 6.1
(fig:error_decomposition_summary).

Self-contained: all numbers are hard-coded from Table 5.4
(tab:latent_edm_results), so this script has NO project dependencies and can be
run anywhere with matplotlib.

Two panels:
  LEFT  - step-1 RMSE, stacked into:
            lower segment = diffusion prediction error (RMSE vs D(E(x)))
            upper segment = autoencoder reconstruction bias
                            (= RMSE vs x  minus  RMSE vs D(E(x)))
          for x2 (hd32), x4 (hd32), x4 (hd128).
  RIGHT - step-50 RMSE for the same three configs, with the pixel-space EDM
          baseline (5.88) drawn as a dashed reference line.

The figure visualises the "dual bottleneck": autoencoder reconstruction bias is
the differentiator at step 1 (early horizon), while model capacity / compression
loss governs step 50 (late horizon).

Usage:
    python plot_dual_bottleneck.py --out figures/error_decomposition_summary.pdf
"""

import os
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --- Numbers from Table 5.4 (tab:latent_edm_results) ---
# step-1 RMSE vs x  and  vs D(E(x)):
#   x2  (hd32):  1.05 -> 0.79
#   x4  (hd32):  1.57 -> 1.00
#   x4  (hd128): 1.53 -> 0.94
# step-50 RMSE (vs x):
#   x2 hd32 = 7.24,  x4 hd32 = 7.36,  x4 hd128 = 6.81
# pixel baseline RMSE@50 = 5.88
LABELS = [r"$2\times$" "\n(hd32)", r"$4\times$" "\n(hd32)", r"$4\times$" "\n(hd128)"]
STEP1_TOTAL = [1.05, 1.57, 1.53]      # RMSE vs x
STEP1_DIFF = [0.79, 1.00, 0.94]       # RMSE vs D(E(x))  (diffusion error)
STEP50 = [7.24, 7.36, 6.81]
PIXEL_BASELINE_50 = 5.88

# Colours (consistent with the thesis figures / defense deck)
NAVY = "#1E2C4F"
TEAL = "#2C7A7B"
AMBER = "#C8772E"
GREY = "#9AA6B2"
SLATE = "#333B47"

FS_TITLE = 14
FS_LABEL = 13
FS_TICK = 12
FS_ANNOT = 11


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/error_decomposition_summary.pdf")
    args = ap.parse_args()

    ae_bias = [t - d for t, d in zip(STEP1_TOTAL, STEP1_DIFF)]
    x = range(len(LABELS))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    fig.subplots_adjust(wspace=0.28, left=0.08, right=0.97, top=0.88, bottom=0.13)

    # ---------- LEFT: step-1 stacked decomposition ----------
    ax1.bar(x, STEP1_DIFF, width=0.6, color=TEAL,
            label="Diffusion prediction error  (vs $\\mathcal{D}(\\mathcal{E}(x))$)")
    ax1.bar(x, ae_bias, width=0.6, bottom=STEP1_DIFF, color=AMBER,
            label="Autoencoder reconstruction bias")
    for i, tot in enumerate(STEP1_TOTAL):
        ax1.text(i, tot + 0.03, f"{tot:.2f}", ha="center", va="bottom",
                 fontsize=FS_ANNOT, fontweight="bold", color=NAVY)
    ax1.set_title("Step-1 RMSE: early-horizon decomposition",
                  fontsize=FS_TITLE, fontweight="bold", color=NAVY, pad=10)
    ax1.set_ylabel("RMSE (pixel space)", fontsize=FS_LABEL, color=SLATE)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(LABELS, fontsize=FS_LABEL, color=SLATE)
    ax1.set_ylim(0, 1.95)
    ax1.legend(loc="upper left", fontsize=9.5, frameon=False)
    ax1.tick_params(labelsize=FS_TICK, colors=SLATE)

    # ---------- RIGHT: step-50 RMSE vs baseline ----------
    colours = [GREY, GREY, NAVY]
    ax2.bar(x, STEP50, width=0.6, color=colours)
    for i, v in enumerate(STEP50):
        ax2.text(i, v + 0.06, f"{v:.2f}", ha="center", va="bottom",
                 fontsize=FS_ANNOT, fontweight="bold", color=NAVY)
    ax2.axhline(PIXEL_BASELINE_50, color=TEAL, lw=2, ls="--")
    ax2.text(-0.45, PIXEL_BASELINE_50 + 0.08,
             f"Pixel EDM baseline ({PIXEL_BASELINE_50:.2f})",
             color=TEAL, fontsize=FS_ANNOT, va="bottom", ha="left")
    ax2.set_title("Step-50 RMSE: late-horizon performance",
                  fontsize=FS_TITLE, fontweight="bold", color=NAVY, pad=10)
    ax2.set_ylabel("RMSE (pixel space)", fontsize=FS_LABEL, color=SLATE)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(LABELS, fontsize=FS_LABEL, color=SLATE)
    ax2.set_ylim(0, 8.6)
    ax2.set_xlim(-0.6, 2.6)
    ax2.tick_params(labelsize=FS_TICK, colors=SLATE)

    for ax in (ax1, ax2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color(GREY)
        ax.spines["bottom"].set_color(GREY)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved -> {args.out}")
    print("Step-1 totals (vs x):", STEP1_TOTAL)
    print("Step-1 diffusion (vs D(E(x))):", STEP1_DIFF)
    print("Step-1 AE bias (difference):", [round(b, 2) for b in ae_bias])
    print("Step-50 RMSE:", STEP50, " | pixel baseline:", PIXEL_BASELINE_50)


if __name__ == "__main__":
    main()
