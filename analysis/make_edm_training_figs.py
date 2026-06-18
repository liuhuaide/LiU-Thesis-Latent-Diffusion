"""
Generate two publication-quality training-dynamics figures for the EDM baseline,
from the W&B-exported CSVs.

Figure A (edm_training_val.pdf):
    2x2 panel (a-d):
      (a) val_mean_loss
      (b) val_rmse_0_step_1
      (c) val_rmse_1_step_1
      (d) val_mean_spsk_ratio

Figure B (edm_training_loss.pdf):
    Single panel: smoothed per-step train_loss (no raw shaded band).

X-axis is epoch (50 epochs total). Step <-> epoch is recovered from
the val CSVs (one validation point per epoch).
"""
import csv
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 用法:  python make_edm_training_figs.py [CSV_DIR] [OUT_DIR]
# 或设环境变量 EDM_FIG_CSV_DIR / EDM_FIG_OUT_DIR。默认读当前目录、写当前目录。
CSV_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("EDM_FIG_CSV_DIR", "."))
OUT_DIR = Path(sys.argv[2] if len(sys.argv) > 2 else os.environ.get("EDM_FIG_OUT_DIR", "."))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -- common style ----------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})
LINE_COLOR = "#1f4e79"   # academic deep blue
GRID_COLOR = "#dddddd"

# -- read a CSV ------------------------------------------------------
def read_csv(name, value_col=1):
    """Return (step_array, value_array). value_col=1 is the main metric column."""
    path = CSV_DIR / name
    steps, vals = [], []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            try:
                s = int(float(row[0]))
                v = float(row[value_col])
                steps.append(s)
                vals.append(v)
            except (ValueError, IndexError):
                pass
    return np.array(steps), np.array(vals)

# Build the step -> epoch map using val_mean_loss (50 evenly-spaced epochs)
val_steps, _ = read_csv("val_mean_loss.csv")
N_EPOCHS = len(val_steps)               # 50
# epoch index i (1..50) corresponds to step val_steps[i-1]
EPOCH_TO_STEP = val_steps               # length-50 array
STEP_TO_EPOCH = lambda s: np.interp(s, EPOCH_TO_STEP, np.arange(1, N_EPOCHS + 1))

def val_epoch_axis():
    return np.arange(1, N_EPOCHS + 1)

# -- moving-average smoothing for noisy train loss -------------------
def smooth(x, window):
    x = np.asarray(x, dtype=float)
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    # 'same' length, but edge values are biased; we trim the edges later
    return np.convolve(x, kernel, mode="same")

# =========================================================================
# Figure A: 2x2 validation panels
# =========================================================================
metrics = [
    ("val_mean_loss.csv",       "Validation loss",
                                "weighted MSE", "a"),
    ("val_rmse_0_step_1.csv",   "One-step validation RMSE  (upper PV, level 0)",
                                "RMSE", "b"),
    ("val_rmse_1_step_1.csv",   "One-step validation RMSE  (lower PV, level 1)",
                                "RMSE", "c"),
    ("val_mean_spsk_ratio.csv", "Mean spread-skill ratio",
                                "spread / skill", "d"),
]

fig, axes = plt.subplots(2, 2, figsize=(9.0, 5.6))
axes = axes.flatten()
epochs = val_epoch_axis()

for ax, (fname, title, ylabel, letter) in zip(axes, metrics):
    _, vals = read_csv(fname)
    ax.plot(epochs, vals, color=LINE_COLOR, linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", linestyle="-", linewidth=0.5, color=GRID_COLOR)
    ax.set_xlim(1, N_EPOCHS)
    # tag the subplot letter
    ax.text(0.02, 0.98, f"({letter})", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="left")

# y limits tuned per panel
axes[0].set_ylim(0.010, 0.035)   # val_mean_loss
axes[3].axhline(1.0, color="#888", linestyle="--", linewidth=0.8)  # SSR=1 reference
axes[3].set_ylim(0.95, 1.10)

plt.tight_layout()
out_a = OUT_DIR / "edm_training_val.pdf"
out_a_png = OUT_DIR / "edm_training_val.png"
fig.savefig(out_a, bbox_inches="tight")
fig.savefig(out_a_png, dpi=180, bbox_inches="tight")
plt.close(fig)
print("saved:", out_a)

# =========================================================================
# Figure B: smoothed per-step train loss
# =========================================================================
tr_steps, tr_vals = read_csv("train_loss_step.csv")

# Map steps to epochs (continuous)
tr_epochs = STEP_TO_EPOCH(tr_steps)

# Smooth: moving average over ~200 logged points (≈ 1 epoch since wandb log step≈16)
WINDOW = 200
tr_smooth = smooth(tr_vals, WINDOW)
# Trim half-window from each side to avoid edge bias
half = WINDOW // 2
tr_epochs_trim = tr_epochs[half:-half]
tr_smooth_trim = tr_smooth[half:-half]

fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.plot(tr_epochs_trim, tr_smooth_trim, color=LINE_COLOR, linewidth=1.6,
        label="smoothed (moving avg)")
ax.set_title("Per-step training loss (smoothed)")
ax.set_xlabel("Epoch")
ax.set_ylabel("weighted MSE")
ax.grid(True, which="both", linestyle="-", linewidth=0.5, color=GRID_COLOR)
ax.set_xlim(1, N_EPOCHS)
ax.set_ylim(0.0, 0.10)

plt.tight_layout()
out_b = OUT_DIR / "edm_training_loss.pdf"
out_b_png = OUT_DIR / "edm_training_loss.png"
fig.savefig(out_b, bbox_inches="tight")
fig.savefig(out_b_png, dpi=180, bbox_inches="tight")
plt.close(fig)
print("saved:", out_b)

# =========================================================================
# Report key numbers we'll need for the prose
# =========================================================================
print("\n--- numbers for the thesis text ---")
def first_last_avg(name):
    _, v = read_csv(name)
    return v[0], v[-1], float(np.mean(v[-5:]))
for f in ["val_mean_loss.csv", "val_rmse_0_step_1.csv",
         "val_rmse_1_step_1.csv", "val_mean_spsk_ratio.csv"]:
    a, b, c = first_last_avg(f)
    print(f"{f:30s} start={a:.4f}  final={b:.4f}  last-5-avg={c:.4f}")

# smoothed train_loss final
print(f"train_loss (smoothed) early epochs ~ {tr_smooth_trim[:200].mean():.4f},  "
      f"final {tr_smooth_trim[-200:].mean():.4f}")
