"""
Re-plot Figure 5.3 (pixel-space EDM ensemble forecast) WITHOUT the spurious
"Obs, sigma=0, p=0%" panel.

New top-row layout:  Truth | |Mean - Truth| | Ensemble Mean | Ensemble Std
Bottom row:          four individual ensemble members (with RMSE)

Reuses the exact data-loading, model-loading and rollout logic from
eval_latent_pipeline.py so it runs inside your existing environment.

Usage (on the remote machine, project root):
    python plot_fig53_ensemble.py \
        --pixel_edm_ckpt saved_models/EDM_Local_Run_MSE-EDM-32-02_25_03-2991/min_val_loss.ckpt \
        --step 30 \
        --ensemble_size 20 \
        --traj_idx 0 \
        --level 0 \
        --out figures/fig53_ensemble.pdf

Notes:
  --step is the 1-based rollout step to visualise (1..pred_steps). Whatever you
    pass here is the number you should cite in the caption.
  --ensemble_size should match what you used for the test results (20).
  Set --seed for reproducibility of the sampled members.
"""

import os
import sys
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- Reuse everything from the existing evaluation script ---
from eval.eval_latent_pipeline import (
    load_pixel_test_data,
    load_pixel_edm,
    predict_pixel_edm,
    DEVICE,
)
import data.SQG.constants as SGConstants  # for scalefact


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pixel_edm_ckpt", required=True,
                    help="Path to the pixel-space EDM checkpoint (EDM-32).")
    ap.add_argument("--data_path", default="data/SQG/dataset",
                    help="Pixel data root (same as eval_latent_pipeline.py).")
    ap.add_argument("--init_states", type=int, default=2)
    ap.add_argument("--pred_steps", type=int, default=50)
    ap.add_argument("--ensemble_size", type=int, default=20)
    ap.add_argument("--step", type=int, default=30,
                    help="1-based rollout step to visualise. CITE THIS in the caption.")
    ap.add_argument("--traj_idx", type=int, default=0,
                    help="Which test trajectory to plot.")
    ap.add_argument("--level", type=int, default=0,
                    help="PV vertical level / channel to display (0 or 1).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cmap", default="RdBu_r")
    ap.add_argument("--out", default="figures/fig53_ensemble.pdf")
    return ap.parse_args()


def rmse_phys(pred_field, truth_field, pixel_std_c, scalefact):
    """RMSE in physical units for a single (H,W) field of one channel,
    matching the de-standardisation used in compute_metrics()."""
    # pred_field, truth_field are standardised single-channel (H, W) tensors.
    mse = torch.mean((pred_field - truth_field) ** 2)
    return float(torch.sqrt(mse) * pixel_std_c * scalefact)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    assert 1 <= args.step <= args.pred_steps, \
        f"--step must be in [1, {args.pred_steps}]"

    scalefact = SGConstants.scalefact

    # ---- Load data (standardised) + stats ----
    print("Loading pixel-space test data...")
    test_data, pixel_mean, pixel_std = load_pixel_test_data(
        args.data_path, init_states=args.init_states, pred_length=args.pred_steps
    )
    print(f"  {len(test_data)} test trajectories loaded")

    # ---- Load pixel EDM ----
    pixel_edm = load_pixel_edm(args.pixel_edm_ckpt)

    # ---- Run ensemble rollout for the chosen trajectory only ----
    init_single = test_data[args.traj_idx][0].unsqueeze(0)  # (1, N_init, 2, 64, 64)
    target_std = test_data[args.traj_idx][1]                 # (pred_steps, 2, 64, 64) standardised

    print(f"  Rolling out {args.ensemble_size} members for {args.pred_steps} steps...")
    predictions, _ = predict_pixel_edm(
        pixel_edm, init_single, args.pred_steps, args.ensemble_size
    )
    # predictions: (1, ensemble_size, pred_steps, 2, 64, 64) standardised
    preds = predictions[0]  # (E, pred_steps, 2, 64, 64)

    # ---- Select the step and channel ----
    s = args.step - 1                      # 0-based index
    c = args.level
    pstd_c = pixel_std.view(-1)[c]         # scalar std for this channel

    members_std = preds[:, s, c]           # (E, 64, 64) standardised
    truth_std = target_std[s, c]           # (64, 64) standardised
    mean_std = members_std.mean(dim=0)     # (64, 64)
    std_map = members_std.std(dim=0)       # (64, 64)
    abs_err = torch.abs(mean_std - truth_std)  # (64, 64) standardised units

    # ---- Convert displayed fields to physical units ----
    def to_phys(field_std):
        return ((field_std * pstd_c) + pixel_mean.view(-1)[c]) * scalefact

    truth_phys = to_phys(truth_std).numpy()
    mean_phys = to_phys(mean_std).numpy()
    members_phys = [to_phys(members_std[m]).numpy() for m in range(members_std.shape[0])]
    # error map in physical units (a scale factor on the standardised abs error)
    err_phys = (abs_err * pstd_c * scalefact).numpy()
    std_phys = (std_map * pstd_c * scalefact).numpy()

    # ---- RMSE (physical units) for mean and the 4 displayed members ----
    mean_rmse = rmse_phys(mean_std, truth_std, pstd_c, scalefact)
    member_rmses = [rmse_phys(members_std[m], truth_std, pstd_c, scalefact)
                    for m in range(4)]

    # symmetric color scale for the PV fields (centered on zero, matches RdBu_r
    # and the AE reconstruction figures)
    vmax = max(abs(truth_phys).max(), abs(mean_phys).max(),
               *[abs(m).max() for m in members_phys[:4]])
    vmin = -vmax

    # ---- Plot: 2 rows x 4 cols ----
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    FS_TITLE = 16   # >= 11pt thesis body text

    def show(ax, data, title, cmap=args.cmap, vmin=vmin, vmax=vmax):
        # No colorbars: this is a qualitative figure, and per-panel colorbars
        # would shrink only some panels and make the grid uneven.
        ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(title, fontsize=FS_TITLE)
        ax.set_xticks([]); ax.set_yticks([])

    # Top row: Truth | |Mean - Truth| | Mean | Std
    show(axes[0, 0], truth_phys, "Truth")
    # error map and std map are non-negative -> sequential "Reds" (consistent
    # with the reconstruction-error figures), each on its own auto-scaled range.
    show(axes[0, 1], err_phys, "|Mean \u2212 Truth|", cmap="Reds", vmin=0, vmax=err_phys.max())
    show(axes[0, 2], mean_phys, f"Mean ({mean_rmse:.3f})")
    show(axes[0, 3], std_phys, "Std", cmap="Reds", vmin=0, vmax=std_phys.max())

    # Bottom row: four members
    for m in range(4):
        show(axes[1, m], members_phys[m], f"Member {m+1} ({member_rmses[m]:.3f})")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01,
                        wspace=0.05, hspace=0.12)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"\nSaved figure to {args.out}")
    print(f"  Rollout step shown : {args.step}  (cite this in the caption)")
    print(f"  Trajectory index   : {args.traj_idx}")
    print(f"  PV level/channel   : {args.level}")
    print(f"  Ensemble size      : {args.ensemble_size}")
    print(f"  Mean RMSE          : {mean_rmse:.3f}")
    print(f"  Member RMSEs       : {[round(r,3) for r in member_rmses]}")


if __name__ == "__main__":
    main()
