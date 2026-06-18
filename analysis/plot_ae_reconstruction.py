"""
Visualize autoencoder reconstruction quality at different compression rates.

Generates:
  1. Side-by-side comparison: Original | x2 | x4 | x8 | x16 reconstruction
  2. Error maps: |Original - Reconstruction| for each compression rate
  3. Per channel (Level 0 and Level 1)

Usage:
    python plot_ae_reconstruction.py
    python plot_ae_reconstruction.py --sample_idx 5 --timestep 50
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from networks.autoencoder import get_autoencoder
from data.sqg_dataloader import SQGDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAT_CHANNELS_MAP = {2: 16, 4: 8, 8: 4, 16: 2}
CHANNEL_NAMES = ["Level 0 (upper PV)", "Level 1 (lower PV)"]

# --- Unified styling across all figures ---
# PV is a signed field -> diverging colormap, symmetric about zero.
CMAP = "RdBu_r"      # PV fields
CMAP_ERR = "Reds"    # absolute-error maps (non-negative, sequential)
# Font sizes: kept at least as large as the 11pt thesis body text.
FS_TITLE = 15        # panel titles (column headers, per-panel labels)
FS_LABEL = 15        # row labels
FS_SUP = 16          # figure suptitle
FS_CBAR = 13         # colorbar label
FS_TICK = 12         # colorbar tick labels


def load_ae(compression):
    """Load trained autoencoder."""
    lat_ch = LAT_CHANNELS_MAP[compression]
    ae = get_autoencoder(
        pix_channels=2, lat_channels=lat_ch, spatial=2,
        arch="dcae", saturation="softclip2",
        hid_channels=(64, 128, 256), hid_blocks=(3, 3, 3),
        periodic=True, identity_init=True,
    )
    weight_path = f"saved_models/ae_x{compression}/best.pth"
    ae.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    ae = ae.to(DEVICE)
    ae.eval()
    return ae


def plot_reconstruction_comparison(original, reconstructions, channel, save_path, 
                                    sample_idx, timestep):
    """
    Plot original vs all compression rates side by side.
    
    original: (H, W) numpy array
    reconstructions: dict {compression: (H, W) numpy array}
    """
    compressions = sorted(reconstructions.keys())
    n_cols = 1 + len(compressions)  # original + reconstructions

    # Make room on the right for a dedicated colorbar axis.
    # The leftmost column is the original; each subsequent column is a reconstruction.
    fig = plt.figure(figsize=(4 * n_cols + 0.6, 4.0))
    gs = fig.add_gridspec(
        nrows=1, ncols=n_cols + 1,
        width_ratios=[1.0] * n_cols + [0.04],
        wspace=0.05,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_cols)]
    cax  = fig.add_subplot(gs[0, n_cols])

    # Shared symmetric colorbar range centered on zero (PV is a signed field)
    vmax = float(np.abs(original).max())
    vmin = -vmax

    # Plot original
    im = axes[0].imshow(original, cmap=CMAP, vmin=vmin, vmax=vmax, origin="lower")
    axes[0].set_title("Original", fontsize=FS_TITLE, fontweight="bold")
    axes[0].set_xticks([]); axes[0].set_yticks([])
    for s in axes[0].spines.values(): s.set_visible(False)

    # Plot each reconstruction
    for i, comp in enumerate(compressions):
        recon = reconstructions[comp]
        axes[i + 1].imshow(recon, cmap=CMAP, vmin=vmin, vmax=vmax, origin="lower")
        rmse = np.sqrt(np.mean((original - recon) ** 2))
        axes[i + 1].set_title(f"$\\times${comp}\nRMSE = {rmse:.4f}", fontsize=FS_TITLE)
        axes[i + 1].set_xticks([]); axes[i + 1].set_yticks([])
        for s in axes[i + 1].spines.values(): s.set_visible(False)

    # Dedicated colorbar in its own axis (not overlapping the panels)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("PV", fontsize=FS_CBAR)
    cbar.ax.tick_params(labelsize=FS_TICK)

    fig.suptitle(f"{CHANNEL_NAMES[channel]} — Sample {sample_idx}, Timestep {timestep}",
                 fontsize=FS_SUP, fontweight="bold", y=1.08)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_error_maps(original, reconstructions, channel, save_path,
                    sample_idx, timestep):
    """
    Plot absolute error maps |original - reconstruction| for each compression rate.
    """
    compressions = sorted(reconstructions.keys())
    n_cols = len(compressions)

    # Dedicated right-side colorbar axis, same approach as the reconstruction figure.
    fig = plt.figure(figsize=(4 * n_cols + 0.6, 4.0))
    gs = fig.add_gridspec(
        nrows=1, ncols=n_cols + 1,
        width_ratios=[1.0] * n_cols + [0.04],
        wspace=0.05,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_cols)]
    cax  = fig.add_subplot(gs[0, n_cols])

    # Shared error colorbar range (errors are non-negative, sequential cmap)
    max_err = max(np.abs(original - reconstructions[c]).max() for c in compressions)

    for i, comp in enumerate(compressions):
        error = np.abs(original - reconstructions[comp])
        im = axes[i].imshow(error, cmap=CMAP_ERR, vmin=0, vmax=max_err, origin="lower")
        rmse = np.sqrt(np.mean((original - reconstructions[comp]) ** 2))
        axes[i].set_title(f"$\\times${comp}\nRMSE = {rmse:.4f}", fontsize=FS_TITLE)
        axes[i].set_xticks([]); axes[i].set_yticks([])
        for s in axes[i].spines.values(): s.set_visible(False)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("|error|", fontsize=FS_CBAR)
    cbar.ax.tick_params(labelsize=FS_TICK)

    fig.suptitle(f"Reconstruction error — {CHANNEL_NAMES[channel]} — Sample {sample_idx}, Timestep {timestep}",
                 fontsize=FS_SUP, fontweight="bold", y=1.08)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_multi_timestep(original_traj, recon_trajs, channel, timesteps, 
                        save_path, sample_idx):
    """
    Plot original and reconstructions across multiple timesteps.
    Rows: Original, then one compression ratio per row (x2, x4, x8, x16).
    Columns: different timesteps.
    """
    compressions = sorted(recon_trajs.keys())
    n_rows = 1 + len(compressions)
    n_cols = len(timesteps)

    # Larger local font sizes for this multi-panel grid only (the other figures
    # keep the global FS_* sizes).
    FS_TITLE_MT = 24   # column titles (t = ...)
    FS_LABEL_MT = 24   # row labels (compression ratio)
    FS_SUP_MT = 26     # figure suptitle

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))
    
    # Symmetric color range centered on zero (consistent with the other figures)
    vmax = float(np.abs(original_traj[:, channel]).max())
    vmin = -vmax
    
    row_labels = ["Original"] + [rf"$\times${c} reconstruction" for c in compressions]
    
    for col, t in enumerate(timesteps):
        # Original (top row)
        axes[0, col].imshow(original_traj[t, channel], cmap=CMAP, 
                           vmin=vmin, vmax=vmax, origin="lower")
        axes[0, col].set_title(f"t = {t}", fontsize=FS_TITLE_MT, pad=10)
        
        # Each compression ratio on its own row
        for row, comp in enumerate(compressions):
            axes[row + 1, col].imshow(recon_trajs[comp][t, channel], cmap=CMAP,
                                      vmin=vmin, vmax=vmax, origin="lower")
    
    # Hide ticks but KEEP the axes so the row labels stay visible.
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Row labels (one compression ratio per row), placed to the left of column 0.
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=FS_LABEL_MT, fontweight="bold",
                                rotation=90, labelpad=16, va="center")
    
    fig.suptitle(f"{CHANNEL_NAMES[channel]} — Reconstruction across time (Sample {sample_idx})",
                 fontsize=FS_SUP_MT, fontweight="bold", y=0.97)
    # Roomy, error-map-like margins. Fixed (no tight_layout / no bbox_inches="tight")
    # so the larger row labels are never clipped and the spacing stays consistent
    # with the reconstruction-error figures.
    fig.subplots_adjust(left=0.115, right=0.99, top=0.92, bottom=0.02,
                        wspace=0.05, hspace=0.05)
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize AE reconstruction")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Which test sample to visualize (default: 0)")
    parser.add_argument("--timestep", type=int, default=50,
                        help="Which timestep to visualize (default: 50)")
    parser.add_argument("--compression", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--save_dir", type=str, default="results/ae_eval")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Load test data (standardized)
    dataset = SQGDataset("data/SQG/dataset/test", stats_dir="data/SQG/dataset/train")
    
    # Get a single sample
    x_standardized = dataset[args.sample_idx].unsqueeze(0).to(DEVICE)  # (1, 2, 64, 64)
    original_np = x_standardized[0].cpu().numpy()  # (2, 64, 64)

    print(f"Sample {args.sample_idx}, standardized range: [{original_np.min():.3f}, {original_np.max():.3f}]")

    # Reconstruct with each AE
    reconstructions = {}  # {compression: (2, 64, 64) numpy}
    
    for comp in args.compression:
        ae = load_ae(comp)
        with torch.no_grad():
            y, z = ae(x_standardized)
        reconstructions[comp] = y[0].cpu().numpy()
        print(f"  x{comp}: latent shape {tuple(z.shape)}, recon RMSE = {np.sqrt(np.mean((original_np - reconstructions[comp])**2)):.4f}")
        del ae
        torch.cuda.empty_cache()

    # === Plot 1: Side-by-side reconstruction comparison (per channel) ===
    for ch in range(2):
        recon_ch = {comp: reconstructions[comp][ch] for comp in args.compression}
        
        save_path = os.path.join(args.save_dir, f"reconstruction_ch{ch}_t{args.timestep}.png")
        plot_reconstruction_comparison(
            original_np[ch], recon_ch, ch, save_path,
            args.sample_idx, args.timestep
        )
        print(f"Saved: {save_path}")

    # === Plot 2: Error maps (per channel) ===
    for ch in range(2):
        recon_ch = {comp: reconstructions[comp][ch] for comp in args.compression}
        
        save_path = os.path.join(args.save_dir, f"error_map_ch{ch}_t{args.timestep}.png")
        plot_error_maps(
            original_np[ch], recon_ch, ch, save_path,
            args.sample_idx, args.timestep
        )
        print(f"Saved: {save_path}")

    # === Plot 3: Multi-timestep comparison ===
    # Load a full trajectory and reconstruct multiple timesteps
    print("\nGenerating multi-timestep comparison...")
    
    # Load raw trajectory file for multiple timesteps
    import glob
    test_files = sorted(glob.glob("data/SQG/dataset/test/sqg_N64_3hrly_*.npy"))
    if args.sample_idx < len(test_files):
        raw_traj = np.load(test_files[args.sample_idx]).astype(np.float32)  # (101, 2, 64, 64)
        
        # Standardize
        stats_dir = "data/SQG/dataset/train"
        data_mean = torch.load(os.path.join(stats_dir, "data_mean.pt"), weights_only=True).numpy()
        data_std = torch.load(os.path.join(stats_dir, "data_std.pt"), weights_only=True).numpy()
        traj_standardized = (raw_traj - data_mean.reshape(1, -1, 1, 1)) / data_std.reshape(1, -1, 1, 1)
        
        # Reconstruct full trajectory with each AE
        timesteps_to_show = [0, 25, 50, 75, 100]
        recon_trajs = {}
        
        for comp in args.compression:
            ae = load_ae(comp)
            with torch.no_grad():
                traj_tensor = torch.tensor(traj_standardized, dtype=torch.float32).to(DEVICE)
                # Encode/decode in batches
                recon_list = []
                for start in range(0, len(traj_tensor), 32):
                    end = min(start + 32, len(traj_tensor))
                    y, _ = ae(traj_tensor[start:end])
                    recon_list.append(y.cpu().numpy())
                recon_trajs[comp] = np.concatenate(recon_list, axis=0)
            del ae
            torch.cuda.empty_cache()
        
        for ch in range(2):
            save_path = os.path.join(args.save_dir, f"multi_timestep_ch{ch}.png")
            plot_multi_timestep(
                traj_standardized, recon_trajs, ch, timesteps_to_show,
                save_path, args.sample_idx
            )
            print(f"Saved: {save_path}")

    print(f"\nAll plots saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
