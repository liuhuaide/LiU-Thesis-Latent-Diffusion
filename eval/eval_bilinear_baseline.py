"""
Bilinear interpolation baseline for autoencoder evaluation.

This provides a lower bound: simply downsample the input with bilinear
interpolation and upsample back, with no learned parameters.

Usage:
    python eval_bilinear_baseline.py

Output:
    RMSE, VRMSE, LSD metrics for bilinear down/upsample (64→16→64)
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch.utils.data import DataLoader
from data.sqg_dataloader import SQGDataset


# ============ Helpers (same as eval_sqg_ae_.py) ============

def compute_psd(field):
    """Compute radially averaged power spectrum for a 2D field (H, W)."""
    f = torch.fft.rfft2(field)
    psd2d = f.abs().square()
    H, W = field.shape
    ky = torch.fft.fftfreq(H) * H
    kx = torch.fft.rfftfreq(W) * W
    KX, KY = torch.meshgrid(kx, ky, indexing="xy")
    K = torch.sqrt(KX**2 + KY**2).long()
    kmax = min(H, W) // 2
    psd1d = torch.zeros(kmax)
    for k in range(kmax):
        mask = K == k
        if mask.any():
            psd1d[k] = psd2d[mask].mean()
    return psd1d


def log_spectral_distance(psd_true, psd_pred, eps=1e-10):
    """Log Spectral Distance between two power spectra."""
    log_ratio = torch.log10(psd_pred + eps) - torch.log10(psd_true + eps)
    return torch.sqrt((log_ratio ** 2).mean()).item()


# ============ Main ============

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Bilinear interpolation baseline")
    parser.add_argument("--latent_size", type=int, default=16,
                        help="Spatial size after downsampling (default: 16)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_dir", type=str, default="results/ae_eval_bilinear")
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # Load test data
    dataset = SQGDataset("data/SQG/dataset/test", stats_dir="data/SQG/dataset/train")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    rmse_list = []
    vrmse_list = []
    lsd_per_channel = {ch: [] for ch in range(2)}
    psd_true_accum = {ch: [] for ch in range(2)}
    psd_pred_accum = {ch: [] for ch in range(2)}

    with torch.no_grad():
        for x in loader:
            x = x.to(DEVICE)
            H, W = x.shape[-2], x.shape[-1]

            # Bilinear downsample → upsample
            z = F.interpolate(x, size=(args.latent_size, args.latent_size),
                              mode="bilinear", align_corners=False)
            y = F.interpolate(z, size=(H, W),
                              mode="bilinear", align_corners=False)

            # RMSE per channel
            rmse = (x - y).square().mean(dim=(-2, -1)).sqrt()  # (B, 2)
            rmse_list.append(rmse.cpu())

            # VRMSE per channel
            mse_per_ch = (x - y).square().mean(dim=(-2, -1))
            var_per_ch = x.var(dim=(-2, -1))
            vrmse = torch.sqrt(mse_per_ch / (var_per_ch + 1e-2))
            vrmse_list.append(vrmse.cpu())

            # Power spectra and LSD
            for i in range(x.shape[0]):
                for ch in range(2):
                    psd_x = compute_psd(x[i, ch].cpu())
                    psd_y = compute_psd(y[i, ch].cpu())
                    lsd_per_channel[ch].append(log_spectral_distance(psd_x, psd_y))
                    psd_true_accum[ch].append(psd_x)
                    psd_pred_accum[ch].append(psd_y)

    # Aggregate
    rmse_all = torch.cat(rmse_list, dim=0)
    rmse_mean = rmse_all.mean().item()

    vrmse_all = torch.cat(vrmse_list, dim=0)
    vrmse_per_ch = vrmse_all.mean(dim=0)
    vrmse_mean = vrmse_all.mean().item()

    lsd_per_ch = {ch: np.mean(lsd_per_channel[ch]) for ch in range(2)}
    lsd_mean = np.mean([lsd_per_ch[0], lsd_per_ch[1]])

    # ============================================================
    # Per-trajectory aggregation: mean ± std across the test trajectories.
    # Test loader has shuffle=False and 100 snapshots per .npy file,
    # so sample index i belongs to trajectory (i // 100).
    # ============================================================
    STEPS_PER_TRAJ = dataset.steps_per_file
    n_traj = len(dataset.files)

    rmse_per_snapshot = rmse_all.mean(dim=1).numpy()                       # (N,)
    rmse_per_traj = rmse_per_snapshot.reshape(n_traj, STEPS_PER_TRAJ).mean(axis=1)
    rmse_traj_mean = float(np.mean(rmse_per_traj))
    rmse_traj_std  = float(np.std (rmse_per_traj, ddof=1)) if n_traj > 1 else 0.0

    lsd_flat = np.stack([np.array(lsd_per_channel[0]),
                         np.array(lsd_per_channel[1])], axis=1)             # (N, 2)
    lsd_per_snapshot = lsd_flat.mean(axis=1)
    lsd_per_traj = lsd_per_snapshot.reshape(n_traj, STEPS_PER_TRAJ).mean(axis=1)
    lsd_traj_mean = float(np.mean(lsd_per_traj))
    lsd_traj_std  = float(np.std (lsd_per_traj, ddof=1)) if n_traj > 1 else 0.0

    # Print results
    print(f"\nBilinear Baseline (64 → {args.latent_size} → 64)")
    print(f"  Latent shape: (2, {args.latent_size}, {args.latent_size})")
    print(f"  RMSE: {rmse_traj_mean:.4f} +/- {rmse_traj_std:.4f}  "
          f"(mean +/- std across {n_traj} trajectories)")
    print(f"  LSD:  {lsd_traj_mean:.4f} +/- {lsd_traj_std:.4f}")
    print(f"  [snapshot-level for reference: RMSE={rmse_mean:.4f}, "
          f"LSD={lsd_mean:.4f}, VRMSE={vrmse_mean:.4f}]")

    # Save power spectra plot
    CHANNEL_NAMES = ["Level 0 (upper)", "Level 1 (lower)"]
    for ch in range(2):
        mean_psd_true = torch.stack(psd_true_accum[ch]).mean(dim=0)
        mean_psd_pred = torch.stack(psd_pred_accum[ch]).mean(dim=0)

        fig, ax = plt.subplots(figsize=(8, 5))
        wavenumbers = np.arange(1, len(mean_psd_true))
        ax.loglog(wavenumbers, mean_psd_true.numpy()[1:], 'k-', linewidth=2, label="Ground truth")
        ax.loglog(wavenumbers, mean_psd_pred.numpy()[1:], 'r--', linewidth=1.5, label="Bilinear baseline")
        ax.set_xlabel("Wavenumber", fontsize=12)
        ax.set_ylabel("Power", fontsize=12)
        ax.set_title(f"Power Spectra — {CHANNEL_NAMES[ch]}", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fname = os.path.join(args.save_dir, f"power_spectra_bilinear_ch{ch}.png")
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"Saved: {fname}")

    # Save summary
    summary_path = os.path.join(args.save_dir, "bilinear_baseline_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Bilinear Baseline (64 -> {args.latent_size} -> 64)\n")
        f.write(f"Latent shape: (2, {args.latent_size}, {args.latent_size})\n")
        f.write(f"RMSE:  {rmse_mean:.4f}\n")
        f.write(f"VRMSE: {vrmse_mean:.4f} (ch0={vrmse_per_ch[0]:.4f}, ch1={vrmse_per_ch[1]:.4f})\n")
        f.write(f"LSD:   {lsd_mean:.4f} (ch0={lsd_per_ch[0]:.4f}, ch1={lsd_per_ch[1]:.4f})\n")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
