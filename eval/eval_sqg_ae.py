"""
Evaluate trained autoencoders at different compression rates.

Metrics (per supervisor's requirements):
  1. RMSE — reconstruction error
  2. Power Spectra — visual comparison of spatial frequency content
  3. Log Spectral Distance (LSD) — scalar metric for spectral quality

Usage:
    python eval_sqg_ae.py                        # evaluate all compression rates
    python eval_sqg_ae.py --compression 2 4      # evaluate only x2 and x4

Output:
    results/ae_eval/                             # summary table + spectra plots
                                                 # + per-trajectory .npy arrays
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from networks.autoencoder import get_autoencoder
from data.sqg_dataloader import SQGDataset

# ============ Helpers ============

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
    parser = argparse.ArgumentParser(description="Evaluate SQG Autoencoders")
    parser.add_argument("--compression", type=int, nargs="+", default=[2, 4, 8, 16],
                        help="Compression rates to evaluate (default: 2 4 8 16)")
    parser.add_argument("--version", type=str, default=None,
                        help="Version suffix, e.g. v2 (default: None = v1 behavior)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--attention_stage", type=int, default=None,
                        help="Stage index with self-attention (must match training config)")
    parser.add_argument("--attention_heads", type=int, default=4,
                        help="Number of attention heads (default: 4)")
    parser.add_argument("--hid_channels", type=int, nargs=3, default=[64, 128, 256],
                        help="Hidden channels per stage (must match training config)")
    parser.add_argument("--skip_mode", type=str, default=None, choices=["bilinear", "nearest"],
                        help="Latent skip connection mode (must match training config)")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Output directory (default: results/ae_eval or results/ae_eval_v2)")
    args = parser.parse_args()

    version_suffix = f"_{args.version}" if args.version else ""
    if args.save_dir is None:
        args.save_dir = f"results/ae_eval{version_suffix}"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    LAT_CHANNELS_MAP = {1: 32, 2: 16, 4: 8, 8: 4, 16: 2}
    CHANNEL_NAMES = ["Level 0 (upper)", "Level 1 (lower)"]

    attn_heads = {args.attention_stage: args.attention_heads} if args.attention_stage is not None else {}
    hid_ch = tuple(args.hid_channels)

    results = {}

    # Collect spectra across all compression rates for combined plot
    all_psd_true = {ch: [] for ch in range(2)}
    all_psd_pred = {comp: {ch: [] for ch in range(2)} for comp in args.compression}

    for compression in args.compression:
        lat_ch = LAT_CHANNELS_MAP[compression]

        # Instantiate and load model
        ae = get_autoencoder(
            pix_channels=2, lat_channels=lat_ch, spatial=2,
            arch="dcae", saturation="softclip2",
            hid_channels=hid_ch, hid_blocks=(3, 3, 3),
            attention_heads=attn_heads,
            skip_mode=args.skip_mode,
            periodic=True, identity_init=True,
        )
        ae = ae.to(DEVICE)

        weight_path = f"saved_models/ae_x{compression}{version_suffix}/best.pth"
        try:
            ae.load_state_dict(torch.load(weight_path, map_location=DEVICE))
            ae.eval()
        except FileNotFoundError:
            print(f"WARNING: {weight_path} not found, skipping x{compression}")
            continue

        # Load test data (using training stats for normalization)
        dataset = SQGDataset("data/SQG/dataset/test", stats_dir="data/SQG/dataset/train")
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        rmse_list = []
        vrmse_list = []
        lsd_per_channel = {ch: [] for ch in range(2)}
        psd_true_accum = {ch: [] for ch in range(2)}
        psd_pred_accum = {ch: [] for ch in range(2)}

        # --- per-trajectory bookkeeping ---
        # Test loader has shuffle=False and 100 snapshots per .npy file (one trajectory),
        # so sample index i belongs to trajectory (i // 100).
        STEPS_PER_TRAJ = dataset.steps_per_file
        n_traj = len(dataset.files)

        with torch.no_grad():
            for x in loader:
                x = x.to(DEVICE)
                y, z = ae(x)

                # RMSE per channel
                rmse = (x - y).square().mean(dim=(-2, -1)).sqrt()  # (B, 2)
                rmse_list.append(rmse.cpu())

                # VRMSE per channel: sqrt( MSE / Var(x) )
                mse_per_ch = (x - y).square().mean(dim=(-2, -1))  # (B, 2)
                var_per_ch = x.var(dim=(-2, -1))  # (B, 2)
                vrmse = torch.sqrt(mse_per_ch / (var_per_ch + 1e-2))  # (B, 2)
                vrmse_list.append(vrmse.cpu())

                # Power spectra and LSD per channel, per sample
                for i in range(x.shape[0]):
                    for ch in range(2):
                        psd_x = compute_psd(x[i, ch].cpu())
                        psd_y = compute_psd(y[i, ch].cpu())

                        lsd_per_channel[ch].append(log_spectral_distance(psd_x, psd_y))
                        psd_true_accum[ch].append(psd_x)
                        psd_pred_accum[ch].append(psd_y)

        # Aggregate RMSE (snapshot-level, kept for backwards compatibility)
        rmse_all = torch.cat(rmse_list, dim=0)  # (N, 2)
        rmse_per_ch = rmse_all.mean(dim=0)       # (2,)
        rmse_mean = rmse_all.mean().item()

        # Aggregate VRMSE
        vrmse_all = torch.cat(vrmse_list, dim=0)  # (N, 2)
        vrmse_per_ch = vrmse_all.mean(dim=0)       # (2,)
        vrmse_mean = vrmse_all.mean().item()

        # Aggregate LSD (snapshot-level, kept for backwards compatibility)
        lsd_per_ch = {ch: np.mean(lsd_per_channel[ch]) for ch in range(2)}
        lsd_mean = np.mean([lsd_per_ch[0], lsd_per_ch[1]])

        # ============================================================
        # Per-trajectory aggregation: mean ± std across the 10 trajectories
        # For each trajectory, average its 100 snapshots first, then take
        # mean / std over trajectories. This is the statistic asked for
        # in the thesis (std across trajectories).
        # ============================================================
        # RMSE: rmse_all has shape (N=n_traj*STEPS, 2). Mean over channels first
        # to match the "mean over both channels" column in the table.
        rmse_per_snapshot = rmse_all.mean(dim=1).numpy()       # (N,)
        rmse_per_traj = rmse_per_snapshot.reshape(n_traj, STEPS_PER_TRAJ).mean(axis=1)
        rmse_traj_mean = float(np.mean(rmse_per_traj))
        rmse_traj_std  = float(np.std (rmse_per_traj, ddof=1)) if n_traj > 1 else 0.0

        # LSD: lsd_per_channel[ch] is a list of length N; same trajectory layout.
        lsd_flat = np.stack([np.array(lsd_per_channel[0]),
                             np.array(lsd_per_channel[1])], axis=1)   # (N, 2)
        lsd_per_snapshot = lsd_flat.mean(axis=1)
        lsd_per_traj = lsd_per_snapshot.reshape(n_traj, STEPS_PER_TRAJ).mean(axis=1)
        lsd_traj_mean = float(np.mean(lsd_per_traj))
        lsd_traj_std  = float(np.std (lsd_per_traj, ddof=1)) if n_traj > 1 else 0.0

        # ---- save & print per-trajectory arrays for the paired t-test ----
        np.save(os.path.join(args.save_dir, f"rmse_per_traj_x{compression}.npy"), rmse_per_traj)
        np.save(os.path.join(args.save_dir, f"lsd_per_traj_x{compression}.npy"),  lsd_per_traj)
        print(f"  rmse_per_traj = {np.array2string(rmse_per_traj, precision=4, separator=', ')}")
        print(f"  lsd_per_traj  = {np.array2string(lsd_per_traj,  precision=4, separator=', ')}")

        # Aggregate spectra (mean across samples)
        for ch in range(2):
            mean_psd_true = torch.stack(psd_true_accum[ch]).mean(dim=0)
            mean_psd_pred = torch.stack(psd_pred_accum[ch]).mean(dim=0)
            all_psd_pred[compression][ch] = mean_psd_pred
            if isinstance(all_psd_true[ch], list):  # only collect ground truth once
                all_psd_true[ch] = mean_psd_true

        results[compression] = {
            "rmse_mean": rmse_mean,
            "rmse_ch0": rmse_per_ch[0].item(),
            "rmse_ch1": rmse_per_ch[1].item(),
            "vrmse_mean": vrmse_mean,
            "vrmse_ch0": vrmse_per_ch[0].item(),
            "vrmse_ch1": vrmse_per_ch[1].item(),
            "lsd_mean": lsd_mean,
            "lsd_ch0": lsd_per_ch[0],
            "lsd_ch1": lsd_per_ch[1],
            # ---- per-trajectory (the numbers to put in the thesis table) ----
            "rmse_traj_mean": rmse_traj_mean,
            "rmse_traj_std":  rmse_traj_std,
            "lsd_traj_mean":  lsd_traj_mean,
            "lsd_traj_std":   lsd_traj_std,
            "n_traj": n_traj,
        }

        print(f"x{compression}: "
              f"RMSE = {rmse_traj_mean:.4f} +/- {rmse_traj_std:.4f}  |  "
              f"LSD = {lsd_traj_mean:.4f} +/- {lsd_traj_std:.4f}   "
              f"(mean +/- std across {n_traj} trajectories)")

        del ae
        torch.cuda.empty_cache()

    # ============ Plot Power Spectra (per channel) ============
    for ch in range(2):
        fig, ax = plt.subplots(figsize=(8, 5))
        wavenumbers = np.arange(1, len(all_psd_true[ch]))

        # Ground truth
        ax.loglog(wavenumbers, all_psd_true[ch].numpy()[1:],
                  'k-', linewidth=2, label="Ground truth")

        # Each compression rate
        colors = {2: "tab:blue", 4: "tab:green", 8: "tab:orange", 16: "tab:red"}
        for comp in args.compression:
            if comp in all_psd_pred and ch in all_psd_pred[comp] and len(all_psd_pred[comp][ch]) > 0:
                psd = all_psd_pred[comp][ch]
                ax.loglog(wavenumbers, psd.numpy()[1:],
                          color=colors.get(comp, "gray"), linewidth=1.5,
                          linestyle="--", label=f"AE x{comp}{version_suffix}")

        ax.set_xlabel("Wavenumber", fontsize=12)
        ax.set_ylabel("Power", fontsize=12)
        ax.set_title(f"Power Spectra — {CHANNEL_NAMES[ch]}", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fname = os.path.join(args.save_dir, f"power_spectra_ch{ch}.png")
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"Saved: {fname}")

    # ============ Summary Table ============
    print("\n" + "=" * 84)
    print(f"{'Compression':<14} {'RMSE (mean)':<14} {'VRMSE (mean)':<14} {'LSD (mean)':<14} {'VRMSE (ch0)':<14} {'VRMSE (ch1)':<14}")
    print("-" * 84)
    for comp in sorted(results.keys()):
        r = results[comp]
        print(f"x{comp:<13} {r['rmse_mean']:<14.4f} {r['vrmse_mean']:<14.4f} {r['lsd_mean']:<14.4f} {r['vrmse_ch0']:<14.4f} {r['vrmse_ch1']:<14.4f}")
    print("=" * 84)

    # Save results to text file
    summary_path = os.path.join(args.save_dir, "ae_eval_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"{'Compression':<14} {'RMSE (mean)':<14} {'VRMSE (mean)':<14} {'LSD (mean)':<14} "
                f"{'VRMSE (ch0)':<14} {'VRMSE (ch1)':<14} {'RMSE (ch0)':<14} {'RMSE (ch1)':<14} "
                f"{'LSD (ch0)':<14} {'LSD (ch1)':<14}\n")
        f.write("-" * 126 + "\n")
        for comp in sorted(results.keys()):
            r = results[comp]
            f.write(f"x{comp:<13} {r['rmse_mean']:<14.4f} {r['vrmse_mean']:<14.4f} {r['lsd_mean']:<14.4f} "
                    f"{r['vrmse_ch0']:<14.4f} {r['vrmse_ch1']:<14.4f} {r['rmse_ch0']:<14.4f} {r['rmse_ch1']:<14.4f} "
                    f"{r['lsd_ch0']:<14.4f} {r['lsd_ch1']:<14.4f}\n")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
