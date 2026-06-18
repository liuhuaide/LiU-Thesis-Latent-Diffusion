"""
End-to-end evaluation of Latent EDM models.

Pipeline:
  1. Load pixel-space test data
  2. Encode init_states → latent (frozen AE encoder)
  3. Latent EDM autoregressive prediction (multiple steps)
  4. Decode predictions → pixel space (frozen AE decoder)
  5. Compute metrics vs pixel ground truth: RMSE, CRPS, Spread-Skill, Power Spectra
  6. Measure inference time per step

Usage:
  # Evaluate Latent EDM x4
  python eval_latent_pipeline.py --compression 4 \
      --latent_edm_ckpt saved_models/Latent_EDM_x4_Run-LatentEDM-32-03_22_07-1458/min_val_loss.ckpt

  # Evaluate all compression rates
  python eval_latent_pipeline.py --compression 2 4 8 16 \
      --latent_edm_ckpt saved_models/latent_x2/min_val_loss.ckpt \
                        saved_models/latent_x4/min_val_loss.ckpt \
                        saved_models/latent_x8/min_val_loss.ckpt \
                        saved_models/latent_x16/min_val_loss.ckpt

  # Also include pixel-space EDM baseline in comparison
  python eval_latent_pipeline.py --compression 4 \
      --latent_edm_ckpt saved_models/latent_x4/min_val_loss.ckpt \
      --pixel_edm_ckpt saved_models/EDM_Local_Run_MSE-EDM-32-02_25_03-2991/min_val_loss.ckpt
"""

import os
import sys
import time
import argparse
import glob
import json

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from networks.autoencoder import get_autoencoder
from metrics.metrics import mse, mae, crps_ens, power_spectrum, radial_average
import data.SQG.constants as SQGConstants

# ============ Configuration ============
PIXEL_DATA_ROOT = "data/SQG/dataset"
LATENT_DATA_ROOT = "data/SQG"
LAT_CHANNELS_MAP = {2: 16, 4: 8, 8: 4, 16: 2}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============ Data Loading ============

def load_pixel_test_data(data_path, init_states=2, pred_length=50, subsample_step=1):
    """
    Load pixel-space test trajectories manually (not using SQGForecastDataset)
    to avoid any standardization — we want raw pixel data for final metrics.

    Returns list of (init_states_tensor, target_states_tensor) tuples,
    both standardized for model input, plus the raw stats for de-standardization.
    """
    stats_dir = os.path.join(data_path, "train")
    data_mean = torch.load(os.path.join(stats_dir, "data_mean.pt"),
                           weights_only=True).view(1, -1, 1, 1)
    data_std = torch.load(os.path.join(stats_dir, "data_std.pt"),
                          weights_only=True).view(1, -1, 1, 1)

    test_dir = os.path.join(data_path, "test")
    npy_files = sorted(glob.glob(os.path.join(test_dir, "sqg_N64_3hrly_*.npy")))
    if not npy_files:
        raise ValueError(f"No .npy test files found in {test_dir}")

    sample_length = 1 + (init_states - 1 + pred_length) * subsample_step

    trajectories = []
    for f in npy_files:
        traj = torch.tensor(np.load(f), dtype=torch.float32)  # (101, 2, 64, 64)
        sample = traj[:sample_length:subsample_step]  # (init_states + pred_length, 2, 64, 64)

        # Standardize
        sample_std = (sample - data_mean) / data_std

        init = sample_std[:init_states]        # (init_states, 2, 64, 64)
        target = sample_std[init_states:]      # (pred_length, 2, 64, 64)
        target_raw = sample[init_states:]      # (pred_length, 2, 64, 64) un-standardized

        trajectories.append((init, target, target_raw))

    return trajectories, data_mean, data_std


# ============ Model Loading ============

def load_autoencoder(compression):
    """Load frozen autoencoder for a given compression rate."""
    lat_channels = LAT_CHANNELS_MAP[compression]
    ae = get_autoencoder(
        pix_channels=2, lat_channels=lat_channels, spatial=2,
        arch="dcae", saturation="softclip2",
        hid_channels=(64, 128, 256), hid_blocks=(3, 3, 3),
        periodic=True, identity_init=True,
    )
    weight_path = f"saved_models/ae_x{compression}/best.pth"
    ae.load_state_dict(torch.load(weight_path, map_location="cpu"))
    ae = ae.to(DEVICE).eval()
    print(f"  Loaded AE x{compression} from {weight_path}")
    return ae


def load_latent_edm(ckpt_path, compression):
    """Load trained Latent EDM from Lightning checkpoint."""
    from forecasting.models.latent_edm import LatentEDM

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = ckpt["hyper_parameters"]["args"]

    # Handle both Namespace and dict
    if isinstance(hparams, dict):
        from argparse import Namespace
        hparams = Namespace(**hparams)

    # Fix data_path if stats file not found at recorded path
    if not os.path.exists(os.path.join(hparams.data_path, "data_mean.pt")):
        train_path = os.path.join(hparams.data_path, "train")
        if os.path.exists(os.path.join(train_path, "data_mean.pt")):
            hparams.data_path = train_path

    model = LatentEDM(hparams)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(DEVICE).eval()
    print(f"  Loaded Latent EDM x{compression} from {ckpt_path}")
    return model


def load_pixel_edm(ckpt_path):
    """Load trained pixel-space EDM from Lightning checkpoint."""
    from forecasting.models.edm import EDM

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = ckpt["hyper_parameters"]["args"]

    if isinstance(hparams, dict):
        from argparse import Namespace
        hparams = Namespace(**hparams)

    # Fix data_path: ARModel.__init__ loads stats from this path,
    # and the pixel stats live in the train/ subdirectory
    if not os.path.exists(os.path.join(hparams.data_path, "data_mean.pt")):
        train_path = os.path.join(hparams.data_path, "train")
        if os.path.exists(os.path.join(train_path, "data_mean.pt")):
            hparams.data_path = train_path

    model = EDM(hparams)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(DEVICE).eval()
    print(f"  Loaded Pixel EDM from {ckpt_path}")
    return model


# ============ Inference ============

@torch.no_grad()
def predict_latent_edm(ae, edm, init_states_batch, pred_steps, ensemble_size=5):
    """
    Full latent EDM inference pipeline (batched over trajectories):
      pixel init → encode → latent AR prediction → decode → pixel prediction

    Args:
        ae: frozen autoencoder
        edm: trained LatentEDM model
        init_states_batch: (B, init_states, 2, 64, 64) standardized pixel input
        pred_steps: number of AR steps
        ensemble_size: number of ensemble members

    Returns:
        predictions: (B, ensemble_size, pred_steps, 2, 64, 64) pixel-space predictions
        timing: dict with 'diffusion', 'encode', 'decode', 'total' times in seconds
    """
    B, N_init, C, H, W = init_states_batch.shape
    latent_mean = edm.data_mean.to(DEVICE)
    latent_std = edm.data_std.to(DEVICE)

    # --- Encode ---
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    t_enc_start = time.time()

    # Encode all init frames: (B*N_init, 2, 64, 64) → (B*N_init, lat_ch, 16, 16)
    flat_init = init_states_batch.reshape(B * N_init, C, H, W).to(DEVICE)
    flat_latent = ae.encode(flat_init)
    lat_ch = flat_latent.shape[1]
    init_latent = flat_latent.reshape(B, N_init, lat_ch, flat_latent.shape[2], flat_latent.shape[3])
    init_latent = (init_latent - latent_mean.unsqueeze(0)) / latent_std.unsqueeze(0)

    if DEVICE.type == "cuda": torch.cuda.synchronize()
    t_encode = time.time() - t_enc_start

    # --- Diffusion AR rollout (batched over B trajectories) ---
    ensemble_predictions_latent = []

    if DEVICE.type == "cuda": torch.cuda.synchronize()
    t_diff_start = time.time()

    for _ in range(ensemble_size):
        latent_preds = []
        current_init = init_latent.clone()  # (B, N_init, lat_ch, 16, 16)

        for step in range(pred_steps):
            pred_latent = edm.predict_step(current_init)  # (B, lat_ch, 16, 16)
            latent_preds.append(pred_latent)
            current_init = torch.cat(
                (current_init[:, 1:], pred_latent.unsqueeze(1)), dim=1
            )

        # (pred_steps, B, lat_ch, 16, 16) → (B, pred_steps, lat_ch, 16, 16)
        latent_seq = torch.stack(latent_preds, dim=1)
        ensemble_predictions_latent.append(latent_seq)

    if DEVICE.type == "cuda": torch.cuda.synchronize()
    t_diffusion = time.time() - t_diff_start

    # --- Decode ---
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    t_dec_start = time.time()

    ensemble_predictions = []
    for latent_seq in ensemble_predictions_latent:
        # (B, pred_steps, lat_ch, 16, 16) → de-standardize
        latent_seq_destd = latent_seq * latent_std.unsqueeze(0) + latent_mean.unsqueeze(0)
        # Decode: flatten to (B*pred_steps, lat_ch, 16, 16)
        flat_lat = latent_seq_destd.reshape(B * pred_steps, lat_ch,
                                            latent_seq.shape[3], latent_seq.shape[4])
        flat_pix = ae.decode(flat_lat, noisy=False)  # (B*pred_steps, 2, 64, 64)
        pixel_seq = flat_pix.reshape(B, pred_steps, flat_pix.shape[1],
                                     flat_pix.shape[2], flat_pix.shape[3])
        ensemble_predictions.append(pixel_seq.cpu())

    if DEVICE.type == "cuda": torch.cuda.synchronize()
    t_decode = time.time() - t_dec_start

    # (B, ensemble_size, pred_steps, 2, 64, 64)
    predictions = torch.stack(ensemble_predictions, dim=1)

    # Also return latent-space predictions for latent metrics
    # ensemble_predictions_latent: list of (B, pred_steps, lat_ch, 16, 16) standardized
    latent_preds_destd = []
    latent_preds_std = []
    for latent_seq in ensemble_predictions_latent:
        latent_seq_destd = latent_seq * latent_std.unsqueeze(0) + latent_mean.unsqueeze(0)
        latent_preds_destd.append(latent_seq_destd.cpu())
        latent_preds_std.append(latent_seq.cpu())
    # (B, ensemble_size, pred_steps, lat_ch, 16, 16)
    latent_predictions = torch.stack(latent_preds_destd, dim=1)
    latent_predictions_std = torch.stack(latent_preds_std, dim=1)

    timing = {
        "encode": t_encode,
        "diffusion": t_diffusion,
        "decode": t_decode,
        "total": t_encode + t_diffusion + t_decode,
        "diffusion_per_step": t_diffusion / (pred_steps * ensemble_size),
    }

    return predictions, latent_predictions, latent_predictions_std, timing


@torch.no_grad()
def predict_pixel_edm(edm, init_states_batch, pred_steps, ensemble_size=5):
    """
    Pixel-space EDM inference (batched over trajectories).

    Args:
        init_states_batch: (B, init_states, 2, 64, 64) standardized

    Returns:
        predictions: (B, ensemble_size, pred_steps, 2, 64, 64) standardized
        timing: dict
    """
    init = init_states_batch.to(DEVICE)  # (B, N_init, 2, 64, 64)

    ensemble_predictions = []

    if DEVICE.type == "cuda": torch.cuda.synchronize()
    t_diff_start = time.time()

    for _ in range(ensemble_size):
        preds = []
        current_init = init.clone()

        for step in range(pred_steps):
            pred = edm.predict_step(current_init)  # (B, 2, 64, 64)
            preds.append(pred)
            current_init = torch.cat(
                (current_init[:, 1:], pred.unsqueeze(1)), dim=1
            )

        # (pred_steps, B, 2, 64, 64) → (B, pred_steps, 2, 64, 64)
        pixel_seq = torch.stack(preds, dim=1)
        ensemble_predictions.append(pixel_seq.cpu())

    if DEVICE.type == "cuda": torch.cuda.synchronize()
    t_diffusion = time.time() - t_diff_start

    # (B, ensemble_size, pred_steps, 2, 64, 64)
    predictions = torch.stack(ensemble_predictions, dim=1)

    timing = {
        "encode": 0.0,
        "diffusion": t_diffusion,
        "decode": 0.0,
        "total": t_diffusion,
        "diffusion_per_step": t_diffusion / (pred_steps * ensemble_size),
    }

    return predictions, timing


# ============ Metrics ============

def compute_metrics(predictions, targets, pixel_mean, pixel_std):
    """
    Compute all metrics in pixel space.

    Args:
        predictions: (ensemble_size, pred_steps, 2, 64, 64) standardized
        targets: (pred_steps, 2, 64, 64) standardized
        pixel_mean, pixel_std: for de-standardization

    Returns:
        dict of metrics, each (pred_steps,) or (pred_steps, 2)
    """
    scalefact = SQGConstants.scalefact

    # Ensemble mean
    ens_mean = predictions.mean(dim=0)  # (pred_steps, 2, 64, 64)

    # De-standardize for physical-unit metrics
    ens_mean_phys = (ens_mean * pixel_std + pixel_mean) * scalefact
    targets_phys = (targets * pixel_std + pixel_mean) * scalefact

    # RMSE per step per variable: (pred_steps, 2)
    step_mse = mse(ens_mean, targets, mean_vars=False)  # (pred_steps, 2)
    step_rmse_phys = torch.sqrt(step_mse) * pixel_std.view(1, -1) * scalefact

    # CRPS per step per variable: (pred_steps, 2)
    step_crps = crps_ens(predictions, targets, mean_vars=False, ens_dim=0)
    step_crps_phys = step_crps * pixel_std.view(1, -1) * scalefact

    # Spread-Skill
    spread_sq = torch.var(predictions, dim=0).mean(dim=(-2, -1))  # (pred_steps, 2)
    ens_mse_val = mse(ens_mean, targets, mean_vars=False)  # (pred_steps, 2)
    spread = torch.sqrt(spread_sq)
    skill = torch.sqrt(ens_mse_val)
    n_ens = predictions.shape[0]
    spsk = np.sqrt((n_ens + 1) / n_ens) * (spread / skill)

    # Power spectra (on physical-unit data)
    # power_spectrum expects (B, C, H, W) → (B, H, W), then radial_average → (B, R)
    # We compute per time step: input (1, 2, 64, 64) → output (1, R)
    pred_rad = []
    target_rad = []
    for t in range(ens_mean_phys.shape[0]):
        ps_pred = power_spectrum(ens_mean_phys[t:t+1])   # (1, 2, 64, 64) → (1, 64, 64)
        ps_tgt = power_spectrum(targets_phys[t:t+1])     # (1, 2, 64, 64) → (1, 64, 64)
        pred_rad.append(radial_average(ps_pred).squeeze(0))   # (R,)
        target_rad.append(radial_average(ps_tgt).squeeze(0))  # (R,)

    return {
        "rmse": step_rmse_phys.numpy(),           # (pred_steps, 2)
        "crps": step_crps_phys.numpy(),            # (pred_steps, 2) physical units
        "spsk_ratio": spsk.numpy(),                # (pred_steps, 2)
        "spectra_pred": torch.stack(pred_rad).numpy(),    # (pred_steps, R)
        "spectra_truth": torch.stack(target_rad).numpy(), # (pred_steps, R)
    }


def compute_latent_metrics(latent_predictions, latent_targets, latent_predictions_std, latent_targets_std):
    """
    Compute RMSE in latent space: E(x) vs LEDM prediction (before decoding).

    Args:
        latent_predictions: (ensemble_size, pred_steps, lat_ch, 16, 16) de-standardized
        latent_targets: (pred_steps, lat_ch, 16, 16) de-standardized
        latent_predictions_std: (ensemble_size, pred_steps, lat_ch, 16, 16) standardized
        latent_targets_std: (pred_steps, lat_ch, 16, 16) standardized

    Returns:
        dict with latent-space RMSE per step (raw and standardized)
    """
    # --- Raw (de-standardized) latent RMSE ---
    ens_mean = latent_predictions.mean(dim=0)
    step_mse = (ens_mean - latent_targets).square().mean(dim=(-2, -1))
    step_rmse = torch.sqrt(step_mse)
    step_rmse_mean = step_rmse.mean(dim=-1)

    # --- Standardized latent RMSE (comparable across compression rates) ---
    ens_mean_std = latent_predictions_std.mean(dim=0)
    step_mse_std = (ens_mean_std - latent_targets_std).square().mean(dim=(-2, -1))
    step_rmse_std = torch.sqrt(step_mse_std)
    step_rmse_std_mean = step_rmse_std.mean(dim=-1)

    return {
        "latent_rmse": step_rmse.numpy(),                   # (pred_steps, lat_ch)
        "latent_rmse_mean": step_rmse_mean.numpy(),         # (pred_steps,)
        "latent_rmse_std": step_rmse_std.numpy(),           # (pred_steps, lat_ch)
        "latent_rmse_std_mean": step_rmse_std_mean.numpy(), # (pred_steps,)
    }


# ============ Plotting ============

def plot_rmse_comparison(all_results, output_dir, step_hours=3):
    """Plot RMSE vs time step for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    var_names = ["Level 0", "Level 1"]

    for var_i in range(2):
        ax = axes[var_i]
        for name, res in all_results.items():
            rmse = res["rmse"][:, var_i]
            steps = np.arange(1, len(rmse) + 1) * step_hours
            ax.plot(steps, rmse, label=name, linewidth=2)
        ax.set_xlabel("Lead Time (hours)")
        ax.set_ylabel("RMSE")
        ax.set_title(f"RMSE — {var_names[var_i]}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "rmse_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_crps_comparison(all_results, output_dir, step_hours=3):
    """Plot CRPS vs time step for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    var_names = ["Level 0", "Level 1"]

    for var_i in range(2):
        ax = axes[var_i]
        for name, res in all_results.items():
            crps = res["crps"][:, var_i]
            steps = np.arange(1, len(crps) + 1) * step_hours
            ax.plot(steps, crps, label=name, linewidth=2)
        ax.set_xlabel("Lead Time (hours)")
        ax.set_ylabel("CRPS")
        ax.set_title(f"CRPS — {var_names[var_i]}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "crps_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_spsk_comparison(all_results, output_dir, step_hours=3):
    """Plot Spread-Skill Ratio vs time step."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    var_names = ["Level 0", "Level 1"]

    for var_i in range(2):
        ax = axes[var_i]
        for name, res in all_results.items():
            spsk = res["spsk_ratio"][:, var_i]
            steps = np.arange(1, len(spsk) + 1) * step_hours
            ax.plot(steps, spsk, label=name, linewidth=2)
        ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.5, label="Ideal (1.0)")
        ax.set_xlabel("Lead Time (hours)")
        ax.set_ylabel("Spread-Skill Ratio")
        ax.set_title(f"Spread-Skill Ratio — {var_names[var_i]}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "spsk_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_spectra_comparison(all_results, output_dir, time_steps=[1, 25, 50]):
    """Plot power spectra at selected time steps."""
    fig, axes = plt.subplots(1, len(time_steps), figsize=(6 * len(time_steps), 5))
    if len(time_steps) == 1:
        axes = [axes]

    # Use consistent colors for each model
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results) + 1))

    for ax, t in zip(axes, time_steps):
        t_idx = t - 1  # 0-indexed

        for i, (name, res) in enumerate(all_results.items()):
            if t_idx < res["spectra_pred"].shape[0]:
                spec = res["spectra_pred"][t_idx]  # (R,) — 1D radial spectrum
                x = np.arange(1, len(spec))
                ax.loglog(x, spec[1:], label=name, linewidth=2, color=colors[i])

        # Plot truth (from first model's result)
        first_res = next(iter(all_results.values()))
        if t_idx < first_res["spectra_truth"].shape[0]:
            spec_truth = first_res["spectra_truth"][t_idx]  # (R,)
            x = np.arange(1, len(spec_truth))
            ax.loglog(x, spec_truth[1:], "k--", label="Truth", linewidth=2)

        ax.set_xlabel("Wavenumber")
        ax.set_ylabel("Power")
        ax.set_title(f"Power Spectrum (step {t})")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "spectra_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_timing_bar(all_results, output_dir):
    """Stacked bar chart: encode / diffusion / decode per step."""
    names = list(all_results.keys())

    encode_times = [all_results[n].get("timing_breakdown", {}).get("encode", 0)
                    / max(all_results[n].get("total_time", 1), 1e-9)
                    * all_results[n].get("time_per_step", 0)
                    for n in names]
    diff_times = [all_results[n].get("diffusion_per_step", all_results[n].get("time_per_step", 0))
                  for n in names]
    decode_times = [all_results[n].get("timing_breakdown", {}).get("decode", 0)
                    / max(all_results[n].get("total_time", 1), 1e-9)
                    * all_results[n].get("time_per_step", 0)
                    for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: total time per step (for overall comparison)
    ax = axes[0]
    total_times = [all_results[n]["time_per_step"] for n in names]
    bars = ax.bar(names, total_times, color=plt.cm.Set2(np.linspace(0, 1, len(names))))
    ax.set_ylabel("Time per step (seconds)")
    ax.set_title("Total Inference Time per Step")
    for bar, t in zip(bars, total_times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{t*1000:.0f}ms", ha="center", va="bottom", fontsize=10)

    # Right: diffusion-only time (the fair comparison)
    ax = axes[1]
    bars = ax.bar(names, [d * 1000 for d in diff_times],
                  color=plt.cm.Set2(np.linspace(0, 1, len(names))))
    ax.set_ylabel("Time per diffusion step (ms)")
    ax.set_title("Pure Diffusion Time per Step (excl. AE)")
    for bar, t in zip(bars, diff_times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{t*1000:.1f}ms", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    path = os.path.join(output_dir, "inference_time.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ============ Main ============

def main():
    parser = argparse.ArgumentParser(description="Evaluate Latent EDM pipeline")
    parser.add_argument("--compression", type=int, nargs="+", default=[4],
                        help="Compression rates to evaluate (default: 4)")
    parser.add_argument("--latent_edm_ckpt", type=str, nargs="+", required=True,
                        help="Checkpoint paths for each compression rate (same order)")
    parser.add_argument("--pixel_edm_ckpt", type=str, default=None,
                        help="Pixel-space EDM checkpoint for baseline comparison")
    parser.add_argument("--data_path", type=str, default=PIXEL_DATA_ROOT,
                        help="Path to pixel-space dataset root")
    parser.add_argument("--pred_steps", type=int, default=50,
                        help="Number of AR prediction steps (default: 50)")
    parser.add_argument("--ensemble_size", type=int, default=20,
                        help="Number of ensemble members (default: 20)")
    parser.add_argument("--init_states", type=int, default=2,
                        help="Number of initial conditioning states (default: 2)")
    parser.add_argument("--ae_target", action="store_true",
                        help="Use AE-reconstructed target D(E(x)) instead of raw x")
    parser.add_argument("--sampler", type=str, default=None, choices=["heun", "edm", "ddpm"],
                        help="Override sampler for evaluation (default: use model's sampler)")
    parser.add_argument("--output_dir", type=str, default="eval_results",
                        help="Output directory for plots and metrics")
    parser.add_argument("--step_hours", type=int, default=3,
                        help="Hours per time step (default: 3)")

    args = parser.parse_args()

    assert len(args.compression) == len(args.latent_edm_ckpt), \
        "Number of compression rates must match number of checkpoint paths"

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load pixel test data ----
    print("Loading pixel-space test data...")
    test_data, pixel_mean, pixel_std = load_pixel_test_data(
        args.data_path, init_states=args.init_states, pred_length=args.pred_steps
    )
    print(f"  Loaded {len(test_data)} test trajectories")

    all_results = {}

    # ---- Evaluate Latent EDM at each compression rate ----
    for comp, ckpt in zip(args.compression, args.latent_edm_ckpt):
        print(f"\n{'='*60}")
        print(f"Evaluating Latent EDM x{comp}")
        print(f"{'='*60}")

        ae = load_autoencoder(comp)
        edm = load_latent_edm(ckpt, comp)

        # Override sampler if specified
        if args.sampler:
            edm.args.sampler = args.sampler
            print(f"  Sampler overridden to: {args.sampler}")

        all_metrics = []
        total_timing = {"encode": 0, "diffusion": 0, "decode": 0, "total": 0}

        # Batch all test trajectories together
        init_batch = torch.stack([d[0] for d in test_data], dim=0)      # (B, init_states, 2, 64, 64)
        targets_std_batch = torch.stack([d[1] for d in test_data], dim=0)  # (B, pred_steps, 2, 64, 64)

        # Optionally reconstruct targets through AE: D(E(x))
        if args.ae_target:
            print(f"  Reconstructing targets through AE (D(E(x)))...")
            with torch.no_grad():
                B, T, C, H, W = targets_std_batch.shape
                flat_targets = targets_std_batch.reshape(B * T, C, H, W).to(DEVICE)
                # Encode and decode in batches to avoid OOM
                recon_chunks = []
                chunk_size = 64
                for start in range(0, flat_targets.shape[0], chunk_size):
                    end = min(start + chunk_size, flat_targets.shape[0])
                    chunk = flat_targets[start:end]
                    z = ae.encode(chunk)
                    recon = ae.decode(z, noisy=False)
                    recon_chunks.append(recon.cpu())
                targets_std_batch = torch.cat(recon_chunks, dim=0).reshape(B, T, C, H, W)
            print(f"  Target mode: AE-reconstructed D(E(x))")

        print(f"  Running batched inference (B={init_batch.shape[0]})...")
        predictions, latent_predictions, latent_predictions_std, timing = predict_latent_edm(
            ae, edm, init_batch, args.pred_steps, args.ensemble_size
        )
        # predictions: (B, ensemble_size, pred_steps, 2, 64, 64)
        # latent_predictions: (B, ensemble_size, pred_steps, lat_ch, 16, 16)
        total_timing = timing
        B = init_batch.shape[0]
        total_steps = args.pred_steps * args.ensemble_size * B

        # Encode target states to latent space for latent-space metrics
        print(f"  Encoding targets to latent space for latent metrics...")
        latent_mean = edm.data_mean.cpu()
        latent_std = edm.data_std.cpu()
        with torch.no_grad():
            B_t, T_t, C_t, H_t, W_t = targets_std_batch.shape
            flat_tgt = targets_std_batch.reshape(B_t * T_t, C_t, H_t, W_t).to(DEVICE)
            latent_tgt_chunks = []
            chunk_size = 64
            for start in range(0, flat_tgt.shape[0], chunk_size):
                end = min(start + chunk_size, flat_tgt.shape[0])
                z_tgt = ae.encode(flat_tgt[start:end])
                latent_tgt_chunks.append(z_tgt.cpu())
            latent_targets_batch = torch.cat(latent_tgt_chunks, dim=0)
            lat_ch = latent_targets_batch.shape[1]
            latent_targets_batch = latent_targets_batch.reshape(B_t, T_t, lat_ch,
                                                                 latent_targets_batch.shape[2],
                                                                 latent_targets_batch.shape[3])
            # Standardized version of latent targets
            latent_targets_batch_std = (latent_targets_batch - latent_mean.view(1, 1, -1, 1, 1)) / latent_std.view(1, 1, -1, 1, 1)

        # Compute metrics per trajectory, then average
        for i in range(len(test_data)):
            batch_metrics = compute_metrics(
                predictions[i],  # (ensemble_size, pred_steps, 2, 64, 64)
                targets_std_batch[i],  # (pred_steps, 2, 64, 64)
                pixel_mean, pixel_std
            )
            # Latent-space metrics (raw + standardized)
            batch_latent_metrics = compute_latent_metrics(
                latent_predictions[i],          # (ensemble_size, pred_steps, lat_ch, 16, 16) de-std
                latent_targets_batch[i],        # (pred_steps, lat_ch, 16, 16) de-std
                latent_predictions_std[i],      # (ensemble_size, pred_steps, lat_ch, 16, 16) std
                latent_targets_batch_std[i],    # (pred_steps, lat_ch, 16, 16) std
            )
            batch_metrics.update(batch_latent_metrics)
            all_metrics.append(batch_metrics)

        # Average metrics across test trajectories
        avg_metrics = {}
        for key in all_metrics[0]:
            avg_metrics[key] = np.mean([m[key] for m in all_metrics], axis=0)

        avg_metrics["time_per_step"] = total_timing["total"] / total_steps
        avg_metrics["diffusion_per_step"] = total_timing["diffusion"] / total_steps
        avg_metrics["total_time"] = total_timing["total"]
        avg_metrics["timing_breakdown"] = {
            k: total_timing[k] for k in total_timing
        }

        all_results[f"Latent EDM x{comp}"] = avg_metrics
        print(f"  Mean RMSE (step 1):  {avg_metrics['rmse'][0].mean():.4f}")
        print(f"  Mean RMSE (step 50): {avg_metrics['rmse'][-1].mean():.4f}")
        print(f"  Mean Latent RMSE (step 1):  {avg_metrics['latent_rmse_mean'][0]:.4f}")
        print(f"  Mean Latent RMSE (step 50): {avg_metrics['latent_rmse_mean'][-1]:.4f}")
        print(f"  Std Latent RMSE (step 1):   {avg_metrics['latent_rmse_std_mean'][0]:.4f}")
        print(f"  Std Latent RMSE (step 50):  {avg_metrics['latent_rmse_std_mean'][-1]:.4f}")
        print(f"  Timing breakdown:")
        print(f"    Encode:    {total_timing['encode']:.2f}s")
        print(f"    Diffusion: {total_timing['diffusion']:.2f}s  ({avg_metrics['diffusion_per_step']*1000:.1f}ms/step)")
        print(f"    Decode:    {total_timing['decode']:.2f}s")
        print(f"    Total:     {total_timing['total']:.2f}s")

        # Free GPU memory
        del ae, edm
        torch.cuda.empty_cache()

    # ---- Evaluate Pixel EDM baseline (optional) ----
    if args.pixel_edm_ckpt:
        print(f"\n{'='*60}")
        print(f"Evaluating Pixel EDM baseline")
        print(f"{'='*60}")

        pixel_edm = load_pixel_edm(args.pixel_edm_ckpt)

        # Override sampler if specified
        if args.sampler:
            pixel_edm.args.sampler = args.sampler
            print(f"  Sampler overridden to: {args.sampler}")

        all_metrics = []

        # Batch all test trajectories
        init_batch = torch.stack([d[0] for d in test_data], dim=0)
        targets_std_batch = torch.stack([d[1] for d in test_data], dim=0)

        print(f"  Running batched inference (B={init_batch.shape[0]})...")
        predictions, timing = predict_pixel_edm(
            pixel_edm, init_batch, args.pred_steps, args.ensemble_size
        )
        total_timing = timing
        B = init_batch.shape[0]
        total_steps = args.pred_steps * args.ensemble_size * B

        for i in range(len(test_data)):
            batch_metrics = compute_metrics(
                predictions[i],
                targets_std_batch[i],
                pixel_mean, pixel_std
            )
            all_metrics.append(batch_metrics)

        avg_metrics = {}
        for key in all_metrics[0]:
            avg_metrics[key] = np.mean([m[key] for m in all_metrics], axis=0)

        avg_metrics["time_per_step"] = total_timing["total"] / total_steps
        avg_metrics["diffusion_per_step"] = total_timing["diffusion"] / total_steps
        avg_metrics["total_time"] = total_timing["total"]
        avg_metrics["timing_breakdown"] = {
            k: total_timing[k] for k in total_timing
        }

        all_results["Pixel EDM"] = avg_metrics
        print(f"  Mean RMSE (step 1):  {avg_metrics['rmse'][0].mean():.4f}")
        print(f"  Mean RMSE (step 50): {avg_metrics['rmse'][-1].mean():.4f}")
        print(f"  Diffusion per step: {avg_metrics['diffusion_per_step']*1000:.1f}ms")

        del pixel_edm
        torch.cuda.empty_cache()

    # ---- Generate plots ----
    print(f"\n{'='*60}")
    print("Generating comparison plots...")
    print(f"{'='*60}")

    plot_rmse_comparison(all_results, args.output_dir, args.step_hours)
    plot_crps_comparison(all_results, args.output_dir, args.step_hours)
    plot_spsk_comparison(all_results, args.output_dir, args.step_hours)
    plot_spectra_comparison(all_results, args.output_dir)
    plot_timing_bar(all_results, args.output_dir)

    # ---- Save numerical results ----
    summary = {
        "_config": {
            "target_mode": "ae_reconstructed" if args.ae_target else "raw",
            "ensemble_size": args.ensemble_size,
            "pred_steps": args.pred_steps,
        }
    }
    for name, res in all_results.items():
        entry = {
            "rmse_step1_mean": float(res["rmse"][0].mean()),
            "rmse_step25_mean": float(res["rmse"][24].mean()) if len(res["rmse"]) > 24 else None,
            "rmse_step50_mean": float(res["rmse"][-1].mean()),
            "crps_step1_mean": float(res["crps"][0].mean()),
            "crps_step50_mean": float(res["crps"][-1].mean()),
            "time_per_step": float(res["time_per_step"]),
            "diffusion_per_step": float(res["diffusion_per_step"]),
            "total_time": float(res["total_time"]),
            "timing_breakdown": res.get("timing_breakdown", {}),
        }
        # Add latent metrics if available (only for Latent EDM)
        if "latent_rmse_mean" in res:
            entry["latent_rmse_step1"] = float(res["latent_rmse_mean"][0])
            entry["latent_rmse_step50"] = float(res["latent_rmse_mean"][-1])
            entry["latent_rmse_std_step1"] = float(res["latent_rmse_std_mean"][0])
            entry["latent_rmse_std_step50"] = float(res["latent_rmse_std_mean"][-1])
        summary[name] = entry

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {summary_path}")

    # Print summary table
    target_mode = "AE-reconstructed D(E(x))" if args.ae_target else "raw ground truth x"
    print(f"\n{'='*90}")
    print(f"SUMMARY  (target: {target_mode})")
    print(f"{'='*90}")
    print(f"{'Model':<20} {'RMSE@1':>8} {'RMSE@50':>8} {'CRPS@50':>8} {'LatRMSE@1':>10} {'LatRMSE@50':>11} {'StdLatR@1':>10} {'StdLatR@50':>11} {'Diff/step':>11}")
    print("-" * 110)
    for name, s in summary.items():
        if name.startswith("_"):
            continue
        lat1 = f"{s['latent_rmse_step1']:>10.4f}" if "latent_rmse_step1" in s else f"{'—':>10}"
        lat50 = f"{s['latent_rmse_step50']:>11.4f}" if "latent_rmse_step50" in s else f"{'—':>11}"
        slat1 = f"{s['latent_rmse_std_step1']:>10.4f}" if "latent_rmse_std_step1" in s else f"{'—':>10}"
        slat50 = f"{s['latent_rmse_std_step50']:>11.4f}" if "latent_rmse_std_step50" in s else f"{'—':>11}"
        print(f"{name:<20} {s['rmse_step1_mean']:>8.4f} {s['rmse_step50_mean']:>8.4f} "
              f"{s['crps_step50_mean']:>8.4f} {lat1} {lat50} {slat1} {slat50} "
              f"{s['diffusion_per_step']*1000:>8.1f}ms")

    print(f"\nAll results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
