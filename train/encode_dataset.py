"""
Pre-encode SQG pixel data → latent space using trained autoencoders.

Usage:
    python encode_dataset.py                          # encode all compression rates
    python encode_dataset.py --compression 4          # encode only x4
    python encode_dataset.py --compression 4 8        # encode x4 and x8

Output structure:
    data/SQG/dataset_latent_x2/train/     *.npy + data_mean.pt, data_std.pt, diff_mean.pt, diff_std.pt
    data/SQG/dataset_latent_x2/validation/ ...
    data/SQG/dataset_latent_x2/test/       ...
    data/SQG/dataset_latent_x4/...
    ...
"""

import os
import glob
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from networks.autoencoder import get_autoencoder

# ============ Configuration ============
PIXEL_DATA_ROOT = "data/SQG/dataset"
LATENT_DATA_ROOT = "data/SQG"
SPLITS = ["train", "validation", "test"]

# Compression rate -> latent channels
LAT_CHANNELS_MAP = {2: 16, 4: 8, 8: 4, 16: 2}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_FRAMES = 64  # Number of frames to encode at once (adjust if OOM)


def load_autoencoder(compression, version_suffix=""):
    """Load a trained autoencoder for a given compression rate."""
    lat_channels = LAT_CHANNELS_MAP[compression]
    ae = get_autoencoder(
        pix_channels=2,
        lat_channels=lat_channels,
        spatial=2,
        arch="dcae",
        saturation="softclip2",
        hid_channels=(64, 128, 256),
        hid_blocks=(3, 3, 3),
        periodic=True,
        identity_init=True,
    )
    weight_path = f"saved_models/ae_x{compression}{version_suffix}/best.pth"
    ae.load_state_dict(torch.load(weight_path, map_location=DEVICE, weights_only=True))
    ae = ae.to(DEVICE)
    ae.eval()
    print(f"Loaded AE x{compression} from {weight_path}")
    return ae


def load_pixel_stats():
    """Load pixel-space normalization stats (from training set)."""
    stats_dir = os.path.join(PIXEL_DATA_ROOT, "train")
    data_mean = torch.load(os.path.join(stats_dir, "data_mean.pt"), weights_only=True)
    data_std = torch.load(os.path.join(stats_dir, "data_std.pt"), weights_only=True)
    return data_mean, data_std


@torch.no_grad()
def encode_trajectory(ae, trajectory, pixel_mean, pixel_std):
    """
    Encode a full trajectory from pixel space to latent space.

    Args:
        ae: trained autoencoder
        trajectory: numpy array (T, 2, 64, 64) raw pixel values
        pixel_mean: (2,) mean for standardization
        pixel_std: (2,) std for standardization

    Returns:
        latent_trajectory: numpy array (T, lat_channels, 16, 16)
    """
    T = trajectory.shape[0]
    # Convert to tensor and standardize
    traj_tensor = torch.tensor(trajectory, dtype=torch.float32)
    mean = pixel_mean.view(1, -1, 1, 1)
    std = pixel_std.view(1, -1, 1, 1)
    traj_tensor = (traj_tensor - mean) / std

    # Encode in batches to avoid OOM
    latent_chunks = []
    for start in range(0, T, BATCH_FRAMES):
        end = min(start + BATCH_FRAMES, T)
        batch = traj_tensor[start:end].to(DEVICE)
        z = ae.encode(batch)
        latent_chunks.append(z.cpu())

    latent_trajectory = torch.cat(latent_chunks, dim=0).numpy()
    return latent_trajectory


def compute_latent_stats(latent_dir, split="train"):
    """
    Compute and save latent-space normalization statistics from encoded training data.
    Computes: data_mean, data_std, diff_mean, diff_std

    Matches the methodology in compute_data_stats.py:
    - data_mean/std: computed on raw latent data, mean over (T, H, W), keep C
    - diff_mean/std: computed on STANDARDIZED latent data, then differenced

    These are needed by SQGForecastDataset and the EDM model for standardization.
    """
    train_dir = os.path.join(latent_dir, split)
    files = sorted(glob.glob(os.path.join(train_dir, "*.npy")))
    print(f"Computing latent stats from {len(files)} files in {train_dir}...")

    # ---- Step 1: Compute data_mean and data_std ----
    # Using E[X] and E[X²] method (same as compute_data_stats.py)
    means_list = []
    squares_list = []

    for f in tqdm(files, desc="Stats: data_mean/std"):
        data = np.load(f).astype(np.float32)  # (T, C, H, W)
        data_t = torch.from_numpy(data)

        # Mean and mean(x²) over (T, H, W), keep C → shape (C,)
        means_list.append(data_t.mean(dim=(0, 2, 3)))
        squares_list.append((data_t ** 2).mean(dim=(0, 2, 3)))

    all_means = torch.stack(means_list)      # (N_files, C)
    all_squares = torch.stack(squares_list)  # (N_files, C)

    data_mean = all_means.mean(dim=0)         # (C,)
    second_moment = all_squares.mean(dim=0)   # (C,)
    data_std = torch.sqrt((second_moment - data_mean ** 2).clamp_min(1e-12))  # (C,)

    print(f"  data_mean: {data_mean}")
    print(f"  data_std:  {data_std}")

    # ---- Step 2: Compute diff_mean and diff_std ----
    # On STANDARDIZED data first, then compute diffs (same as compute_data_stats.py)
    diff_means_list = []
    diff_squares_list = []

    for f in tqdm(files, desc="Stats: diff_mean/std"):
        data = np.load(f).astype(np.float32)
        data_t = torch.from_numpy(data)

        # Standardize using data_mean/std
        standardized = (data_t - data_mean.view(1, -1, 1, 1)) / data_std.view(1, -1, 1, 1)

        # Compute consecutive diffs on standardized data
        diffs = standardized[1:] - standardized[:-1]  # (T-1, C, H, W)

        diff_means_list.append(diffs.mean(dim=(0, 2, 3)))
        diff_squares_list.append((diffs ** 2).mean(dim=(0, 2, 3)))

    all_diff_means = torch.stack(diff_means_list)
    all_diff_squares = torch.stack(diff_squares_list)

    diff_mean = all_diff_means.mean(dim=0)
    diff_second_moment = all_diff_squares.mean(dim=0)
    diff_std = torch.sqrt((diff_second_moment - diff_mean ** 2).clamp_min(1e-12))

    print(f"  diff_mean: {diff_mean}")
    print(f"  diff_std:  {diff_std}")

    # ---- Step 3: Save to all splits ----
    for s in SPLITS:
        out_dir = os.path.join(latent_dir, s)
        os.makedirs(out_dir, exist_ok=True)
        torch.save(data_mean.cpu(), os.path.join(out_dir, "data_mean.pt"))
        torch.save(data_std.cpu(), os.path.join(out_dir, "data_std.pt"))
        torch.save(diff_mean.cpu(), os.path.join(out_dir, "diff_mean.pt"))
        torch.save(diff_std.cpu(), os.path.join(out_dir, "diff_std.pt"))

    return data_mean, data_std, diff_mean, diff_std


def encode_split(ae, compression, split, pixel_mean, pixel_std, version_suffix=""):
    """Encode all trajectory files in one split."""
    input_dir = os.path.join(PIXEL_DATA_ROOT, split)
    output_dir = os.path.join(LATENT_DATA_ROOT, f"dataset_latent_x{compression}{version_suffix}", split)
    os.makedirs(output_dir, exist_ok=True)

    # Find all .npy trajectory files (exclude stat files)
    npy_files = sorted(glob.glob(os.path.join(input_dir, "sqg_N64_3hrly_*.npy")))
    if not npy_files:
        print(f"  No .npy files found in {input_dir}, skipping.")
        return

    print(f"  Encoding {len(npy_files)} files from {split}...")

    for f in tqdm(npy_files, desc=f"  {split}"):
        filename = os.path.basename(f)
        output_path = os.path.join(output_dir, filename)

        # Skip if already encoded
        if os.path.exists(output_path):
            continue

        # Load raw pixel trajectory
        trajectory = np.load(f)  # (101, 2, 64, 64)

        # Encode to latent
        latent_traj = encode_trajectory(ae, trajectory, pixel_mean, pixel_std)

        # Save
        np.save(output_path, latent_traj)

    print(f"  Done: {split} -> {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Encode SQG dataset to latent space")
    parser.add_argument(
        "--compression", type=int, nargs="+", default=[2, 4, 8, 16],
        help="Compression rates to encode (default: 2 4 8 16)"
    )
    parser.add_argument(
        "--version", type=str, default=None,
        help="Version suffix, e.g. v2b (default: None = v1 behavior)"
    )
    args = parser.parse_args()

    version_suffix = f"_{args.version}" if args.version else ""

    # Load pixel-space stats once
    pixel_mean, pixel_std = load_pixel_stats()
    print(f"Pixel stats loaded: mean={pixel_mean}, std={pixel_std}")

    for comp in args.compression:
        assert comp in LAT_CHANNELS_MAP, f"Unsupported compression rate: {comp}"
        print(f"\n{'='*60}")
        print(f"Encoding dataset with x{comp} compression")
        print(f"{'='*60}")

        # Load AE for this compression rate
        ae = load_autoencoder(comp, version_suffix)

        # Verify shapes
        dummy = torch.randn(1, 2, 64, 64).to(DEVICE)
        z = ae.encode(dummy)
        print(f"  Shape check: (1, 2, 64, 64) -> {tuple(z.shape)}")

        # Encode each split
        for split in SPLITS:
            encode_split(ae, comp, split, pixel_mean, pixel_std, version_suffix)

        # Compute and save latent-space stats (from training set)
        latent_dir = os.path.join(LATENT_DATA_ROOT, f"dataset_latent_x{comp}{version_suffix}")
        compute_latent_stats(latent_dir, split="train")

        # Free GPU memory
        del ae
        torch.cuda.empty_cache()

    print(f"\nAll done! Latent datasets saved under {LATENT_DATA_ROOT}/dataset_latent_x*{version_suffix}/")


if __name__ == "__main__":
    main()
