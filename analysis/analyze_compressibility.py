"""
Compressibility analysis for the SQG data at 64x64.

Produces two model-independent / linear-reference diagnostics to SUPPORT (not
prove) the interpretation that the 64x64 SQG field carries limited spatial
redundancy and is therefore hard to compress to a small latent space:

  (1) Cumulative radial power-spectrum energy fraction
      - "How much of the total spectral energy lies in the lowest k wavenumbers?"
      - Fully model-independent (a property of the data, not of any compressor),
        so it avoids the linear-vs-nonlinear caveat of PCA.
      - Uses the SAME power_spectrum() / radial binning as metrics.py, but sums
        the per-ring TOTAL energy (not the per-ring average used for the spectra
        plots), which is the correct quantity for an energy-fraction.

  (2) PCA cumulative explained-variance
      - "How many linear components are needed to explain X% of the variance?"
      - This is a LINEAR reference only: a non-linear autoencoder could in
        principle do better, so report it as a conservative linear baseline,
        NOT as the compressibility limit of the DCAE.

Both are offered as evidence consistent with the "hard-to-compress" hypothesis,
not as a proof. Establishing it rigorously would need a matched cross-resolution
study (see thesis Section 6.3).

Usage (remote machine, project root):
    python analyze_compressibility.py \
        --data_path data/SQG/dataset/train \
        --stats_dir data/SQG/dataset/train \
        --max_samples 2000 \
        --out_dir figures/

Outputs:
    figures/spectrum_energy_fraction.pdf   (+ printed table)
    figures/pca_explained_variance.pdf     (+ printed table)
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Self-contained: no project imports. We read the .npy trajectory files and the
# .pt normalization stats directly, replicating SQGDataset's standardization
# (x - mean) / std and metrics.py's power_spectrum, so the script runs regardless
# of how your project package is structured.


# -----------------------------------------------------------------------------
# Plot style shared by both diagnostic figures
# -----------------------------------------------------------------------------
NAVY = "#1E2C4F"
TEAL = "#2C7A7B"
ORANGE = "#C8772E"
GREY = "#9AA6B2"
LIGHT_GREY = "#E7ECF1"
TEXT = "#2F3745"


def set_plot_style():
    """Small, thesis-friendly matplotlib style."""
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "axes.edgecolor": "#A9B4C0",
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def clean_axis(ax):
    """Remove visual clutter and apply a light horizontal grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#A9B4C0")
    ax.spines["bottom"].set_color("#A9B4C0")
    ax.tick_params(colors=TEXT, length=4, width=0.8)
    ax.yaxis.grid(True, color=LIGHT_GREY, linewidth=0.8)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


def boxed_text(ax, text, xy=(0.97, 0.08), ha="right", va="bottom"):
    """Consistent text box used for the threshold summaries."""
    ax.text(
        xy[0], xy[1], text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=10,
        color=TEXT,
        linespacing=1.35,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="#D7DEE8",
            linewidth=0.8,
            alpha=0.96,
        ),
    )


# -----------------------------------------------------------------------------
# Data and spectrum helpers
# -----------------------------------------------------------------------------
def power_spectrum(x):
    """2D power spectrum, averaged over channels. Identical to metrics.py.
    x: (B, C, H, W) -> returns (B, H, W)."""
    fft2 = torch.fft.fft2(x, norm="ortho")
    fftshift = torch.fft.fftshift(fft2, dim=(-2, -1))
    psd2D = torch.abs(fftshift) ** 2
    return psd2D.mean(1)


# Latent dimensionalities of the compression configs (per Table 4.3):
# total latent size = c * 16 * 16. Input = 2*64*64 = 8192 elements.
LATENT_SIZES = {
    "x1": 32 * 16 * 16,
    "x2": 16 * 16 * 16,
    "x4": 8 * 16 * 16,
    "x8": 4 * 16 * 16,
    "x16": 2 * 16 * 16,
}
INPUT_ELEMENTS = 2 * 64 * 64  # 8192


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True,
                    help="Directory of .npy trajectory files (e.g. .../train).")
    ap.add_argument("--stats_dir", default=None,
                    help="Dir with data_mean.pt/data_std.pt (defaults to data_path).")
    ap.add_argument("--max_samples", type=int, default=2000,
                    help="Number of snapshots to use (subsampled across the set).")
    ap.add_argument("--use_raw", action="store_true",
                    help="Analyse RAW (un-standardised) fields. Default uses the "
                         "standardised fields the AE actually sees.")
    ap.add_argument("--out_dir", default="figures")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def load_snapshots(args):
    """Load up to max_samples standardised snapshots of shape (N, 2, 64, 64),
    reading .npy trajectory files and .pt stats directly (no project imports)."""
    data_dir = Path(args.data_path)
    stats_dir = Path(args.stats_dir or args.data_path)

    files = sorted(data_dir.glob("*.npy"))
    assert len(files) > 0, f"No .npy files found in {data_dir}"
    steps_per_file = 100  # each trajectory file holds 100 snapshots

    mean = torch.load(stats_dir / "data_mean.pt", weights_only=True).float()  # (2,)
    std = torch.load(stats_dir / "data_std.pt", weights_only=True).float()    # (2,)

    n_total = len(files) * steps_per_file
    n = min(args.max_samples, n_total)
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(n_total, size=n, replace=False))
    print(f"Loading {n} / {n_total} snapshots from {len(files)} files...")

    fields = []
    for flat_i in idx:
        file_idx = flat_i // steps_per_file
        step_idx = flat_i % steps_per_file
        arr = np.load(files[file_idx], mmap_mode="r")[step_idx]  # (2, 64, 64)
        x = torch.from_numpy(np.array(arr)).float()
        if not args.use_raw:
            # standardise exactly as SQGDataset does
            x = (x - mean[:, None, None]) / std[:, None, None]
        fields.append(x)
    X = torch.stack(fields)  # (N, 2, 64, 64)
    print(f"  loaded tensor {tuple(X.shape)}  "
          f"({'raw' if args.use_raw else 'standardised'})")
    return X


# -----------------------------------------------------------------------------
# (1) Power-spectrum cumulative energy fraction
# -----------------------------------------------------------------------------
def spectrum_energy_fraction(X, out_dir):
    """
    Compute the radially-binned TOTAL energy per wavenumber, averaged over
    samples, then the cumulative fraction of total energy vs wavenumber.
    """
    # 2D PSD averaged over channels: (N, H, W)
    psd2d = power_spectrum(X)  # uses fft2(norm="ortho"); |.|^2; mean over C

    # Per-ring TOTAL energy (sum), NOT the per-ring mean used for spectrum plots.
    # We replicate the radial binning but keep the bincount sum (tbin) directly.
    psd2d = psd2d.cpu()
    N, H, W = psd2d.shape
    cy, cx = H // 2, W // 2
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    r = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).to(torch.int64)
    R = int(r.max().item()) + 1

    ring_energy = torch.zeros(R)
    for b in range(N):
        ring_energy += torch.bincount(
            r.flatten(),
            weights=psd2d[b].flatten(),
            minlength=R,
        )
    ring_energy /= N  # average total energy per ring over samples

    total = ring_energy.sum()
    cum_frac = torch.cumsum(ring_energy, dim=0) / total  # cumulative fraction
    cum_np = cum_frac.numpy()
    k = np.arange(R)

    # Report: wavenumbers needed to reach common thresholds
    print("\n=== Power-spectrum cumulative energy fraction ===")
    print(f"  Max resolved wavenumber (radius): {R - 1}")
    threshold_ks = {}
    for thr in (0.90, 0.95, 0.99):
        kk = int(np.searchsorted(cum_np, thr))
        threshold_ks[thr] = kk
        print(f"  Wavenumbers to reach {int(thr * 100)}% of energy: k = {kk}  "
              f"(= {kk}/{R - 1} = {100 * kk / (R - 1):.0f}% of resolved band)")

    # energy fraction captured by the lowest few k
    for kk in (1, 2, 3, 5, 8, 10):
        if kk < R:
            print(f"  Lowest k<= {kk:2d} wavenumbers hold {100 * cum_frac[kk].item():.1f}% of energy")

    # Plot
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.plot(k, cum_np, color=NAVY, lw=2.5)

    # Threshold guides: markers on the curve + compact summary box.
    for thr, kk in threshold_ks.items():
        ax.axhline(thr, color=GREY, ls="--", lw=0.9, alpha=0.85)
        ax.axvline(kk, ymax=thr / 1.02, color=GREY, ls="--", lw=0.8, alpha=0.55)
        ax.scatter([kk], [cum_np[kk]], s=28, color=ORANGE, zorder=5)

    summary_lines = [
        "Energy threshold",
        *[f"{int(thr * 100)}%: $k={kk}$" for thr, kk in threshold_ks.items()],
    ]
    boxed_text(ax, "\n".join(summary_lines), xy=(0.97, 0.08), ha="right", va="bottom")

    ax.set_xlabel(r"Radial wavenumber $k$")
    ax.set_ylabel("Cumulative spectral-energy fraction")
    ax.set_title(r"Cumulative spectral energy of the $64\times64$ SQG data",
                 fontweight="bold", color=NAVY, pad=12)
    ax.set_xlim(0, R - 1)
    ax.set_ylim(0, 1.02)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    clean_axis(ax)

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "spectrum_energy_fraction.pdf")
    fig.savefig(p, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"  saved -> {p}")
    return cum_np


# -----------------------------------------------------------------------------
# (2) PCA cumulative explained variance (LINEAR reference only)
# -----------------------------------------------------------------------------
def pca_explained_variance(X, out_dir):
    """
    Flatten each snapshot to a vector and compute PCA explained-variance ratio
    via SVD on the centred data matrix.
    """
    N = X.shape[0]
    flat = X.reshape(N, -1).numpy().astype(np.float64)  # (N, D), D=2*64*64=8192
    D = flat.shape[1]
    mean = flat.mean(axis=0, keepdims=True)
    Xc = flat - mean

    # economy SVD; singular values^2 give variance per component
    # (limited by min(N, D) components)
    print("\n=== PCA cumulative explained variance (LINEAR reference) ===")
    print(f"  Data matrix: {N} samples x {D} dims; max components = {min(N, D)}")
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2
    evr = var / var.sum()
    cum = np.cumsum(evr)

    threshold_components = {}
    for thr in (0.90, 0.95, 0.99):
        ncomp = int(np.searchsorted(cum, thr)) + 1
        threshold_components[thr] = ncomp
        print(f"  Components to explain {int(thr * 100)}% variance: {ncomp}  "
              f"(= {100 * ncomp / D:.1f}% of the {D} input dims)")

    # How much variance is explained at each compression budget's latent size?
    print("  Variance explained by a LINEAR subspace of the latent size:")
    latent_evr = {}
    for label, lat in LATENT_SIZES.items():
        if lat <= len(cum):
            latent_evr[label] = float(cum[lat - 1])
            print(f"    {label:>3s}: linear dim {lat:4d}  ->  "
                  f"{100 * cum[lat - 1]:.2f}% variance "
                  f"(latent is {100 * lat / INPUT_ELEMENTS:.0f}% of input size)")
        else:
            print(f"    {label:>3s}: linear dim {lat:4d}  -> exceeds available "
                  f"components ({len(cum)})")

    # Plot
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ks = np.arange(1, len(cum) + 1)
    ax.plot(ks, cum, color=NAVY, lw=2.5)

    # Threshold lines and compact summary box.
    for thr, ncomp in threshold_components.items():
        ax.axhline(thr, color=GREY, ls="--", lw=0.9, alpha=0.85)
        ax.axvline(ncomp, ymax=thr / 1.02, color=GREY, ls="--", lw=0.8, alpha=0.45)
        ax.scatter([ncomp], [cum[ncomp - 1]], s=28, color=ORANGE, zorder=5)

    summary_lines = [
        "Explained variance",
        *[f"{int(thr * 100)}%: {ncomp} PCs" for thr, ncomp in threshold_components.items()],
    ]
    boxed_text(ax, "\n".join(summary_lines), xy=(0.97, 0.08), ha="right", va="bottom")

    # Mark latent budgets that are within the available PCA rank.
    for label in ("x16", "x8"):
        lat = LATENT_SIZES[label]
        if lat <= len(cum):
            ax.axvline(lat, color=TEAL, ls=":", lw=1.3, alpha=0.9)
            ax.text(
                lat + 18,
                0.18,
                f"{label}: {100 * cum[lat - 1]:.1f}%",
                rotation=90,
                va="bottom",
                ha="left",
                fontsize=9.5,
                color=TEAL,
            )

    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title(r"Linear compressibility reference for the $64\times64$ SQG data",
                 fontweight="bold", color=NAVY, pad=12)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, len(cum))
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    clean_axis(ax)

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "pca_explained_variance.pdf")
    fig.savefig(p, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"  saved -> {p}")
    return cum


def main():
    args = parse_args()
    set_plot_style()
    torch.manual_seed(args.seed)
    X = load_snapshots(args)
    spectrum_energy_fraction(X, args.out_dir)
    pca_explained_variance(X, args.out_dir)
    print("\nDone. Both diagnostics SUPPORT (not prove) the limited-redundancy "
          "interpretation; report them with hedged language and note that a "
          "matched cross-resolution study would be needed to establish it.")


if __name__ == "__main__":
    main()
