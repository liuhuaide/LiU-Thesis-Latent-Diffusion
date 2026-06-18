"""
Inference Time Benchmark — Compare Pixel EDM vs Latent EDM at different compression rates.

This script does NOT require trained models. It creates randomly initialized models
and measures pure forward-pass (denoise) time, because computation time depends only
on network architecture, not on weight values.

Usage:
    PYTHONPATH=$(pwd) python benchmark_inference.py
"""

import torch
import time
import numpy as np
from argparse import Namespace

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ Configuration ============
# Compression rate → (lat_channels, spatial_size)
CONFIGS = {
    "Pixel EDM (64×64)": {"lat_channels": 2, "nx": 64, "init_states": 2},
    "Latent x2 (16ch, 16×16)": {"lat_channels": 16, "nx": 16, "init_states": 2},
    "Latent x4 (8ch, 16×16)":  {"lat_channels": 8,  "nx": 16, "init_states": 2},
    "Latent x8 (4ch, 16×16)":  {"lat_channels": 4,  "nx": 16, "init_states": 2},
    "Latent x16 (2ch, 16×16)": {"lat_channels": 2,  "nx": 16, "init_states": 2},
}

BATCH_SIZES = [1, 4, 10, 32]
N_WARMUP = 10
N_REPEATS = 50
SAMPLER_STEPS = 20  # Heun sampler: each AR step = sampler_steps * 2 denoise calls


def make_dummy_args(lat_channels, nx, init_states):
    """Create minimal args to instantiate SongUNet."""
    return Namespace(
        nx=nx,
        lat_channels=lat_channels,
        init_states=init_states,
        resample_filter=[1, 3, 3, 1],
        channel_mult=[2, 2, 2, 2],
        encoder_type="residual",
        attn_resolutions=[1],
        channel_mult_emb=4,
        channel_mult_noise=1,
        hidden_dim=32,
    )


def build_unet(args):
    """Build a SongUNet with given configuration."""
    from networks.diffusion_networks import SongUNet

    in_ch = args.lat_channels * args.init_states + args.lat_channels
    out_ch = args.lat_channels

    model = SongUNet(
        img_resolution=torch.as_tensor(args.nx),
        in_channels=in_ch,
        out_channels=out_ch,
        embedding_type="fourier",
        resample_filter=args.resample_filter,
        channel_mult=args.channel_mult,
        encoder_type=args.encoder_type,
        attn_resolutions=args.attn_resolutions,
        channel_mult_emb=args.channel_mult_emb,
        channel_mult_noise=args.channel_mult_noise,
    )
    return model.to(DEVICE).eval()


@torch.no_grad()
def benchmark_denoise(model, x_shape, label_shape, batch_size):
    """Benchmark a single denoise call (one UNet forward pass)."""
    x = torch.randn(batch_size, *x_shape, device=DEVICE)
    sigma = torch.ones(batch_size, device=DEVICE)
    labels = torch.randn(batch_size, *label_shape, device=DEVICE)

    # Warmup
    for _ in range(N_WARMUP):
        model(x, sigma, class_labels=labels)
    torch.cuda.synchronize()

    # Benchmark
    t_start = time.time()
    for _ in range(N_REPEATS):
        model(x, sigma, class_labels=labels)
    torch.cuda.synchronize()
    elapsed = (time.time() - t_start) / N_REPEATS

    return elapsed


def main():
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("Inference Time Benchmark")
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Warmup: {N_WARMUP}, Repeats: {N_REPEATS}")
    print(f"Sampler steps: {SAMPLER_STEPS} (Heun → ~{SAMPLER_STEPS * 2} denoise calls per AR step)")
    print("=" * 70)

    results = {}  # {config_name: {batch_size: denoise_time_ms}}

    for name, cfg in CONFIGS.items():
        print(f"\n--- {name} ---")
        args = make_dummy_args(**cfg)
        model = build_unet(args)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        in_ch = cfg["lat_channels"]
        cond_ch = cfg["lat_channels"] * cfg["init_states"]
        nx = cfg["nx"]
        x_shape = (in_ch, nx, nx)
        label_shape = (cond_ch, nx, nx)

        results[name] = {}

        for bs in BATCH_SIZES:
            try:
                t = benchmark_denoise(model, x_shape, label_shape, bs)
                t_ms = t * 1000
                ar_step_ms = t_ms * SAMPLER_STEPS * 2  # Heun: 2 denoise per sampler step
                results[name][bs] = t_ms
                print(f"  BS={bs:>3d}: {t_ms:>8.2f}ms/denoise  →  {ar_step_ms:>8.0f}ms/AR_step")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    results[name][bs] = float("nan")
                    print(f"  BS={bs:>3d}: OOM!")
                    torch.cuda.empty_cache()
                else:
                    raise

        del model
        torch.cuda.empty_cache()

    # ============ Summary Table ============
    print(f"\n{'=' * 70}")
    print("SUMMARY: Denoise time (ms) per call")
    print(f"{'=' * 70}")

    header = f"{'Model':<30}" + "".join(f"{'BS='+str(bs):>12}" for bs in BATCH_SIZES)
    print(header)
    print("-" * len(header))

    for name in results:
        row = f"{name:<30}"
        for bs in BATCH_SIZES:
            t = results[name].get(bs, float("nan"))
            if np.isnan(t):
                row += f"{'OOM':>12}"
            else:
                row += f"{t:>10.2f}ms"
        print(row)

    # ============ Speedup relative to Pixel EDM ============
    pixel_key = "Pixel EDM (64×64)"
    if pixel_key in results:
        print(f"\nSpeedup vs Pixel EDM:")
        for name in results:
            if name == pixel_key:
                continue
            row = f"  {name:<28}"
            for bs in BATCH_SIZES:
                t_pixel = results[pixel_key].get(bs, float("nan"))
                t_latent = results[name].get(bs, float("nan"))
                if np.isnan(t_pixel) or np.isnan(t_latent) or t_latent == 0:
                    row += f"{'N/A':>12}"
                else:
                    row += f"{t_pixel/t_latent:>10.1f}x  "
            print(row)

    # ============ Plot ============
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: denoise time vs batch size
    ax = axes[0]
    for name in results:
        bs_list = []
        times = []
        for bs in BATCH_SIZES:
            t = results[name].get(bs, float("nan"))
            if not np.isnan(t):
                bs_list.append(bs)
                times.append(t)
        ax.plot(bs_list, times, "o-", label=name, linewidth=2, markersize=6)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Denoise Time (ms)")
    ax.set_title("Single Denoise Call")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(BATCH_SIZES)

    # Right: estimated AR step time (Heun sampler) vs batch size
    ax = axes[1]
    for name in results:
        bs_list = []
        times = []
        for bs in BATCH_SIZES:
            t = results[name].get(bs, float("nan"))
            if not np.isnan(t):
                bs_list.append(bs)
                times.append(t * SAMPLER_STEPS * 2)  # Heun
        ax.plot(bs_list, times, "o-", label=name, linewidth=2, markersize=6)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("AR Step Time (ms)")
    ax.set_title(f"Full AR Step (Heun, {SAMPLER_STEPS} steps)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(BATCH_SIZES)

    plt.tight_layout()
    plt.savefig("eval_results/benchmark_inference.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to eval_results/benchmark_inference.png")


if __name__ == "__main__":
    main()
