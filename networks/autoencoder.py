r"""Auto-encoder building blocks."""

__all__ = [
    "AutoEncoder",
    "AutoEncoderLoss",
    "get_autoencoder",
]

import math
import torch
import torch.nn as nn

from einops import rearrange
from omegaconf import DictConfig
from torch import Tensor
from torch.nn.functional import cosine_similarity
from typing import Any, Dict, Optional, Sequence, Tuple

from .dcae import DCDecoder, DCEncoder, LatentSkipConnection



class AutoEncoder(nn.Module):
    r"""Creates an auto-encoder module.

    Arguments:
        encoder: An encoder module.
        decoder: A decoder module.
        saturation: The type of latent saturation.
        noise: The latent noise's standard deviation.
        latent_skip: Optional LatentSkipConnection module.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        saturation: str = "softclip2",
        noise: float = 0.0,
        latent_skip: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.saturation = saturation
        self.noise = noise
        self.latent_skip = latent_skip

    def saturate(self, x: Tensor) -> Tensor:
        if self.saturation is None:
            return x
        elif self.saturation == "softclip":
            return x / (1 + abs(x) / 5)
        elif self.saturation == "softclip2":
            return x * torch.rsqrt(1 + torch.square(x / 5))
        elif self.saturation == "tanh":
            return torch.tanh(x / 5) * 5
        elif self.saturation == "arcsinh":
            return torch.arcsinh(x)
        elif self.saturation == "rmsnorm":
            return x * torch.rsqrt(torch.mean(torch.square(x), dim=1, keepdim=True) + 1e-5)
        else:
            raise ValueError(f"unknown saturation '{self.saturation}'")

    def encode(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        z = self.saturate(z)
        return z

    def decode(self, z: Tensor, noisy: bool = True) -> Tensor:
        if noisy and self.noise > 0:
            z = z + self.noise * torch.randn_like(z)

        return self.decoder(z, latent_skip=self.latent_skip)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z = self.encode(x)
        y = self.decode(z)
        return y, z


class AutoEncoderLoss(nn.Module):
    r"""Creates a weighted auto-encoder loss module."""

    def __init__(
        self,
        losses: Sequence[str] = ["mse"],  # noqa: B006
        weights: Sequence[float] = [1.0],  # noqa: B006
    ):
        super().__init__()

        assert len(losses) == len(weights)

        self.losses = list(losses)
        self.register_buffer("weights", torch.as_tensor(weights))

    def forward(self, autoencoder: AutoEncoder, x: Tensor, **kwargs) -> Tensor:
        r"""
        Arguments:
            x: A clean tensor :math:`x`, with shape :math:`(B, C, ...)`.
            kwargs: Optional keyword arguments.

        Returns:
            The weighted loss.
        """

        y, z = autoencoder(x, **kwargs)

        # Keep original 4D tensors for all loss branches
        x_orig, y_orig = x, y

        values = []

        for loss in self.losses:
            if loss == "mse":
                l = (x_orig - y_orig).square().mean()
            elif loss == "mae":
                l = (x_orig - y_orig).abs().mean()
            elif loss == "vmse":
                xf = rearrange(x_orig, "B C ... -> B C (...)")
                yf = rearrange(y_orig, "B C ... -> B C (...)")
                l = (xf - yf).square().mean(dim=2) / (xf.var(dim=2) + 1e-2)
                l = l.mean()
            elif loss == "vrmse":
                xf = rearrange(x_orig, "B C ... -> B C (...)")
                yf = rearrange(y_orig, "B C ... -> B C (...)")
                l = (xf - yf).square().mean(dim=2) / (xf.var(dim=2) + 1e-2)
                l = torch.sqrt(l).mean()
            elif loss == "spectral":
                # Spectral loss: MSE of log-magnitude spectra
                # L = mean( (log|FFT2D(x)| - log|FFT2D(y)|)^2 )
                eps = 1e-7
                fx = torch.fft.rfft2(x_orig)  # (B, C, H, W//2+1) complex
                fy = torch.fft.rfft2(y_orig)
                log_mag_x = torch.log(fx.abs() + eps)
                log_mag_y = torch.log(fy.abs() + eps)
                l = (log_mag_x - log_mag_y).square().mean()
            elif loss == "similarity":
                f = rearrange(z, "B ... -> B (...)")
                l = cosine_similarity(f[None, :], f[:, None], dim=-1)
                l = l[*torch.triu_indices(*l.shape, offset=1, device=l.device)]
                l = l.mean()
            else:
                raise ValueError(f"unknown loss '{loss}'.")

            values.append(l)

        values = torch.stack(values)

        return torch.vdot(self.weights, values)


def get_autoencoder(
    pix_channels: int,
    lat_channels: int,
    spatial: int = 2,
    # Arch
    arch: Optional[str] = None,
    saturation: str = "softclip2",
    # Asymmetry
    encoder_only: Dict[str, Any] = {},  # noqa: B006
    decoder_only: Dict[str, Any] = {},  # noqa: B006
    # Noise
    latent_noise: float = 0.0,
    # Skip connection
    skip_mode: Optional[str] = None,  # "bilinear", "nearest", or None
    # Ignore
    name: Optional[str] = None,
    loss: Optional[DictConfig] = None,
    # Passthrough
    **kwargs,
) -> AutoEncoder:
    r"""Instantiates an auto-encoder.

    Arguments:
        skip_mode: If set, adds a LatentSkipConnection to the decoder.
            'bilinear' or 'nearest' interpolation. None = no skip (default).
    """

    # Extract hid_channels and stride for skip target computation
    hid_channels = kwargs.get("hid_channels", (64, 128, 256))
    stride = kwargs.get("stride", 2)
    if isinstance(stride, int):
        stride = [stride] * spatial

    if arch in (None, "dcae"):
        encoder = DCEncoder(
            in_channels=pix_channels,
            out_channels=lat_channels,
            spatial=spatial,
            **encoder_only,
            **kwargs,
        )

        decoder = DCDecoder(
            in_channels=lat_channels,
            out_channels=pix_channels,
            spatial=spatial,
            **decoder_only,
            **kwargs,
        )
    else:
        raise NotImplementedError()

    # Build LatentSkipConnection if requested
    latent_skip = None
    if skip_mode is not None:
        # Decoder ascent order is reversed: deepest stage first
        # For hid_channels=(64,128,256), ascent processes [256, 128, 64]
        # Spatial sizes: 16→32→64 (for 2 levels of stride-2)
        reversed_channels = list(reversed(hid_channels))
        n_stages = len(hid_channels)

        # Compute spatial sizes at each decoder stage
        # Deepest = latent size, then multiply by stride per level at each upsample
        # stride=[2,2] means each spatial dim doubles per level
        base_size = 16  # latent spatial size (fixed for your pipeline)
        stride_factor = stride[0]  # per-dimension stride (e.g. 2)
        target_channels_list = []
        target_sizes = []

        for stage_idx in range(n_stages):
            target_channels_list.append(reversed_channels[stage_idx])
            scale = stride_factor ** stage_idx
            size = base_size * scale
            target_sizes.append([size] * spatial)

        latent_skip = LatentSkipConnection(
            lat_channels=lat_channels,
            target_channels_list=target_channels_list,
            target_sizes=target_sizes,
            spatial=spatial,
            mode=skip_mode,
            periodic=kwargs.get("periodic", False),
        )

    autoencoder = AutoEncoder(
        encoder,
        decoder,
        saturation=saturation,
        noise=latent_noise,
        latent_skip=latent_skip,
    )

    return autoencoder

