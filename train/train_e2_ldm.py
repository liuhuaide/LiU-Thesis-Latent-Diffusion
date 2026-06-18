"""
Joint training of E₂ (new encoder) + Latent EDM.

Pipeline:
  1. Load pixel-space data (init_states, target_states) from SQGForecastDataset
  2. E₂(pixel init_states) → latent init_states     [E₂ trainable]
  3. E₁(pixel target_states) → latent target          [E₁ frozen]
  4. LDM trains on (latent init, latent target)        [LDM trainable]
  5. Loss = latent-space diffusion loss

E₂ architecture: same as E₁ (DCEncoder) but with in_channels=4 (2 frames × 2 PV channels)
E₁ and Decoder D are frozen (loaded from pre-trained AE).

Usage:
  export PYTHONPATH=$(pwd)
  nohup python train_e2_ldm.py \
      --ae_ckpt saved_models/ae_x4/best.pth \
      --compression 4 \
      --epochs 50 \
      --batch_size 32 \
      --lr 1e-3 \
      --wandb_run_name "E2_LDM_x4" \
      > train_e2_ldm.log 2>&1 &
"""

import os
import sys
import time
import random
import argparse

import torch
import pytorch_lightning as pl
from lightning_fabric.utilities import seed
from pytorch_lightning.callbacks import LearningRateMonitor

from networks.autoencoder import get_autoencoder
from networks.dcae import DCEncoder
from data.SQG.QG_dataset import SQGForecastDataset

# ============ E2-LDM Wrapper Model ============

class E2LatentEDM(pl.LightningModule):
    """
    Wraps E₂ (trainable encoder) + LatentEDM (trainable) + E₁/D (frozen).
    
    Training loop:
      1. Pixel init_states (B, init_states, 2, 64, 64)
         → E₂ encodes each frame → latent init (B, init_states, lat_ch, 16, 16)
      2. Pixel target_states (B, ar_steps, 2, 64, 64)
         → E₁ encodes each frame → latent target (B, ar_steps, lat_ch, 16, 16)
      3. LDM.predict_step_train(latent_init, latent_target) → loss
      4. Gradients flow to E₂ and LDM, not E₁.
    """

    def __init__(self, args, e2_encoder, latent_edm, ae_e1):
        super().__init__()
        self.save_hyperparameters(args)
        self.args = args

        # E₂: trainable new encoder (in_channels=4)
        self.e2_encoder = e2_encoder

        # Latent EDM: trainable
        self.latent_edm = latent_edm

        # E₁: frozen encoder (for computing targets)
        self.ae_e1 = ae_e1
        self.ae_e1.eval()
        for p in self.ae_e1.parameters():
            p.requires_grad_(False)

    def encode_with_e2(self, pixel_states):
        """
        Encode pixel states using E₂.
        
        Args:
            pixel_states: (B, N_frames, 2, 64, 64)
        Returns:
            latent_states: (B, N_frames, lat_ch, 16, 16)
        """
        B, N, C, H, W = pixel_states.shape
        # Concatenate frames along channel dim: (B, N*C, H, W)
        x_concat = pixel_states.reshape(B, N * C, H, W)
        # E₂ encodes to single latent: (B, lat_ch, 16, 16)
        z = self.e2_encoder(x_concat)
        # Apply saturation (same as AE)
        z = z * torch.rsqrt(1 + torch.square(z / 5))
        return z

    def encode_with_e1(self, pixel_states):
        """
        Encode pixel states using frozen E₁ (one frame at a time).
        
        Args:
            pixel_states: (B, N_frames, 2, 64, 64)
        Returns:
            latent_states: (B, N_frames, lat_ch, 16, 16)
        """
        B, N, C, H, W = pixel_states.shape
        flat = pixel_states.reshape(B * N, C, H, W)
        with torch.no_grad():
            z = self.ae_e1.encode(flat)  # (B*N, lat_ch, 16, 16)
        return z.reshape(B, N, *z.shape[1:])

    def standardize_latent(self, z):
        """Standardize latent using LDM's stats."""
        return (z - self.latent_edm.data_mean) / self.latent_edm.data_std

    def training_step(self, batch):
        pixel_init, pixel_target = batch
        # pixel_init: (B, init_states, 2, 64, 64)
        # pixel_target: (B, ar_steps, 2, 64, 64)

        # E₂ encode init states → single latent representation
        # E₂ takes all init frames concatenated as input
        latent_init_raw = self.encode_with_e2(pixel_init)  # (B, lat_ch, 16, 16)

        # E₁ encode target states (frozen)
        latent_target_raw = self.encode_with_e1(pixel_target)  # (B, ar_steps, lat_ch, 16, 16)

        # Standardize latents using LDM's stats
        latent_init_std = self.standardize_latent(latent_init_raw)
        latent_target_std = self.standardize_latent(latent_target_raw)

        # For ar_steps=1: we need init_states format (B, N_init, lat_ch, H, W)
        # E₂ gives us a single latent, but LDM expects (B, init_states, lat_ch, H, W)
        # We duplicate it to match expected shape (or reshape as needed)
        # Actually, LDM's predict_step_train expects:
        #   init_states: (B, N_steps, lat_ch, H, W)
        #   true_state: (B, lat_ch, H, W)

        # Build init_states for LDM: we need init_states=2 frames in latent space
        # Option: E₂ gives one latent from 2 pixel frames, but LDM expects 2 latent frames
        # Solution: also encode each pixel frame individually with E₂'s first C channels
        # Actually simpler: encode each frame with E₁ for init, use E₂ only for a richer encoding
        # 
        # Wait - let's reconsider. The original LDM uses:
        #   init_states = (B, 2, lat_ch, 16, 16) — 2 latent frames from E₁
        # 
        # In the new scheme, E₂ takes (X_{t-1}, X_t) as 4-channel input and produces
        # a single latent. But LDM expects 2 latent frames as conditioning.
        #
        # Simplest approach: use E₁ to encode each init frame individually (frozen),
        # then E₂ output is NOT used for LDM conditioning but instead we train E₂
        # to produce a better latent encoding. But that changes the architecture...
        #
        # Actually the teacher's idea is:
        #   L_t = E₂(X_t, X_{t-1})  — E₂ encodes 2 frames into 1 latent
        #   LDM(L_t) → L̂_{t+1}     — LDM predicts from this single latent
        #
        # So LDM needs to accept single-frame conditioning (init_states=1).
        # Let's reshape accordingly:
        
        latent_init_for_ldm = latent_init_std.unsqueeze(1)  # (B, 1, lat_ch, H, W)

        # Unroll prediction
        total_loss = 0.0
        ar_steps = pixel_target.shape[1]

        current_init = latent_init_for_ldm  # (B, 1, lat_ch, H, W)

        for step in range(ar_steps):
            target = latent_target_std[:, step]  # (B, lat_ch, H, W)
            _, step_loss = self.latent_edm.predict_step_train(current_init, target)
            total_loss = total_loss + step_loss.mean()

            # For next step: use predicted state (or target for teacher forcing)
            # Using teacher forcing for stability:
            current_init = target.unsqueeze(1)  # (B, 1, lat_ch, H, W)

        total_loss = total_loss / ar_steps

        self.log("train_loss", total_loss, prog_bar=True)
        return total_loss

    def validation_step(self, batch):
        pixel_init, pixel_target = batch

        latent_init_raw = self.encode_with_e2(pixel_init)
        latent_target_raw = self.encode_with_e1(pixel_target)

        latent_init_std = self.standardize_latent(latent_init_raw)
        latent_target_std = self.standardize_latent(latent_target_raw)

        latent_init_for_ldm = latent_init_std.unsqueeze(1)

        total_loss = 0.0
        ar_steps = pixel_target.shape[1]
        current_init = latent_init_for_ldm

        for step in range(ar_steps):
            target = latent_target_std[:, step]
            _, step_loss = self.latent_edm.predict_step_train(current_init, target)
            total_loss = total_loss + step_loss.mean()
            current_init = target.unsqueeze(1)

        total_loss = total_loss / ar_steps

        self.log("val_mean_loss", total_loss, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        # Only optimize E₂ and LDM parameters (E₁ is frozen)
        params = list(self.e2_encoder.parameters()) + list(self.latent_edm.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.args.epochs
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


# ============ Main ============

def list_of_ints(arg):
    return list(map(int, arg.split(',')))


def main():
    parser = argparse.ArgumentParser(description="Joint E2 + Latent EDM Training")

    # Data
    parser.add_argument("--pixel_data_path", type=str, default="data/SQG/dataset",
                        help="Path to pixel-space dataset root")
    parser.add_argument("--latent_stats_path", type=str, default=None,
                        help="Path to latent stats (default: data/SQG/dataset_latent_x{comp}/train)")

    # AE
    parser.add_argument("--ae_ckpt", type=str, required=True,
                        help="Path to pre-trained AE checkpoint (E₁)")
    parser.add_argument("--compression", type=int, default=4)

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ar_steps", type=int, default=1)
    parser.add_argument("--init_states", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_workers", type=int, default=4)

    # LDM backbone
    parser.add_argument("--sampler", type=str, default="heun")
    parser.add_argument("--sampler_steps", type=int, default=20)
    parser.add_argument("--loss", type=str, default="mse")
    parser.add_argument("--resample_filter", type=list_of_ints, default="1,3,3,1")
    parser.add_argument("--channel_mult", type=list_of_ints, default="2,2,2,2")
    parser.add_argument("--encoder_type", type=str, default="residual")
    parser.add_argument("--attn_resolutions", type=list_of_ints, default="1")
    parser.add_argument("--channel_mult_emb", type=int, default=4)
    parser.add_argument("--channel_mult_noise", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=32)

    # Wandb
    parser.add_argument("--wandb_run_name", type=str, default="E2_LDM")
    parser.add_argument("--wandb_project", type=str, default="SQG_Local_Train")

    args = parser.parse_args()

    # Derived
    LAT_CHANNELS_MAP = {1: 32, 2: 16, 4: 8, 8: 4, 16: 2}
    args.lat_channels = LAT_CHANNELS_MAP[args.compression]
    args.nx = 16  # latent spatial size

    if args.latent_stats_path is None:
        args.latent_stats_path = f"data/SQG/dataset_latent_x{args.compression}/train"

    # Set args needed by LatentEDM but not directly used here
    args.data_path = args.latent_stats_path
    args.eval = None
    args.metrics_watch = ["val_rmse"]
    args.val_steps_to_log = [1, 25, 50]
    args.step_length = 3
    args.ar_steps_eval = 1
    args.pred_residual = False  # E₂ and E₁ produce different latent spaces, residual is invalid

    # Args required by ARModel / ARProbModel / LatentEDM but not used in this script
    args.restore_opt = False
    args.n_example_pred = 1
    args.ensemble_size = 5
    args.min_lr = 1e-5
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.spectral_loss_weight = 0
    args.sampler_steps = 20

    seed.seed_everything(args.seed)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Load pre-trained AE (E₁ + D) ----
    print("Loading pre-trained AE (E₁)...")
    ae = get_autoencoder(
        pix_channels=2, lat_channels=args.lat_channels, spatial=2,
        arch="dcae", saturation="softclip2",
        hid_channels=(64, 128, 256), hid_blocks=(3, 3, 3),
        periodic=True, identity_init=True,
    )
    ae.load_state_dict(torch.load(args.ae_ckpt, map_location="cpu", weights_only=True))
    ae.eval()
    print(f"  Loaded E₁ from {args.ae_ckpt}")

    # ---- Create E₂ (new encoder, in_channels=4) ----
    print("Creating E₂ (in_channels=4)...")
    e2_encoder = DCEncoder(
        in_channels=2 * args.init_states,  # 4 channels (2 frames × 2 PV)
        out_channels=args.lat_channels,
        hid_channels=(64, 128, 256),
        hid_blocks=(3, 3, 3),
        periodic=True,
        identity_init=True,
    )
    print(f"  E₂ params: {sum(p.numel() for p in e2_encoder.parameters()):,}")

    # ---- Create Latent EDM ----
    print("Creating Latent EDM...")
    from forecasting.models.latent_edm import LatentEDM

    # LDM expects init_states=1 now (E₂ encodes 2 pixel frames → 1 latent)
    args_ldm = argparse.Namespace(**vars(args))
    args_ldm.init_states = 1  # E₂ output is single frame
    latent_edm = LatentEDM(args_ldm)
    print(f"  LDM params: {sum(p.numel() for p in latent_edm.parameters()):,}")

    # ---- Create joint model ----
    model = E2LatentEDM(args, e2_encoder, latent_edm, ae)

    # ---- Data loaders (pixel space!) ----
    # SQGForecastDataset loads stats from data_path, so both train and val
    # should point to directories that contain data_mean.pt and data_std.pt
    pixel_train_path = os.path.join(args.pixel_data_path, "train")
    pixel_val_path = os.path.join(args.pixel_data_path, "validation")

    print(f"Loading pixel-space data...")
    print(f"  Train: {pixel_train_path}")
    print(f"  Val:   {pixel_val_path}")

    train_loader = torch.utils.data.DataLoader(
        SQGForecastDataset(
            data_path=pixel_train_path,
            subsample_step=args.step_length // 3,
            nx=64,
            pred_length=args.ar_steps,
            init_states=args.init_states,
            split="train",
            standardize=True,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
    )

    val_loader = torch.utils.data.DataLoader(
        SQGForecastDataset(
            data_path=pixel_val_path,
            subsample_step=args.step_length // 3,
            nx=64,
            pred_length=args.ar_steps,
            init_states=args.init_states,
            split="val",
            standardize=True,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
    )

    # ---- Trainer ----
    random_run_id = random.randint(0, 9999)
    run_name = f"{args.wandb_run_name}-E2LDM-{args.hidden_dim}-{time.strftime('%m_%d_%H')}-{random_run_id:04d}"

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=f"saved_models/{run_name}",
            filename="min_val_loss",
            monitor="val_mean_loss",
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval='epoch'),
    ]

    logger = pl.loggers.WandbLogger(
        project=args.wandb_project,
        name=run_name,
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        deterministic=True,
        accelerator="gpu" if DEVICE == "cuda" else "cpu",
        devices=1,
        log_every_n_steps=10,
        callbacks=callbacks,
        logger=logger,
        check_val_every_n_epoch=1,
    )

    trainer.fit(model, train_loader, val_loader)
    print(f"Training complete. Model saved to saved_models/{run_name}/")


if __name__ == "__main__":
    main()
