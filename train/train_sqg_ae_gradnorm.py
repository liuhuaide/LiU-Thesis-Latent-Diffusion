import torch
import argparse
import wandb
from torch.utils.data import DataLoader
from pathlib import Path
from networks.autoencoder import get_autoencoder, AutoEncoderLoss
from data.sqg_dataloader import SQGDataset  # Note: the imported name must match your file/class name

# ============ Config ============
parser = argparse.ArgumentParser(description="Train SQG Autoencoder")
parser.add_argument("--compression", type=int, default=4, help="Target compression rate (1, 2, 4, 8, 16)")
parser.add_argument("--version", type=str, default=None, help="Version suffix, e.g. v2 (default: None = v1 behavior)")
parser.add_argument("--spectral_weight", type=float, default=0.0, help="Weight for spectral loss (0 = no spectral loss)")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
parser.add_argument("--attention_stage", type=int, default=None, help="Stage index to add self-attention (e.g. 2 for Stage 2)")
parser.add_argument("--attention_heads", type=int, default=4, help="Number of attention heads (default: 4)")
parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "psgd"], help="Optimizer (default: adamw)")
parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: auto per compression for adamw, 1e-3 for psgd)")
parser.add_argument("--hid_channels", type=int, nargs=3, default=[64, 128, 256], help="Hidden channels per stage (default: 64 128 256)")
parser.add_argument("--skip_mode", type=str, default=None, choices=["bilinear", "nearest"], help="Latent skip connection mode (default: None = no skip)")
args = parser.parse_args()
COMPRESSION = args.compression
VERSION = args.version

LAT_CHANNELS = {1: 32, 2: 16, 4: 8, 8: 4, 16: 2}[COMPRESSION]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = args.epochs
BATCH_SIZE = 32
LR_MAP = {1: 1e-4, 2: 1e-4, 4: 3e-5, 8: 1e-5, 16: 5e-6}
LR = LR_MAP[COMPRESSION]

# Save path: ae_x4 (v1) or ae_x4_v2 (with version)
version_suffix = f"_{VERSION}" if VERSION else ""
SAVE_DIR = Path(f"saved_models/ae_x{COMPRESSION}{version_suffix}")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ============ Data ============
# Both train and val use the statistics from the train directory
train_dataset = SQGDataset("data/SQG/dataset/train", stats_dir="data/SQG/dataset/train")
val_dataset   = SQGDataset("data/SQG/dataset/validation", stats_dir="data/SQG/dataset/train")

train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ============ Model ============
attn_heads = {args.attention_stage: args.attention_heads} if args.attention_stage is not None else {}
hid_ch = tuple(args.hid_channels)
autoencoder = get_autoencoder(
    pix_channels=2,
    lat_channels=LAT_CHANNELS,
    spatial=2,
    arch="dcae",
    saturation="softclip2",
    hid_channels=hid_ch,
    hid_blocks=(3, 3, 3),
    attention_heads=attn_heads,
    skip_mode=args.skip_mode,
    periodic=True,
    identity_init=True,
)
autoencoder = autoencoder.to(DEVICE)
print(f"hid_channels: {hid_ch}")
if args.skip_mode:
    print(f"Latent skip: {args.skip_mode}")
if attn_heads:
    print(f"Self-attention enabled: {attn_heads}")

# ============ Loss ============
if args.spectral_weight > 0:
    loss_fn = AutoEncoderLoss(
        losses=["vmse", "spectral"],
        weights=[1.0, args.spectral_weight],
    ).to(DEVICE)
    loss_desc = f"vmse + {args.spectral_weight}×spectral"
else:
    loss_fn = AutoEncoderLoss(losses=["vmse"], weights=[1.0]).to(DEVICE)
    loss_desc = "vmse"
print(f"Loss: {loss_desc}")

# ============ Optimizer ============
if args.optimizer == "psgd":
    from kron_torch import Kron
    psgd_lr = args.lr if args.lr is not None else 1e-3
    optimizer = Kron(autoencoder.parameters(), lr=psgd_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    print(f"Optimizer: PSGD Kron (lr={psgd_lr})")
else:
    adamw_lr = args.lr if args.lr is not None else LR
    optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=adamw_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    print(f"Optimizer: AdamW (lr={adamw_lr})")

# ============ Verify compression rate ============
with torch.no_grad():
    dummy = torch.randn(1, 2, 64, 64).to(DEVICE)
    z = autoencoder.encode(dummy)
    print(f"Input: {dummy.shape} -> Latent: {z.shape}")
    print(f"Actual compression rate: {dummy.numel() / z.numel():.1f}x")

# ============ Training ============
wandb_name = f"ae_x{COMPRESSION}{version_suffix}"
wandb.init(project="sqg-autoencoder", name=wandb_name)

best_val_loss = float("inf")

for epoch in range(EPOCHS):
    # --- train ---
    autoencoder.train()
    train_loss = 0.0
    max_grad_norm = 0.0  # peak pre-clip gradient norm within this epoch
    for x in train_loader:
        x = x.to(DEVICE)
        if args.optimizer == "psgd":
            # PSGD requires closure; zero_grad inside because closure may be called multiple times
            _gn = [0.0]  # container to carry the norm out of the closure
            def closure():
                optimizer.zero_grad()
                loss = loss_fn(autoencoder, x)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
                _gn[0] = float(gn)
                return loss
            loss = optimizer.step(closure)
            grad_norm = _gn[0]
        else:
            loss = loss_fn(autoencoder, x)
            optimizer.zero_grad()
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0))
            optimizer.step()
        max_grad_norm = max(max_grad_norm, grad_norm)
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # --- val ---
    autoencoder.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x in val_loader:
            x = x.to(DEVICE)
            loss = loss_fn(autoencoder, x)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    scheduler.step()
    print(f"Epoch {epoch+1}/{EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | max_grad_norm={max_grad_norm:.3f}")
    wandb.log({"train_loss": train_loss, "val_loss": val_loss, "max_grad_norm": max_grad_norm, "epoch": epoch+1})

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # Save as best.pth
        torch.save(autoencoder.state_dict(), SAVE_DIR / "best.pth")
        print(f"  🌟 [new best] Validation error reached a new low! Model saved to best.pth")
    
torch.save(autoencoder.state_dict(), SAVE_DIR / "state.pth")
print(f"Model saved to {SAVE_DIR}")
