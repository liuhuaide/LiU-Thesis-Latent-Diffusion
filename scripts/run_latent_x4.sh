#!/bin/bash

# Keep offline if you don't want to sync training logs to the web;
# comment this line out if you want to view charts on the wandb website
export WANDB_MODE=offline 

echo "🚀 Starting Latent EDM (x4 compression) training..."
echo "🖥️  GPU check: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Set the Python import path and launch training
PYTHONPATH=$(pwd) python3 forecasting/trainer.py \
    --model LatentEDM \
    --data_path data/SQG/dataset_latent_x4/train \
    --train_data_path data/SQG/dataset_latent_x4/train \
    --val_data_path data/SQG/dataset_latent_x4/validation \
    --nx 16 \
    --lat_channels 8 \
    --init_states 2 \
    --ar_steps 1 \
    --loss mse \
    --pred_residual \
    --sampler heun \
    --sampler_steps 20 \
    --batch_size 32 \
    --epochs 50 \
    --lr 1e-3 \
    --wandb_run_name "Latent_EDM_x4_Run" \
    --wandb_project SQG_Local_Train

echo "✅ Training command finished!"
