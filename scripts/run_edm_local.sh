#!/bin/bash
export WANDB_MODE=offline
# Configuration
# -----------------------------
DATA_PATH="/local/data2/huali824/mt-huaide-liu/data/SQG/dataset"
EXP_NAME="EDM_Local_Run_MSE"

echo "🚀 Starting EDM model training..."
echo "📂 Data path: $DATA_PATH"
echo "🖥️  GPU check: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Run training
# Note: batch_size is set to 32; lower it (e.g. 16) if you run out of GPU memory
PYTHONPATH=$(pwd) python3 forecasting/trainer.py \
    --model EDM \
    --data_path "$DATA_PATH" \
    --wandb_run_name "$EXP_NAME" \
    --val_steps_to_log 1 \
    --ar_steps_eval 1 \
    --init_states 2 \
    --pred_residual \
    --step_length 3 \
    --loss mse \
    --batch_size 32 \
    --epochs 50 \
    --n_workers 8 \
    --wandb_project SQG_Local_Train

echo "✅ Training script launched!"
