#!/bin/bash
export WANDB_MODE=offline
# 配置部分
# -----------------------------
DATA_PATH="/local/data2/huali824/mt-huaide-liu/data/SQG/dataset"
EXP_NAME="EDM_Local_Run_MSE"

echo "🚀 开始训练 EDM 模型..."
echo "📂 数据路径: $DATA_PATH"
echo "🖥️  显卡检查: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# 运行训练
# 注意：batch_size 我设置了 32，如果显存不够可以改小（比如 16）
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

echo "✅ 训练脚本已启动！"
