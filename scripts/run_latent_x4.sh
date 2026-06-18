#!/bin/bash

# 如果你不想把训练日志同步到网页端，可以保持 offline；
# 如果你想在 wandb 网页上看图表，可以把这行注释掉
export WANDB_MODE=offline 

echo "🚀 开始训练 Latent EDM (x4 压缩) 模型..."
echo "🖥️  显卡检查: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# 设置 Python 寻包路径并启动训练
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

echo "✅ 训练命令执行完毕！"
