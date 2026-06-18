#!/bin/bash

# 0. 设置基础路径 (你的大硬盘路径)
BASE_DIR="/local/data2/huali824/mt-huaide-liu/data/SQG/dataset"
echo "数据将存储在: $BASE_DIR"

# 自动创建文件夹 (解决 PermissionError 关键一步！)
mkdir -p "${BASE_DIR}/train"
mkdir -p "${BASE_DIR}/validation"
mkdir -p "${BASE_DIR}/test"

# 1. 生成训练集 (Training) - 2000 条
echo "Step 1: Generating Training Data (2000 trajectories)..."
PYTHONPATH=$(pwd) python3 data/SQG/sqg_nature_run.py \
    --N 64 \
    --hrs 3 \
    --n_traj 2000 \
    --n_times 100 \
    --data_path "${BASE_DIR}/train"

# 2. 生成验证集 (Validation) - 10 条
echo "Step 2: Generating Validation Data (10 trajectories)..."
PYTHONPATH=$(pwd) python3 data/SQG/sqg_nature_run.py \
    --N 64 \
    --hrs 3 \
    --n_traj 10 \
    --n_times 100 \
    --data_path "${BASE_DIR}/validation"

# 3. 生成测试集 (Testing) - 10 条
echo "Step 3: Generating Testing Data (10 trajectories)..."
PYTHONPATH=$(pwd) python3 data/SQG/sqg_nature_run.py \
    --N 64 \
    --hrs 3 \
    --n_traj 10 \
    --n_times 100 \
    --data_path "${BASE_DIR}/test"

# 4. 计算均值和方差 (Compute Stats)
echo "Step 4: Computing Data Statistics..."
# 修正：只需要指定 data_path，它会自动把 .pt 文件存在该目录下
PYTHONPATH=$(pwd) python3 data/compute_data_stats.py \
    --data_path "${BASE_DIR}/train"

echo "✅ 所有任务完成！请检查 ${BASE_DIR}/train 下是否有 .pt 文件。"
