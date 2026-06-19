#!/bin/bash

# 0. Set base path (your large-disk path)
BASE_DIR="/local/data2/huali824/mt-huaide-liu/data/SQG/dataset"
echo "Data will be stored in: $BASE_DIR"

# Auto-create folders (key step to avoid PermissionError!)
mkdir -p "${BASE_DIR}/train"
mkdir -p "${BASE_DIR}/validation"
mkdir -p "${BASE_DIR}/test"

# 1. Generate training set (Training) - 2000 trajectories
echo "Step 1: Generating Training Data (2000 trajectories)..."
PYTHONPATH=$(pwd) python3 data/SQG/sqg_nature_run.py \
    --N 64 \
    --hrs 3 \
    --n_traj 2000 \
    --n_times 100 \
    --data_path "${BASE_DIR}/train"

# 2. Generate validation set (Validation) - 10 trajectories
echo "Step 2: Generating Validation Data (10 trajectories)..."
PYTHONPATH=$(pwd) python3 data/SQG/sqg_nature_run.py \
    --N 64 \
    --hrs 3 \
    --n_traj 10 \
    --n_times 100 \
    --data_path "${BASE_DIR}/validation"

# 3. Generate test set (Testing) - 10 trajectories
echo "Step 3: Generating Testing Data (10 trajectories)..."
PYTHONPATH=$(pwd) python3 data/SQG/sqg_nature_run.py \
    --N 64 \
    --hrs 3 \
    --n_traj 10 \
    --n_times 100 \
    --data_path "${BASE_DIR}/test"

# 4. Compute mean and variance (Compute Stats)
echo "Step 4: Computing Data Statistics..."
# Note: only data_path is needed; the .pt stats files are saved into that directory automatically
PYTHONPATH=$(pwd) python3 data/compute_data_stats.py \
    --data_path "${BASE_DIR}/train"

echo "✅ All tasks done! Check ${BASE_DIR}/train for the .pt files."
