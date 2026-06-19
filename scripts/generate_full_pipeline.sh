#!/bin/bash

# ================= Configuration =================
# Base path
BASE_DIR="/local/data2/huali824/mt-huaide-liu/data/SQG/dataset"

# Training set config (2000 trajectories total)
TRAIN_TOTAL=2000
JOBS=10  # use 10 cores for training-set generation
TRAIN_PER_JOB=$((TRAIN_TOTAL / JOBS))

# Validation and test set config (10 trajectories each)
VAL_TEST_COUNT=10
# ===========================================

# 1. Prepare directories (auto-clean old data to avoid confusion)
echo "🧹 Cleaning old data and creating directories..."
rm -rf "${BASE_DIR}/train" "${BASE_DIR}/validation" "${BASE_DIR}/test"
mkdir -p "${BASE_DIR}/train"
mkdir -p "${BASE_DIR}/validation"
mkdir -p "${BASE_DIR}/test"
mkdir -p logs

echo "🚀 [Stage 1] Launching data-generation jobs at full speed..."
echo "------------------------------------------------"

# --- A. Launch parallel training-set jobs (uses 10 cores) ---
for i in $(seq 1 $JOBS); do
    echo "   Running: training-set job $i/$JOBS (generating $TRAIN_PER_JOB trajectories)..."
    (
        PYTHONPATH=$(pwd) python3 data/SQG/sqg_nature_run.py \
            --N 64 --hrs 3 --n_times 100 \
            --n_traj $TRAIN_PER_JOB \
            --data_path "${BASE_DIR}/train" \
            > "logs/train_job_${i}.log" 2>&1
    ) & 
    sleep 1 # avoid concurrent-write conflicts
done

# --- B. Launch validation-set job (uses the 11th core) ---
echo "   Running: validation-set generation (Validation)..."
(
    PYTHONPATH=$(pwd) python3 data/SQG/sqg_nature_run.py \
        --N 64 --hrs 3 --n_times 100 \
        --n_traj $VAL_TEST_COUNT \
        --data_path "${BASE_DIR}/validation" \
        > "logs/val_job.log" 2>&1
) &

# --- C. Launch test-set job (uses the 12th core) ---
echo "   Running: test-set generation (Test)..."
(
    PYTHONPATH=$(pwd) python3 data/SQG/sqg_nature_run.py \
        --N 64 --hrs 3 --n_times 100 \
        --n_traj $VAL_TEST_COUNT \
        --data_path "${BASE_DIR}/test" \
        > "logs/test_job.log" 2>&1
) &

echo "------------------------------------------------"
echo "✅ All generation jobs are running in the background! The CPU should be fully loaded."
echo "⏳ Waiting for all jobs to finish (estimated 4-6 hours)..."

# --- D. Wait for all background jobs to finish ---
wait

echo "🎉 [Stage 2] Data generation complete! Computing statistics..."

# --- E. Compute mean and variance ---
# Note: statistics are computed from the training set (train) only
PYTHONPATH=$(pwd) python3 data/compute_data_stats.py \
    --data_path "${BASE_DIR}/train" \
    > "logs/stats.log" 2>&1

echo "================================================"
echo "✅✅✅ Full pipeline finished! All data and statistics files are ready."
echo "📂 Data location: $BASE_DIR"
echo "================================================"
