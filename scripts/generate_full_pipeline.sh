#!/bin/bash

# ================= 配置区域 =================
# 基础路径
BASE_DIR="/local/data2/huali824/mt-huaide-liu/data/SQG/dataset"

# 训练集配置 (总共 2000 条)
TRAIN_TOTAL=2000
JOBS=10  # 使用 10 个核心跑训练
TRAIN_PER_JOB=$((TRAIN_TOTAL / JOBS))

# 验证和测试集配置 (各 10 条)
VAL_TEST_COUNT=10
# ===========================================

# 1. 准备目录 (自动清理旧数据，防止混淆)
echo "🧹 清理旧数据并创建目录..."
rm -rf "${BASE_DIR}/train" "${BASE_DIR}/validation" "${BASE_DIR}/test"
mkdir -p "${BASE_DIR}/train"
mkdir -p "${BASE_DIR}/validation"
mkdir -p "${BASE_DIR}/test"
mkdir -p logs

echo "🚀 [阶段1] 全速启动数据生成任务..."
echo "------------------------------------------------"

# --- A. 启动并行训练任务 (占用 10 个核) ---
for i in $(seq 1 $JOBS); do
    echo "   Running: 训练集任务 $i/$JOBS (生成 $TRAIN_PER_JOB 条)..."
    (
        PYTHONPATH=$(pwd) python3 data/SQG/sqg_nature_run.py \
            --N 64 --hrs 3 --n_times 100 \
            --n_traj $TRAIN_PER_JOB \
            --data_path "${BASE_DIR}/train" \
            > "logs/train_job_${i}.log" 2>&1
    ) & 
    sleep 1 # 防止并发写入冲突
done

# --- B. 启动验证集任务 (占用第 11 个核) ---
echo "   Running: 验证集生成 (Validation)..."
(
    PYTHONPATH=$(pwd) python3 data/SQG/sqg_nature_run.py \
        --N 64 --hrs 3 --n_times 100 \
        --n_traj $VAL_TEST_COUNT \
        --data_path "${BASE_DIR}/validation" \
        > "logs/val_job.log" 2>&1
) &

# --- C. 启动测试集任务 (占用第 12 个核) ---
echo "   Running: 测试集生成 (Test)..."
(
    PYTHONPATH=$(pwd) python3 data/SQG/sqg_nature_run.py \
        --N 64 --hrs 3 --n_times 100 \
        --n_traj $VAL_TEST_COUNT \
        --data_path "${BASE_DIR}/test" \
        > "logs/test_job.log" 2>&1
) &

echo "------------------------------------------------"
echo "✅ 所有生成任务已在后台运行！CPU 应该已满载。"
echo "⏳ 正在等待所有任务完成 (预计 4-6 小时)..."

# --- D. 等待所有后台任务结束 ---
wait

echo "🎉 [阶段2] 数据生成完毕！开始计算统计量..."

# --- E. 计算均值和方差 ---
# 注意：这里只根据训练集(train)来计算
PYTHONPATH=$(pwd) python3 data/compute_data_stats.py \
    --data_path "${BASE_DIR}/train" \
    > "logs/stats.log" 2>&1

echo "================================================"
echo "✅✅✅ 全流程结束！所有数据和统计文件已就绪。"
echo "📂 数据位置: $BASE_DIR"
echo "================================================"
