#!/bin/bash

# 遇到错误直接退出
set -e

echo "🚀 开始 Autoencoder 自动化训练与评估流程..."
echo "================================================="

# 依次训练 4 个不同压缩率的模型
for c in 4 8 16; do
    echo "⏳ [$(date +'%Y-%m-%d %H:%M:%S')] 正在训练 x${c} 压缩率模型..."
    
    # 调用 Python 脚本并传入参数
    python train_sqg_ae.py --compression $c
    
    echo "✅ x${c} 训练完成！"
    echo "-------------------------------------------------"
done

echo "🎉 所有模型训练完毕！开始执行联合评估..."
echo "================================================="

# 统一进行评估
python eval_sqg_ae.py

echo "🏁 全流程结束！请查看上方输出的 RMSE 和 LSD 结果。"
