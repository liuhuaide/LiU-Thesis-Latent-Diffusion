#!/bin/bash

# Exit immediately on error
set -e

echo "🚀 Starting automated autoencoder training and evaluation pipeline..."
echo "================================================="

# Train 4 models at different compression rates in sequence
for c in 4 8 16; do
    echo "⏳ [$(date +'%Y-%m-%d %H:%M:%S')] Training x${c} compression model..."
    
    # Call the Python script with arguments
    python train_sqg_ae.py --compression $c
    
    echo "✅ x${c} training complete!"
    echo "-------------------------------------------------"
done

echo "🎉 All models trained! Starting joint evaluation..."
echo "================================================="

# Run evaluation uniformly
python eval_sqg_ae.py

echo "🏁 Pipeline finished! See the RMSE and LSD results printed above."
