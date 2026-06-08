#!/bin/bash

# Exit immediately if any script fails along the line
set -e

echo "=== [1/3] Running Data Preprocessing ==="
python3 src/preprocessing_data.py

echo "=== [2/3] Executing Model Training ==="
python3 src/model_training.py

echo "=== [3/3] Initiating SMOTE Fine-Tuning & Evaluation ==="
python3 src/optimization_evaluation.py

echo "========================================================="
echo "Pipeline Completed Successfully! Summary metrics printed above."
echo "========================================================="