#!/bin/bash
# 完整训练流程脚本

set -e  # 遇到错误立即退出

echo "=================================="
echo "SFT + DPO Training Pipeline"
echo "=================================="

# 切换到项目目录
cd "$(dirname "$0")/.."

# 1. 数据预处理（可选，主要用于检查数据）
echo ""
echo "=================================="
echo "Step 1: Data Preparation"
echo "=================================="
python src/data_utils.py

# 2. SFT训练
echo ""
echo "=================================="
echo "Step 2: SFT Training"
echo "=================================="
python src/sft_train.py

# 3. DPO训练
echo ""
echo "=================================="
echo "Step 3: DPO Training"
echo "=================================="
python src/dpo_train.py

# 4. 评估
echo ""
echo "=================================="
echo "Step 4: Evaluation"
echo "=================================="
python src/evaluate.py

echo ""
echo "=================================="
echo "All steps completed!"
echo "=================================="
