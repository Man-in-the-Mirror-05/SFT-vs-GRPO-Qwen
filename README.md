# Lightweight Math Reasoning with SFT + DPO

基于 **Qwen3.5-0.8B** 的轻量级数学推理模型微调项目，使用 SFT（监督微调）和 DPO（直接偏好优化）提升小学数学应用题（GSM8K）的解题能力。

## 📋 项目简介

本项目验证了在极低资源条件下（单卡 6GB 显存）通过高效微调技术获得可用推理能力的可行性。

**核心特点：**
- 🎯 **轻量化模型**：使用 **Qwen3.5-0.8B**（亚 1B 级别）
- 💾 **低资源友好**：单卡 6GB 显存可完成全流程
- 🚀 **高效训练**：LoRA + 梯度累积，5 小时完成 SFT+DPO
- 📊 **完整流程**：数据工程 → SFT → DPO → 评估

## 🏗️ 项目结构

```
.
├── src/
│   ├── data_utils.py       # GSM8K 数据预处理和格式化
│   ├── sft_train.py        # 监督微调训练
│   ├── dpo_train.py        # DPO 偏好对齐训练
│   ├── evaluate.py         # GSM8K 测试集评估
│   └── inference.py        # 交互式推理
├── config.py               # 全局配置（Qwen3.5-0.8B）
├── requirements.txt        # 依赖列表
├── scripts/
│   ├── run_all.sh          # Linux 一键训练脚本
│   └── run_all.bat         # Windows 一键训练脚本
├── outputs/                # 模型输出目录
├── data/                   # 数据缓存目录
└── README.md
```

## 🚀 快速开始

### 环境准备

```bash
# 创建环境
conda create -n math-sft python=3.10
conda activate math-sft

# 安装依赖
pip install -r requirements.txt
```

### 一键训练

```bash
# Windows
scripts\run_all.bat

# Linux/Mac
bash scripts/run_all.sh
```

### 分步执行

```bash
# 1. SFT 训练（约 1-2 小时）
python src/sft_train.py --num_samples 3000

# 2. DPO 训练（约 20-30 分钟）
python src/dpo_train.py --num_samples 200

# 3. 评估（约 30 分钟）
python src/evaluate.py --num_samples 200

# 4. 交互式推理
python src/inference.py --model_path outputs/sft_model --mode interactive
```
## 💡 常见问题

### Q: 显存不足怎么办？

A: 修改 `config.py`：
```python
MAX_SEQ_LENGTH = 256  # 从 512 减小
LORA_R = 8            # 从 16 减小
```

### Q: 训练时间太长？

A: 使用命令行参数减少样本数：
```bash
python src/sft_train.py --num_samples 1500
```

### Q: 如何恢复训练？

A: 训练支持 checkpoint 恢复，重新运行会自动提示。

## 📊 技术方案

### 训练策略

| 阶段 | 方法 | 关键参数 | 说明 |
|------|------|---------|------|
| SFT | QLoRA | r=16, α=32, target=[q_proj, v_proj] | 监督微调，学习推理格式 |
| DPO | Direct Preference Optimization | β=0.1 | 偏好对齐，提升回答质量 |

### 数据工程

- **数据集**：GSM8K（小学数学应用题）
- **格式化**：转换为对话格式（system/user/assistant）
- **偏好对构造**：`chosen`（完整推理）vs `rejected`（直接答案）

### 显存优化

- 4-bit 量化（QLoRA）
- Gradient Checkpointing
- 梯度累积（有效 batch size=8）
- 序列长度截断（max_length=512）

## 📈 实验结果

在 GSM8K 测试集上的准确率对比：

| 模型 | 准确率 | 提升 |
|------|--------|------|
| Base (Qwen3.5-0.8B) | ~28% | - |
| SFT | ~42% | +14% |
| SFT + DPO | ~45% | +17% |

*注：实际结果可能因随机性和评估样本数有所波动*

## 🎯 关键优化点

1. **数据格式化**：将 GSM8K 的原始格式转换为对话式，分离推理过程和最终答案
2. **LoRA 配置**：仅微调 q_proj 和 v_proj，平衡效果和效率
3. **早停策略**：监控训练 loss，避免过拟合（观察到 1 个 epoch 即可收敛）
4. **偏好对齐**：使用 DPO 替代 PPO，节省显存同时提升回答质量


## 📚 参考资料

- [GSM8K Dataset](https://github.com/openai/grade-school-math)
- [Qwen3.5 Model](https://github.com/QwenLM/Qwen)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [DPO Paper](https://arxiv.org/abs/2305.18290)

## 📝 License

MIT License

## 🙏 Acknowledgements

- 感谢 Qwen 团队开源的 **Qwen3.5** 模型
- 感谢 HuggingFace 的 Transformers 和 PEFT 库
