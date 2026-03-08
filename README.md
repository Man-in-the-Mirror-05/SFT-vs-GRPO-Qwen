# RLVR vs SFT: Qwen2.5-1.5B on GSM8K

使用 **Qwen2.5-1.5B-Instruct** 在 **GSM8K** 数据集上对比 **SFT** 和 **GRPO** 训练效果。

## 项目特点

- ✅ **低显存可跑** - 使用 LoRA + BF16/FP16，无需多卡
- ✅ **完整流程** - 数据准备、SFT 训练、GRPO 训练、评测对比
- ✅ **标准实现** - 使用 TRL 库的 GRPOTrainer，非简化版
- ✅ **实测结果** - SFT +8%，GRPO +2%（单条样本限制）

## 环境要求

```bash
# 基础环境
Python >= 3.10
CUDA >= 11.8

# 安装依赖
pip install torch transformers datasets peft pandas
pip install trl>=0.8.0  # GRPO 必需
```

## 项目结构

```
.
├── data/                          # 数据集
│   ├── sft_train.parquet          # SFT 训练数据 (7473条)
│   ├── gsm8k_test.parquet         # 测试数据 (1319条)
│   ├── single_rlvr.parquet        # 单条 RLVR 数据
│   └── ...
├── outputs/                       # 训练输出
│   ├── sft_lora/                  # SFT 模型
│   └── grpo_trl/                  # GRPO 模型
├── train_sft.py                   # SFT 训练脚本
├── train_grpo.py                  # GRPO 训练脚本 (TRL)
├── eval_all_models.py             # 多模型评测对比
└── prepare_sft_data.py            # 数据准备工具
```

## 快速开始

### 1. SFT 训练

```bash
python train_sft.py
```

- **显存占用**: ~4GB
- **训练时间**: ~3小时 (RTX 4050)
- **检查点**: 每 100 步保存
- **输出**: `outputs/sft_lora/`

### 2. GRPO 训练

```bash
python train_grpo.py
```

- **显存占用**: ~4-6GB (取决于配置)
- **训练时间**: ~30分钟 (单条样本 50 step)
- **检查点**: 每 5 步保存
- **输出**: `outputs/grpo_trl/`

### 3. 评测对比

```bash
python eval_all_models.py
```

评测 4 个模型各 100 题：
- Base Model (原始模型)
- SFT-100 (SFT 100 步检查点)
- SFT-200 (SFT 200 步检查点)
- GRPO-TRL (GRPO 最终模型)

## 实验结果

| 模型 | 准确率 (100题) | vs Base | 结论 |
|------|--------------|---------|------|
| **Base** | 32.0% | +0.0% | 基准 |
| **SFT-100** | 39.0% | +7.0% | ✅ 显著提升 |
| **SFT-200** | 40.0% | +8.0% | ✅ **最佳** |
| **GRPO-TRL** | 34.0% | +2.0% | ✅ 微提升 |

> 注：GRPO 使用单条样本 (single_rlvr)，数据多样性不足。使用完整 GSM8K 训练集可能效果更佳。

## 关键发现

### 1. SFT 有效
- SFT 训练后准确率从 32% → 40% (+8%)
- 说明在小模型上 SFT 仍能提升数学推理能力

### 2. GRPO 原理验证
- 即使单条样本，GRPO 也能 +2%
- 标准 TRL 实现比手写简化版更稳定

### 3. 低显存可行性
- 完整训练流程可在笔记本 RTX 4050 上跑通
- 无需多卡，无需 A100

## 配置说明

### SFT 配置
```python
LoRA: r=8, alpha=16
Learning Rate: 5e-5
Batch Size: 1 (gradient_accumulation=4)
Max Length: 1024
Epochs: 3
Quantization: 4-bit NF4
```

### GRPO 配置
```python
LoRA: r=16, alpha=32
Learning Rate: 5e-6
Generations (G): 2
Max Completion Length: 256
Temperature: 0.9
KL Coefficient (beta): 0.01
Quantization: BF16/FP16
```

## 进阶优化

### 提升 GRPO 效果
1. **使用完整数据集**:
   ```python
   # train_grpo.py 中修改
   DATA_PATH = "./data/train.parquet"  # 7473条
   NUM_STEPS = 500
   ```

2. **增大探索**:
   ```python
   NUM_GENERATIONS = 4
   TEMPERATURE = 1.0
   ```

3. **启用 vLLM 加速** (需 8GB+ 显存):
   ```python
   use_vllm=True
   vllm_gpu_memory_utilization=0.5
   ```

### 提升 SFT 效果
1. **更大 LoRA**:
   ```python
   LORA_R = 16
   LORA_ALPHA = 32
   ```

2. **更长训练**:
   ```python
   NUM_EPOCHS = 5
   ```

## 查看训练日志

```bash
# SFT 日志
tensorboard --logdir=./outputs/sft_lora/runs

# 访问 http://localhost:6006
```

## 常见问题

**Q: 显存溢出？**  
A: 减小 `MAX_COMPLETION_LENGTH` 或 `NUM_GENERATIONS`

**Q: 训练卡住？**  
A: GRPO 生成阶段较慢，耐心等待。首次运行需下载模型 (~3GB)

**Q: 效果不如论文？**  
A: 论文使用完整训练集 + 多卡。单条样本 GRPO 本就效果有限。

## 参考

- 原项目: [RLVR-vs-SFT-Qwen2.5-1.5b](https://github.com/jayminban/RLVR-vs-SFT-Qwen2.5-1.5b)
- TRL 库: [Hugging Face TRL](https://github.com/huggingface/trl)
- GSM8K: [Grade School Math 8K](https://github.com/openai/grade-school-math)

## 许可证

MIT License (详见 LICENSE 文件)
