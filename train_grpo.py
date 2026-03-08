"""
GRPO Training with TRL
使用 bf16 + Gradient Checkpointing + LoRA
"""
import os
import re
import json
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

try:
    from trl import GRPOTrainer, GRPOConfig
except ImportError:
    print("Please install: pip install trl>=0.8.0")
    raise

# ============ 配置 ============
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_PATH = "./data/single_rlvr.parquet"
OUTPUT_DIR = "./outputs/grpo_trl"

# 模型配置
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# GRPO 配置 - 加速版
NUM_GENERATIONS = 2
MAX_PROMPT_LENGTH = 256
MAX_COMPLETION_LENGTH = 256
TEMPERATURE = 0.9
TOP_P = 0.95

# 训练配置
NUM_TRAIN_EPOCHS = 1
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 5e-6
WARMUP_STEPS = 10
SAVE_STEPS = 5
LOGGING_STEPS = 5

print("=" * 70)
print("GRPO with TRL")
print("=" * 70)

# ============ Reward Function ============
def extract_boxed(text):
    if not text:
        return None
    match = re.search(r'\\boxed\{([^{}]*)\}', text)
    return match.group(1).strip() if match else None

def extract_number(text):
    if not text:
        return None
    match = re.search(r'-?\d+(?:,\d{3})*(?:\.\d+)?', str(text))
    return match.group(0).replace(',', '') if match else None

def reward_func(completions, **kwargs):
    """TRL reward function"""
    rewards = []
    for completion in completions:
        text = completion if isinstance(completion, str) else str(completion)
        pred = extract_boxed(text)
        if pred is None:
            pred = extract_number(text)
        
        # Single rlr 数据集的答案是 5
        if pred and pred.isdigit():
            rewards.append(1.0 if int(pred) == 5 else 0.0)
        else:
            rewards.append(0.0)
    return rewards

# ============ 加载数据 ============
print("\n[1/4] Loading dataset...")
df = pd.read_parquet(DATA_PATH)
sample = df.iloc[0]
prompt_msgs = sample['prompt'].tolist() if hasattr(sample['prompt'], 'tolist') else sample['prompt']
ground_truth = str(sample['reward_model']['ground_truth'])
print(f"Ground truth: {ground_truth}")

# 创建训练数据（减少数据量加速）
train_data = []
for _ in range(50):
    train_data.append({
        'prompt': prompt_msgs,
        'ground_truth': ground_truth,
    })
dataset = Dataset.from_list(train_data)

# ============ 加载 Tokenizer ============
print("\n[2/4] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# ============ 加载模型 ============
print("\n[3/4] Loading model (BF16 + LoRA)...")

# 检测支持的 dtype
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    torch_dtype = torch.bfloat16
    print("Using bfloat16")
else:
    torch_dtype = torch.float16
    print("Using float16")

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch_dtype,
)

# 开启 Gradient Checkpointing 省显存
model.gradient_checkpointing_enable()

# 准备 LoRA
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 修复 TRL + PEFT 兼容性问题
if not hasattr(model, "warnings_issued"):
    model.warnings_issued = {}

# ============ 配置 GRPO ============
print("\n[4/4] Configuring GRPO Trainer...")

grpo_config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
    fp16=not (torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False),
    gradient_checkpointing=True,
    report_to="none",
    
    # GRPO 特定配置
    num_generations=NUM_GENERATIONS,
    max_prompt_length=MAX_PROMPT_LENGTH,
    max_completion_length=MAX_COMPLETION_LENGTH,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    use_vllm=False,
    beta=0.01,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=reward_func,
    args=grpo_config,
    train_dataset=dataset,
)

# ============ 训练 ============
print("\n" + "=" * 70)
print("Starting GRPO Training...")
print("=" * 70)

trainer.train()

# 保存
print("\nSaving final model...")
trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))

print("\n" + "=" * 70)
print("Training completed!")
print(f"Model saved to: {OUTPUT_DIR}/final")
print("=" * 70)

# 保存信息
info = {
    "model": MODEL_NAME,
    "torch_dtype": str(torch_dtype),
    "num_train_epochs": NUM_TRAIN_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "num_generations": NUM_GENERATIONS,
    "lora_r": LORA_R,
    "lora_alpha": LORA_ALPHA,
}
with open(os.path.join(OUTPUT_DIR, "training_info.json"), "w") as f:
    json.dump(info, f, indent=2)
