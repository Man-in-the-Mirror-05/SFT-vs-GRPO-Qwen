"""
SFT Training with QLoRA
"""
import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ============ 配置 ============
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "sft_train.parquet")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "sft_lora")

# LoRA 配置
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# 训练配置
MAX_LENGTH = 1024
BATCH_SIZE = 1
GRAD_ACCUM = 4
LEARNING_RATE = 5e-5
NUM_EPOCHS = 1
SAVE_STEPS = 100
LOG_STEPS = 10

print("=" * 70)
print("SFT Training with QLoRA")
print("=" * 70)
print(f"Model: {MODEL_NAME}")
print(f"LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
print(f"Train Epochs: {NUM_EPOCHS}")
print("=" * 70)

# ============ 加载 Tokenizer ============
print("\n[1/4] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ============ 加载模型（4-bit）============
print("\n[2/4] Loading model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# ============ 准备 LoRA 训练 ============
print("\n[3/4] Preparing LoRA...")
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============ 加载训练数据 ============
print("\n[4/4] Loading training data...")
train_df = pd.read_parquet(DATA_PATH)
print(f"Train samples: {len(train_df)}")

def format_sample(row):
    messages = row['messages']
    if hasattr(messages, 'tolist'):
        messages = messages.tolist()
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

train_dataset = Dataset.from_pandas(train_df)
train_dataset = train_dataset.map(format_sample, remove_columns=train_dataset.column_names)

def tokenize_function(examples):
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

print("Tokenizing...")
tokenized_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# ============ 训练 ============
print("\nStarting training...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_steps=int(len(tokenized_dataset) * NUM_EPOCHS / BATCH_SIZE / GRAD_ACCUM * 0.05),
    logging_steps=LOG_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=2,
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    gradient_checkpointing=True,
    report_to=["tensorboard"],
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_strategy="steps",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# 开始训练
trainer.train()

# 保存模型
print("\nSaving model...")
model.save_pretrained(os.path.join(OUTPUT_DIR, "lora_model"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "lora_model"))

print("\n" + "=" * 70)
print("Training completed!")
print(f"Model saved to: {os.path.join(OUTPUT_DIR, 'lora_model')}")
print(f"TensorBoard: tensorboard --logdir={os.path.join(OUTPUT_DIR, 'logs')}")
print("=" * 70)
