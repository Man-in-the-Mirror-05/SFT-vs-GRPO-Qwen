"""
SFT训练脚本
基于 Qwen3.5-0.8B，使用标准transformers实现
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from src.data_utils import load_gsm8k_dataset


def setup_model_and_tokenizer():
    """设置模型和tokenizer"""
    print(f"Loading Qwen3.5 model: {MODEL_NAME}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 加载模型（不使用4-bit量化，避免兼容性问题）
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,  # 使用fp16节省显存
    )
    
    # 设置LoRA配置
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # 应用LoRA
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


def format_chat_template(examples):
    """格式化对话模板"""
    texts = []
    for messages in examples["messages"]:
        text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                text += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                text += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        texts.append(text)
    return {"text": texts}


def tokenize_function(examples, tokenizer, max_length=MAX_SEQ_LENGTH):
    """Tokenize文本"""
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )
    # 对于因果语言模型，labels和input_ids相同
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs


def train(num_train_samples=None):
    """主训练函数"""
    # 设置随机种子
    torch.manual_seed(SEED)
    
    # 创建输出目录
    os.makedirs(SFT_OUTPUT_DIR, exist_ok=True)
    
    # 加载数据
    print("Loading dataset...")
    train_dataset, _ = load_gsm8k_dataset()
    
    # 限制样本数（如果指定）
    if num_train_samples and num_train_samples < len(train_dataset):
        print(f"Using {num_train_samples} / {len(train_dataset)} samples for quick training")
        train_dataset = train_dataset.select(range(num_train_samples))
    else:
        print(f"Using all {len(train_dataset)} samples")
    
    # 格式化数据
    print("Formatting dataset...")
    train_dataset = train_dataset.map(
        format_chat_template,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    print(f"\nFormatted train sample:\n{train_dataset[0]['text'][:500]}...")
    
    # 设置模型和tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    
    # Tokenize数据
    print("Tokenizing dataset...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"]
    )
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=SFT_OUTPUT_DIR,
        num_train_epochs=SFT_NUM_EPOCHS,
        per_device_train_batch_size=SFT_BATCH_SIZE,
        gradient_accumulation_steps=SFT_GRADIENT_ACCUMULATION_STEPS,
        learning_rate=SFT_LEARNING_RATE,
        warmup_ratio=SFT_WARMUP_RATIO,
        logging_steps=SFT_LOGGING_STEPS,
        save_steps=SFT_SAVE_STEPS,
        save_total_limit=2,
        fp16=True,  # 使用fp16
        optim="adamw_torch",
        logging_dir=f"{SFT_OUTPUT_DIR}/logs",
        report_to="none",  # 不使用wandb
        remove_unused_columns=False,
        seed=SEED,
        dataloader_num_workers=0,
    )
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    
    # 开始训练
    print("\n" + "="*50)
    print("Starting SFT training...")
    print("="*50 + "\n")
    
    trainer.train()
    
    # 保存模型
    print("\nSaving model...")
    trainer.save_model(SFT_OUTPUT_DIR)
    tokenizer.save_pretrained(SFT_OUTPUT_DIR)
    
    print(f"\n✅ SFT training completed! Model saved to: {SFT_OUTPUT_DIR}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SFT Training")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of training samples to use (default: all)")
    args = parser.parse_args()
    
    train(num_train_samples=args.num_samples)
