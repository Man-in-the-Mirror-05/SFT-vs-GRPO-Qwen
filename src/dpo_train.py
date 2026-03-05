"""
DPO训练脚本 - 基于 Qwen3.5-0.8B
"""

import os
import sys
import gc
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_NAME, DPO_BETA, DPO_LEARNING_RATE, DPO_NUM_EPOCHS, SFT_OUTPUT_DIR, DPO_OUTPUT_DIR, SEED


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DPODataset(Dataset):
    """DPO数据集 - 更短序列节省显存"""
    def __init__(self, data, tokenizer, max_length=200):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        chosen = self.tokenizer(
            item["prompt"] + item["chosen"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        rejected = self.tokenizer(
            item["prompt"] + item["rejected"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        return {
            "chosen_ids": chosen["input_ids"].squeeze(0),
            "chosen_mask": chosen["attention_mask"].squeeze(0),
            "rejected_ids": rejected["input_ids"].squeeze(0),
            "rejected_mask": rejected["attention_mask"].squeeze(0),
        }


def compute_log_probs(model, input_ids, attention_mask):
    """计算log概率 - 使用fp16节省显存，计算时转fp32"""
    # 前向传播
    with torch.cuda.amp.autocast(dtype=torch.float16):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    
    # 计算时转fp32保证数值稳定
    logits = logits.float()
    labels = input_ids[:, 1:].contiguous()
    logits = logits[:, :-1, :].contiguous()
    mask = attention_mask[:, 1:].contiguous().float()
    
    log_probs = F.log_softmax(logits, dim=-1)
    
    batch_size, seq_len = labels.shape
    labels = labels.clamp(0, log_probs.size(-1) - 1)
    
    token_log_probs = log_probs.gather(2, labels.unsqueeze(2)).squeeze(2)
    masked_log_probs = token_log_probs * mask
    sum_log_probs = masked_log_probs.sum(dim=1)
    token_count = mask.sum(dim=1).clamp(min=1.0)
    
    return sum_log_probs / token_count


def save_checkpoint(model, tokenizer, step, save_dir):
    """保存checkpoint"""
    checkpoint_dir = os.path.join(save_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    print(f"  Checkpoint saved: step {step}")


def train_epoch(model, ref_model, dataloader, optimizer, tokenizer, device, beta=0.1, save_dir=None):
    """训练一个epoch - 超保守版本"""
    model.train()
    ref_model.eval()
    
    total_loss = 0
    total_acc = 0
    valid_steps = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # 每10步清理一次显存
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
        
        # 数据移到设备
        chosen_ids = batch["chosen_ids"].to(device)
        chosen_mask = batch["chosen_mask"].to(device)
        rejected_ids = batch["rejected_ids"].to(device)
        rejected_mask = batch["rejected_mask"].to(device)
        
        # Policy log probs
        policy_chosen = compute_log_probs(model, chosen_ids, chosen_mask)
        policy_rejected = compute_log_probs(model, rejected_ids, rejected_mask)
        
        # Reference log probs (无梯度)
        with torch.no_grad():
            ref_chosen = compute_log_probs(ref_model, chosen_ids, chosen_mask)
            ref_rejected = compute_log_probs(ref_model, rejected_ids, rejected_mask)
        
        # DPO loss
        policy_diff = policy_chosen - policy_rejected
        ref_diff = ref_chosen - ref_rejected
        logits = beta * (policy_diff - ref_diff)
        
        loss = -F.logsigmoid(logits).mean()
        
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        with torch.no_grad():
            acc = (logits > 0).float().mean().item()
        
        total_loss += loss.item()
        total_acc += acc
        valid_steps += 1
        
        if (batch_idx + 1) % 20 == 0:
            avg_loss = total_loss / valid_steps
            avg_acc = total_acc / valid_steps
            print(f"  Step {batch_idx+1}/{len(dataloader)} | Loss: {loss.item():.4f} | Acc: {acc:.3f} | Avg Loss: {avg_loss:.4f}")
        
        # 保存checkpoint
        if save_dir and (batch_idx + 1) % 50 == 0:
            save_checkpoint(model, tokenizer, batch_idx + 1, save_dir)
            torch.cuda.empty_cache()
    
    if valid_steps == 0:
        return 0.0, 0.0
    return total_loss / valid_steps, total_acc / valid_steps


def create_data(dataset, num_samples=200):
    """创建偏好对 - 默认减少到200对节省显存"""
    data = []
    for i in range(min(num_samples, len(dataset))):
        ex = dataset[i]
        question = ex["question"]
        reasoning = ex["reasoning"]
        answer = ex["answer"]
        
        system = "You are a helpful math assistant. Please solve the math problem step by step, and provide the final answer at the end."
        prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        
        # 截断reasoning节省长度
        if len(reasoning) > 150:
            reasoning = reasoning[:150]
        
        chosen = f"{reasoning}\n答案是：{answer}<|im_end|>"
        rejected = f"答案是：{answer}<|im_end|>"
        
        data.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    
    return data


def load_model(model_name, sft_path, is_trainable=False):
    """加载模型 - 使用fp16节省显存"""
    print(f"Loading model...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(sft_path, trust_remote_code=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 使用fp16加载节省显存
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # 启用gradient checkpointing
    if hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()
    
    model = PeftModel.from_pretrained(base_model, sft_path, is_trainable=is_trainable)
    return model, tokenizer


def main():
    import argparse
    parser = argparse.ArgumentParser(description="DPO Training")
    parser.add_argument("--num_samples", type=int, default=200,
                        help="Number of preference pairs (default: 200)")
    parser.add_argument("--max_length", type=int, default=200,
                        help="Max sequence length (default: 200)")
    args = parser.parse_args()
    
    set_seed(SEED)
    
    if not os.path.exists(SFT_OUTPUT_DIR):
        print(f"❌ SFT model not found: {SFT_OUTPUT_DIR}")
        return
    
    os.makedirs(DPO_OUTPUT_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # 加载模型
    print("\nLoading policy model...")
    model, tokenizer = load_model(MODEL_NAME, SFT_OUTPUT_DIR, is_trainable=True)
    model.print_trainable_parameters()
    
    print("\nLoading reference model...")
    ref_model, _ = load_model(MODEL_NAME, SFT_OUTPUT_DIR, is_trainable=False)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    # 准备数据
    print("\nPreparing data...")
    from src.data_utils import load_gsm8k_dataset
    _, train_ds = load_gsm8k_dataset()
    pref_data = create_data(train_ds, num_samples=args.num_samples)
    dataset = DPODataset(pref_data, tokenizer, max_length=args.max_length)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    print(f"Dataset: {len(dataset)} pairs, max_length={args.max_length}")
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=DPO_LEARNING_RATE)
    
    # 训练
    print("\n" + "="*50)
    print("DPO Training (Memory Efficient)")
    print("="*50)
    
    for epoch in range(DPO_NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{DPO_NUM_EPOCHS}")
        avg_loss, avg_acc = train_epoch(
            model, ref_model, dataloader, optimizer, tokenizer, device,
            beta=DPO_BETA, save_dir=DPO_OUTPUT_DIR
        )
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Acc: {avg_acc:.3f}")
    
    # 保存
    print("\nSaving model...")
    model.save_pretrained(DPO_OUTPUT_DIR)
    tokenizer.save_pretrained(DPO_OUTPUT_DIR)
    
    info = {"completed": True, "final_loss": avg_loss, "final_acc": avg_acc}
    with open(os.path.join(DPO_OUTPUT_DIR, "training_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"✅ Saved to: {DPO_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
