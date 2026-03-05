"""
评估脚本
评估 Qwen3.5-0.8B 在GSM8K测试集上的准确率
"""

import os
import sys
import re
import json
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_NAME, MAX_NEW_TOKENS, SFT_OUTPUT_DIR, DPO_OUTPUT_DIR
from src.data_utils import load_gsm8k_dataset


def load_model_for_eval(model_path: str):
    """加载模型用于评估"""
    print(f"Loading model from: {model_path}")
    
    # 加载tokenizer - 从基础模型加载避免tiktoken问题
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except:
        # 如果从保存的模型加载失败，从基础模型加载
        print("Loading tokenizer from base model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 加载基础模型
    print(f"Loading Qwen3.5 base model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    # 加载LoRA权重
    print(f"Loading LoRA weights from: {model_path}")
    model = PeftModel.from_pretrained(model, model_path)
    
    model.eval()
    return model, tokenizer


def generate_answer(model, tokenizer, question: str) -> str:
    """生成答案"""
    system_msg = "You are a helpful math assistant. Please solve the math problem step by step, and provide the final answer at the end."
    
    # 构造prompt
    prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    
    # 编码
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 解码
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # 提取生成的答案部分
    response = generated_text.split("<|im_start|>assistant\n")[-1]
    response = response.replace("<|im_end|>", "").strip()
    
    return response


def extract_final_number(text: str) -> str:
    """从生成的文本中提取最终数字答案"""
    # 尝试匹配"答案是：X"格式
    match = re.search(r"答案是[:：]\s*(-?\d+(?:\.\d+)?)", text)
    if match:
        return match.group(1).strip()
    
    # 尝试匹配"#### X"格式
    match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if match:
        return match.group(1).strip()
    
    # 尝试匹配最后一行或最后几个数字
    lines = text.strip().split("\n")
    for line in reversed(lines):
        numbers = re.findall(r"-?\d+(?:\.\d+)?", line)
        if numbers:
            return numbers[-1].strip()
    
    return text.strip()


def normalize_answer(answer: str) -> str:
    """规范化答案"""
    answer = answer.replace(",", "")
    answer = answer.strip()
    
    try:
        if "." in answer:
            num = float(answer)
            if num == int(num):
                return str(int(num))
            return str(num)
        else:
            return str(int(answer))
    except:
        return answer


def evaluate_model(model_path: str, output_file: str = None, num_samples: int = None):
    """评估模型"""
    # 加载模型
    model, tokenizer = load_model_for_eval(model_path)
    
    # 加载测试数据
    _, test_dataset = load_gsm8k_dataset()
    
    # 确定评估样本数
    if num_samples is not None and num_samples > 0:
        eval_samples = min(num_samples, len(test_dataset))
        print(f"\nEvaluating on {eval_samples} samples (quick mode)...")
    else:
        eval_samples = len(test_dataset)
        print(f"\nEvaluating on {eval_samples} samples (full mode)...")
    
    correct = 0
    total = 0
    results = []
    
    for i in tqdm(range(eval_samples)):
        example = test_dataset[i]
        question = example["question"]
        ground_truth = normalize_answer(example["answer"])
        
        # 生成答案
        generated = generate_answer(model, tokenizer, question)
        predicted = normalize_answer(extract_final_number(generated))
        
        # 判断是否正确
        is_correct = (predicted == ground_truth)
        if is_correct:
            correct += 1
        total += 1
        
        # 保存结果
        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "generated": generated,
            "predicted": predicted,
            "correct": is_correct
        })
        
        # 每50个样本打印一次进度
        if (i + 1) % 50 == 0:
            acc = correct / total * 100
            print(f"\nProgress: {i+1}/{eval_samples}, Current Acc: {acc:.2f}%")
    
    # 计算准确率
    accuracy = correct / total * 100
    
    print("\n" + "="*50)
    print(f"Evaluation Results:")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("="*50)
    
    # 保存结果
    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        output = {
            "model_path": model_path,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults saved to: {output_file}")
    
    return accuracy


def evaluate_base_model(output_file: str = None, num_samples: int = None):
    """评估原始（基础）模型"""
    print(f"Loading base model: {MODEL_NAME}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 加载基础模型（不加载LoRA）
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model.eval()
    
    # 加载测试数据
    _, test_dataset = load_gsm8k_dataset()
    
    # 确定评估样本数
    if num_samples is not None and num_samples > 0:
        eval_samples = min(num_samples, len(test_dataset))
        print(f"\nEvaluating BASE MODEL on {eval_samples} samples (quick mode)...")
    else:
        eval_samples = len(test_dataset)
        print(f"\nEvaluating BASE MODEL on {eval_samples} samples (full mode)...")
    
    correct = 0
    total = 0
    results = []
    
    for i in tqdm(range(eval_samples)):
        example = test_dataset[i]
        question = example["question"]
        ground_truth = normalize_answer(example["answer"])
        
        # 生成答案
        generated = generate_answer(model, tokenizer, question)
        predicted = normalize_answer(extract_final_number(generated))
        
        # 判断是否正确
        is_correct = (predicted == ground_truth)
        if is_correct:
            correct += 1
        total += 1
        
        # 保存结果
        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "generated": generated,
            "predicted": predicted,
            "correct": is_correct
        })
        
        # 每50个样本打印一次进度
        if (i + 1) % 50 == 0:
            acc = correct / total * 100
            print(f"\nProgress: {i+1}/{eval_samples}, Current Acc: {acc:.2f}%")
    
    # 计算准确率
    accuracy = correct / total * 100
    
    print("\n" + "="*50)
    print(f"BASE MODEL Results:")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("="*50)
    
    # 保存结果
    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        output = {
            "model_path": MODEL_NAME,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults saved to: {output_file}")
    
    return accuracy


def main():
    """主函数：评估所有模型"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate models on GSM8K")
    parser.add_argument("--num_samples", type=int, default=None, 
                        help="Number of samples to evaluate (default: all)")
    args = parser.parse_args()
    
    num_samples = args.num_samples
    if num_samples:
        print(f"\n⚡ Quick mode: evaluating {num_samples} samples per model")
    else:
        print(f"\n📊 Full mode: evaluating all samples")
    
    results_summary = {}
    
    # 1. 评估原始模型
    print("\n" + "="*60)
    print("Evaluating BASE MODEL (Original)")
    print("="*60)
    try:
        base_acc = evaluate_base_model(output_file="outputs/base_results.json", num_samples=num_samples)
        results_summary["Base (Original)"] = base_acc
    except Exception as e:
        print(f"Error evaluating base model: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. 评估SFT模型
    if os.path.exists(SFT_OUTPUT_DIR):
        print("\n" + "="*60)
        print("Evaluating SFT MODEL")
        print("="*60)
        try:
            sft_acc = evaluate_model(
                SFT_OUTPUT_DIR, 
                output_file="outputs/sft_results.json",
                num_samples=num_samples
            )
            results_summary["SFT"] = sft_acc
        except Exception as e:
            print(f"Error evaluating SFT model: {e}")
    else:
        print(f"\n⚠️ SFT model not found at {SFT_OUTPUT_DIR}")
    
    # 3. 评估DPO模型
    if os.path.exists(DPO_OUTPUT_DIR):
        print("\n" + "="*60)
        print("Evaluating DPO MODEL")
        print("="*60)
        try:
            dpo_acc = evaluate_model(
                DPO_OUTPUT_DIR,
                output_file="outputs/dpo_results.json",
                num_samples=num_samples
            )
            results_summary["DPO"] = dpo_acc
        except Exception as e:
            print(f"Error evaluating DPO model: {e}")
    else:
        print(f"\n⚠️ DPO model not found at {DPO_OUTPUT_DIR}")
    
    # 打印总结
    if results_summary:
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        for model_name, acc in results_summary.items():
            print(f"{model_name}: {acc:.2f}%")
        
        # 计算提升
        if "Base (Original)" in results_summary and "SFT" in results_summary:
            sft_improvement = results_summary["SFT"] - results_summary["Base (Original)"]
            print(f"SFT Improvement: +{sft_improvement:.2f}%")
        
        if "SFT" in results_summary and "DPO" in results_summary:
            dpo_improvement = results_summary["DPO"] - results_summary["SFT"]
            total_improvement = results_summary["DPO"] - results_summary["Base (Original)"]
            print(f"DPO Improvement: +{dpo_improvement:.2f}%")
            print(f"Total Improvement: +{total_improvement:.2f}%")
        
        print("="*60)
        
        # 保存总结
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/eval_summary.json", "w") as f:
            json.dump(results_summary, f, indent=2)


if __name__ == "__main__":
    main()
