"""
评测所有模型：Base、SFT-100、SFT-200、GRPO-TRL
100 题评测，完整对比
"""
import os
import re
import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

TEST_PATH = "./data/gsm8k_test.parquet"
BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
EVAL_SAMPLES = 100

def extract_boxed(text):
    match = re.search(r'\\boxed\{([^{}]*)\}', text)
    return match.group(1).strip() if match else None

def extract_number(text):
    match = re.search(r'-?\d+(?:,\d{3})*(?:\.\d+)?', str(text))
    return match.group(0).replace(',', '') if match else str(text).strip()

def numeric_match(gold, pred):
    try:
        g = float(str(gold).replace(',', ''))
        p = float(str(pred).replace(',', ''))
        return abs(g - p) < 1e-6
    except:
        return str(gold).strip() == str(pred).strip()

def evaluate_model(model, tokenizer, test_df, num_samples=100, model_name="Model"):
    """评测模型"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")
    
    device = next(model.parameters()).device
    model.eval()
    
    correct = 0
    total = min(num_samples, len(test_df))
    
    for i in range(total):
        row = test_df.iloc[i]
        messages = row['prompt'] if 'prompt' in row else row['messages'][:1]
        if hasattr(messages, 'tolist'):
            messages = messages.tolist()
        
        if 'reward_model' in row:
            gold = str(row['reward_model']['ground_truth'])
        elif 'ground_truth' in row:
            gold = str(row['ground_truth'])
        else:
            continue
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        pred = extract_boxed(response)
        if pred is None:
            pred = extract_number(response)
        
        if numeric_match(gold, pred):
            correct += 1
        
        if (i + 1) % 5 == 0:
            print(f"  Progress: {i+1}/{total}, Acc: {correct/(i+1)*100:.1f}%")
    
    accuracy = correct / total * 100
    print(f"\n  Final Accuracy: {accuracy:.1f}% ({correct}/{total})")
    return accuracy

def load_and_evaluate(base_model_name, model_path, test_df, tokenizer, model_name, results_dict, key):
    """加载并评测模型"""
    if os.path.exists(model_path):
        print(f"\nLoading {model_name} from {model_path}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        acc = evaluate_model(model, tokenizer, test_df, EVAL_SAMPLES, model_name)
        results_dict[key] = acc
        del model
        del base_model
        torch.cuda.empty_cache()
        return acc
    else:
        print(f"\n{model_name} not found: {model_path}")
        results_dict[key] = None
        return None

def main():
    print("="*60)
    print("Model Comparison - 100 Samples")
    print("Models: Base, SFT-100, SFT-200, GRPO-TRL-v2")
    print("="*60)
    
    # 加载测试数据
    print(f"\nLoading test data: {TEST_PATH}")
    test_df = pd.read_parquet(TEST_PATH)
    print(f"Total samples: {len(test_df)}, Evaluating: {EVAL_SAMPLES}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    
    # 加载 tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # ========== 1. Base Model ==========
    print("\n" + "="*60)
    print("1/4: Base Model")
    print("="*60)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    base_acc = evaluate_model(base_model, tokenizer, test_df, EVAL_SAMPLES, "Base Model")
    results["Base"] = base_acc
    del base_model
    torch.cuda.empty_cache()
    
    # ========== 2. SFT Checkpoint-100 ==========
    print("\n" + "="*60)
    print("2/4: SFT Checkpoint-100")
    print("="*60)
    load_and_evaluate(
        BASE_MODEL_NAME, 
        "./outputs/sft_lora/checkpoint-100",
        test_df, tokenizer, "SFT Checkpoint-100", results, "SFT-100"
    )
    
    # ========== 3. SFT Checkpoint-200 ==========
    print("\n" + "="*60)
    print("3/4: SFT Checkpoint-200")
    print("="*60)
    load_and_evaluate(
        BASE_MODEL_NAME,
        "./outputs/sft_lora/checkpoint-200",
        test_df, tokenizer, "SFT Checkpoint-200", results, "SFT-200"
    )
    
    # ========== 4. GRPO TRL==========
    print("\n" + "="*60)
    print("4/4: GRPO TRL")
    print("="*60)
    load_and_evaluate(
        BASE_MODEL_NAME,
        "./outputs/grpo_trl/final",
        test_df, tokenizer, "GRPO-TRL-v2 Final", results, "GRPO-TRL"
    )
    
    # ========== 对比结果 ==========
    print("\n" + "="*60)
    print("FINAL COMPARISON - 100 Samples")
    print("="*60)
    
    base_acc = results.get("Base", 0)
    print(f"{'Model':<20} {'Accuracy':<12} {'vs Base':<12} {'Status':<15}")
    print("-" * 60)
    
    for name, acc in sorted(results.items()):
        if acc is not None:
            diff = acc - base_acc
            diff_str = f"{diff:+.1f}%"
            if diff > 0:
                status = "✅ Improved"
            elif diff == 0:
                status = "➡️  Same"
            else:
                status = "❌ Degraded"
            print(f"{name:<20} {acc:>8.1f}%    {diff_str:<12} {status:<15}")
        else:
            print(f"{name:<20} {'N/A':<12} {'N/A':<12} {'❌ Missing':<15}")
    
    # 保存结果
    output_file = "./comparison_all_models_100.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    print("="*60)

if __name__ == "__main__":
    main()
