"""
数据预处理工具
处理GSM8K数据集，转换为 Qwen3.5 对话格式
"""

import re
import json
from typing import Dict, List, Tuple
from datasets import load_dataset, Dataset


def extract_answer(text: str) -> str:
    """
    从GSM8K的答案中提取最终数字答案
    答案格式通常是：...#### 42
    """
    if "####" in text:
        # 提取####后的数字
        match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
        if match:
            return match.group(1).strip()
    return text.strip()


def extract_reasoning(text: str) -> str:
    """
    从GSM8K的答案中提取推理过程（去掉####部分）
    """
    if "####" in text:
        return text.split("####")[0].strip()
    return text.strip()


def format_gsm8k_to_conversation(example: Dict) -> Dict:
    """
    将GSM8K数据转换为对话格式
    
    输入格式：
    {
        "question": "问题文本",
        "answer": "推理过程#### 最终答案"
    }
    
    输出格式：
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "问题"},
            {"role": "assistant", "content": "推理过程\n答案是：X"}
        ],
        "answer": "X"
    }
    """
    question = example["question"]
    answer_text = example["answer"]
    
    # 提取推理过程和最终答案
    reasoning = extract_reasoning(answer_text)
    final_answer = extract_answer(answer_text)
    
    # 构造对话格式
    system_msg = "You are a helpful math assistant. Please solve the math problem step by step, and provide the final answer at the end."
    
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": question},
        {"role": "assistant", "content": f"{reasoning}\n答案是：{final_answer}"}
    ]
    
    return {
        "messages": messages,
        "question": question,
        "reasoning": reasoning,
        "answer": final_answer
    }


def create_dpo_preference_data(
    dataset: Dataset, 
    sft_model_path: str,
    num_samples: int = 1000
) -> Dataset:
    """
    为DPO创建偏好对数据
    chosen: GSM8K的正确答案
    rejected: SFT前模型的错误输出（或较差的输出）
    
    简化版本：直接使用不同的答案格式构造偏好对
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    print(f"Creating DPO preference data with {num_samples} samples...")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def format_for_dpo(example):
        """将数据格式化为DPO需要的prompt/chosen/rejected格式"""
        question = example["question"]
        correct_answer = example["answer"]
        reasoning = example.get("reasoning", "")
        
        system_msg = "You are a helpful math assistant. Please solve the math problem step by step, and provide the final answer at the end."
        prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        
        # Chosen: 正确的详细推理
        chosen = f"{reasoning}\n答案是：{correct_answer}<|im_end|>"
        
        # Rejected: 较差的回答（简化版或错误格式）
        # 这里构造一个推理不完整的版本作为rejected
        rejected = f"答案是：{correct_answer}<|im_end|>"
        
        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "question": question,
            "answer": correct_answer
        }
    
    # 只取前num_samples个样本
    small_dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    # 转换为DPO格式
    dpo_dataset = small_dataset.map(format_for_dpo, remove_columns=small_dataset.column_names)
    
    return dpo_dataset


def load_gsm8k_dataset() -> Tuple[Dataset, Dataset]:
    """
    加载并预处理GSM8K数据集
    返回：(训练集, 测试集)
    """
    print("Loading GSM8K dataset...")
    
    # 加载数据集
    train_dataset = load_dataset("gsm8k", "main", split="train")
    test_dataset = load_dataset("gsm8k", "main", split="test")
    
    print(f"Original train size: {len(train_dataset)}")
    print(f"Original test size: {len(test_dataset)}")
    
    # 转换为对话格式
    train_dataset = train_dataset.map(format_gsm8k_to_conversation)
    test_dataset = test_dataset.map(format_gsm8k_to_conversation)
    
    print("Dataset preprocessing completed!")
    
    return train_dataset, test_dataset


def save_dataset_info(dataset: Dataset, output_path: str):
    """保存数据集信息用于检查"""
    info = {
        "size": len(dataset),
        "columns": list(dataset.features.keys()),
        "sample": dataset[0] if len(dataset) > 0 else None
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    
    print(f"Dataset info saved to {output_path}")


if __name__ == "__main__":
    # 测试数据加载
    train_ds, test_ds = load_gsm8k_dataset()
    
    print("\n=== Train Sample ===")
    print(json.dumps(train_ds[0], ensure_ascii=False, indent=2))
    
    print("\n=== Test Sample ===")
    print(json.dumps(test_ds[0], ensure_ascii=False, indent=2))
