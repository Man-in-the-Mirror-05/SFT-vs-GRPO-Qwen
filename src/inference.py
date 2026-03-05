"""
推理脚本
使用训练好的 Qwen3.5-0.8B 模型进行推理
"""

import os
import sys
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_NAME, MAX_NEW_TOKENS


class MathSolver:
    """Qwen3.5 数学问题求解器"""
    
    def __init__(self, model_path: str):
        """初始化模型"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # 加载tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except:
            print("Loading tokenizer from base model...")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载基础模型
        print(f"Loading base model: {MODEL_NAME}")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        
        # 加载LoRA权重
        print(f"Loading LoRA weights from: {model_path}")
        self.model = PeftModel.from_pretrained(model, model_path)
        self.model.eval()
        print("Model loaded successfully!")
    
    def solve(self, question: str, temperature: float = 0.0) -> str:
        """解决数学问题"""
        system_msg = "You are a helpful math assistant. Please solve the math problem step by step, and provide the final answer at the end."
        
        # 构造prompt
        prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        
        # 编码
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # 生成参数
        gen_kwargs = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9
        else:
            gen_kwargs["do_sample"] = False
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # 解码
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # 提取助手回复
        response = generated_text.split("<|im_start|>assistant\n")[-1]
        response = response.replace("<|im_end|>", "").strip()
        
        return response


def interactive_mode(solver: MathSolver):
    """交互式推理模式"""
    print("\n" + "="*60)
    print("Math Solver Interactive Mode")
    print("Type 'quit' or 'exit' to quit")
    print("="*60 + "\n")
    
    while True:
        print("\nEnter your math question:")
        question = input("> ").strip()
        
        if question.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        
        if not question:
            continue
        
        print("\nSolving...")
        answer = solver.solve(question)
        
        print("\n" + "-"*60)
        print("Answer:")
        print("-"*60)
        print(answer)
        print("-"*60)


def demo_mode(solver: MathSolver):
    """演示模式"""
    demo_questions = [
        "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
        "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
    ]
    
    print("\n" + "="*60)
    print("Demo Mode: Solving sample questions")
    print("="*60)
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n{'='*60}")
        print(f"Question {i}:")
        print(f"{'='*60}")
        print(question)
        
        print("\nSolving...")
        answer = solver.solve(question)
        
        print(f"\n{'-'*60}")
        print("Answer:")
        print(f"{'-'*60}")
        print(answer)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Math Solver Inference")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./outputs/dpo_model",
        help="Path to the model"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["interactive", "demo"],
        default="interactive",
        help="Inference mode"
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Single question to solve"
    )
    
    args = parser.parse_args()
    
    # 检查模型路径
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model path does not exist: {args.model_path}")
        print("Please check the path or run training first.")
        return
    
    # 初始化求解器
    print(f"Loading model from: {args.model_path}")
    solver = MathSolver(args.model_path)
    
    # 根据模式运行
    if args.question:
        print(f"\nQuestion: {args.question}")
        answer = solver.solve(args.question)
        print(f"\nAnswer:\n{answer}")
    elif args.mode == "interactive":
        interactive_mode(solver)
    else:
        demo_mode(solver)


if __name__ == "__main__":
    main()
