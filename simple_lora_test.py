#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用LLaMA-Factory的推理功能快速验证LoRA适配器效果
"""

import json
import sys
import os

# 设置环境变量避免分布式初始化问题
os.environ["LOCAL_RANK"] = "0"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"

# 添加LLaMA-Factory路径
sys.path.append('/home/ziqiang/LLaMA-Factory/src')

from llamafactory.hparams import get_infer_args
from llamafactory.model import load_model, load_tokenizer

def test_lora_inference():
    """使用LLaMA-Factory进行LoRA推理测试"""
    
    # 配置推理参数
    infer_args = {
        "model_name_or_path": "/data/models/Qwen3-8B",
        "adapter_name_or_path": "/home/ziqiang/LLaMA-Factory/saves/Qwen3-8B/lora/train_2025-09-26-10-43/checkpoint-100",
        "finetuning_type": "lora",
        "template": "qwen",
        "infer_dtype": "float16",
        "do_sample": True,
        "temperature": 0.1,
        "enable_thinking": False,
        "max_new_tokens": 512,
    }
    
    # 加载测试数据
    test_data_path = "/home/ziqiang/LLaMA-Factory/data/dataset/9_17/demo_1.json"
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print("正在加载模型和tokenizer...")
    
    # 加载模型和tokenizer
    args = get_infer_args(infer_args)
    model_args, data_args, finetuning_args, generating_args = args
    
    # 直接使用transformers加载tokenizer避免LLaMA-Factory的兼容性问题
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, 
        trust_remote_code=True
    )
    
    model = load_model(tokenizer, model_args, finetuning_args)
    
    print("模型加载完成，开始测试...")
    
    # 测试第一个样本
    sample = test_data[0]
    conversations = sample["conversations"]
    
    # 构建输入
    input_text = ""
    for conv in conversations:
        if conv["from"] == "human":
            input_text += f"<|im_start|>user\n{conv['value']}<|im_end|>\n"
        elif conv["from"] == "function_call":
            input_text += f"<|im_start|>assistant\n<tool_call>\n{conv['value']}\n</tool_call><|im_end|>\n"
        elif conv["from"] == "observation":
            input_text += f"<|im_start|>function\n{conv['value']}<|im_end|>\n"
        elif conv["from"] == "gpt":
            target_output = conv['value']
            break
    
    # 添加assistant开始标记
    input_text += "<|im_start|>assistant\n"
    
    print("="*60)
    print("测试输入:")
    print(input_text[:300] + "..." if len(input_text) > 300 else input_text)
    print("\n目标输出:")
    print(target_output[:300] + "..." if len(target_output) > 300 else target_output)
    print("="*60)
    
    # 编码输入
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 生成预测
    with model.inference_context():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 解码输出
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predicted_text = full_output[len(input_text):].strip()
    
    print("\n模型预测输出:")
    print(predicted_text[:500] + "..." if len(predicted_text) > 500 else predicted_text)
    print("="*60)
    
    # 简单分析
    print("\n分析结果:")
    if "retrieval_tool" in predicted_text:
        print("✅ 预测包含retrieval_tool调用")
    else:
        print("❌ 预测未包含retrieval_tool调用")
        
    if "长城景区" in predicted_text:
        print("✅ 预测包含长城景区关键词")
    else:
        print("❌ 预测未包含长城景区关键词")
        
    if "manage_order_status" in predicted_text:
        print("✅ 预测包含manage_order_status调用")
    else:
        print("❌ 预测未包含manage_order_status调用")
    
    print("\n测试完成!")

if __name__ == "__main__":
    test_lora_inference()
