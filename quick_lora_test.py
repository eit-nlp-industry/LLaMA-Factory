#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速验证LoRA适配器效果的脚本
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sys
import os

def load_lora_model(base_model_path, lora_checkpoint_path):
    """加载基础模型和LoRA适配器"""
    print(f"正在加载基础模型: {base_model_path}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        enable_thinking=False,
        low_cpu_mem_usage=True
    )
    
    print(f"正在加载LoRA适配器: {lora_checkpoint_path}")
    
    # 加载LoRA适配器
    model = PeftModel.from_pretrained(model, lora_checkpoint_path)
    
    return model, tokenizer

def test_prediction(model, tokenizer, test_data):
    """测试模型预测效果"""
    print("\n" + "="*50)
    print("开始测试LoRA适配器效果")
    print("="*50)
    
    for i, item in enumerate(test_data):
        print(f"\n测试样本 {i+1}:")
        print("-" * 30)
        
        # 构建输入
        conversations = item["conversations"]
        input_text = ""
        
        # 构建对话历史
        for conv in conversations:
            if conv["from"] == "human":
                input_text += f"<|im_start|>user\n{conv['value']}<|im_end|>\n"
            elif conv["from"] == "function_call":
                input_text += f"<|im_start|>assistant\n<tool_call>\n{conv['value']}\n</tool_call><|im_end|>\n"
            elif conv["from"] == "observation":
                input_text += f"<|im_start|>function\n{conv['value']}<|im_end|>\n"
            elif conv["from"] == "gpt":
                # 这是目标输出，我们不需要包含在输入中
                target_output = conv['value']
                break
        
        # 添加assistant开始标记
        input_text += "<|im_start|>assistant\n"
        
        print(f"输入: {input_text[:200]}...")
        print(f"目标输出: {target_output[:200]}...")
        
        # 编码输入
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 生成预测
        with torch.no_grad():
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
        
        print(f"模型预测: {predicted_text[:]}...")
        
        # 简单比较
        if "retrieval_tool" in predicted_text and "长城景区" in predicted_text:
            print("✅ 预测包含关键信息")
        else:
            print("❌ 预测可能有问题")
        
        print("-" * 30)

def main():
    # 配置路径
    base_model_path = "/data/models/Qwen3-8B"
    lora_checkpoint_path = "/home/ziqiang/LLaMA-Factory/saves/Qwen3-8B/lora/train_2025-09-26-10-43/checkpoint-100"
    test_data_path = "/home/ziqiang/LLaMA-Factory/data/dataset/9_17/demo_1.json"
    
    # 检查路径是否存在
    if not os.path.exists(base_model_path):
        print(f"错误: 基础模型路径不存在: {base_model_path}")
        return
    
    if not os.path.exists(lora_checkpoint_path):
        print(f"错误: LoRA检查点路径不存在: {lora_checkpoint_path}")
        return
    
    if not os.path.exists(test_data_path):
        print(f"错误: 测试数据路径不存在: {test_data_path}")
        return
    
    try:
        # 加载测试数据
        print("正在加载测试数据...")
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        # 加载模型
        model, tokenizer = load_lora_model(base_model_path, lora_checkpoint_path)
        
        # 测试预测
        test_prediction(model, tokenizer, test_data)
        
        print("\n" + "="*50)
        print("测试完成!")
        print("="*50)
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
