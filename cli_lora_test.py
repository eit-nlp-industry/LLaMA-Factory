#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用LLaMA-Factory命令行工具进行LoRA推理测试
"""

import subprocess
import json
import os

def test_lora_with_cli():
    """使用LLaMA-Factory CLI进行LoRA推理测试"""
    
    # 测试数据
    test_data_path = "/home/ziqiang/LLaMA-Factory/data/dataset/9_17/demo_1.json"
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # 获取测试输入
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
    
    # 创建临时输入文件
    temp_input_file = "/tmp/test_input.txt"
    with open(temp_input_file, 'w', encoding='utf-8') as f:
        f.write(input_text)
    
    # 构建LLaMA-Factory推理命令
    cmd = [
        "python", "-m", "llamafactory.cli.infer",
        "--model_name_or_path", "/data/models/Qwen3-8B",
        "--adapter_name_or_path", "/home/ziqiang/LLaMA-Factory/saves/Qwen3-8B/lora/train_2025-09-26-10-43/checkpoint-100",
        "--finetuning_type", "lora",
        "--template", "qwen",
        "--infer_dtype", "float16",
        "--do_sample", "true",
        "--temperature", "0.1",
        "--max_new_tokens", "512",
        "--input_file", temp_input_file,
        "--output_file", "/tmp/test_output.txt"
    ]
    
    print("正在运行推理命令...")
    print(" ".join(cmd))
    
    try:
        # 设置环境变量
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "6"
        env["LOCAL_RANK"] = "0"
        env["RANK"] = "0"
        env["WORLD_SIZE"] = "1"
        env["MASTER_ADDR"] = "localhost"
        env["MASTER_PORT"] = "12355"
        
        # 运行推理
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd="/home/ziqiang/LLaMA-Factory")
        
        print("推理完成!")
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        # 读取输出结果
        if os.path.exists("/tmp/test_output.txt"):
            with open("/tmp/test_output.txt", 'r', encoding='utf-8') as f:
                output = f.read()
            print("\n模型预测输出:")
            print(output[:500] + "..." if len(output) > 500 else output)
            
            # 简单分析
            print("\n分析结果:")
            if "retrieval_tool" in output:
                print("✅ 预测包含retrieval_tool调用")
            else:
                print("❌ 预测未包含retrieval_tool调用")
                
            if "长城景区" in output:
                print("✅ 预测包含长城景区关键词")
            else:
                print("❌ 预测未包含长城景区关键词")
                
            if "manage_order_status" in output:
                print("✅ 预测包含manage_order_status调用")
            else:
                print("❌ 预测未包含manage_order_status调用")
        
    except Exception as e:
        print(f"推理过程中出现错误: {e}")
    
    finally:
        # 清理临时文件
        if os.path.exists(temp_input_file):
            os.remove(temp_input_file)
        if os.path.exists("/tmp/test_output.txt"):
            os.remove("/tmp/test_output.txt")
    
    print("\n测试完成!")

if __name__ == "__main__":
    test_lora_with_cli()
