#!/usr/bin/env python3
"""
测试真实的ShareGPT数据处理流程
"""

import json
import sys
import os
sys.path.append('/home/ziqiang/LLaMA-Factory/src')

from llamafactory.data.converter import get_dataset_converter
from llamafactory.data.parser import get_dataset_list
from llamafactory.hparams import DataArguments
from llamafactory.data.template import get_template_and_fix_tokenizer
from transformers import AutoTokenizer

def test_real_sharegpt_flow():
    """测试真实的ShareGPT数据流"""
    
    print("🚀 开始测试真实的ShareGPT数据流")
    print("=" * 80)
    
    # 1. 设置参数（模拟训练脚本的参数）
    data_args = DataArguments(
        dataset_dir="data",
        dataset="mixed_training_data_09_17",
        template="qwen3",
        cutoff_len=8192,
        preprocessing_num_workers=1,  # 使用单进程避免输出分散
        streaming=False,
        overwrite_cache=True,  # 强制重新处理，不使用缓存
    )
    
    # 2. 加载tokenizer和template
    model_name = "/data/models/Qwen3-8B"
    print(f"📥 加载tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    print(f"✅ 已加载template: {template.__class__.__name__}")
    
    # 3. 获取数据集配置
    print("📥 获取数据集配置...")
    dataset_list = get_dataset_list(data_args.dataset, data_args.dataset_dir)
    dataset_attr = dataset_list[0]
    print(f"✅ 数据集配置: {dataset_attr}")
    
    # 4. 加载数据样本
    print("📥 加载数据样本...")
    data_file = "/home/ziqiang/LLaMA-Factory/data/dataset/9_17/demo.json"
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    sample = data[0]
    print(f"✅ 已加载样本，包含 {len(sample['conversations'])} 条对话")
    
    # 5. 数据转换
    print("🔄 开始数据转换...")
    converter = get_dataset_converter(dataset_attr.formatting, dataset_attr, data_args)
    print(f"✅ 使用转换器: {converter.__class__.__name__}")
    
    converted_sample = converter(sample)
    print(f"✅ 转换完成")
    
    # 6. 测试template的encode_multiturn
    print("🔄 测试template.encode_multiturn...")
    
    # 准备messages
    messages = []
    for conv in sample['conversations']:
        if conv["from"] == "human":
            messages.append({"role": "user", "content": conv["value"]})
        elif conv["from"] == "gpt":
            messages.append({"role": "assistant", "content": conv["value"]})
        elif conv["from"] == "function_call":
            messages.append({"role": "function", "content": conv["value"]})
        elif conv["from"] == "observation":
            messages.append({"role": "observation", "content": conv["value"]})
    
    print(f"📊 转换后的messages数量: {len(messages)}")
    
    # 调用encode_multiturn（这里应该会触发我们的print语句）
    pairs = template.encode_multiturn(tokenizer, messages, sample.get("system", ""), sample.get("tools", ""))
    
    print(f"✅ encode_multiturn完成，生成了 {len(pairs)} 个pairs")
    
    print("\n" + "=" * 80)
    print("🎉 测试完成!")
    print("=" * 80)

if __name__ == "__main__":
    test_real_sharegpt_flow()
