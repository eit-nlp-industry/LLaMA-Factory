#!/usr/bin/env python3
"""
调试ShareGPT训练过程中的对话pair分割情况
"""

import json
import os
from datetime import datetime
from transformers import AutoTokenizer
from src.llamafactory.data.template import TEMPLATES

def log_debug(msg, log_file="/home/ziqiang/LLaMA-Factory/pair_debug.log"):
    """调试日志函数"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    log_entry = f"{timestamp} | INFO | {msg}\n"
    
    # 写入日志文件
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_entry)
        f.flush()
    
    # 同时输出到控制台
    print(f"DEBUG | {msg}")

def analyze_conversation_pairs():
    """分析对话pair分割情况"""
    
    # 清空日志文件
    log_file = "/home/ziqiang/LLaMA-Factory/pair_debug.log"
    if os.path.exists(log_file):
        os.remove(log_file)
    
    log_debug("=== 开始分析ShareGPT对话pair分割 ===")
    
    # 1. 加载tokenizer和template
    model_name = "/data/models/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    template = TEMPLATES["qwen3"]
    template.fix_special_tokens(tokenizer)
    
    log_debug(f"已加载tokenizer和template: {template.__class__.__name__}")
    
    # 2. 加载一个示例数据
    data_file = "/home/ziqiang/LLaMA-Factory/data/dataset/9_17/mixed_training_data_9_17.json"
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 取第一个样本进行分析
    sample = data[0]
    conversations = sample["conversations"]
    system = sample.get("system", "")
    tools = sample.get("tools", "")
    
    log_debug(f"样本包含 {len(conversations)} 条对话")
    log_debug(f"System: {system[:100]}..." if system else "System: (空)")
    log_debug(f"Tools: {tools[:100]}..." if tools else "Tools: (空)")
    
    # 3. 转换为messages格式
    messages = []
    for conv in conversations:
        if conv["from"] == "human":
            messages.append({"role": "user", "content": conv["value"]})
        elif conv["from"] == "gpt":
            messages.append({"role": "assistant", "content": conv["value"]})
        elif conv["from"] == "function_call":
            messages.append({"role": "function", "content": conv["value"]})
        elif conv["from"] == "observation":
            messages.append({"role": "observation", "content": conv["value"]})
    
    log_debug(f"转换后的messages数量: {len(messages)}")
    for i, msg in enumerate(messages):
        content_preview = msg["content"][:50].replace("\n", "\\n")
        log_debug(f"  Message {i+1}: {msg['role']} - {content_preview}...")
    
    # 4. 调用template的_encode方法查看编码过程
    log_debug("\n=== 开始编码过程 ===")
    encoded_messages = template._encode(tokenizer, messages, system, tools)
    
    log_debug(f"编码后的消息段数量: {len(encoded_messages)}")
    for i, encoded in enumerate(encoded_messages):
        log_debug(f"  编码段 {i+1}: {len(encoded)} tokens")
        # 解码前50个token看内容
        if len(encoded) > 0:
            preview = tokenizer.decode(encoded[:min(50, len(encoded))], skip_special_tokens=False)
            preview = preview.replace("\n", "\\n")
            log_debug(f"    内容预览: {preview}...")
    
    # 5. 调用encode_multiturn查看pair分割
    log_debug("\n=== 开始pair分割 ===")
    pairs = template.encode_multiturn(tokenizer, messages, system, tools)
    
    log_debug(f"分割后的pairs数量: {len(pairs)}")
    for i, (source_ids, target_ids) in enumerate(pairs):
        log_debug(f"\n--- Pair {i+1} ---")
        log_debug(f"Source长度: {len(source_ids)} tokens")
        log_debug(f"Target长度: {len(target_ids)} tokens")
        
        # 解码source和target
        if len(source_ids) > 0:
            source_text = tokenizer.decode(source_ids, skip_special_tokens=False)
            source_preview = source_text[:200].replace("\n", "\\n")
            log_debug(f"Source内容: {source_preview}...")
        
        if len(target_ids) > 0:
            target_text = tokenizer.decode(target_ids, skip_special_tokens=False)
            target_preview = target_text[:200].replace("\n", "\\n")
            log_debug(f"Target内容: {target_preview}...")
    
    # 6. 模拟截断过程
    log_debug("\n=== 模拟截断过程 ===")
    cutoff_len = 8192
    total_length = 1 if template.efficient_eos else 0  # 为eos token预留空间
    
    log_debug(f"截断长度: {cutoff_len}")
    log_debug(f"初始长度: {total_length}")
    
    for turn_idx, (source_ids, target_ids) in enumerate(pairs):
        original_source_len = len(source_ids)
        original_target_len = len(target_ids)
        remaining_budget = cutoff_len - total_length
        
        log_debug(f"\nPair {turn_idx + 1}:")
        log_debug(f"  原始长度: source={original_source_len}, target={original_target_len}")
        log_debug(f"  剩余预算: {remaining_budget}")
        
        if total_length >= cutoff_len:
            log_debug(f"  ⚠️ 预算耗尽，丢弃此pair及后续pairs")
            break
        
        # 简化的长度推断逻辑
        source_len = min(original_source_len, remaining_budget // 2)
        target_len = min(original_target_len, remaining_budget - source_len)
        
        log_debug(f"  截断后长度: source={source_len}, target={target_len}")
        
        if source_len < original_source_len:
            log_debug(f"  ⚠️ source被截断: {original_source_len - source_len} tokens")
        if target_len < original_target_len:
            log_debug(f"  ⚠️ target被截断: {original_target_len - target_len} tokens")
        
        total_length += source_len + target_len
        log_debug(f"  当前累计长度: {total_length}/{cutoff_len}")
    
    log_debug(f"\n=== 分析完成 ===")
    log_debug(f"日志已保存到: {log_file}")

if __name__ == "__main__":
    analyze_conversation_pairs()
