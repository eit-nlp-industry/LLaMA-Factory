#!/usr/bin/env python3
"""
计算训练数据的token数量统计脚本
用于确定cutoff_len的值
使用Qwen3-8B模型的tokenizer
"""

import json
import sys
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

def build_qwen3_prompt(instruction, input_text, output_text, tokenizer):
    """
    根据Qwen3模板格式构建完整的训练文本
    Qwen3格式：
    - user: <|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n
    - assistant: {{content}}<|im_end|>\n
    - 没有system部分（与qwen不同）
    """
    # 构建用户输入部分
    if input_text:
        user_content = f"{instruction}\n\n{input_text}"
    else:
        user_content = instruction
    
    # 构建完整对话文本（按照Qwen3模板格式）
    # user部分
    user_text = f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
    
    # assistant部分（output）
    assistant_text = f"{output_text}<|im_end|>"
    
    # 完整文本
    full_text = user_text + assistant_text
    
    return full_text

def count_tokens(tokenizer, text):
    """计算文本的token数量"""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens), tokens

def analyze_dataset(json_file_path, model_path):
    """
    分析数据集，统计每条数据的token数
    """
    print(f"📊 开始分析数据集: {json_file_path}")
    print(f"🤖 使用模型: {model_path}")
    print("=" * 80)
    
    # 加载tokenizer
    print("⏳ 正在加载tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print("✅ Tokenizer加载成功")
        print(f"   Vocab size: {len(tokenizer)}")
    except Exception as e:
        print(f"❌ Tokenizer加载失败: {e}")
        return
    
    # 读取数据集
    print(f"📖 正在读取数据集...")
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ 数据集读取成功，共 {len(data)} 条数据")
    except Exception as e:
        print(f"❌ 数据集读取失败: {e}")
        return
    
    # 统计信息
    total_samples = len(data)
    token_counts = []
    token_distribution = []
    
    print(f"\n🔍 开始统计token数...")
    print("=" * 80)
    
    # 遍历每条数据
    for idx, sample in enumerate(data):
        if (idx + 1) % 10 == 0 or (idx + 1) == total_samples:
            print(f"处理进度: {idx + 1}/{total_samples} ({(idx + 1)*100/total_samples:.1f}%)")
        
        # 获取字段
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        output_text = sample.get("output", "")
        
        # 构建完整文本
        try:
            full_text = build_qwen3_prompt(instruction, input_text, output_text, tokenizer)
            token_count, tokens = count_tokens(tokenizer, full_text)
            token_counts.append(token_count)
            
            # 分别计算每个字段的token数和字符数，以便对比
            instruction_tokens, _ = count_tokens(tokenizer, instruction)
            input_tokens, _ = count_tokens(tokenizer, input_text)
            output_tokens, _ = count_tokens(tokenizer, output_text)
            
            token_distribution.append({
                "index": idx,
                "token_count": token_count,  # 完整格式化后的token数
                "instruction_tokens": instruction_tokens,  # instruction字段的token数
                "input_tokens": input_tokens,  # input字段的token数
                "output_tokens": output_tokens,  # output字段的token数
                "instruction_len": len(instruction),  # instruction字段的字符数
                "input_len": len(input_text),  # input字段的字符数
                "output_len": len(output_text),  # output字段的字符数
            })
        except Exception as e:
            print(f"⚠️ 处理第 {idx} 条数据时出错: {e}")
            token_distribution.append({
                "index": idx,
                "token_count": -1,
                "error": str(e)
            })
    
    # 计算统计信息
    if not token_counts:
        print("❌ 没有成功处理任何数据")
        return
    
    token_counts_sorted = sorted(token_counts)
    total_tokens = sum(token_counts)
    mean_tokens = np.mean(token_counts)
    median_tokens = np.median(token_counts)
    std_tokens = np.std(token_counts)
    min_tokens = min(token_counts)
    max_tokens = max(token_counts)
    
    # 计算分位数
    p25 = np.percentile(token_counts, 25)
    p50 = np.percentile(token_counts, 50)  # 中位数
    p75 = np.percentile(token_counts, 75)
    p90 = np.percentile(token_counts, 90)
    p95 = np.percentile(token_counts, 95)
    p99 = np.percentile(token_counts, 99)
    
    # 输出统计结果
    print("\n" + "=" * 80)
    print("📊 Token统计结果")
    print("=" * 80)
    print(f"总样本数: {total_samples}")
    print(f"总Token数: {total_tokens:,}")
    print(f"平均Token数: {mean_tokens:.2f}")
    print(f"Token数标准差: {std_tokens:.2f}")
    
    print(f"\n📈 Token范围统计:")
    print(f"   最小token数: {min_tokens}")
    print(f"   最大token数: {max_tokens}")
    print(f"   中位数token数: {median_tokens:.2f}")
    
    print(f"\n📊 Token分位数统计:")
    print(f"   P25 (25%分位数): {p25:.2f}")
    print(f"   P50 (50%分位数/中位数): {p50:.2f}")
    print(f"   P75 (75%分位数): {p75:.2f}")
    print(f"   P90 (90%分位数): {p90:.2f}")
    print(f"   P95 (95%分位数): {p95:.2f}")
    print(f"   P99 (99%分位数): {p99:.2f}")
    
    # 统计分布
    print(f"\n📊 Token分布统计:")
    ranges = [
        (0, 1000, "0-1000"),
        (1000, 2000, "1000-2000"),
        (2000, 4000, "2000-4000"),
        (4000, 6000, "4000-6000"),
        (6000, 8000, "6000-8000"),
        (8000, 10000, "8000-10000"),
        (10000, 12000, "10000-12000"),
        (12000, 16000, "12000-16000"),
        (16000, 20000, "16000-20000"),
        (20000, float('inf'), ">20000")
    ]
    
    for min_val, max_val, label in ranges:
        if max_val == float('inf'):
            count = sum(1 for tc in token_counts if tc > min_val)
        else:
            count = sum(1 for tc in token_counts if min_val < tc <= max_val)
        percentage = count * 100 / len(token_counts) if token_counts else 0
        print(f"   {label:15s}: {count:4d} ({percentage:5.2f}%)")
    
    # 推荐cutoff_len值
    print(f"\n💡 Cutoff_len建议:")
    print(f"   如果希望保留90%的数据: {int(p90)}")
    print(f"   如果希望保留95%的数据: {int(p95)}")
    print(f"   如果希望保留99%的数据: {int(p99)}")
    print(f"   如果想保留所有数据: {max_tokens}")
    print(f"   建议值（保留95%数据，留有余量）: {int(p95) + 100}")
    
    # 统计超过常见cutoff_len值的样本数
    common_cutoffs = [2048, 4096, 8192, 16384, 32768]
    print(f"\n📋 不同cutoff_len值下的数据保留情况:")
    for cutoff in common_cutoffs:
        if cutoff >= max_tokens:
            valid = len(token_counts)
            print(f"   cutoff_len={cutoff:5d}: {valid}/{total_samples} ({valid*100/total_samples:.2f}%) ✅ 所有数据可保留")
            break
        else:
            valid = sum(1 for tc in token_counts if tc <= cutoff)
            lost = total_samples - valid
            print(f"   cutoff_len={cutoff:5d}: {valid}/{total_samples} ({valid*100/total_samples:.2f}%) 保留, {lost} ({lost*100/total_samples:.2f}%) 会被截断")
    
    # 保存详细统计到文件
    output_file = Path(json_file_path).parent / "token_statistics.json"
    stats = {
        "model_path": model_path,
        "dataset_path": str(json_file_path),
        "total_samples": total_samples,
        "statistics": {
            "total_tokens": int(total_tokens),
            "mean": float(mean_tokens),
            "median": float(median_tokens),
            "std": float(std_tokens),
            "min": int(min_tokens),
            "max": int(max_tokens),
            "p25": float(p25),
            "p50": float(p50),
            "p75": float(p75),
            "p90": float(p90),
            "p95": float(p95),
            "p99": float(p99),
        },
        "recommended_cutoff_len": int(p95) + 100,
        "token_distribution": token_distribution
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n💾 详细统计已保存到: {output_file}")
    
    print("\n" + "=" * 80)
    print("✅ 分析完成！")
    print("=" * 80)

def main():
    """主函数"""
    # 默认参数
    json_file_path = "/home/ziqiang/LLaMA-Factory/data/dataset/01_17/train_sft.json"
    model_path = "/data/models/Qwen3-8B"
    
    # 从命令行参数获取
    if len(sys.argv) > 1:
        json_file_path = sys.argv[1]
    if len(sys.argv) > 2:
        model_path = sys.argv[2]
    
    # 检查文件是否存在
    if not Path(json_file_path).exists():
        print(f"❌ 数据文件不存在: {json_file_path}")
        return
    
    if not Path(model_path).exists():
        print(f"❌ 模型路径不存在: {model_path}")
        return
    
    # 执行分析
    analyze_dataset(json_file_path, model_path)

if __name__ == "__main__":
    main()

