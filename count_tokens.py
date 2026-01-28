#!/usr/bin/env python3
"""
Token统计脚本
检查训练集中每条数据的token数是否小于cutoff_len (8192)
使用Qwen2.5的tokenizer进行统计
"""

import json
import sys
from pathlib import Path
from transformers import AutoTokenizer

def count_tokens_for_sample(tokenizer, instruction, input_text, output_text, template="qwen"):
    """
    计算单条训练样本的token数
    根据LLaMA-Factory的template格式进行编码
    """
    # 根据qwen template格式构建完整文本
    # Qwen template格式: <|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{instruction}\n{input}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>
    
    system_text = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    
    # 构建完整对话文本
    if input_text:
        user_text = f"{instruction}\n\n{input_text}"
    else:
        user_text = instruction
    
    # 构建完整prompt（包含system和user部分）
    prompt_text = f"<|im_start|>system\n{system_text}<|im_end|>\n<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
    
    # 构建完整文本（包含output）
    full_text = f"{prompt_text}{output_text}<|im_end|>"
    
    # 编码并计算token数
    tokens = tokenizer.encode(full_text, add_special_tokens=False)
    token_count = len(tokens)
    
    return token_count, tokens

def analyze_dataset(json_file_path, model_path, cutoff_len=8192):
    """
    分析数据集，统计每条数据的token数
    """
    print(f"📊 开始分析数据集: {json_file_path}")
    print(f"🤖 使用模型: {model_path}")
    print(f"📏 Cutoff长度: {cutoff_len}")
    print("=" * 80)
    
    # 加载tokenizer
    print("⏳ 正在加载tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print("✅ Tokenizer加载成功")
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
    valid_samples = 0
    invalid_samples = 0
    token_counts = []
    invalid_indices = []
    
    print(f"\n🔍 开始统计token数...")
    print("=" * 80)
    
    # 遍历每条数据
    for idx, sample in enumerate(data):
        if idx % 100 == 0:
            print(f"处理进度: {idx}/{total_samples} ({idx*100/total_samples:.1f}%)")
        
        # 获取字段
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        output_text = sample.get("output", "")
        
        # 计算token数
        try:
            token_count, _ = count_tokens_for_sample(
                tokenizer, instruction, input_text, output_text
            )
            token_counts.append(token_count)
            
            if token_count <= cutoff_len:
                valid_samples += 1
            else:
                invalid_samples += 1
                invalid_indices.append({
                    "index": idx,
                    "token_count": token_count,
                    "instruction": instruction[:100] + "..." if len(instruction) > 100 else instruction,
                    "input": input_text[:100] + "..." if len(input_text) > 100 else input_text,
                    "output": output_text[:100] + "..." if len(output_text) > 100 else output_text,
                })
        except Exception as e:
            print(f"⚠️ 处理第 {idx} 条数据时出错: {e}")
            invalid_samples += 1
            invalid_indices.append({
                "index": idx,
                "token_count": -1,
                "error": str(e)
            })
    
    # 输出统计结果
    print("\n" + "=" * 80)
    print("📊 统计结果")
    print("=" * 80)
    print(f"总样本数: {total_samples}")
    print(f"✅ 有效样本数 (token <= {cutoff_len}): {valid_samples} ({valid_samples*100/total_samples:.2f}%)")
    print(f"❌ 无效样本数 (token > {cutoff_len}): {invalid_samples} ({invalid_samples*100/total_samples:.2f}%)")
    
    if token_counts:
        print(f"\n📈 Token统计:")
        print(f"   最小token数: {min(token_counts)}")
        print(f"   最大token数: {max(token_counts)}")
        print(f"   平均token数: {sum(token_counts)/len(token_counts):.2f}")
        print(f"   中位数token数: {sorted(token_counts)[len(token_counts)//2]}")
        
        # 统计分布
        print(f"\n📊 Token分布:")
        ranges = [
            (0, 1000, "0-1000"),
            (1000, 2000, "1000-2000"),
            (2000, 4000, "2000-4000"),
            (4000, 6000, "4000-6000"),
            (6000, 8192, "6000-8192"),
            (8192, float('inf'), ">8192")
        ]
        for min_val, max_val, label in ranges:
            count = sum(1 for tc in token_counts if min_val < tc <= max_val)
            print(f"   {label}: {count} ({count*100/len(token_counts):.2f}%)")
    
    # 输出无效样本详情
    if invalid_indices:
        print(f"\n❌ 超过cutoff_len的样本详情 (前10条):")
        print("=" * 80)
        for item in invalid_indices[:10]:
            print(f"\n索引: {item['index']}")
            print(f"Token数: {item['token_count']}")
            if 'error' in item:
                print(f"错误: {item['error']}")
            else:
                print(f"Instruction: {item['instruction']}")
                print(f"Input: {item['input']}")
                print(f"Output: {item['output']}")
        
        if len(invalid_indices) > 10:
            print(f"\n... 还有 {len(invalid_indices) - 10} 条无效样本未显示")
        
        # 保存无效样本索引到文件
        invalid_file = Path(json_file_path).parent / "invalid_samples.json"
        with open(invalid_file, 'w', encoding='utf-8') as f:
            json.dump(invalid_indices, f, ensure_ascii=False, indent=2)
        print(f"\n💾 无效样本详情已保存到: {invalid_file}")
    
    print("\n" + "=" * 80)
    if invalid_samples == 0:
        print("✅ 所有样本的token数都在cutoff_len范围内！")
    else:
        print(f"⚠️ 有 {invalid_samples} 条样本超过cutoff_len，需要处理！")
    print("=" * 80)

def main():
    """主函数"""
    # 默认参数
    json_file_path = "/home/ziqiang/LLaMA-Factory/data/dataset/11_25/price_service.json"
    model_path = "/data/models/qwen2.5"
    cutoff_len = 8192
    
    # 从命令行参数获取
    if len(sys.argv) > 1:
        json_file_path = sys.argv[1]
    if len(sys.argv) > 2:
        model_path = sys.argv[2]
    if len(sys.argv) > 3:
        cutoff_len = int(sys.argv[3])
    
    # 检查文件是否存在
    if not Path(json_file_path).exists():
        print(f"❌ 文件不存在: {json_file_path}")
        return
    
    # 执行分析
    analyze_dataset(json_file_path, model_path, cutoff_len)

if __name__ == "__main__":
    main()






