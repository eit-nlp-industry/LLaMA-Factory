#!/usr/bin/env python3
"""
测试新的pair构造逻辑
"""

import sys
import os
sys.path.append('/home/ziqiang/LLaMA-Factory/src')

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.hparams import DataArguments, ModelArguments
from transformers import AutoTokenizer

def test_pair_construction():
    print("🧪 测试新的pair构造逻辑")
    print("=" * 80)
    
    # 1. 设置参数
    data_args = DataArguments(
        dataset="mixed_training_data_09_17",
        dataset_dir="data",
        cutoff_len=8192,
        template="qwen3",
        enable_thinking=False
    )
    model_args = ModelArguments(
        model_name_or_path="/data/models/Qwen3-8B"
    )
    
    # 2. 加载tokenizer和template
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    
    print(f"✅ 已加载template: {template.__class__.__name__}")
    
    # 3. 模拟一个多轮对话
    messages = [
        {"role": "user", "content": "报表编号L20250611的收入类型分布情况能给我看看吗"},
        {"role": "function", "content": '{"name":"retrieval_tool","arguments":{"query":"分析报表编号L20250611的收入类型构成，基于财务报表数据分析各业态的收入占比和金额分布，以饼图格式展示，用于收入结构分析和业态表现评估。","source_filter":"toollist","user_id":136451106,"top_k":5}}'},
        {"role": "observation", "content": '[{"name":"analyze_revenue_by_type","description":"收入类型构成分析工具"}]'},
        {"role": "assistant", "content": "根据查询结果，以下是收入类型分布情况..."}
    ]
    
    system = "# 工具\n\n你可以调用一个或多个函数来协助处理用户查询。"
    tools = '[{"name":"retrieval_tool","description":"根据用户的问题，在知识库中搜索相关信息。"}]'
    
    print(f"📊 输入messages数量: {len(messages)}")
    print(f"📊 system长度: {len(system)}")
    print(f"📊 tools长度: {len(tools)}")
    
    # 4. 调用encode_multiturn
    print("\n🔄 调用encode_multiturn...")
    pairs = template.encode_multiturn(tokenizer, messages, system, tools)
    
    print(f"✅ 生成了 {len(pairs)} 个pairs")
    
    # 5. 分析每个pair
    for i, (source_ids, target_ids) in enumerate(pairs):
        print(f"\n--- Pair {i+1} ---")
        print(f"📏 source长度: {len(source_ids)} tokens")
        print(f"📏 target长度: {len(target_ids)} tokens")
        
        # 解码source内容
        source_text = tokenizer.decode(source_ids, skip_special_tokens=False)
        print(f"📤 Source内容预览:")
        print(source_text[:500] + "..." if len(source_text) > 500 else source_text)
        
        # 解码target内容
        target_text = tokenizer.decode(target_ids, skip_special_tokens=False)
        print(f"📥 Target内容预览:")
        print(target_text[:200] + "..." if len(target_text) > 200 else target_text)
        
        # 检查source是否包含system信息
        if "system" in source_text.lower():
            print("✅ Source包含system信息")
        else:
            print("❌ Source不包含system信息")
    
    print("\n" + "=" * 80)
    print("🎉 测试完成!")

if __name__ == "__main__":
    test_pair_construction()
