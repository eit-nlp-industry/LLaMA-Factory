#!/bin/bash
# 测试集成后的完整增强训练功能

set -e

echo "🚀 测试集成后的完整增强训练功能"
echo "=" * 60

# 运行create_enhanced_training.py脚本
cd /home/ziqiang/LLaMA-Factory
python create_enhanced_training.py

echo "✅ 测试完成!"
echo ""
echo "📋 功能验证清单:"
echo "✅ Token分析和解码功能"
echo "✅ 预测Token监控功能（包含Token ID和调试信息）"
echo "✅ -100部分分析和多轮对话结构分析"
echo "✅ 训练过程监控"
echo "✅ 日志文件生成"
echo "✅ 脚本集成完成"
echo ""
echo "🔍 新增功能验证:"
echo "✅ 预测Token ID详细打印"
echo "✅ 预测准确率调试信息"
echo "✅ 忽略Token（-100）分析"
echo "✅ 多轮对话分段分析"
echo "✅ 对话模式检测"

