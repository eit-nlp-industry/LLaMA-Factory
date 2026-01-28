#!/bin/bash
# Token-level Loss 分析示例脚本

# 设置训练输出目录（根据你的实际训练结果修改）
TRAIN_OUTPUT_DIR="/home/ziqiang/LLaMA-Factory/saves/Qwen3-8B/lora/train_2026-01-12-11-20"
TOKEN_LOSS_DIR="${TRAIN_OUTPUT_DIR}/token_loss_data"
ANALYSIS_OUTPUT_DIR="./token_loss_analysis_results"

# 检查token_loss_data目录是否存在
if [ ! -d "$TOKEN_LOSS_DIR" ]; then
    echo "❌ Token loss data directory not found: $TOKEN_LOSS_DIR"
    echo "   请确保训练已经完成并生成了token_loss_data目录"
    exit 1
fi

# 创建分析输出目录
mkdir -p "$ANALYSIS_OUTPUT_DIR"

echo "📊 开始分析Token-level Loss数据..."
echo "   数据目录: $TOKEN_LOSS_DIR"
echo "   输出目录: $ANALYSIS_OUTPUT_DIR"
echo ""

# 运行分析脚本
python scripts/analyze_token_loss.py \
    --token_loss_dir "$TOKEN_LOSS_DIR" \
    --output_dir "$ANALYSIS_OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 分析完成！"
    echo "📁 结果文件："
    echo "   - ${ANALYSIS_OUTPUT_DIR}/token_loss_analysis_report.md (综合分析报告)"
    echo "   - ${ANALYSIS_OUTPUT_DIR}/high_loss_tokens.csv (高频高loss token)"
    echo "   - ${ANALYSIS_OUTPUT_DIR}/token_type_analysis.csv (token类型分析)"
    echo "   - ${ANALYSIS_OUTPUT_DIR}/position_analysis.csv (位置分析)"
    echo "   - ${ANALYSIS_OUTPUT_DIR}/topk_prediction_analysis.csv (Top-k预测分析)"
    echo "   - ${ANALYSIS_OUTPUT_DIR}/position_loss_analysis.png (位置loss可视化)"
    echo ""
    echo "📖 查看报告:"
    echo "   cat ${ANALYSIS_OUTPUT_DIR}/token_loss_analysis_report.md"
else
    echo "❌ 分析失败，请检查错误信息"
    exit 1
fi
