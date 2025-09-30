#!/usr/bin/env python3
"""
使用真实测试数据运行评估的脚本
"""

import json
import os
from datetime import datetime
from eval_by_training_flow import TrainingFlowEvaluator

def main():
    """主函数"""
    print("=== 真实数据评估开始 ===")
    
    # 配置
    test_file = "/home/ziqiang/LLaMA-Factory/data/dataset/9_17/9.17_train_data_top5_final.json"
    output_file = "/home/ziqiang/LLaMA-Factory/real_evaluation_results.json"
    
    # 检查文件是否存在
    if not os.path.exists(test_file):
        print(f"错误：测试数据文件不存在: {test_file}")
        return
    
    # 检查API服务是否可用
    print("检查API服务状态...")
    try:
        import requests
        response = requests.get("http://localhost:8021/v1/models", timeout=5)
        if response.status_code == 200:
            print("✅ API服务正常运行")
        else:
            print(f"⚠️ API服务响应异常: {response.status_code}")
    except Exception as e:
        print(f"❌ API服务连接失败: {str(e)}")
        print("请确保模型服务运行在 http://localhost:8021")
        return
    
    # 创建评估器
    print("初始化评估器...")
    evaluator = TrainingFlowEvaluator(
        model_endpoint="http://localhost:8021/v1/completions",
        judge_endpoint="http://localhost:8021/v1/completions"
    )
    
    # 获取数据统计信息
    print("分析测试数据...")
    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    total_samples = len(test_data)
    print(f"测试数据总量: {total_samples} 条样本")
    
    # 计算预计时间
    estimated_time = total_samples * 30  # 假设每个样本30秒
    print(f"预计评估时间: {estimated_time//60} 分钟")
    
    # 询问是否继续
    choice = input("是否开始评估？(y/n): ").lower().strip()
    if choice != 'y':
        print("评估已取消")
        return
    
    # 开始评估
    print(f"\n开始评估 {total_samples} 条样本...")
    start_time = datetime.now()
    
    try:
        results = evaluator.evaluate_dataset(test_file, output_file)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n=== 评估完成 ===")
        print(f"总耗时: {duration}")
        print(f"平均每条样本: {duration.total_seconds()/total_samples:.2f} 秒")
        
        # 显示结果摘要
        stats = results["stats"]
        print(f"\n=== 评估结果摘要 ===")
        
        if stats["function_call"]["total"] > 0:
            print(f"Function Call评估:")
            print(f"  总数: {stats['function_call']['total']}")
            print(f"  名称准确率: {stats['function_call']['name_accuracy']:.2%}")
            print(f"  参数准确率: {stats['function_call']['args_accuracy']:.2%}")
            print(f"  整体准确率: {stats['function_call']['overall_accuracy']:.2%}")
        
        if stats["assistant_response"]["total"] > 0:
            print(f"\nAssistant Response评估:")
            print(f"  总数: {stats['assistant_response']['total']}")
            print(f"  平均分数: {stats['assistant_response']['average_score']:.2f}/5")
            
            if stats["assistant_response"]["score_distribution"]:
                print(f"  分数分布:")
                for range_str, count in stats["assistant_response"]["score_distribution"].items():
                    print(f"    {range_str}: {count} 条")
        
        print(f"\n详细结果已保存到: {output_file}")
        print(f"评估日志: {evaluator.log_file}")
        
    except KeyboardInterrupt:
        print("\n\n评估被用户中断")
    except Exception as e:
        print(f"\n评估过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

def quick_test():
    """快速测试前几个样本"""
    print("=== 快速测试模式 ===")
    
    test_file = "/home/ziqiang/LLaMA-Factory/data/dataset/9_17/9.17_train_data_top5_final.json"
    
    # 加载前5个样本进行测试
    with open(test_file, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    
    test_data = all_data[:5]  # 只取前5个样本
    test_file_small = "/home/ziqiang/LLaMA-Factory/quick_test_data.json"
    
    with open(test_file_small, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print(f"创建快速测试数据: {len(test_data)} 条样本")
    
    # 创建评估器
    evaluator = TrainingFlowEvaluator(
        model_endpoint="http://localhost:8021/v1/completions",
        judge_endpoint="http://localhost:8021/v1/completions"
    )
    
    # 运行快速测试
    try:
        results = evaluator.evaluate_dataset(test_file_small, "/home/ziqiang/LLaMA-Factory/quick_test_results.json")
        
        stats = results["stats"]
        print(f"\n=== 快速测试结果 ===")
        print(f"Function Call - 整体准确率: {stats['function_call']['overall_accuracy']:.2%}")
        print(f"Assistant Response - 平均分数: {stats['assistant_response']['average_score']:.2f}/5")
        
    except Exception as e:
        print(f"快速测试失败: {str(e)}")
    
    # 清理临时文件
    if os.path.exists(test_file_small):
        os.remove(test_file_small)

if __name__ == "__main__":
    print("选择运行模式:")
    print("1. 完整评估 (1544条样本)")
    print("2. 快速测试 (5条样本)")
    
    choice = input("请选择 (1/2): ").strip()
    
    if choice == "1":
        main()
    elif choice == "2":
        quick_test()
    else:
        print("无效选择")
