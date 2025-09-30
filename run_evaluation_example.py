#!/usr/bin/env python3
"""
运行评估的示例脚本
"""

import json
from eval_by_training_flow import TrainingFlowEvaluator

def create_sample_test_data():
    """创建示例测试数据"""
    test_data = [
        {
            "conversations": [
                {
                    "from": "human",
                    "value": "帮我看看2025年6月到8月各种收入类型的占比情况"
                },
                {
                    "from": "function_call",
                    "value": "{\"name\": \"retrieval_tool\", \"arguments\": {\"query\": \"帮我看看2025年6月到8月各种收入类型的占比情况\", \"source_filter\": \"toollist\", \"user_id\": 136451106, \"top_k\": 10}}"
                },
                {
                    "from": "observation",
                    "value": "[{\"name\": \"calculate_revenue_statistics\", \"description\": \"收入统计计算工具，基于财务报表数据分析门票、交通、餐饮、住宿、导游、其他等6类业态收入占比和金额分布\"}]"
                },
                {
                    "from": "function_call",
                    "value": "{\"name\": \"analyze_revenue_by_type\", \"arguments\": {\"start_date\": \"2025-06-01\", \"end_date\": \"2025-08-31\"}}"
                },
                {
                    "from": "observation",
                    "value": "[{\"type\": \"text\", \"text\": \"收入类型分析报告\\n\\n分析期间：2025-06-01 至 2025-08-31\\n总收入：¥1,285,893.00\\n\\n各业态收入分布：\\n• 门票: ¥605,697.00 (47.1%)\\n• 交通: ¥650,221.00 (50.6%)\\n• 餐饮: ¥25,975.00 (2.0%)\\n• 住宿: ¥4,000.00 (0.3%)\"}]"
                },
                {
                    "from": "gpt",
                    "value": "根据2025年6月到8月的收入数据分析，各种收入类型的占比情况如下：\\n\\n## 收入类型分析报告\\n\\n**分析期间：** 2025-06-01 至 2025-08-31  \\n**总收入：** ¥1,285,893.00\\n\\n### 各业态收入分布：\\n\\n- **交通：** ¥650,221.00 (50.6%) - 主要收入来源\\n- **门票：** ¥605,697.00 (47.1%) - 重要收入组成\\n- **餐饮：** ¥25,975.00 (2.0%) - 辅助收入\\n- **住宿：** ¥4,000.00 (0.3%) - 少量收入\\n\\n从数据可以看出，交通和门票是主要的收入来源，合计占比达到97.7%，其中交通收入略高于门票收入。餐饮和住宿收入相对较少，仅占总收入的2.3%。"
                }
            ],
            "system": "\\n\\n# 工具\\n\\n你可以调用一个或多个函数来协助处理用户查询。\\n\\n在 <tools></tools> XML 标签中提供了可用的函数签名：\\n<tools>\\n</tools>\\n\\n你在输出时必须严格遵循以下规则：\\n\\n1. 如果需要调用函数，则 **只能输出一个函数调用**，格式如下：\\n<tool_call>\\n{\"name\": <function-name>, \"arguments\": <args-dict>}\\n</tool_call>",
            "tools": "[{\"name\": \"retrieval_tool\", \"description\": \"根据用户的问题，在知识库中搜索相关信息。可以指定知识来源（如工具库、对话历史或具体的'建德'、'新昌'文档库），并返回相关的工具、对话或文档信息，以便进一步处理用户查询。\"}]"
        }
    ]
    
    return test_data

def test_with_mock_api():
    """使用模拟API进行测试"""
    print("=== 使用模拟API测试评估流程 ===")
    
    # 创建评估器 - 使用模拟API端点
    evaluator = TrainingFlowEvaluator(
        model_endpoint="http://localhost:8021/v1/completions",
        judge_endpoint="http://localhost:8021/v1/completions"
    )
    
    # 重写API调用方法，返回模拟结果
    def mock_model_api(prompt):
        """模拟模型API调用"""
        if "retrieval_tool" in prompt:
            return '<think>\\n\\n</think>\\n\\n<tool_call>\\n{"name": "retrieval_tool", "arguments": {"query": "帮我看看2025年6月到8月各种收入类型的占比情况", "source_filter": "toollist", "user_id": 136451106, "top_k": 10}}\\n</tool_call>'
        elif "analyze_revenue_by_type" in prompt:
            return '<think>\\n\\n</think>\\n\\n<tool_call>\\n{"name": "analyze_revenue_by_type", "arguments": {"start_date": "2025-06-01", "end_date": "2025-08-31"}}\\n</tool_call>'
        else:
            return "根据2025年6月到8月的收入数据分析，各种收入类型的占比情况如下：\\n\\n## 收入类型分析报告\\n\\n**分析期间：** 2025-06-01 至 2025-08-31\\n**总收入：** ¥1,285,893.00"
    
    def mock_judge_api(prompt):
        """模拟Judge API调用"""
        return """准确性分数：4/5
完整性分数：4/5  
相关性分数：5/5
清晰度分数：4/5
总体分数：4/5
评估理由：回复内容准确且相关，涵盖了主要信息，表述清晰，整体质量较高。"""
    
    # 替换API调用方法
    original_call_model = evaluator.call_model_api
    original_judge = evaluator.llm_judge_response
    
    evaluator.call_model_api = lambda prompt, max_tokens=512: mock_model_api(prompt)
    evaluator.llm_judge_response = lambda pred, exp, ctx="": {
        "accuracy": 4.0, "completeness": 4.0, "relevance": 5.0, 
        "clarity": 4.0, "overall": 4.0
    }
    
    # 创建测试数据
    test_data = create_sample_test_data()
    
    # 保存测试数据到文件
    test_file = "/home/ziqiang/LLaMA-Factory/sample_test_data.json"
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print(f"测试数据已保存到: {test_file}")
    
    # 运行评估
    try:
        results = evaluator.evaluate_dataset(test_file, "/home/ziqiang/LLaMA-Factory/mock_eval_results.json")
        
        print("\\n=== 评估结果摘要 ===")
        stats = results["stats"]
        
        print(f"Function Call评估:")
        print(f"  总数: {stats['function_call']['total']}")
        print(f"  名称准确率: {stats['function_call']['name_accuracy']:.2%}")
        print(f"  参数准确率: {stats['function_call']['args_accuracy']:.2%}")
        print(f"  整体准确率: {stats['function_call']['overall_accuracy']:.2%}")
        
        print(f"\\nAssistant Response评估:")
        print(f"  总数: {stats['assistant_response']['total']}")
        print(f"  平均分数: {stats['assistant_response']['average_score']:.2f}/5")
        
        print("\\n详细结果已保存到: /home/ziqiang/LLaMA-Factory/mock_eval_results.json")
        
    except Exception as e:
        print(f"评估过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # 恢复原始方法
    evaluator.call_model_api = original_call_model
    evaluator.llm_judge_response = original_judge

def show_usage_instructions():
    """显示使用说明"""
    print("\\n=== 使用说明 ===")
    print("1. 确保你的模型API服务正在运行在 http://localhost:8021")
    print("2. 准备测试数据文件，格式应该与训练数据相同（ShareGPT格式）")
    print("3. 修改 eval_by_training_flow.py 中的 main() 函数，设置正确的测试文件路径")
    print("4. 运行评估脚本：")
    print("   python eval_by_training_flow.py")
    print("\\n评估将按照训练时的pair分割方式进行，包括：")
    print("- Function call的名称和参数准确性评估")
    print("- Assistant response的LLM judge评估")
    print("- 详细的评估日志和统计结果")

def main():
    """主函数"""
    # 运行模拟测试
    test_with_mock_api()
    
    # 显示使用说明
    show_usage_instructions()

if __name__ == "__main__":
    main()
