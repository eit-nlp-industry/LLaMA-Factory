#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 toolsssss.json 转换为 function_call_context_audit.json 格式的脚本
"""

import json
import asyncio
import aiohttp
import os
from typing import Dict, Any, List
from datetime import datetime
import random


# Qwen API配置（vLLM兼容）
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://125.122.38.32:8021")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "")
QWEN_MODEL_NAME = "/data/models/Qwen3-8B"
QWEN_API_URL = f"{VLLM_BASE_URL.rstrip('/')}/v1/chat/completions"


# 从 rewrite_long_data.py 中提取的 API 调用函数
async def call_qwen_api(session: aiohttp.ClientSession, prompt: str, max_retries: int = 3) -> str:
    """
    调用 Qwen API 生成响应
    """
    headers = {
        "Content-Type": "application/json"
    }
    if VLLM_API_KEY:
        headers["Authorization"] = f"Bearer {VLLM_API_KEY}"
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    payload = {
        "model": QWEN_MODEL_NAME,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 500,
        "stream": False,
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    }
    
    for attempt in range(max_retries):
        try:
            async with session.post(QWEN_API_URL, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
                if response.status == 200:
                    result = await response.json()
                    return result['choices'][0]['message']['content'].strip()
                else:
                    error_text = await response.text()
                    print(f"API 错误 (尝试 {attempt + 1}/{max_retries}): HTTP {response.status}")
                    print(f"错误详情: {error_text}")
                    
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        return f"基于工具返回结果生成响应。"
                        
        except asyncio.TimeoutError:
            print(f"API 超时 (尝试 {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                return f"基于工具返回结果生成响应。"
                
        except Exception as e:
            print(f"API 调用异常 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                return f"基于工具返回结果生成响应。"
    
    return f"基于工具返回结果生成响应。"


async def generate_gpt_response(session: aiohttp.ClientSession, query: str, tool_response: List[Dict]) -> str:
    """
    根据用户查询和工具返回结果生成 GPT 响应
    """
    # 解析工具返回结果
    service_data = None
    for item in tool_response:
        if item.get("type") == "service_data":
            service_data = item
            break
    
    if not service_data:
        return "操作已完成。"
    
    success = service_data.get("success", False)
    msg = service_data.get("msg", "")
    data = service_data.get("data", "")
    
    # 构建提示词
    prompt = f"""你是一个智能助手，需要根据用户的请求和工具执行结果，生成简洁友好的回复。

用户请求：{query}

工具执行结果：
- 是否成功：{success}
- 消息：{msg}
- 数据：{data}

请生成一个简短、友好、自然的回复（不超过30个字），告知用户操作结果。只输出回复内容，不要有任何其他说明。

示例：
- 如果成功新增：已成功新增审批人XXX。
- 如果成功修改：已成功将审批人更新为XXX。
- 如果成功删除：已成功删除审批人XXX。
- 如果失败：操作失败：XXX已经存在，无法重复添加。

现在请生成回复："""
    
    response = await call_qwen_api(session, prompt)
    return response


# 系统提示词模板
SYSTEM_TEMPLATE = """
# 工具

你可以调用一个或多个函数来协助处理用户查询。

在 <tools></tools> XML 标签中提供了可用的函数签名：
<tools>

</tools>

你在输出时必须严格遵循以下规则：

1. 如果需要调用函数，则 **只能输出一个函数调用**，JSON格式如下：
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

正确输出示例：
<tool_call>
{"name": "function_name", "arguments": {"arg1": "value1", "arg2": 123}}
</tool_call>

2. 如果你已经从工具返回结果或已有推理得出足够信息，必须立即停止调用工具，并输出最终答案，格式如下：
<answer>
你的最终答案在这里
</answer>

3. **智能流程阶段判断**：
- 仔细分析下方的对话流程历史，了解当前处于哪个阶段
- 如果看到Assistant已经调用过工具且User已经提供了<tool_response>...</tool_response>，说明工具调用已完成
- 如果工具返回空数据（如总收入为0、空列表等），应生成解释性答案而不是重复调用
- 如果已经获得足够信息回答用户问题，立即生成最终答案

4. **严格禁止以下行为**：
- 在同一轮输出中同时给出函数调用和最终答案  
- 使用完全相同的参数重复调用同一个工具
- 在工具已经返回结果（包括空结果）后，继续调用相同工具
- 忽略对话流程历史中已有的工具调用和响应信息

**特别注意：通过查看下方的对话流程历史，你可以清楚地看到：**
- 之前的User和Assistant交互
- 已经执行过的工具调用  
- 工具返回的具体结果
- 当前对话进行到了哪个阶段

**错误处理和数据质量判断**：
- 如果工具返回 `success: False` 或明确的错误码（如status_code: 3001），说明操作失败
- 如果工具返回总收入为0、空列表或空图表数据，说明查询条件下确实没有数据
- 如果工具返回错误信息（如"资源不存在"、"参数无效"等），不应重复调用
- 此时应该生成解释性答案，说明具体的错误原因或数据状况
- **绝对不要**因为工具返回错误或空数据就重复调用同一工具

**工具调用历史检查**：
- 在每次调用工具前，必须检查对话历史中是否已经调用过相同工具
- 如果历史中存在相同工具调用且已有返回结果，必须基于该结果生成答案
- 如果上一次调用失败，应分析失败原因并解释给用户，而不是重试

记住：基于对话流程历史判断当前阶段，一旦能够生成答案就立即输出，避免无意义的工具重复调用。

# 前端页面信息处理规则
当前页面信息：审批流程配置页面，用于编辑和配置审批流程模板，包括设置审批人员、抄送人员和审批节点等
当前页面参数：{
  "id": {
    "description": "审批模板ID",
    "value": 105
  }
}
指定工具列表：{
  "jsonrpc": "2.0",
  "id": "string",
  "result": {
    "tools": [
      {
        "name": "get_approval_list",
        "chinese_name": "审批列表查询工具",
        "description": "查询指定用户ID的审批列表，可以根据用户是申请人还是审批人，以及审批状态进行筛选。",
        "category": "approval",
        "enabled": true,
        "source": "internal_service",
        "inputSchema": {
          "type": "object",
          "properties": {
            "user_id": {
              "type": "integer",
              "description": "用户的唯一标识ID。此参数通常由系统根据当前登录用户自动填充。",
              "label": "用户的唯一标识ID。此参数通常由系统根据当前登录用户自动填充。",
              "format": null,
              "pattern": null,
              "examples": null,
              "default": null,
              "enum": null
            },
            "user_role": {
              "type": "string",
              "description": "用户角色，查询时所使用的用户角色",
              "label": "用户角色，查询时所使用的用户角色",
              "format": null,
              "pattern": null,
              "examples": [
                "approver"
              ],
              "default": null,
              "enum": [
                "applicant",
                "approver"
              ]
            },
            "approval_status": {
              "type": "string",
              "description": "审批状态，根据审批状态进行筛选",
              "label": "审批状态，根据审批状态进行筛选",
              "format": null,
              "pattern": null,
              "examples": [
                "pending"
              ],
              "default": null,
              "enum": [
                "pending",
                "approved",
                "rejected",
                "cancelled"
              ]
            }
          },
          "required": [
            "user_id",
            "user_role"
          ],
          "additionalProperties": null
        }
      },
      {
        "name": "update_approval_template",
        "chinese_name": "审批模板更新工具",
        "description": "对审批流模板中的审批人和抄送人进行新增、更新、删除等操作",
        "category": "approval",
        "enabled": true,
        "source": "internal_service",
        "inputSchema": {
          "type": "object",
          "properties": {
            "approval_template_id": {
              "type": "integer",
              "description": "唯一标识ID，要编辑的审批模板对应唯一标识ID，是一个大于0的整数。",
              "label": "唯一标识ID，要编辑的审批模板对应唯一标识ID，是一个大于0的整数。"
            },
            "operate_entity": {
              "type": "string",
              "description": "用于定位新增或修改或删除的位置的参考对象，而非目标对象。可以是已有审批人/抄送人的姓名（2-4个中文字符），也可以是序号（大于0的整数字符串），表示在其前后插入或替换。如果不需要定位，则为空字符串。例如：在第1位新增时填写 '1'，在张三后新增时填写 '张三'。",
              "examples": [
                "张三",
                "1",
                ""
              ],
              "label": "用于定位新增或修改或删除的位置的参考对象，而非目标对象。可以是已有审批人/抄送人的姓名（2-4个中文字符），也可以是序号（大于0的整数字符串），表示在其前后插入或替换。如果不需要定位，则为空字符串。例如：在第1位新增时填写 '1'，在张三后新增时填写 '张三'。"
            },
            "entity_type": {
              "type": "string",
              "description": "操作对象类型，要操作的对象对应的类型，分为审批人：approver、抄送人：cc",
              "enum": [
                "approver",
                "cc"
              ],
              "examples": [
                "approver"
              ],
              "label": "操作对象类型，要操作的对象对应的类型，分为审批人：approver、抄送人：cc"
            },
            "operate_type": {
              "type": "string",
              "description": "操作类型：create（新增）、update（修改）、delete（删除）。",
              "enum": [
                "create",
                "update",
                "delete"
              ],
              "examples": [
                "create"
              ],
              "label": "操作类型：create（新增）、update（修改）、delete（删除）。"
            },
            "target_values": {
              "type": "array",
              "description": "要新增或修改成的审批人/抄送人姓名列表。删除操作时必须为空数组。",
              "examples": [
                [
                  "张三",
                  "李四"
                ],
                []
              ],
              "label": "要新增或修改成的审批人/抄送人姓名列表。删除操作时必须为空数组。"
            }
          },
          "required": [
            "approval_template_id",
            "operate_entity",
            "entity_type",
            "operate_type",
            "target_values"
          ]
        }
      }
    ]
  }
}
目前涉及前端页面信息，请遵循以下规则：
1. 检查能否基于当前页面信息直接生成答案，若能则立即输出最终答案
2. 若需调用工具，优先使用当前页面参数匹配指定工具列表中的工具
3. 否则使用当前页面信息作为上下文进行常规工具调用
4. 输出**必须**仍然遵循常规工具调用的输出格式

"""

TOOLS_TEMPLATE = '[{"name": "retrieval_tool", "chinese_name": "retrieval_tool", "description": "根据用户的问题，在知识库中搜索相关信息。可以指定知识来源（如工具库、对话历史或具体的\'建德\'、\'新昌\'文档库），并返回最匹配的结果。", "category": "nlp", "enabled": true, "source": "internal_service", "inputSchema": {"type": "object", "properties": {"query": {"type": "string", "description": "用户的查询内容或问题", "label": "用户的查询内容或问题", "format": null, "pattern": null, "examples": null, "default": null, "enum": null}, "top_k": {"type": "integer", "description": "可选：需要返回的最相关结果的数量", "label": "可选：需要返回的最相关结果的数量", "format": null, "pattern": null, "examples": null, "default": 3, "enum": null}, "source_filter": {"type": "string", "description": "必选：指定检索的知识库来源以缩小搜索范围。\'toollist\'搜索mcp工具库，\'xinchang\'搜索新昌的导游手册，\'jiande\'搜索建德的导游手册。", "label": "必选：指定检索的知识库来源以缩小搜索范围。\'toollist\'搜索mcp工具库，\'xinchang\'搜索新昌的导游手册，\'jiande\'搜索建德的导游手册。", "format": null, "pattern": null, "examples": ["toollist", "jiande"], "default": null, "enum": ["toollist", "jiande", "xinchang"]}, "user_id": {"type": "integer", "description": "必选，用户的ID，用于确认身份", "label": "必选，用户的ID，用于确认身份", "format": null, "pattern": null, "examples": null, "default": null, "enum": null}}, "required": ["query", "source_filter", "user_id"], "additionalProperties": null}}]'


def generate_random_time() -> str:
    """生成随机时间戳"""
    year = 2025
    month = random.randint(9, 10)
    day = random.randint(1, 28)
    hour = random.randint(0, 23)
    minute = random.randint(0, 59)
    return f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}"


async def convert_item(session: aiohttp.ClientSession, item: Dict[str, Any], index: int, total: int) -> Dict[str, Any]:
    """
    转换单个数据项
    """
    print(f"处理 [{index + 1}/{total}]: {item['query']}")
    
    query = item['query']
    input_data = item['input']
    output_data = item['output']
    
    # 检查并处理 user_id
    user_id = None
    arguments = input_data['arguments'].copy()
    
    if 'user_id' in arguments:
        user_id = arguments['user_id']
        # 如果 user_id 超过两位数，生成新的两位数
        if user_id > 99:
            user_id = random.randint(10, 99)
            arguments['user_id'] = user_id
    
    # 构建 function_call
    function_call_value = json.dumps({
        "name": input_data['name'],
        "arguments": arguments
    }, ensure_ascii=False)
    
    # 构建 observation
    observation_value = json.dumps(output_data, ensure_ascii=False)
    
    # 生成 gpt 响应
    gpt_value = await generate_gpt_response(session, query, output_data)
    
    # 构建对话
    conversations = [
        {
            "from": "human",
            "value": query
        },
        {
            "from": "function_call",
            "value": function_call_value
        },
        {
            "from": "observation",
            "value": observation_value
        },
        {
            "from": "gpt",
            "value": gpt_value
        }
    ]
    
    # 构建完整数据项
    result = {
        "conversations": conversations,
        "system": SYSTEM_TEMPLATE,
        "tools": TOOLS_TEMPLATE,
        "time": generate_random_time()
    }
    
    # 如果有 user_id，添加到顶层
    if user_id is not None:
        result['user_id'] = user_id
    
    print(f"  ✓ 生成响应: {gpt_value}")
    if user_id is not None:
        print(f"  ✓ user_id: {user_id}")
    
    return result


def get_item_key(conversations: List[Dict]) -> str:
    """
    生成数据项的唯一键
    使用 query + function_call 的组合作为键
    """
    if len(conversations) < 2:
        return ""
    
    query = conversations[0].get('value', '')
    function_call = conversations[1].get('value', '')
    
    return f"{query}||{function_call}"


async def main():
    """主函数"""
    input_file = "/home/ziqiang/LLaMA-Factory/data/dataset/10_22/toolsssss.json"
    output_file = "/home/ziqiang/LLaMA-Factory/data/function_call_data/function_call_context_audit.json"
    
    # 读取输入文件
    print("读取输入文件...")
    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    print(f"共 {len(input_data)} 条数据需要转换")
    
    # 读取现有输出文件
    print("读取现有输出文件...")
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        print(f"现有 {len(existing_data)} 条数据")
    except FileNotFoundError:
        existing_data = []
        print("输出文件不存在，将创建新文件")
    
    # 构建现有数据的key集合，用于去重
    print("构建现有数据索引...")
    existing_keys = set()
    for item in existing_data:
        if 'conversations' in item:
            key = get_item_key(item['conversations'])
            if key:
                existing_keys.add(key)
    print(f"现有唯一key: {len(existing_keys)} 个")
    
    # 创建 aiohttp session
    async with aiohttp.ClientSession() as session:
        # 转换数据
        print("\n开始转换数据...")
        new_data = []
        skipped_count = 0
        
        for i, item in enumerate(input_data):
            try:
                converted = await convert_item(session, item, i, len(input_data))
                
                # 检查是否重复
                item_key = get_item_key(converted['conversations'])
                if item_key in existing_keys:
                    print(f"  ⊗ 跳过重复数据")
                    skipped_count += 1
                    continue
                
                new_data.append(converted)
                existing_keys.add(item_key)  # 添加到已有key集合，避免本次转换中的重复
                
                # 每处理 10 条数据暂停一下
                if (i + 1) % 10 == 0:
                    print(f"\n已处理 {i + 1} 条，跳过重复 {skipped_count} 条，暂停 2 秒...\n")
                    await asyncio.sleep(2)
                    
            except Exception as e:
                print(f"  ✗ 处理失败: {e}")
                continue
    
    # 合并数据
    print(f"\n合并数据...")
    combined_data = existing_data + new_data
    
    # 保存到输出文件
    print(f"保存到输出文件...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 完成！")
    print(f"  输入数据: {len(input_data)} 条")
    print(f"  原有数据: {len(existing_data)} 条")
    print(f"  跳过重复: {skipped_count} 条")
    print(f"  新增数据: {len(new_data)} 条")
    print(f"  总计数据: {len(combined_data)} 条")
    print(f"  输出文件: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())

