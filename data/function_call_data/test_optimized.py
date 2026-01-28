#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试优化后的Qwen API配置
"""

import asyncio
import aiohttp
import time
import json

QWEN_API_URL = "http://125.122.38.32:8021/v1/chat/completions"
QWEN_MODEL_NAME = "/data/models/Qwen3-8B"

async def test_optimized_compression():
    """测试优化后的压缩配置"""
    
    headers = {"Content-Type": "application/json"}
    
    # 模拟简化的压缩任务
    mock_obs = json.dumps([{"id": i, "data": f"item_{i}"} for i in range(20)], ensure_ascii=False)
    mock_gpt = "找到20条数据：" + "，".join([f"item_{i}" for i in range(20)])
    
    prompt = f"""压缩任务：500 tokens -> 100 tokens

数据样本:
{mock_obs}

gpt样本:
{mock_gpt}

要求：保留5条数据，保持JSON格式

必须按此格式输出：
<observation>
压缩后的数据
</observation>

<gpt>
调整后的回答
</gpt>"""
    
    messages = [{"role": "user", "content": prompt}]
    
    data = {
        "model": QWEN_MODEL_NAME,
        "messages": messages,
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens": 2048,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False}
    }
    
    print("=" * 60)
    print("测试最低负载配置")
    print("=" * 60)
    print(f"Model: {QWEN_MODEL_NAME}")
    print(f"Temperature: 0.1 (最低)")
    print(f"Max tokens: 2048")
    print(f"Timeout: 60秒")
    print("\n发送请求...\n")
    
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(QWEN_API_URL, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=60)) as response:
                elapsed = time.time() - start_time
                print(f"✓ 收到响应，耗时: {elapsed:.2f}秒")
                print(f"状态码: {response.status}\n")
                
                if response.status == 200:
                    result = await response.json()
                    content = result['choices'][0]['message']['content'].strip()
                    
                    print(f"响应长度: {len(content)} 字符\n")
                    print(f"响应内容:\n{content}\n")
                    
                    # 检查格式
                    import re
                    obs_match = re.search(r'<observation>(.*?)</observation>', content, re.DOTALL)
                    gpt_match = re.search(r'<gpt>(.*?)</gpt>', content, re.DOTALL)
                    
                    if obs_match and gpt_match:
                        print("✓ 格式正确！")
                        print(f"  - observation长度: {len(obs_match.group(1).strip())}")
                        print(f"  - gpt长度: {len(gpt_match.group(1).strip())}")
                    else:
                        print("✗ 格式错误！缺少标签")
                    
                    print("\n✓ 测试成功！")
                else:
                    error = await response.text()
                    print(f"✗ API返回错误:\n{error}\n")
                    
        except asyncio.TimeoutError:
            print(f"✗ 请求超时（60秒）\n")
        except Exception as e:
            print(f"✗ 请求失败: {e}\n")

if __name__ == "__main__":
    asyncio.run(test_optimized_compression())



