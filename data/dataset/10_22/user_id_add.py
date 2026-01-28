import json
import random
import re

# 读取JSON文件
with open('/home/ziqiang/LLaMA-Factory/data/function_call_data/function_call_context_audit.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 处理每条记录
for item in data:
    user_id = None
    
    # 在conversations中查找function_call，提取user_id
    for conv in item.get('conversations', []):
        if conv.get('from') == 'function_call':
            try:
                # 解析function_call的value（JSON字符串）
                func_call = json.loads(conv['value'])
                if 'arguments' in func_call:
                    args = func_call['arguments']
                    if isinstance(args, str):
                        args = json.loads(args)
                    if 'user_id' in args:
                        user_id = args['user_id']
                        break
            except:
                pass
    
    # 如果找到了user_id
    if user_id is not None:
        # 如果user_id超过两位数（>99），生成新的两位数
        if user_id > 99:
            new_user_id = random.randint(10, 99)
            
            # 替换所有function_call中的user_id
            for conv in item.get('conversations', []):
                if conv.get('from') == 'function_call':
                    try:
                        # 使用正则表达式替换user_id值
                        old_value = conv['value']
                        # 匹配 "user_id":数字 或 "user_id": 数字
                        new_value = re.sub(
                            r'"user_id"\s*:\s*' + str(user_id) + r'\b',
                            f'"user_id":{new_user_id}',
                            old_value
                        )
                        conv['value'] = new_value
                    except:
                        pass
            
            user_id = new_user_id
        
        # 在item最后添加user_id字段（在time字段之后）
        item['user_id'] = user_id

# 保存修改后的文件
with open('/home/ziqiang/LLaMA-Factory/data/function_call_data/function_call_context_audit.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("处理完成！")
print(f"共处理了 {len(data)} 条记录")