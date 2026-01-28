#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import random
from datetime import datetime, timedelta

def generate_random_time():
    """生成随机时间戳"""
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 12, 31)
    time_delta = end_date - start_date
    random_days = random.randint(0, time_delta.days)
    random_hours = random.randint(0, 23)
    random_minutes = random.randint(0, 59)
    random_date = start_date + timedelta(days=random_days, hours=random_hours, minutes=random_minutes)
    return random_date.strftime("%Y-%m-%d %H:%M")

# 系统提示和工具配置
SYSTEM_PROMPT = """

# 工具

你可以调用一个或多个函数来协助处理用户查询。

在 <tools></tools> XML 标签中提供了可用的函数签名：
<tools>
</tools>

你在输出时必须严格遵循以下规则：

1. 如果需要调用函数，则 **只能输出一个函数调用**，格式如下：
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
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
- 如果工具返回错误信息（如'资源不存在'、'参数无效'等），不应重复调用
- 此时应该生成解释性答案，说明具体的错误原因或数据状况
- **绝对不要**因为工具返回错误或空数据就重复调用同一工具

**工具调用历史检查**：
- 在每次调用工具前，必须检查对话历史中是否已经调用过相同工具
- 如果历史中存在相同工具调用且已有返回结果，必须基于该结果生成答案
- 如果上一次调用失败，应分析失败原因并解释给用户，而不是重试
记住：基于对话流程历史判断当前阶段，一旦能够生成答案就立即输出，避免无意义的工具重复调用。
"""

TOOLS = '[{"name":"retrieval_tool","description":"根据用户的问题，在知识库中搜索相关信息。可以指定知识来源（如工具库、对话历史或具体的\'建德\'、\'新昌\'文档库），并返回最匹配的结果。","category":"nlp","inputSchema":{"type":"object","properties":{"query":{"type":"string","description":"用户的查询内容或问题"},"top_k":{"type":"integer","description":"可选：需要返回的最相关结果的数量","default":3},"source_filter":{"type":"string","description":"必选：指定检索的知识库来源以缩小搜索范围。\'toollist\'搜索mcp工具库，\'xinchang\'搜索新昌的导游手册，\'jiande\'搜索建德的导游手册。","examples":["toollist","jiande"],"enum":["toollist","jiande","xinchang"]},"user_id":{"type":"integer","description":"必选，用户的ID，用于确认身份"}},"required":["query","source_filter","user_id"]}}]'

# 中文QA对
chinese_qa = [
    # 日常问候
    ("你好", "你好！很高兴为你服务，有什么我可以帮助你的吗？"),
    ("早上好", "早上好！新的一天开始了，希望你有个美好的一天！"),
    ("晚安", "晚安！祝你有个好梦，明天见！"),
    
    # 自我介绍
    ("你是谁", "我是一个AI助手，致力于帮助你解答问题和完成任务。"),
    ("你能做什么", "我可以回答各种问题，提供信息查询、日常对话、知识解答等服务。如果需要特定领域的深度信息，我也可以帮你检索相关资料。"),
    ("介绍一下你自己", "我是一个智能对话助手，基于大型语言模型训练而成。我可以理解自然语言，回答问题，进行对话交流，必要时还能调用工具获取更专业的信息。"),
    
    # 常识问答
    ("什么是人工智能", "人工智能（AI）是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统，如学习、推理、问题解决、感知和语言理解等。"),
    ("地球有多大", "地球的赤道半径约为6378公里，极半径约为6357公里，平均半径约为6371公里。地球的表面积约为5.1亿平方公里。"),
    ("为什么天空是蓝色的", "天空呈现蓝色是因为大气中的气体分子和微小颗粒会散射太阳光。蓝光的波长较短，更容易被散射，所以我们看到的天空主要是蓝色的。这种现象叫做瑞利散射。"),
    ("什么是光合作用", "光合作用是植物、藻类和某些细菌利用阳光能量，将二氧化碳和水转化为葡萄糖等有机物，并释放氧气的过程。这是地球上最重要的生物化学反应之一。"),
    
    # 数学计算
    ("1加1等于几", "1加1等于2。"),
    ("圆周率是多少", "圆周率π约等于3.14159265358979323846...，它是一个无理数，小数部分无限不循环。"),
    ("什么是质数", "质数是指大于1的自然数中，除了1和它本身以外不再有其他因数的数。例如2、3、5、7、11、13等都是质数。"),
    
    # 生活常识
    ("感冒了怎么办", "感冒时建议多休息、多喝水、保持室内空气流通。如果症状严重或持续不退，应及时就医。注意保暖，避免着凉，可以吃一些富含维生素C的水果。"),
    ("如何保持健康", "保持健康的关键包括：规律作息、均衡饮食、适量运动、保持良好心态、定期体检。建议每天至少运动30分钟，多吃蔬菜水果，少吃油腻食物，保证充足睡眠。"),
    ("喝水有什么好处", "喝水的好处很多：促进新陈代谢、维持体温、帮助消化、排出废物、保持皮肤水润、预防便秘等。建议成年人每天饮水1500-2000毫升。"),
    
    # 文化知识
    ("中国有多少个省", "中国有23个省、5个自治区、4个直辖市和2个特别行政区，共计34个省级行政区。"),
    ("什么是春节", "春节是中国农历新年，是中华民族最重要的传统节日。通常在公历1月或2月间，人们会贴春联、放鞭炮、吃团圆饭、走亲访友、发红包等，庆祝新年的到来。"),
    ("长城有多长", "中国长城的总长度约为21196.18公里，包括墙体、壕堑、单体建筑等。我们常说的明长城东起辽宁虎山，西至甘肃嘉峪关，全长约8851.8公里。"),
    
    # 科技知识
    ("什么是5G", "5G是第五代移动通信技术，相比4G具有更高的传输速度、更低的延迟和更大的连接容量。5G能够支持更多的智能设备连接，为物联网、自动驾驶等应用提供基础。"),
    ("区块链是什么", "区块链是一种分布式账本技术，通过密码学方法将数据区块按时间顺序链接起来，形成不可篡改的数据链。它具有去中心化、透明度高、安全性强等特点，被广泛应用于数字货币、供应链管理等领域。"),
    
    # 情感对话
    ("我很开心", "太好了！能看到你开心我也很高兴。有什么好事想分享吗？"),
    ("我有点难过", "听到你难过我很关心。如果你愿意的话，可以和我聊聊发生了什么事。有时候倾诉能让心情好一些。"),
    ("谢谢你", "不客气！能帮到你是我的荣幸。如果还有其他需要，随时告诉我。"),
    
    # 建议咨询
    ("推荐一本书", "我推荐《人类简史》，这是以色列历史学家尤瓦尔·赫拉利的作品，以宏大的视角讲述了人类从石器时代到21世纪的发展历程，非常值得一读。"),
    ("学习编程从哪里开始", "建议从Python开始学习编程，它语法简单、应用广泛。可以先学习基础语法，然后通过做小项目来实践。推荐网站如Codecademy、菜鸟教程等，都有很好的入门教程。"),
    
    # 时间相关（需要配合特定时间戳）
    ("现在几点了", "现在是{time}。"),
    ("今天是几月几号", "今天是{date}。"),
    ("现在是早上还是下午", "现在是{period}。"),
    ("今天星期几", "今天是{weekday}。"),
    
    # 天气和季节
    ("春天有什么特点", "春天是万物复苏的季节，气温逐渐回暖，百花盛开，草木发芽。人们常说一年之计在于春，是播种和新生的季节。"),
    ("夏天要注意什么", "夏天要注意防暑降温，多喝水，避免长时间在烈日下暴晒。外出时做好防晒措施，穿着透气的衣物，预防中暑。"),
    
    # 简单对话
    ("嗯", "我在听，请继续说。"),
    ("好的", "好的，明白了。还有其他需要帮助的吗？"),
    ("再见", "再见！期待下次为你服务。"),
    
    # 励志建议
    ("我想变得更优秀", "很棒的目标！变得更优秀需要持续努力：设定明确目标、制定计划、保持学习、接受挑战、从失败中成长。记住，进步是一个过程，每天进步一点点就很了不起。"),
    ("如何提高效率", "提高效率的方法包括：1)制定优先级清单，先做重要紧急的事；2)避免多任务，专注当下；3)使用番茄工作法等时间管理技巧；4)保持工作环境整洁；5)定期休息，保持精力充沛。"),
    
    # 美食相关
    ("如何做番茄炒蛋", "番茄炒蛋的做法：1)鸡蛋打散加少许盐；2)番茄切块；3)热锅倒油，炒蛋至凝固盛出；4)锅中再加油，炒番茄至出汁；5)放入鸡蛋，加盐和少许糖调味，翻炒均匀即可。"),
    ("早餐吃什么好", "健康的早餐应该包含蛋白质、碳水化合物和维生素。建议搭配：鸡蛋/牛奶（蛋白质）+ 面包/粥（碳水）+ 水果/蔬菜（维生素）。避免油腻食物，营养均衡最重要。"),
    
    # 兴趣爱好
    ("推荐一部电影", "我推荐《肖申克的救赎》，这是一部经典的励志电影，讲述了希望与自由的力量。影片节奏流畅，内涵深刻，是影史评分最高的电影之一。"),
    ("如何学习弹吉他", "学习吉他建议：1)先学习基础乐理和和弦；2)每天坚持练习，从简单曲目开始；3)注意手型和姿势；4)可以跟随在线教程或找老师指导；5)多听多练，培养乐感。坚持是关键！"),
    
    # 更多常识
    ("为什么会下雨", "下雨是因为水循环。太阳加热地表水分蒸发上升，遇冷凝结成小水滴形成云。云中水滴越聚越多，当重力超过上升气流时就会降落，形成降雨。"),
    ("什么是温室效应", "温室效应是指大气中的二氧化碳、甲烷等温室气体吸收地面辐射的热量，使地球表面温度升高的现象。适度的温室效应维持地球温度，但过度则会导致全球变暖。"),
]

# 英文QA对
english_qa = [
    ("Hello", "Hello! How can I help you today?"),
    ("What's your name?", "I'm an AI assistant designed to help answer questions and assist with various tasks."),
    ("How are you?", "I'm functioning well, thank you for asking! How can I assist you today?"),
    ("What is machine learning?", "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make decisions."),
    ("Tell me a joke", "Why don't scientists trust atoms? Because they make up everything!"),
    ("What is the capital of France?", "The capital of France is Paris, known for its iconic landmarks like the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral."),
    ("How do I stay motivated?", "Staying motivated involves setting clear goals, breaking them into smaller tasks, celebrating small wins, maintaining a positive mindset, and surrounding yourself with supportive people. Remember why you started and keep that vision in mind."),
    ("What is climate change?", "Climate change refers to long-term shifts in global temperatures and weather patterns. While natural factors play a role, human activities, particularly burning fossil fuels, have been the dominant cause since the mid-20th century."),
    ("Explain photosynthesis", "Photosynthesis is the process by which green plants use sunlight to synthesize nutrients from carbon dioxide and water. It generates oxygen as a byproduct and is fundamental to life on Earth."),
    ("What is the Internet?", "The Internet is a global network of interconnected computers that allows information to be shared worldwide. It enables communication, access to information, online services, and has revolutionized how we live and work."),
    ("Good morning", "Good morning! I hope you have a wonderful day ahead. What can I help you with?"),
    ("Thank you", "You're very welcome! Feel free to reach out if you need anything else."),
    ("What time is it now?", "It's {time} right now."),
    ("What's today's date?", "Today is {date}."),
    ("Goodbye", "Goodbye! Have a great day and feel free to come back anytime!"),
]

def get_time_info(time_str):
    """根据时间字符串获取相关信息"""
    dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M")
    hour = dt.hour
    
    if 5 <= hour < 12:
        period = "上午"
    elif 12 <= hour < 18:
        period = "下午"
    else:
        period = "晚上"
    
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    weekday = weekdays[dt.weekday()]
    
    return {
        "time": time_str.split()[1],
        "date": time_str.split()[0],
        "period": period,
        "weekday": weekday
    }

def create_qa_item(question, answer, time_str):
    """创建单个QA项"""
    # 如果答案中包含占位符，填充时间信息
    if "{" in answer:
        time_info = get_time_info(time_str)
        answer = answer.format(**time_info)
    
    return {
        "conversations": [
            {
                "from": "human",
                "value": question
            },
            {
                "from": "gpt",
                "value": answer
            }
        ],
        "system": SYSTEM_PROMPT,
        "tools": TOOLS,
        "time": time_str
    }

# 生成所有QA数据
all_qa_data = []

# 选择46个中文QA（确保包含时间相关的）
selected_chinese = chinese_qa.copy()
random.shuffle(selected_chinese)

# 确保包含所有时间相关的问题
time_related = [qa for qa in selected_chinese if "{" in qa[1]]
other_qa = [qa for qa in selected_chinese if "{" not in qa[1]]

# 选择46个QA（包含所有时间相关的 + 随机其他QA）
selected_chinese_final = time_related + other_qa[:46-len(time_related)]

# 随机打乱
random.shuffle(selected_chinese_final)

# 生成中文QA
for question, answer in selected_chinese_final:
    time_str = generate_random_time()
    all_qa_data.append(create_qa_item(question, answer, time_str))

# 生成14个英文QA（包含时间相关的）
for question, answer in english_qa:
    time_str = generate_random_time()
    all_qa_data.append(create_qa_item(question, answer, time_str))

# 随机打乱所有数据
random.shuffle(all_qa_data)

# 写入JSON文件
output_file = "/home/ziqiang/LLaMA-Factory/data/function_call_data/QA.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(all_qa_data, f, ensure_ascii=False, indent=2)

print(f"成功生成 {len(all_qa_data)} 条QA数据！")
print(f"已保存到: {output_file}")

