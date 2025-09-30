#### 按字段分类的准确率评估报告

##### 总体统计

- **总样本数**: 394
- **总正确数**: 372
- **总体准确率**: 0.9442 (94.42%)

##### 各字段详细统计

###### team_size

- **样本数**: 30
- **正确数**: 30
- **准确率**: 1.0000 (100.00%)

**正确示例**:

1. ✅ **Line 41**
   - 预测: `{"team_size": 36}`
   - 标签: `{"team_size": 36}`

2. ✅ **Line 51**
   - 预测: `{"team_size": 36}`
   - 标签: `{"team_size": 36}`

3. ✅ **Line 52**
   - 预测: `{"team_size": 50}`
   - 标签: `{"team_size": 50}`

---

###### end_date

- **样本数**: 30
- **正确数**: 30
- **准确率**: 1.0000 (100.00%)

**正确示例**:

1. ✅ **Line 42**
   - 预测: `{"end_date": "2025-07-17 00:00:00"}`
   - 标签: `{"end_date": "2025-07-17 00:00:00"}`

2. ✅ **Line 48**
   - 预测: `{"end_date": "2025-07-17 00:00:00"}`
   - 标签: `{"end_date": "2025-07-17 00:00:00"}`

3. ✅ **Line 74**
   - 预测: `{"end_date": "2025-07-17 00:00:00"}`
   - 标签: `{"end_date": "2025-07-17 00:00:00"}`

---

###### resource_start_time

- **样本数**: 63
- **正确数**: 63
- **准确率**: 1.0000 (100.00%)

**正确示例**:

1. ✅ **Line 43**
   - 预测: `{"resource_start_time": "2025-07-16 00:00:00"}`
   - 标签: `{"resource_start_time": "2025-07-16 00:00:00"}`

2. ✅ **Line 62**
   - 预测: `{"resource_start_time": "2025-07-17 00:00:00"}`
   - 标签: `{"resource_start_time": "2025-07-17 00:00:00"}`

3. ✅ **Line 63**
   - 预测: `{"resource_start_time": "2025-07-16 00:00:00"}`
   - 标签: `{"resource_start_time": "2025-07-16 00:00:00"}`

---

###### start_date

- **样本数**: 30
- **正确数**: 30
- **准确率**: 1.0000 (100.00%)

**正确示例**:

1. ✅ **Line 45**
   - 预测: `{"start_date": "2025-07-17 00:00:00"}`
   - 标签: `{"start_date": "2025-07-17 00:00:00"}`

2. ✅ **Line 56**
   - 预测: `{"start_date": "2025-07-18 00:00:00"}`
   - 标签: `{"start_date": "2025-07-18 00:00:00"}`

3. ✅ **Line 59**
   - 预测: `{"start_date": "2025-07-17 00:00:00"}`
   - 标签: `{"start_date": "2025-07-17 00:00:00"}`

---

###### resource_team_size

- **样本数**: 63
- **正确数**: 63
- **准确率**: 1.0000 (100.00%)

**正确示例**:

1. ✅ **Line 49**
   - 预测: `{"resource_team_size": 19}`
   - 标签: `{"resource_team_size": 19}`

2. ✅ **Line 78**
   - 预测: `{"resource_team_size": 50}`
   - 标签: `{"resource_team_size": 50}`

3. ✅ **Line 82**
   - 预测: `{"resource_team_size": 45}`
   - 标签: `{"resource_team_size": 45}`

---

###### resource_end_time

- **样本数**: 63
- **正确数**: 61
- **准确率**: 0.9683 (96.83%)

**正确示例**:

1. ✅ **Line 44**
   - 预测: `{"resource_end_time": "2025-07-18 00:00:00"}`
   - 标签: `{"resource_end_time": "2025-07-18 00:00:00"}`

2. ✅ **Line 57**
   - 预测: `{"resource_end_time": "2025-07-16 00:00:00"}`
   - 标签: `{"resource_end_time": "2025-07-16 00:00:00"}`

3. ✅ **Line 58**
   - 预测: `{"resource_end_time": "2025-07-16 00:00:00"}`
   - 标签: `{"resource_end_time": "2025-07-16 00:00:00"}`

**错误示例（全部）**:

1. ❌ **Line 85**
   - **指令**: 请从OCR文本中抽取旅行订单中千岛湖好运岛-千岛湖好运岛门票的结束时间。

严格按照以下JSON格式输出：
{
  "resource_end_time": "YYYY-MM-DD"或null
}
   - **输入**: 导游童益梅18857188254，杭州天翼，人数19人，7月16日，好运岛

instruction: 请从OCR文本中抽取旅行订单中千岛湖好运岛-千岛湖好运岛门票的结束时间。

严格按照以下JSON格式输出：
{
  "resource_end_time": "YYYY-MM-DD"或null
}
   - **预测**: `{"resource_end_time": "2025-07-16 00:00:00"}`
   - **标签**: `{"resource_end_time": "2025-07-17 00:00:00"}`

2. ❌ **Line 208**
   - **指令**: 请从OCR文本中抽取旅行订单中新安江-新安江游船的结束时间。

严格按照以下JSON格式输出：
{
  "resource_end_time": "YYYY-MM-DD"或null
}
   - **输入**: 江苏聚庆，7月17日-7月18日新安江梦幻，好运岛，导游邓13867126741，人数85
人

instruction: 请从OCR文本中抽取旅行订单中新安江-新安江游船的结束时间。

严格按照以下JSON格式输出：
{
  "resource_end_time": "YYYY-MM-DD"或null
}
   - **预测**: `{"resource_end_time": "2025-07-18 00:00:00"}`
   - **标签**: `{"resource_end_time": "2025-07-17 00:00:00"}`

---

###### resource_detail

- **样本数**: 62
- **正确数**: 58
- **准确率**: 0.9355 (93.55%)

**正确示例**:

1. ✅ **Line 46**
   - 预测: `{"resource_detail": {"guide_items": "灵栖洞景区导服"}}`
   - 标签: `{"resource_detail": {"guide_items": "灵栖洞景区导服"}}`

2. ✅ **Line 47**
   - 预测: `{"resource_detail": {"ship_combo": null, "ship_items": "船票"}}`
   - 标签: `{"resource_detail": {"ship_combo": null, "ship_items": "船票"}}`

3. ✅ **Line 50**
   - 预测: `{"resource_detail": {"ship_combo": null, "ship_items": "三江口游线"}}`
   - 标签: `{"resource_detail": {"ship_combo": null, "ship_items": "三江口游线"}}`

**错误示例（全部）**:

1. ❌ **Line 228**
   - **指令**: 请从OCR文本中抽取旅行订单中七里扬帆葫芦山庄餐饮的详细信息。

提取字段及可选值：
meal_standard (餐饮标准): "0"、"20"、"25"、"30"、"35"、"40"、"45"、"...
   - **输入**: 7月18日建德
七里扬帆+葫芦峡漂流套票（1.2m
以下儿童80元）
政策：联票125元
餐：40元留10%扣
导游：金杏花13396506951（无
证)
车牌号：浙A60E57
人数：50人

instruction: 请从OCR文本中抽取旅行订单中七里扬帆葫芦山庄餐饮的详细信息。

提取字段及可选值：
meal_standard (餐饮标准): "0"、"20"、"25"、"30"、"35"...
   - **预测**: `{"resource_detail": {"meal_standard": null, "meal_standard_table": null}}`
   - **标签**: `{"resource_detail": {"meal_standard": "40", "meal_standard_table": null}}`

2. ❌ **Line 246**
   - **指令**: 请从OCR文本中抽取旅行订单中新安江游船的详细信息。

提取字段及可选值：
ship_combo (船票套餐类型): "中远程及新增市场价格"、"学生团队价"、"广德"、"旅投员工价"、"景酒打包价"...
   - **输入**: 建德光大，7月18日，5人，江清（乘船）导游：戴艳嫔13868126262

instruction: 请从OCR文本中抽取旅行订单中新安江游船的详细信息。

提取字段及可选值：
ship_combo (船票套餐类型): "中远程及新增市场价格"、"学生团队价"、"广德"、"旅投员工价"、"景酒打包价"、"研学团"、"老年团队价"
ship_items (船只类型): "新安江竹筏漂流"、"新安江...
   - **预测**: `{"resource_detail": {"ship_combo": null, "ship_items": "江清月近人实景演艺船票"}}`
   - **标签**: `{"resource_detail": {"ship_items": "梦幻新安江"}}`

3. ❌ **Line 259**
   - **指令**: 请从OCR文本中抽取旅行订单中千岛湖好运岛门票的详细信息。

提取字段及可选值：
ticket_combo (门票套餐类型): "30年教师证"、"70周岁"、"中远程及新增市场价格"、"儿童"、"其...
   - **输入**: 旅行社：
安徽海屯国际旅行社有限公司
游玩日期：
7月18号好运岛+灵栖洞
人数：10人
导游：许善春13155531899

instruction: 请从OCR文本中抽取旅行订单中千岛湖好运岛门票的详细信息。

提取字段及可选值：
ticket_combo (门票套餐类型): "30年教师证"、"70周岁"、"中远程及新增市场价格"、"儿童"、"其他"、"军官证"、"在校大学生"、"学生团队价...
   - **预测**: `{"resource_detail": {"ticket_combo": null, "ticket_items": "门票"}}`
   - **标签**: `{"resource_detail": {"ticket_combo": "中远程及新增市场价格", "ticket_items": "门票"}}`

4. ❌ **Line 378**
   - **指令**: 请从OCR文本中抽取旅行订单中千岛湖好运岛游船的详细信息。

提取字段及可选值：
ship_combo (船票套餐类型): "中远程及新增市场价格"、"学生团队价"、"广德"、"旅投员工价"、"景酒打...
   - **输入**: 导游童益梅18857188254，杭州
天翼，人数31人，7月29日，好
运岛

instruction: 请从OCR文本中抽取旅行订单中千岛湖好运岛游船的详细信息。

提取字段及可选值：
ship_combo (船票套餐类型): "中远程及新增市场价格"、"学生团队价"、"广德"、"旅投员工价"、"景酒打包价"、"老年团队价"
ship_items (船只类型): "千岛湖好运岛游船"

严格按...
   - **预测**: `{"resource_detail": {"ship_combo": null, "ship_items": "千岛湖好运岛游船"}}`
   - **标签**: `{"resource_detail": {"ship_combo": null}}`

---

###### contacts

- **样本数**: 27
- **正确数**: 24
- **准确率**: 0.8889 (88.89%)

**正确示例**:

1. ✅ **Line 67**
   - 预测: `{"contacts": {"data": [{"phone": "18158519227", "name": "奚侃", "idcard": null}]}}`
   - 标签: `{"contacts": {"data": [{"phone": "18158519227", "name": "奚侃", "idcard": null}]}}`

2. ✅ **Line 71**
   - 预测: `{"contacts": {"data": [{"phone": "13636588126", "name": "张文胜", "idcard": null}]}}`
   - 标签: `{"contacts": {"data": [{"phone": "13636588126", "name": "张文胜", "idcard": null}]}}`

3. ✅ **Line 73**
   - 预测: `{"contacts": {"data": [{"phone": "13857745880", "name": "顾红娟", "idcard": null}]}}`
   - 标签: `{"contacts": {"data": [{"phone": "13857745880", "name": "顾红娟", "idcard": null}]}}`

**错误示例（全部）**:

1. ❌ **Line 105**
   - **指令**: 请从OCR文本中抽取旅行订单的联系人信息。

提取字段：
name (联系人姓名)
phone (联系电话)
idcard (身份证号码，如果没有则为null)

严格按照以下JSON格式输出：
{
...
   - **输入**: 杭州大翼，7月16日七里扬帆景区+葫芦峡漂流，2人
具体联系人：朱敏慧，13506140071
朱敏慧，32040419851108222X
袁也栋，32041120150416011X

instruction: 请从OCR文本中抽取旅行订单的联系人信息。

提取字段：
name (联系人姓名)
phone (联系电话)
idcard (身份证号码，如果没有则为null)

严格按照以下JSON...
   - **预测**: `{"contacts": {"data": [{"phone": "13506140071", "name": "朱敏慧", "idcard": "32040419851108222X"}, {"phone": null, "name": "袁也栋", "idcard": "32041120150416011X"}]}}`
   - **标签**: `{"contacts": {"data": [{"phone": "13506140071", "name": "朱敏慧", "idcard": null}]}}`

2. ❌ **Line 372**
   - **指令**: 请从OCR文本中抽取旅行订单的联系人信息。

提取字段：
name (联系人姓名)
phone (联系电话)
idcard (身份证号码，如果没有则为null)

严格按照以下JSON格式输出：
{
...
   - **输入**: 建德中旅，7.30日，39人，新叶古村，
导游翁1巧君13805701560

instruction: 请从OCR文本中抽取旅行订单的联系人信息。

提取字段：
name (联系人姓名)
phone (联系电话)
idcard (身份证号码，如果没有则为null)

严格按照以下JSON格式输出：
{
  "contacts": {
    "data": [
      {
        "...
   - **预测**: `{"contacts": {"data": [{"phone": "13805701560", "name": "翁1巧君", "idcard": null}]}}`
   - **标签**: `{"contacts": {"data": [{"phone": "13805701560", "name": "翁巧君", "idcard": null}]}}`

3. ❌ **Line 393**
   - **指令**: 请从OCR文本中抽取旅行订单的联系人信息。

提取字段：
name (联系人姓名)
phone (联系电话)
idcard (身份证号码，如果没有则为null)

严格按照以下JSON格式输出：
{
...
   - **输入**: 苏州快乐三义。7.28日，大慈岩
人数10人，朱爱风13862111871

instruction: 请从OCR文本中抽取旅行订单的联系人信息。

提取字段：
name (联系人姓名)
phone (联系电话)
idcard (身份证号码，如果没有则为null)

严格按照以下JSON格式输出：
{
  "contacts": {
    "data": [
      {
        "n...
   - **预测**: `{"contacts": {"data": [{"phone": "13862111871", "name": "朱爱风", "idcard": null}]}}`
   - **标签**: `{"contacts": {"data": [{"phone": "13862111871", "name": "朱爱凤", "idcard": null}]}}`

---

###### name

- **样本数**: 3
- **正确数**: 2
- **准确率**: 0.6667 (66.67%)

**正确示例**:

1. ✅ **Line 21**
   - 预测: `[{"phone": null, "name": "孙忠华", "idcard": "330411196912263618", "gender": "男"}, {"phone": null, "name": "徐方华", "idcard": "33041119690118421X", "gender": "男"}, {"phone": null, "name": "徐松良", "idcard": "330411198607093615", "gender": "男"}, {"phone": null, "name": "伍富强", "idcard": "330411196907193619", "gender": "男"}, {"phone": null, "name": "章青青", "idcard": "342531198603093727", "gender": "女"}, {"phone": null, "name": "陈章哲", "idcard": "330402201405305710", "gender": "男"}, {"phone": null, "name": "张丽群", "idcard": "330411198402063623", "gender": "女"}, {"phone": null, "name": "徐玲丽", "idcard": "330411198703143643", "gender": "女"}, {"phone": null, "name": "徐晨馨", "idcard": "330402201408185726", "gender": "女"}]`
   - 标签: `[{"phone": null, "name": "孙忠华", "idcard": "330411196912263618", "gender": "男"}, {"phone": null, "name": "徐方华", "idcard": "33041119690118421X", "gender": "男"}, {"phone": null, "name": "徐松良", "idcard": "330411198607093615", "gender": "男"}, {"phone": null, "name": "伍富强", "idcard": "330411196907193619", "gender": "男"}, {"phone": null, "name": "章青青", "idcard": "342531198603093727", "gender": "女"}, {"phone": null, "name": "陈章哲", "idcard": "330402201405305710", "gender": "男"}, {"phone": null, "name": "张丽群", "idcard": "330411198402063623", "gender": "女"}, {"phone": null, "name": "徐玲丽", "idcard": "330411198703143643", "gender": "女"}, {"phone": null, "name": "徐晨馨", "idcard": "330402201408185726", "gender": "女"}]`

2. ✅ **Line 35**
   - 预测: `[{"phone": null, "name": "朱敏慧", "idcard": "32040419851108222X", "gender": "女"}, {"phone": null, "name": "袁也栋", "idcard": "32041120150416011X", "gender": "男"}]`
   - 标签: `[{"phone": null, "name": "朱敏慧", "idcard": "32040419851108222X", "gender": "女"}, {"phone": null, "name": "袁也栋", "idcard": "32041120150416011X", "gender": "男"}]`

**错误示例（全部）**:

1. ❌ **Line 28**
   - **指令**: 请从OCR文本中抽取旅行订单中的游客身份证信息。

提取字段及说明：
name (姓名): 游客的真实姓名
idcard (身份证号): 18位身份证号码，支持末位为X的格式
gender (性别):...
   - **输入**: 冯珠奶
330721194302226748
沙利亚
330721196009037218
沙于冬
110108201712038633
胡兰英
330721196112297229
建德光大唐飞13968136568
明天上午好运岛 导游：唐飞
13968136568

instruction: 请从OCR文本中抽取旅行订单中的游客身份证信息。

提取字段及说明：
name (姓名): 游客的真...
   - **预测**: `[{"phone": null, "name": "冯珠奶", "idcard": "330721194302226748", "gender": "女"}, {"phone": null, "name": "沙利亚", "idcard": "330721196009037218", "gender": "女"}, {"phone": null, "name": "沙于冬", "idcard": "110108201712038633", "gender": "女"}, {"phone": null, "name": "胡兰英", "idcard": "330721196112297229", "gender": "女"}, {"phone": "13968136568", "name": "唐飞", "idcard": "330721196112297229", "gender": "女"}]`
   - **标签**: `[{"phone": null, "name": "冯珠奶", "idcard": "330721194302226748", "gender": "女"}, {"phone": null, "name": "沙于冬", "idcard": "330721196009037218", "gender": "男"}, {"phone": null, "name": "胡兰英", "idcard": "110108201712038633", "gender": "男"}, {"phone": null, "name": "胡兰英", "idcard": "330721196112297229", "gender": "女"}]`

---

###### node_name

- **样本数**: 12
- **正确数**: 7
- **准确率**: 0.5833 (58.33%)

**正确示例**:

1. ✅ **Line 7**
   - 预测: `{"node_name": "基地教室1", "conditions": [{"price_type": "半天（含投影）", "price_value": 500, "unit": "元/场", "conditions": []}, {"price_type": "半天（不含投影）", "price_value": 300, "unit": "元/场", "conditions": []}, {"price_type": "全天（含投影）", "price_value": 800, "unit": "元/场", "conditions": []}, {"price_type": "全天（不含投影）", "price_value": 600, "unit": "元/场", "conditions": []}]}`
   - 标签: `{"node_name": "基地教室1", "conditions": [{"price_type": "半天（含投影）", "price_value": 500, "unit": "元/场", "conditions": []}, {"price_type": "半天（不含投影）", "price_value": 300, "unit": "元/场", "conditions": []}, {"price_type": "全天（含投影）", "price_value": 800, "unit": "元/场", "conditions": []}, {"price_type": "全天（不含投影）", "price_value": 600, "unit": "元/场", "conditions": []}]}`

2. ✅ **Line 8**
   - 预测: `{"node_name": "基地教室2", "conditions": [{"price_type": "半天（含投影）", "price_value": 500, "unit": "元/场", "conditions": []}, {"price_type": "半天（不含投影）", "price_value": 300, "unit": "元/场", "conditions": []}, {"price_type": "全天（含投影）", "price_value": 800, "unit": "元/场", "conditions": []}, {"price_type": "全天（不含投影）", "price_value": 600, "unit": "元/场", "conditions": []}]}`
   - 标签: `{"node_name": "基地教室2", "conditions": [{"price_type": "半天（含投影）", "price_value": 500, "unit": "元/场", "conditions": []}, {"price_type": "半天（不含投影）", "price_value": 300, "unit": "元/场", "conditions": []}, {"price_type": "全天（含投影）", "price_value": 800, "unit": "元/场", "conditions": []}, {"price_type": "全天（不含投影）", "price_value": 600, "unit": "元/场", "conditions": []}]}`

3. ✅ **Line 9**
   - 预测: `{"node_name": "基地教室3", "conditions": [{"price_type": "半天（含投影）", "price_value": 500, "unit": "元/场", "conditions": []}, {"price_type": "半天（不含投影）", "price_value": 300, "unit": "元/场", "conditions": []}, {"price_type": "全天（含投影）", "price_value": 800, "unit": "元/场", "conditions": []}, {"price_type": "全天（不含投影）", "price_value": 600, "unit": "元/场", "conditions": []}]}`
   - 标签: `{"node_name": "基地教室3", "conditions": [{"price_type": "半天（含投影）", "price_value": 500, "unit": "元/场", "conditions": []}, {"price_type": "半天（不含投影）", "price_value": 300, "unit": "元/场", "conditions": []}, {"price_type": "全天（含投影）", "price_value": 800, "unit": "元/场", "conditions": []}, {"price_type": "全天（不含投影）", "price_value": 600, "unit": "元/场", "conditions": []}]}`

**错误示例（全部）**:

1. ❌ **Line 2**
   - **指令**: 分析以下价格政策文档，识别节点"灵栖洞西游魔毯上行"的所有价格条件，支持多层条件嵌套：

请返回JSON格式：
{
    "node_name": "灵栖洞西游魔毯上行",
    "conditi...
   - **输入**: 一、灵栖洞西游魔毯上行价格政策：

|  |  |  |
| --- | --- | --- |
| 价格政策 | 单价 | 有效期 |
| 门市价 | 100 | 2025年1月1日-12月31日 |
| 协议成团价 | 30 | 2025年1月1日-12月31日 |
| 协议不成团价 | 30 | 2025年1月1日-12月31日 |
| 非协议成团价 | 100 | 2025年1月1日-12...
   - **预测**: `{"node_name": "灵栖洞西游魔毯上行", "conditions": [{"price_type": "门市价", "price_value": 100, "unit": "人次", "conditions": []}, {"price_type": "协议成团价", "price_value": 30, "unit": "人次", "conditions": []}, {"price_type": "协议不成团价", "price_value": 30, "unit": "人次", "conditions": []}, {"price_type": "非协议成团价", "price_value": 100, "unit": "人次", "conditions": []}, {"price_type": "非协议不成团价", "price_value": 100, "unit": "人次", "conditions": []}, {"price_type": "老年团队价", "price_value": 30, "unit": "人次", "conditions": []}, {"price_type": "学生团队价", "price_value": 30, "unit": "人次", "conditions": []}, {"price_type": "中远程及新增市场价格", "price_value": 30, "unit": "人次", "conditions": []}, {"price_type": "景酒打包价", "price_value": 30, "unit": "人次", "conditions": []}, {"price_type": "旅投员工价", "price_value": 30, "unit": "人次", "conditions": []}]}`
   - **标签**: `{"node_name": "灵栖洞西游魔毯上行", "conditions": [{"price_type": "门市价", "price_value": 100, "unit": "人次", "conditions": []}, {"price_type": "协议成团价", "price_value": 30, "unit": "人次", "conditions": []}, {"price_type": "协议不成团价", "price_value": 30, "unit": "人次", "conditions": []}, {"price_type": "非协议成团价", "price_value": 100, "unit": "人次", "conditions": []}, {"price_type": "非协议不成团价", "price_value": 100, "unit": "人次", "conditions": []}, {"price_type": "老年团队价", "price_value": 30, "unit": "人次", "conditions": [{"condition_type": "年龄条件", "operator": ">=", "value": 55}, {"condition_type": "占团比例", "operator": ">=", "value": 70}]}, {"price_type": "学生团队价", "price_value": 30, "unit": "人次", "conditions": [{"condition_type": "学生比例", "operator": ">=", "value": 80}]}, {"price_type": "中远程及新增市场价格", "price_value": 30, "unit": "人次", "conditions": [{"condition_type": "占团比例", "operator": ">=", "value": 80}]}, {"price_type": "景酒打包价", "price_value": 30, "unit": "人次", "conditions": [{"condition_type": "住宿条件", "operator": "等于", "value": "住酒店"}]}, {"price_type": "旅投员工价", "price_value": 30, "unit": "人次", "conditions": [{"condition_type": "身份条件", "operator": "等于", "value": "旅投员工"}]}]}`

2. ❌ **Line 12**
   - **指令**: 分析以下价格政策文档，识别节点"大慈岩门票"的所有价格条件，支持多层条件嵌套：

请返回JSON格式：
{
    "node_name": "大慈岩门票",
    "conditions": [
...
   - **输入**: |  |  |  |  |
| --- | --- | --- | --- |
| 资源 | 适用对象 | 有效期 | 价格 |
| 大慈岩门票-大慈岩豆腐包制作体验 | 所有 | 2025年1月1日-2025年12月31日 | 大慈岩门票：40 |
| 大慈岩豆腐包制作体验：20 |

instruction: 分析以下价格政策文档，识别节点"大慈岩门票"的所有价格条件，支持多层条件嵌套：

请...
   - **预测**: `{"node_name": "大慈岩门票", "conditions": [{"price_type": "单价", "price_value": 40, "unit": "人次", "conditions": [{"condition_type": "时间", "operator": "在", "value": "2025年1月1日-2025年12月31日", "sub_conditions": [{"condition_type": "适用对象", "operator": "是", "value": "所有"}]}]}]}`
   - **标签**: `{"node_name": "大慈岩门票", "conditions": [{"price_type": "单价", "price_value": 40, "unit": "人次", "conditions": []}]}`

3. ❌ **Line 13**
   - **指令**: 分析以下价格政策文档，识别节点"大慈岩豆腐包制作体验"的所有价格条件，支持多层条件嵌套：

请返回JSON格式：
{
    "node_name": "大慈岩豆腐包制作体验",
    "condi...
   - **输入**: |  |  |  |  |
| --- | --- | --- | --- |
| 资源 | 适用对象 | 有效期 | 价格 |
| 大慈岩门票-大慈岩豆腐包制作体验 | 所有 | 2025年1月1日-2025年12月31日 | 大慈岩门票：40 |
| 大慈岩豆腐包制作体验：20 |

instruction: 分析以下价格政策文档，识别节点"大慈岩豆腐包制作体验"的所有价格条件，支持多层条件嵌...
   - **预测**: `{"node_name": "大慈岩豆腐包制作体验", "conditions": [{"price_type": "单价", "price_value": 20, "unit": "人次", "conditions": [{"condition_type": "时间", "operator": "在", "value": "2025年1月1日-2025年12月31日", "sub_conditions": [{"condition_type": "适用对象", "operator": "所有", "value": ""}]}]}]}`
   - **标签**: `{"node_name": "大慈岩豆腐包制作体验", "conditions": [{"price_type": "单价", "price_value": 20, "unit": "人次", "conditions": []}, {"price_type": "大慈岩门票", "price_value": 40, "unit": "人次", "conditions": []}]}`

4. ❌ **Line 19**
   - **指令**: 分析以下价格政策文档，识别节点"大慈岩门票"的所有价格条件，支持多层条件嵌套：

请返回JSON格式：
{
    "node_name": "大慈岩门票",
    "conditions": [
...
   - **输入**: |  |  |  |  |
| --- | --- | --- | --- |
| 资源 | 适用对象 | 有效期 | 价格 |
| 大慈岩门票-江清月近人实景演艺门票 | 所有 | 2025年1月1日-2025年12月31日 | 大慈岩门票：40 |
| 江清月近人实景演艺门票：90 |

instruction: 分析以下价格政策文档，识别节点"大慈岩门票"的所有价格条件，支持多层条件嵌套：
...
   - **预测**: `{"node_name": "大慈岩门票", "conditions": [{"price_type": "单价", "price_value": 40, "unit": "人次", "conditions": [{"condition_type": "时间", "operator": "在", "value": "2025年1月1日-2025年12月31日", "sub_conditions": [{"condition_type": "适用对象", "operator": "是", "value": "所有"}]}]}]}`
   - **标签**: `{"node_name": "大慈岩门票", "conditions": [{"price_type": "单价", "price_value": 40, "unit": "人次", "conditions": []}]}`

5. ❌ **Line 20**
   - **指令**: 分析以下价格政策文档，识别节点"江清月近人实景演艺门票"的所有价格条件，支持多层条件嵌套：

请返回JSON格式：
{
    "node_name": "江清月近人实景演艺门票",
    "con...
   - **输入**: |  |  |  |  |
| --- | --- | --- | --- |
| 资源 | 适用对象 | 有效期 | 价格 |
| 大慈岩门票-江清月近人实景演艺门票 | 所有 | 2025年1月1日-2025年12月31日 | 大慈岩门票：40 |
| 江清月近人实景演艺门票：90 |

instruction: 分析以下价格政策文档，识别节点"江清月近人实景演艺门票"的所有价格条件，支持多层...
   - **预测**: `{"node_name": "江清月近人实景演艺门票", "conditions": [{"price_type": "单价", "price_value": 90, "unit": "人次", "conditions": [{"condition_type": "时间", "operator": "在", "value": "2025年1月1日-2025年12月31日", "sub_conditions": [{"condition_type": "适用对象", "operator": "是", "value": "所有"}]}]}]}`
   - **标签**: `{"node_name": "江清月近人实景演艺门票", "conditions": [{"price_type": "单价", "price_value": 90, "unit": "人次", "conditions": []}]}`

---

###### unknown

- **样本数**: 7
- **正确数**: 4
- **准确率**: 0.5714 (57.14%)

**正确示例**:

1. ✅ **Line 1**
   - 预测: `["灵栖洞西游魔毯上行"]`
   - 标签: `["灵栖洞西游魔毯上行"]`

2. ✅ **Line 6**
   - 预测: `["基地教室1", "基地教室2", "基地教室3", "基地教室5"]`
   - 标签: `["基地教室1", "基地教室2", "基地教室3", "基地教室5"]`

3. ✅ **Line 11**
   - 预测: `["大慈岩门票", "大慈岩豆腐包制作体验"]`
   - 标签: `["大慈岩门票", "大慈岩豆腐包制作体验"]`

**错误示例（全部）**:

1. ❌ **Line 3**
   - **指令**: 分析以下景区门票/体验活动价格政策文档，识别所有可以分裂的产品节点：

请识别出所有独立的产品/服务项目，返回JSON数组格式：
["产品1", "产品2", "产品3", ...]

注意：
1. ...
   - **输入**: 一、七里扬帆小火车单程价格政策：

|  |  |  |  |
| --- | --- | --- | --- |
| 价格政策 | 小火车单程 | 小火车往返 | 有效期 |
| 门市价 | 15 | 30 | 2025年1月1日-12月31日 |
| 协议成团价 | 10 | 20 | 2025年1月1日-12月31日 |
| 协议不成团价 | 10 | 20 | 2025年1月1日-12月3...
   - **预测**: `["七里扬帆小火车单程", "七里扬帆小火车往返"]`
   - **标签**: `["小火车单程", "小火车往返"]`

2. ❌ **Line 14**
   - **指令**: 分析以下联票套餐价格政策文档，识别所有可以分裂的产品节点：

请识别出所有独立的产品/服务项目，返回JSON数组格式：
["产品1", "产品2", "产品3", ...]

注意：
1. 每个产品应...
   - **输入**: |  |  |  |  |
| --- | --- | --- | --- |
| 资源 | 适用对象 | 有效期 | 价格 |
| 大慈岩门票-大慈岩豆腐包制作体验-新安江游船 | 所有 | 2025年1月1日-2025年12月31日 | 大慈岩门票：25 |
| 大慈岩豆腐包制作体验：20 |
| 新安江游船-梦幻新安江：40 |

instruction: 分析以下联票套餐价格政策文档，识别...
   - **预测**: `["大慈岩门票", "大慈岩豆腐包制作体验", "新安江游船", "梦幻新安江"]`
   - **标签**: `["大慈岩门票", "大慈岩豆腐包制作体验", "新安江游船"]`

3. ❌ **Line 413**
   - **指令**: ## 任务描述 ##
根据以下价格政策信息和OCR文本，提取并计算资源价格。

## 产品信息 ##
产品名称: 大慈岩豆腐包制作体验_产品

## 价格政策条件 ##
产品名称: 大慈岩豆腐包制作体...
   - **输入**: 嘉兴南湖

6月25日 大慈岩豆腐包

人数：9人

导游信息：王导13587654321

instruction: ## 任务描述 ##
根据以下价格政策信息和OCR文本，提取并计算资源价格。

## 产品信息 ##
产品名称: 大慈岩豆腐包制作体验_产品

## 价格政策条件 ##
产品名称: 大慈岩豆腐包制作体验_产品
产品类型: 大慈岩豆腐包制作体验
  None: 门市价 (价格: 3...
   - **预测**: `{"resources": [{"resource_name": "大慈岩豆腐包制作体验", "start_time": "20250625", "end_time": "20250625", "unit_count": "9", "conditions": {"name": "协议不成团价", "children": null}, "notes": "无", "unit": "人次", "usage_info": "无", "unit_price": 20, "total_price": 180}], "product_total_amount": 180, "product_total_unit_amount": 9}`
   - **标签**: `{"resources": [{"resource_name": "大慈岩豆腐包制作体验_产品", "start_time": "20250625", "end_time": "20250625", "unit_count": "9", "conditions": {"name": "协议不成团价", "children": null}, "notes": "协议客户，人数9人不足10人，适用不成团价", "unit": "人次", "usage_info": "9人", "unit_price": 20, "total_price": 180}], "product_total_amount": 180, "product_total_unit_amount": 9}`

---

###### resources

- **样本数**: 4
- **正确数**: 0
- **准确率**: 0.0000 (0.00%)

**错误示例（全部）**:

1. ❌ **Line 409**
   - **指令**: ## 任务描述 ##
根据以下价格政策信息和OCR文本，提取并计算资源价格。

## 产品信息 ##
产品名称: 大慈岩门票_产品

## 价格政策条件 ##
产品名称: 大慈岩门票_产品
产品类型:...
   - **输入**: 杭州绿城

9月15日 大慈岩

人数：10人

导游信息：刘导13800138000

instruction: ## 任务描述 ##
根据以下价格政策信息和OCR文本，提取并计算资源价格。

## 产品信息 ##
产品名称: 大慈岩门票_产品

## 价格政策条件 ##
产品名称: 大慈岩门票_产品
产品类型: 大慈岩门票
  票型: 门市价 (价格: 80 人次)
  票型: 协议成团价 (...
   - **预测**: `{"resources": [{"resource_name": "大慈岩门票", "start_time": "20250915", "end_time": "20250915", "unit_count": "10", "conditions": {"name": "协议成团价", "children": null}, "notes": "无", "unit": "人次", "usage_info": "无", "unit_price": 50, "total_price": 500}], "product_total_amount": 500, "product_total_unit_amount": 10}`
   - **标签**: `{"resources": [{"resource_name": "大慈岩门票", "start_time": "20250915", "end_time": "20250915", "unit_count": "10", "conditions": {"name": "协议成团价", "children": null}, "notes": "协议客户，人数正好等于10人，满足成团条件", "unit": "人次", "usage_info": "10人", "unit_price": 50, "total_price": 500}], "product_total_amount": 500, "product_total_unit_amount": 10}`

2. ❌ **Line 410**
   - **指令**: ## 任务描述 ##
根据以下价格政策信息和OCR文本，提取并计算资源价格。

## 产品信息 ##
产品名称: 灵栖洞门票_产品

## 价格政策条件 ##
产品名称: 灵栖洞门票_产品
产品类型:...
   - **输入**: 盐城海滨

2月20日 灵栖洞

人数：16人

导游信息：孙导13765432109

instruction: ## 任务描述 ##
根据以下价格政策信息和OCR文本，提取并计算资源价格。

## 产品信息 ##
产品名称: 灵栖洞门票_产品

## 价格政策条件 ##
产品名称: 灵栖洞门票_产品
产品类型: 灵栖洞门票
  票型: 门市价 (价格: 80 人次)
  票型: 协议成团价 (...
   - **预测**: `{"resources": [{"resource_name": "灵栖洞门票", "start_time": "20250220", "end_time": "20250220", "unit_count": "16", "conditions": {"name": "非协议不成团价", "children": null}, "notes": "无", "unit": "人次", "usage_info": "无", "unit_price": 72, "total_price": 1152}], "product_total_amount": 1152, "product_total_unit_amount": 16}`
   - **标签**: `{"resources": [{"resource_name": "灵栖洞门票", "start_time": "20250220", "end_time": "20250220", "unit_count": "16", "conditions": {"name": "协议成团价", "children": {"name": "住酒店免门票", "children": null}}, "notes": "盐城属于江浙沪以外的市场，2025年2月20日在1月1日-3月31日期间，享受住酒店免门票政策", "unit": "人次", "usage_info": "住酒店免门票", "unit_price": 50, "total_price": 800}], "product_total_amount": 800, "product_total_unit_amount": 16}`

3. ❌ **Line 411**
   - **指令**: ## 任务描述 ##
根据以下价格政策信息和OCR文本，提取并计算资源价格。

## 产品信息 ##
产品名称: 大慈岩豆腐包制作体验_大慈岩门票_联票

## 价格政策条件 ##
产品名称: 大慈岩...
   - **输入**: 宁波天一

11月8日 大慈岩、大慈岩豆腐包

人数：18人

导游信息：陈导13987654321

instruction: ## 任务描述 ##
根据以下价格政策信息和OCR文本，提取并计算资源价格。

## 产品信息 ##
产品名称: 大慈岩豆腐包制作体验_大慈岩门票_联票

## 价格政策条件 ##
产品名称: 大慈岩豆腐包制作体验_大慈岩门票_联票
产品类型: 大慈岩门票 (价格: 4...
   - **预测**: `{"resources": [{"resource_name": "大慈岩豆腐包制作体验_大慈岩门票_联票", "start_time": "20251108", "end_time": "20251108", "unit_count": "18", "conditions": {"name": "大慈岩门票", "children": {"name": "大慈岩豆腐包制作体验", "children": null}}, "notes": "", "unit": "人次", "usage_info": "大慈岩门票", "unit_price": 40, "total_price": 720}], "product_total_amount": 720, "product_total_unit_amount": 18}`
   - **标签**: `{"resources": [{"resource_name": "大慈岩门票", "start_time": "20251108", "end_time": "20251108", "unit_count": "18", "conditions": {"name": "大慈岩门票", "children": null}, "notes": "联票产品", "unit": "人次", "usage_info": "18人", "unit_price": 40, "total_price": 720}, {"resource_name": "大慈岩豆腐包制作体验", "start_time": "20251108", "end_time": "20251108", "unit_count": "18", "conditions": {"name": "大慈岩豆腐包制作体验", "children": null}, "notes": "联票产品", "unit": "人次", "usage_info": "18人", "unit_price": 40, "total_price": 720}], "product_total_amount": 1440, "product_total_unit_amount": 18}`

4. ❌ **Line 412**
   - **指令**: ## 任务描述 ##
根据以下价格政策信息和OCR文本，提取并计算资源价格。

## 产品信息 ##
产品名称: 新安江游船_产品

## 价格政策条件 ##
产品名称: 新安江游船_产品
产品类型:...
   - **输入**: 温州雁荡

4月12日 新安江龙舟漂游

人数：25人

导游信息：林导13612345678

instruction: ## 任务描述 ##
根据以下价格政策信息和OCR文本，提取并计算资源价格。

## 产品信息 ##
产品名称: 新安江游船_产品

## 价格政策条件 ##
产品名称: 新安江游船_产品
产品类型: 江清月近人实景演艺船票
  票型: 门市价 (价格: 100 人次)
  ...
   - **预测**: `{"resources": [{"resource_name": "新安江龙舟漂游", "start_time": "20250412", "end_time": "20250412", "unit_count": "25", "conditions": {"name": "非协议成团价", "children": {"name": "非协议成团价", "children": null}}, "notes": "无", "unit": "人次", "usage_info": "无", "unit_price": 60, "total_price": 1500}], "product_total_amount": 1500, "product_total_unit_amount": 25}`
   - **标签**: `{"resources": [{"resource_name": "新安江龙舟漂游", "start_time": "20250412", "end_time": "20250412", "unit_count": "25", "conditions": {"name": "协议成团价", "children": null}, "notes": "协议客户，人数25人满足成团条件", "unit": "人次", "usage_info": "25人", "unit_price": 60, "total_price": 1500}], "product_total_amount": 1500, "product_total_unit_amount": 25}`

---
