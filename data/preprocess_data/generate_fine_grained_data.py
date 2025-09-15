#!/usr/bin/env python3
"""
将粗粒度的订单抽取数据转换为细粒度的训练数据
每个input只对应一个特定字段的抽取任务
支持JSON和JSONL两种输入格式
"""

import json
import random
import os
from typing import Dict, List, Any

# 业务范围数据定义
RESOURCE_SUBRESOURCES = {
    "七里扬帆": [
        "七里扬帆草莓采摘入园票", "七里扬帆小火车", "七里扬帆葫芦山庄餐饮", "七里扬帆门票", 
        "七里扬帆游船", "三江口游线（游船）", "七里扬帆停车场", "七里扬帆葫芦峡漂流", "七里扬帆包船"
    ],
    "三都渔村": [
        "三都渔村门票", "三都渔村玻璃水滑道", "三都渔村婚礼表演"
    ],
    "严州古城": [
        "严州古城门票", "严州古城展馆联票+电瓶车", "严州古城摇橹船"
    ],
    "交运公司": [
        "交运包车", "交运租车"
    ],
    "千岛湖好运岛": [
        "千岛湖好运岛草莓采摘（2斤）", "千岛湖好运岛门票", "千岛湖好运岛游船", "千岛湖好运岛停车场"
    ],
    "千鹤妇女精神教育基地": [
        "千鹤门票", "千鹤景区讲解项目", "千鹤会务"
    ],
    "大慈岩": [
        "大慈岩豆腐包制作体验", "大慈岩索道", "大慈岩玻璃栈道", "大慈岩丛林速滑（旱滑道）下行",
        "大慈岩餐饮", "大慈岩中餐", "大慈岩门票", "大慈岩停车场"
    ],
    "宿江公司": [
        "江清月近人实景演艺门票", "江清月近人白天场"
    ],
    "导服中心": [
        "扬帆旅行社全陪导游", "扬帆旅行社地接导游", "大慈岩景区导服", "灵栖洞景区导服",
        "新叶古村景区导服", "千岛湖好运岛景区导服", "七里扬帆景区导服", "严州古城景区导服",
        "新安江景区导服", "千鹤景区导服"
    ],
    "新叶古村": [
        "新叶古村草莓采摘入园票", "新叶古村门票", "新叶古村餐饮", "新叶古村停车场"
    ],
    "新安江": [
        "新安江草莓采摘入园票", "新安江船餐", "新安江中餐", "新安江游船"
    ],
    "景澜酒店": [
        "住宿", "会务"
    ],
    "汉庭酒店": [
        "住宿", "餐饮"
    ],
    "灵栖洞": [
        "灵栖洞豆腐包制作体验", "考拉森林丛林探险", "灵栖洞西游魔毯", "灵栖洞极速滑道",
        "灵栖洞餐饮", "灵栖洞中餐", "灵栖洞门票", "灵栖洞手划船", "灵栖洞停车场", "灵栖洞喊泉"
    ],
    "玉泉寺": [
        "玉泉寺门票"
    ],
    "雷迪森酒店": [
        "住宿", "会务"
    ]
}

# 所有combo/items类型的枚举值定义
COMBO_ITEMS_ENUM_VALUES = {
    "business_items": [
        "三都渔村婚礼表演", "喊泉", "小火车单程", "小火车往返", "考拉森林丛林探险亲子线", 
        "考拉森林丛林探险成人线", "草莓采摘入园票", "草莓采摘（2斤）", "豆腐包制作体验"
    ],
    "guide_items": [
        "七里扬帆景区导服", "严州古城景区导服", "千岛湖好运岛景区导服", "千鹤景区导服", 
        "大慈岩景区导服", "小小讲解员体验", "扬帆旅行社全陪导游", "扬帆旅行社地接导游", 
        "新叶古村景区导服", "新安江景区导服", "民兵体验", "灵栖洞景区导服", "讲解费"
    ],
    "parking_items": [
        "中型", "中巴车", "大型", "大客车", "大巴车", "大车", "小型车", "小客车", "小车", "摩托车"
    ],
    "recreational_combo": [
        "中远程及新增市场价格", "协议不成团价", "协议成团价", "团队价", "学生团队价", "挂牌价", 
        "旅投员工价", "景酒打包价", "老年团队价", "门市价", "非协议不成团价", "非协议成团价"
    ],
    "recreational_items": [
        "丛林速滑（旱滑道）下行", "极速滑道", "玻璃栈道", "玻璃水滑道", "索道上下行", 
        "索道上行", "索道下行", "西游魔毯"
    ],
    "ship_combo": [
        "中远程及新增市场价格", "团队优惠价", "团队优惠价_小扬帆_1500", "团队优惠价_小扬帆_600", 
        "团队优惠价_小扬帆_800", "学生团队价", "广德", "建德市民", "散客挂牌价", "旅投员工价", 
        "景酒打包价", "正常价", "特殊价", "研学团", "老年团队价"
    ],
    "ship_items": [
        "七里扬帆游船", "三江口游线", "千岛湖好运岛游船", "小扬帆", "快艇1号", "快艇2号", 
        "扬帆16号", "扬帆2号", "扬帆3号", "扬帆之星", "摇橹船", "新安江竹筏漂流", "新安江龙舟漂游", 
        "梦幻新安江", "江南秘境1号", "江南秘境2号", "江清月近人实景演艺船票", "浙旅江南秘境", 
        "船票", "葫芦峡漂流", "诗韵新安江（新安江-严州古城航线含自助茶歇、简餐）"
    ],
    "ticket_combo": [
        "", "30年教师证", "70周岁", "中远程及新增市场价格", "儿童", "其他", "军官证", "协议不成团价", 
        "协议成团价", "在校大学生", "学生团队价", "导游证", "广德", "建德就业", "建德市民", 
        "接待免票", "新闻记者证", "旅投员工价", "景酒打包价", "杭州市民卡", "杭州文旅卡/惠民卡", 
        "残疾证", "消防员证", "献血荣誉证", "病故军属家人", "研学团", "老年团队价", 
        "萧山、临平、西湖管委会免票", "退役证", "钱江分免票", "门市价", "青少年优惠", 
        "非协议不成团价", "非协议成团价", "非建德户籍免票", "高层次人才"
    ],
    "ticket_items": [
        "展馆", "江清月近人", "江清月近人白天场", "门票"
    ],
    "meal_standard_table": [
        300, 400, 500, 600, 700, 800, 1000
    ],
    "meal_standard": [
        0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 18
    ],
    "meal_types": [
        "中餐", "前程似锦宴", "金玉满堂宴", "阖家团圆宴"
    ],
    "guide_types": [
        "团队价", "散客"
    ],
    "meeting_place": [
        "基地教室1", "基地教室2", "基地教室3", "基地教室5"
    ],
    "duration_hours": [
        "全天", "半天"
    ],
    "device_name": [
        "投影"
    ],
    "vehicle_type": [
        "14座", "18座", "23-37座", "49-56座"
    ],
    "travel_route": [
        "", "三都", "下涯", "乾潭", "大同", "大慈岩", "大洋", "好运岛", "安仁", "寿昌", "新叶", "李家", "杨村桥", "梅城", "灵栖洞", "航头", "莲花", "长林", "马目"
    ],
    "rent_vehicle_type": [
        "15座全顺", "17座考斯特", "19座金龙中巴", "23座", "28座", "34座", "39座", "50座", "54座", "56座", "5座小汽车", "7座商务车"
    ],
    "rent_travel_route": [
        "其他", "建德市"
    ],
    "room_type": [
        "总套套房", "行政大床房", "行政套房", "豌豆星球太空梦亲子套房", "豌豆星球太空梦亲子房", "豪华双床房", "豪华大床房", "露台亲子房", "露台双床房", "露台大床房", "高级双床房", "高级大床房"
    ],
    "room_agreement": [
        "协议价", "门市价", "团队价", "散客价"
    ],
    "meeting_room_name": [
        "彩虹厅", "新安厅", "新安厅半厅", "沧滩厅", "白沙厅"
    ],
    "meeting_agreement": [
        "协议价", "挂牌价"
    ],
    "coffee_break": [
        "包含", "不包含", "可选"
    ],
    "devices": [
        "投影仪", "音响", "麦克风", "白板", "LED屏", "无"
    ],
    "devices_agreement": [
        "协议价", "门市价", "团队价", "散客价"
    ]
}

# 资源详细信息结构定义 - 按二级资源分类的所有字段及其枚举值
RESOURCE_DETAIL_STRUCTURE = {
    # 七里扬帆停车场
    "七里扬帆停车场": {
        "parking_items": ["大车", "小车"]
    },
    
    # 七里扬帆包船
    "七里扬帆包船": {
        "ship_combo": ["团队优惠价", "团队优惠价_小扬帆_1500", "团队优惠价_小扬帆_600", "团队优惠价_小扬帆_800", "散客挂牌价"],
        "ship_items": ["小扬帆", "快艇1号", "快艇2号", "扬帆16号", "扬帆2号", "扬帆3号", "扬帆之星", "江南秘境1号", "江南秘境2号", "浙旅江南秘境"]
    },
    
    # 七里扬帆小火车
    "七里扬帆小火车": {
        "business_items": ["小火车单程", "小火车往返"],
        "recreational_combo": ["中远程及新增市场价格", "学生团队价", "旅投员工价", "景酒打包价", "老年团队价"]
    },
    
    # 七里扬帆景区导服
    "七里扬帆景区导服": {
        "guide_items": ["七里扬帆景区导服"]
    },
    
    # 七里扬帆游船
    "七里扬帆游船": {
        "ship_combo": ["中远程及新增市场价格", "学生团队价", "旅投员工价", "景酒打包价", "研学团", "老年团队价"],
        "ship_items": ["七里扬帆游船"]
    },
    
    # 七里扬帆草莓采摘入园票
    "七里扬帆草莓采摘入园票": {
        "business_items": ["草莓采摘入园票"]
    },
    
    # 七里扬帆葫芦山庄餐饮
    "七里扬帆葫芦山庄餐饮": {
        "meal_standard": [0, 20, 25, 30, 35, 40, 45, 50, 55, 60],
        "meal_standard_table": [300, 500, 600, 800, 1000]
    },
    
    # 七里扬帆葫芦峡漂流
    "七里扬帆葫芦峡漂流": {
        "ship_combo": ["建德市民"],
        "ship_items": ["葫芦峡漂流"]
    },
    
    # 七里扬帆门票
    "七里扬帆门票": {
        "ticket_combo": ["30年教师证", "70周岁", "中远程及新增市场价格", "儿童", "其他", "军官证", "在校大学生", "学生团队价", "导游证", "建德就业", "建德市民", "接待免票", "新闻记者证", "旅投员工价", "景酒打包价", "杭州市民卡", "杭州文旅卡/惠民卡", "残疾证", "消防员证", "献血荣誉证", "病故军属家人", "研学团", "老年团队价", "萧山、临平、西湖管委会免票", "退役证", "钱江分免票", "青少年优惠", "非建德户籍免票", "高层次人才"],
        "ticket_items": ["门票"]
    },
    
    # 三江口游线（游船）
    "三江口游线（游船）": {
        "ship_combo": ["中远程及新增市场价格", "学生团队价", "旅投员工价", "景酒打包价", "研学团", "老年团队价"],
        "ship_items": ["三江口游线"]
    },
    
    # 三都渔村婚礼表演
    "三都渔村婚礼表演": {
        "business_items": ["三都渔村婚礼表演"],
        "recreational_combo": ["团队价", "挂牌价"]
    },
    
    # 三都渔村玻璃水滑道
    "三都渔村玻璃水滑道": {
        "recreational_combo": ["中远程及新增市场价格", "学生团队价", "旅投员工价", "景酒打包价", "老年团队价"],
        "recreational_items": ["玻璃水滑道"]
    },
    
    # 三都渔村门票
    "三都渔村门票": {
        "ticket_combo": [""],
        "ticket_items": ["门票"]
    },
    
    # 严州古城展馆联票+电瓶车
    "严州古城展馆联票+电瓶车": {
        "ticket_combo": ["研学团"],
        "ticket_items": ["展馆"]
    },
    
    # 严州古城摇橹船
    "严州古城摇橹船": {
        "ship_combo": ["中远程及新增市场价格", "学生团队价", "旅投员工价", "景酒打包价", "研学团", "老年团队价"],
        "ship_items": ["摇橹船"]
    },
    
    # 严州古城景区导服
    "严州古城景区导服": {
        "guide_items": ["严州古城景区导服"]
    },
    
    # 严州古城门票
    "严州古城门票": {
        "ticket_combo": [""],
        "ticket_items": ["门票"]
    },
    
    # 交运包车
    "交运包车": {
        "travel_route": ["", "三都", "下涯", "乾潭", "大同", "大慈岩", "大洋", "好运岛", "安仁", "寿昌", "新叶", "李家", "杨村桥", "梅城", "灵栖洞", "航头", "莲花", "长林", "马目"],
        "vehicle_type": ["14座", "18座", "23-37座", "49-56座"]
    },
    
    # 交运租车
    "交运租车": {
        "rent_travel_route": ["其他", "建德市"],
        "rent_vehicle_type": ["15座全顺", "17座考斯特", "19座金龙中巴", "23座", "28座", "34座", "39座", "50座", "54座", "56座", "5座小汽车", "7座商务车"]
    },
    
    # 会务
    "会务": {
        "meeting_agreement": ["协议价", "挂牌价"],
        "meeting_room_name": ["彩虹厅", "新安厅", "新安厅半厅", "沧滩厅", "白沙厅"]
    },
    
    # 住宿
    "住宿": {
        "room_type": ["总套套房", "行政大床房", "行政套房", "豌豆星球太空梦亲子套房", "豌豆星球太空梦亲子房", "豪华双床房", "豪华大床房", "露台亲子房", "露台双床房", "露台大床房", "高级双床房", "高级大床房"],
        "room_agreement": ["协议价", "门市价", "团队价", "散客价"]
    },
    
    # 千岛湖好运岛停车场
    "千岛湖好运岛停车场": {
        "parking_items": ["中巴车", "大巴车", "小型车"]
    },
    
    # 千岛湖好运岛景区导服
    "千岛湖好运岛景区导服": {
        "guide_items": ["千岛湖好运岛景区导服"]
    },
    
    # 千岛湖好运岛游船
    "千岛湖好运岛游船": {
        "ship_combo": ["中远程及新增市场价格", "学生团队价", "广德", "旅投员工价", "景酒打包价", "老年团队价"],
        "ship_items": ["千岛湖好运岛游船"]
    },
    
    # 千岛湖好运岛草莓采摘（2斤）
    "千岛湖好运岛草莓采摘（2斤）": {
        "business_items": ["草莓采摘（2斤）"]
    },
    
    # 千岛湖好运岛门票
    "千岛湖好运岛门票": {
        "ticket_combo": ["30年教师证", "70周岁", "中远程及新增市场价格", "儿童", "其他", "军官证", "在校大学生", "学生团队价", "导游证", "广德", "建德就业", "建德市民", "接待免票", "新闻记者证", "旅投员工价", "景酒打包价", "杭州市民卡", "杭州文旅卡/惠民卡", "残疾证", "消防员证", "献血荣誉证", "病故军属家人", "老年团队价", "萧山、临平、西湖管委会免票", "退役证", "钱江分免票", "青少年优惠", "非建德户籍免票", "高层次人才"],
        "ticket_items": ["门票"]
    },
    
    # 千鹤会务
    "千鹤会务": {
        "device_name": ["投影"],
        "duration_hours": ["全天", "半天"],
        "meeting_place": ["基地教室1", "基地教室2", "基地教室3", "基地教室5"]
    },
    
    # 千鹤景区导服
    "千鹤景区导服": {
        "guide_items": ["千鹤景区导服"]
    },
    
    # 千鹤景区讲解项目
    "千鹤景区讲解项目": {
        "guide_items": ["小小讲解员体验", "民兵体验", "讲解费"],
        "guide_types": ["团队价", "散客"]
    },
    
    # 千鹤门票
    "千鹤门票": {
        "ticket_items": ["门票"]
    },
    
    # 大慈岩丛林速滑（旱滑道）下行
    "大慈岩丛林速滑（旱滑道）下行": {
        "recreational_combo": ["中远程及新增市场价格", "学生团队价", "旅投员工价", "景酒打包价", "老年团队价"],
        "recreational_items": ["丛林速滑（旱滑道）下行"]
    },
    
    # 大慈岩中餐
    "大慈岩中餐": {
        "meal_types": ["中餐"]
    },
    
    # 大慈岩停车场
    "大慈岩停车场": {
        "parking_items": ["大客车", "小客车", "摩托车"]
    },
    
    # 大慈岩景区导服
    "大慈岩景区导服": {
        "guide_items": ["大慈岩景区导服"]
    },
    
    # 大慈岩玻璃栈道
    "大慈岩玻璃栈道": {
        "recreational_combo": ["中远程及新增市场价格", "学生团队价", "旅投员工价", "景酒打包价", "老年团队价"],
        "recreational_items": ["玻璃栈道"]
    },
    
    # 大慈岩索道
    "大慈岩索道": {
        "recreational_combo": ["中远程及新增市场价格", "协议不成团价", "协议成团价", "学生团队价", "旅投员工价", "景酒打包价", "老年团队价", "门市价", "非协议不成团价", "非协议成团价"],
        "recreational_items": ["索道上下行", "索道上行", "索道下行"]
    },
    
    # 大慈岩豆腐包制作体验
    "大慈岩豆腐包制作体验": {
        "business_items": ["豆腐包制作体验"],
        "recreational_combo": ["中远程及新增市场价格", "学生团队价", "旅投员工价", "景酒打包价", "老年团队价"]
    },
    
    # 大慈岩门票
    "大慈岩门票": {
        "ticket_combo": ["30年教师证", "70周岁", "中远程及新增市场价格", "儿童", "其他", "军官证", "协议不成团价", "协议成团价", "在校大学生", "学生团队价", "导游证", "广德", "建德就业", "建德市民", "接待免票", "新闻记者证", "旅投员工价", "景酒打包价", "杭州市民卡", "杭州文旅卡/惠民卡", "残疾证", "消防员证", "献血荣誉证", "病故军属家人", "老年团队价", "萧山、临平、西湖管委会免票", "退役证", "钱江分免票", "门市价", "青少年优惠", "非协议不成团价", "非协议成团价", "非建德户籍免票", "高层次人才"],
        "ticket_items": ["门票"]
    },
    
    # 大慈岩餐饮
    "大慈岩餐饮": {
        "meal_standard": [0, 20, 25, 30, 35, 40, 45, 50, 55, 60],
        "meal_standard_table": [300, 500, 600, 800, 1000]
    },
    
    # 扬帆旅行社全陪导游
    "扬帆旅行社全陪导游": {
        "guide_items": ["扬帆旅行社全陪导游"]
    },
    
    # 扬帆旅行社地接导游
    "扬帆旅行社地接导游": {
        "guide_items": ["扬帆旅行社地接导游"]
    },
    
    # 新叶古村停车场
    "新叶古村停车场": {
        "parking_items": ["大车", "小车"]
    },
    
    # 新叶古村景区导服
    "新叶古村景区导服": {
        "guide_items": ["新叶古村景区导服"]
    },
    
    # 新叶古村草莓采摘入园票
    "新叶古村草莓采摘入园票": {
        "business_items": ["草莓采摘入园票"]
    },
    
    # 新叶古村门票
    "新叶古村门票": {
        "ticket_combo": ["30年教师证", "70周岁", "中远程及新增市场价格", "儿童", "其他", "军官证", "在校大学生", "学生团队价", "导游证", "广德", "建德就业", "建德市民", "接待免票", "新闻记者证", "旅投员工价", "景酒打包价", "杭州市民卡", "杭州文旅卡/惠民卡", "残疾证", "消防员证", "献血荣誉证", "病故军属家人", "研学团", "老年团队价", "萧山、临平、西湖管委会免票", "退役证", "钱江分免票", "青少年优惠", "非建德户籍免票", "高层次人才"],
        "ticket_items": ["门票"]
    },
    
    # 新叶古村餐饮
    "新叶古村餐饮": {
        "meal_standard": [0, 20, 25, 30, 35, 40, 45, 50, 55, 60],
        "meal_standard_table": [300, 500, 600, 800, 1000]
    },
    
    # 新安江中餐
    "新安江中餐": {
        "meal_types": ["中餐"]
    },
    
    # 新安江景区导服
    "新安江景区导服": {
        "guide_items": ["新安江景区导服"]
    },
    
    # 新安江游船
    "新安江游船": {
        "ship_combo": ["中远程及新增市场价格", "学生团队价", "广德", "旅投员工价", "景酒打包价", "研学团", "老年团队价"],
        "ship_items": ["新安江竹筏漂流", "新安江龙舟漂游", "梦幻新安江", "江清月近人实景演艺船票", "诗韵新安江（新安江-严州古城航线含自助茶歇、简餐）"]
    },
    
    # 新安江船餐
    "新安江船餐": {
        "meal_types": ["前程似锦宴", "金玉满堂宴", "阖家团圆宴"]
    },
    
    # 新安江草莓采摘入园票
    "新安江草莓采摘入园票": {
        "business_items": ["草莓采摘入园票"]
    },
    
    # 江清月近人实景演艺门票
    "江清月近人实景演艺门票": {
        "ticket_combo": ["30年教师证", "70周岁", "中远程及新增市场价格", "儿童", "其他", "军官证", "在校大学生", "学生团队价", "导游证", "广德", "建德就业", "建德市民", "接待免票", "新闻记者证", "旅投员工价", "景酒打包价", "杭州市民卡", "杭州文旅卡/惠民卡", "残疾证", "消防员证", "献血荣誉证", "病故军属家人", "研学团", "老年团队价", "萧山、临平、西湖管委会免票", "退役证", "钱江分免票", "非建德户籍免票", "高层次人才"],
        "ticket_items": ["江清月近人"]
    },
    
    # 江清月近人白天场
    "江清月近人白天场": {
        "ticket_combo": ["研学团"],
        "ticket_items": ["江清月近人白天场"]
    },
    
    # 灵栖洞中餐
    "灵栖洞中餐": {
        "meal_types": ["中餐"]
    },
    
    # 灵栖洞停车场
    "灵栖洞停车场": {
        "parking_items": ["中型", "大型", "小车"]
    },
    
    # 灵栖洞喊泉
    "灵栖洞喊泉": {
        "business_items": ["喊泉"]
    },
    
    # 灵栖洞手划船
    "灵栖洞手划船": {
        "ship_combo": ["正常价", "特殊价"],
        "ship_items": ["船票"]
    },
    
    # 灵栖洞景区导服
    "灵栖洞景区导服": {
        "guide_items": ["灵栖洞景区导服"]
    },
    
    # 灵栖洞极速滑道
    "灵栖洞极速滑道": {
        "recreational_combo": ["中远程及新增市场价格", "学生团队价", "旅投员工价", "景酒打包价", "老年团队价"],
        "recreational_items": ["极速滑道"]
    },
    
    # 灵栖洞西游魔毯
    "灵栖洞西游魔毯": {
        "recreational_combo": ["中远程及新增市场价格", "学生团队价", "旅投员工价", "景酒打包价", "老年团队价"],
        "recreational_items": ["西游魔毯"]
    },
    
    # 灵栖洞豆腐包制作体验
    "灵栖洞豆腐包制作体验": {
        "business_items": ["豆腐包制作体验"],
        "recreational_combo": ["中远程及新增市场价格", "学生团队价", "旅投员工价", "景酒打包价", "老年团队价"]
    },
    
    # 灵栖洞门票
    "灵栖洞门票": {
        "ticket_combo": ["30年教师证", "70周岁", "中远程及新增市场价格", "儿童", "其他", "军官证", "在校大学生", "学生团队价", "导游证", "广德", "建德就业", "建德市民", "接待免票", "新闻记者证", "旅投员工价", "景酒打包价", "杭州市民卡", "杭州文旅卡/惠民卡", "残疾证", "消防员证", "献血荣誉证", "病故军属家人", "研学团", "老年团队价", "萧山、临平、西湖管委会免票", "退役证", "钱江分免票", "青少年优惠", "非建德户籍免票", "高层次人才"],
        "ticket_items": ["门票"]
    },
    
    # 灵栖洞餐饮
    "灵栖洞餐饮": {
        "meal_standard": [0, 20, 25, 30, 35, 40, 45, 50, 55, 60],
        "meal_standard_table": [300, 400, 500, 600, 700, 800, 1000]
    },
    
    # 玉泉寺门票
    "玉泉寺门票": {
        "ticket_items": ["门票"]
    },
    
    # 考拉森林丛林探险
    "考拉森林丛林探险": {
        "business_items": ["考拉森林丛林探险亲子线", "考拉森林丛林探险成人线"],
        "recreational_combo": ["中远程及新增市场价格", "学生团队价", "旅投员工价", "景酒打包价", "老年团队价"]
    },
    
    # 餐饮
    "餐饮": {
        "meal_standard": [18]
    }
}

def get_all_resource_names() -> List[str]:
    """获取所有可能的资源名称"""
    all_names = []
    for resource_main, sub_resources in RESOURCE_SUBRESOURCES.items():
        for sub_resource in sub_resources:
            all_names.append(f"{resource_main}-{sub_resource}")
    return all_names

def get_resource_detail_structure(resource_name: str) -> Dict[str, List]:
    """获取指定资源的详细信息结构"""
    return RESOURCE_DETAIL_STRUCTURE.get(resource_name, {})

def get_field_description(field_type: str) -> str:
    """获取字段类型的中文描述"""
    field_descriptions = {
        "business_items": "业务项目",
        "guide_items": "导游项目",
        "parking_items": "停车类型",
        "recreational_combo": "娱乐套餐类型",
        "recreational_items": "娱乐项目",
        "ship_combo": "船票套餐类型",
        "ship_items": "船只类型",
        "ticket_combo": "门票套餐类型",
        "ticket_items": "票型",
        "meal_standard": "餐饮标准",
        "meal_standard_table": "桌餐标准",
        "meal_types": "餐饮类型",
        "guide_types": "导游类型",
        "meeting_place": "会议场所",
        "duration_hours": "时长",
        "device_name": "设备名称",
        "vehicle_type": "车辆类型",
        "travel_route": "行程路线",
        "rent_vehicle_type": "租车类型",
        "rent_travel_route": "租车路线",
        "room_type": "房型",
        "room_agreement": "房价协议",
        "meeting_room_name": "会议室名称",
        "meeting_agreement": "会议协议",
        "coffee_break": "茶歇",
        "devices": "设备",
        "devices_agreement": "设备协议"
    }
    return field_descriptions.get(field_type, field_type)

def load_original_data(file_path: str) -> List[Dict]:
    """加载原始训练数据，支持JSON和JSONL格式"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 根据文件扩展名判断格式
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.jsonl':
        return load_jsonl_data(file_path)
    elif file_ext == '.json':
        return load_json_data(file_path)
    else:
        # 尝试自动检测格式
        return auto_detect_and_load(file_path)

def load_json_data(file_path: str) -> List[Dict]:
    """加载JSON格式数据"""
    print(f"检测到JSON格式文件: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_jsonl_data(file_path: str) -> List[Dict]:
    """加载JSONL格式数据"""
    print(f"检测到JSONL格式文件: {file_path}")
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # 跳过空行
                continue
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"警告: 第{line_num}行JSON解析失败: {e}")
                print(f"问题行内容: {line[:100]}...")
                continue
    return data

def auto_detect_and_load(file_path: str) -> List[Dict]:
    """自动检测文件格式并加载"""
    print(f"自动检测文件格式: {file_path}")
    
    # 读取文件前几行来判断格式
    with open(file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        f.seek(0)  # 重置文件指针
        
        if first_line.startswith('['):
            # JSON数组格式
            print("检测到JSON数组格式")
            return load_json_data(file_path)
        elif first_line.startswith('{'):
            # 可能是JSONL格式
            print("检测到JSONL格式")
            return load_jsonl_data(file_path)
        else:
            raise ValueError(f"无法识别的文件格式: {file_path}")

def create_field_specific_instruction(field_name: str, field_description: str, resource_name: str = None) -> str:
    """为特定字段创建指令"""
    
    # 特殊处理resource_names字段，包含业务范围
    if field_name == "resource_names":
        all_resource_names = get_all_resource_names()
        resource_list = "、".join(all_resource_names)
        return f"""请从OCR文本中抽取旅行订单中的所有资源名称。

可识别的资源名称包括：{resource_list}

资源别称表述：
- 大慈岩丛林速滑（旱滑道）下行：可能表述为"旱滑道"、"丛林速滑"
- 灵栖洞西游魔毯：可能表述为"飞天魔毯"、"灵栖洞魔毯"
- 灵栖洞极速滑道：可能表述为"灵栖洞滑道"、"灵栖洞速滑"
- 考拉森林丛林探险：可能表述为"考拉森林"、"丛林探险"
- 三江口游线（游船）：可能表述为"富春江游船"
- 严州古城：可能表述为"梅城"
- 三都渔村婚礼表演：可能表述为"三都渔村九姓渔氏水上婚礼表演"、"九姓渔氏"、"婚礼表演"

资源内在联系规则：
1. 七里扬帆景区：
   - 出现"七里扬帆"（无三江口）信息时，包含：七里扬帆-七里扬帆门票、七里扬帆-七里扬帆游船
   - 出现"三江口"信息时，包含：七里扬帆-三江口游线（游船），但不包含七里扬帆门票和七里扬帆游船
   - 出现"葫芦峡漂流"信息时，包含：七里扬帆-七里扬帆门票、七里扬帆-七里扬帆游船

2. 灵栖洞景区：
   - 出现"灵栖洞"信息时，包含：灵栖洞-灵栖洞门票、灵栖洞-灵栖洞手划船

严格按照以下JSON格式输出：
{{
  "resource_names": ["资源主体-资源名称1", "资源主体-资源名称2"] 或 []
}}"""
    
    # 特殊处理resource_detail字段，包含业务范围
    if field_name == "resource_detail" and resource_name:
        detail_structure = get_resource_detail_structure(resource_name)
        if detail_structure:
            field_lines = []
            for field_type, enum_values in detail_structure.items():
                if enum_values:
                    enum_list = "、".join(f'"{str(v)}"' for v in enum_values)
                    field_lines.append(f"{field_type} ({get_field_description(field_type)}): {enum_list}")
                else:
                    field_lines.append(f"{field_type} ({get_field_description(field_type)}): null")
            
            fields_info = "\n".join(field_lines)
            
            # 构建JSON示例结构
            json_fields = []
            for field_type in detail_structure.keys():
                json_fields.append(f'    "{field_type}": 选择值或null')
            json_structure = ",\n".join(json_fields)
            
            return f"""请从OCR文本中抽取旅行订单中{resource_name}的详细信息。

提取字段及可选值：
{fields_info}

严格按照以下JSON格式输出：
{{
  "resource_detail": {{
{json_structure}
  }}
}}"""
        else:
            return f"""请从OCR文本中抽取旅行订单中{resource_name}的详细信息。

严格按照以下JSON格式输出：
{{
  "resource_detail": {{}} 
}}"""
    
    # 基础字段的指令定义
    field_instructions = {
        "team_size": """请从OCR文本中抽取旅行订单的总人数信息。

严格按照以下JSON格式输出：
{
  "team_size": 整数或null
}
注意：订单信息中的导游员、讲解员、司机、领队等人员，不包含在总人数中。
""",
        "start_date": """请从OCR文本中抽取旅行订单的开始日期信息。

严格按照以下JSON格式输出：
{
  "start_date": "YYYY-MM-DD"或null
}""",
        "end_date": """请从OCR文本中抽取旅行订单的结束日期信息。

严格按照以下JSON格式输出：
{
  "end_date": "YYYY-MM-DD"或null
}""",
        "payment_method": """请从OCR文本中抽取旅行订单的支付方式信息。

严格按照以下JSON格式输出：
{
  "payment_method": "支付方式"或null
}""",
        "customer_name": """请从OCR文本中抽取旅行订单的客户名称（通常是旅行社或公司名称）。

严格按照以下JSON格式输出：
{
  "customer_name": "客户名称"或null
}""",
        "customer_market": """请从OCR文本中抽取旅行订单的客户地区，按照'省-市'格式。

严格按照以下JSON格式输出：
{
  "customer_market": "省-市"或null
}""",
        "customer_type": """请从OCR文本中抽取旅行订单的客户类型（如旅行社、机构等）。

严格按照以下JSON格式输出：
{
  "customer_type": "客户类型"或null
}""",
        "notes": """请从OCR文本中抽取旅行订单的备注信息。

严格按照以下JSON格式输出：
{
  "notes": "备注信息"或null
}""",
        "contacts": """请从OCR文本中抽取旅行订单的联系人信息。

提取字段：
name (联系人姓名)
phone (联系电话)
idcard (身份证号码，如果没有则为null)

严格按照以下JSON格式输出：
{
  "contacts": {
    "data": [
      {
        "name": "姓名",
        "phone": "电话", 
        "idcard": "身份证号"或null
      }
    ]
  }
}""",
        "resource_start_time": f"""请从OCR文本中抽取旅行订单中{resource_name or '该资源'}的开始时间。

严格按照以下JSON格式输出：
{{
  "resource_start_time": "YYYY-MM-DD"或null
}}""",
        "resource_end_time": f"""请从OCR文本中抽取旅行订单中{resource_name or '该资源'}的结束时间。

严格按照以下JSON格式输出：
{{
  "resource_end_time": "YYYY-MM-DD"或null
}}
""",
        "resource_team_size": f"""请从OCR文本中抽取旅行订单中{resource_name or '该资源'}的使用人数。

严格按照以下JSON格式输出：
{{
  "resource_team_size": 整数或null
}}
注意：订单信息中的导游员、讲解员、司机、领队等人员，不包含在总人数中。
"""
    }
    
    return field_instructions.get(field_name, f"""请从OCR文本中抽取旅行订单的{field_description}信息。

严格按照以下JSON格式输出：
{{
  "{field_name}": "值"或null
}}""")

def extract_field_value(output_json: Dict, field_path: List[str]) -> Any:
    """从完整输出中提取特定字段的值"""
    current = output_json
    try:
        for key in field_path:
            if isinstance(key, int):
                # 处理数组索引
                if isinstance(current, list) and 0 <= key < len(current):
                    current = current[key]
                else:
                    return None
            else:
                # 处理字典键
                current = current[key]
        return current
    except (KeyError, TypeError, IndexError):
        return None

def format_output_value(value: Any, field_name: str) -> str:
    """格式化输出值，包含字段名"""
    if value is None:
        return f'"{field_name}": null'
    
    if field_name in ["contacts", "resource_names", "resource_detail"]:
        # 对于复杂对象，返回包含字段名的JSON字符串
        return f'"{field_name}": {json.dumps(value, ensure_ascii=False)}'
    elif field_name in ["team_size", "resource_team_size"] and isinstance(value, (int, float)):
        # team_size字段保持为整数格式
        return f'"{field_name}": {value}'
    elif isinstance(value, str):
        return f'"{field_name}": "{value}"'
    elif isinstance(value, (int, float)):
        return f'"{field_name}": "{str(value)}"'
    else:
        return f'"{field_name}": {json.dumps(value, ensure_ascii=False)}'

def extract_resource_names(resource_results: Dict) -> List[str]:
    """从resource_results中提取所有资源名称"""
    if not resource_results or "data" not in resource_results:
        return []
    
    resource_names = []
    for resource in resource_results["data"]:
        if "resource_name" in resource:
            resource_names.append(resource["resource_name"])
    
    return resource_names

def find_resource_by_name(resource_results: Dict, target_name: str) -> Dict:
    """根据资源名称查找对应的资源信息"""
    if not resource_results or "data" not in resource_results:
        return {}
    
    for resource in resource_results["data"]:
        if resource.get("resource_name") == target_name:
            return resource
    
    return {}

def generate_fine_grained_data(original_data: List[Dict]) -> List[Dict]:
    """生成细粒度训练数据"""
    fine_grained_data = []
    
    # 定义基础字段（非资源相关）
    basic_field_definitions = [
        ("team_size", "总人数", ["nonresource_results", "team_size"]), 
        ("start_date", "开始日期", ["nonresource_results", "start_date"]),
        ("end_date", "结束日期", ["nonresource_results", "end_date"]),
        ("payment_method", "支付方式", ["nonresource_results", "payment_method"]),
        ("customer_name", "客户名称", ["nonresource_results", "customer_name"]),
        ("customer_market", "客户地区", ["nonresource_results", "customer_market"]),
        ("customer_type", "客户类型", ["nonresource_results", "customer_type"]),
        ("notes", "备注", ["nonresource_results", "notes"]),
        ("contacts", "联系人信息", ["contacts"])
    ]
    
    for item in original_data:
        input_text = item["input"]
        
        # 解析原始输出
        try:
            output_json = json.loads(item["output"])
        except json.JSONDecodeError:
            print(f"跳过无效JSON: {item['output'][:100]}...")
            continue
        
        # 1. 为基础字段创建训练样本
        for field_name, field_description, field_path in basic_field_definitions:
            field_value = extract_field_value(output_json, field_path)
            
            sample = {
                "instruction": create_field_specific_instruction(field_name, field_description),
                "input": input_text,
                "output": format_output_value(field_value, field_name)
            }
            
            fine_grained_data.append(sample)
        
        # 2. 创建资源名称列表抽取任务
        resource_results = extract_field_value(output_json, ["resource_results"])
        resource_names = extract_resource_names(resource_results)
        
        sample = {
            "instruction": create_field_specific_instruction("resource_names", "资源名称列表"),
            "input": input_text,
            "output": format_output_value(resource_names, "resource_names")
        }
        fine_grained_data.append(sample)
        
        # 3. 为每个资源创建详细的抽取任务
        for resource_name in resource_names:
            resource_info = find_resource_by_name(resource_results, resource_name)
            
            # 资源开始时间
            start_time = resource_info.get("start_time")
            sample = {
                "instruction": create_field_specific_instruction("resource_start_time", "开始时间", resource_name),
                "input": input_text,
                "output": format_output_value(start_time, "resource_start_time")
            }
            fine_grained_data.append(sample)
            
            # 资源结束时间
            end_time = resource_info.get("end_time")
            sample = {
                "instruction": create_field_specific_instruction("resource_end_time", "结束时间", resource_name),
                "input": input_text,
                "output": format_output_value(end_time, "resource_end_time")
            }
            fine_grained_data.append(sample)
            
            # 资源使用人数
            team_size = resource_info.get("team_size")
            sample = {
                "instruction": create_field_specific_instruction("resource_team_size", "使用人数", resource_name),
                "input": input_text,
                "output": format_output_value(team_size, "resource_team_size")
            }
            fine_grained_data.append(sample)
            
            # 资源详细信息
            detail = resource_info.get("detail", {})
            # 从资源名称中提取具体的资源类型（去掉资源主体前缀）
            resource_type = resource_name
            if "-" in resource_name:
                resource_type = resource_name.split("-", 1)[1]  # 取第二部分作为资源类型
            
            sample = {
                "instruction": create_field_specific_instruction("resource_detail", "详细信息", resource_type),
                "input": input_text,
                "output": format_output_value(detail, "resource_detail")
            }
            fine_grained_data.append(sample)
    
    return fine_grained_data

def save_fine_grained_data(data: List[Dict], output_file: str, format_type: str = 'auto'):
    """保存细粒度数据，支持JSON和JSONL格式"""
    if format_type == 'auto':
        # 根据输出文件扩展名自动选择格式
        file_ext = os.path.splitext(output_file)[1].lower()
        if file_ext == '.jsonl':
            format_type = 'jsonl'
        else:
            format_type = 'json'
    
    if format_type == 'jsonl':
        print(f"正在保存为JSONL格式: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    else:
        print(f"正在保存为JSON格式: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='将粗粒度订单数据转换为细粒度训练数据')
    parser.add_argument('--input', '-i', type=str, 
                       default="/home/ziqiang/order_info/database/test/ocr_text_orders_20250812.jsonl",
                       help='输入文件路径 (支持JSON和JSONL格式)')
    parser.add_argument('--output', '-o', type=str,
                       default="/home/ziqiang/order_info/database/dev/ocr_text_orders_20250812_re.jsonl",
                       help='输出文件路径')
    parser.add_argument('--format', '-f', type=str, choices=['json', 'jsonl', 'auto'],
                       default='auto', help='输出格式 (auto: 根据文件扩展名自动选择)')
    parser.add_argument('--shuffle', action='store_true', default=True,
                       help='是否打乱数据顺序')
    
    args = parser.parse_args()
    
    # 加载原始数据
    print("正在加载原始数据...")
    try:
        original_data = load_original_data(args.input)
        print(f"加载了 {len(original_data)} 条原始数据")
    except Exception as e:
        print(f"加载数据失败: {e}")
        return
    
    # 生成细粒度数据
    print("正在生成细粒度训练数据...")
    fine_grained_data = generate_fine_grained_data(original_data)
    print(f"生成了 {len(fine_grained_data)} 条细粒度数据")
    
    # 打乱数据顺序
    if args.shuffle:
        random.shuffle(fine_grained_data)
        print("已打乱数据顺序")
    
    # 保存细粒度数据
    try:
        save_fine_grained_data(fine_grained_data, args.output, args.format)
        print(f"完成！细粒度数据已保存到 {args.output}")
    except Exception as e:
        print(f"保存数据失败: {e}")
        return
    
    # 输出统计信息
    field_counts = {}
    for item in fine_grained_data:
        instruction = item["instruction"]
        # 统计基础字段
        for field_name in ["总人数", "开始日期", "结束日期", "支付方式", "客户名称", "客户地区", "客户类型", "备注", "联系人信息"]:
            if field_name in instruction:
                field_counts[field_name] = field_counts.get(field_name, 0) + 1
                break
        # 统计资源相关字段
        for field_name in ["资源名称列表", "开始时间", "结束时间", "使用人数", "详细信息"]:
            if field_name in instruction and ("资源" in instruction or "resource" in instruction.lower()):
                field_counts[f"资源_{field_name}"] = field_counts.get(f"资源_{field_name}", 0) + 1
                break
    
    print("\n各字段的样本数量:")
    for field, count in field_counts.items():
        print(f"  {field}: {count} 条")

if __name__ == "__main__":
    main()