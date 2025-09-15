#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级数据增强脚本 - 专门针对旅游订单资源抽取

基于你的实际case分析，提供以下增强策略：
1. 针对"新叶古村"、"大慈岩"等低频景点的样本生成
2. 针对"江清月近人"等特殊演艺项目的样本生成
3. 混合资源组合的增强
"""

import json
import random
import copy
from collections import defaultdict
from typing import List, Dict, Any
import re

class AdvancedDataAugmenter:
    def __init__(self):
        # 定义各类变量用于数据增强
        self.dates = [
            "7月15日", "7月16日", "7月19日", "7月21日", "7月22日", "7月25日", 
            "7月26日", "7月27日", "8月1日", "8月2日", "8月3日", "8月5日", "8月6日"
        ]
        
        self.people_counts = [
            "8人", "12人", "15人", "18人", "22人", "25人", "28人", "32人", 
            "35人", "38人", "42人", "45人", "48人", "52人", "55人"
        ]
        
        self.travel_agencies = [
            "杭州天翼", "苏州快乐", "建德光大", "建德中旅", "嘉兴米兰", 
            "浙江吉程", "上海东方", "宁波海天", "温州山水", "台州阳光"
        ]
        
        self.guide_names = [
            ("张", "13567300196"), ("王", "18806210530"), ("李", "13958298707"),
            ("陈", "13868126262"), ("刘", "13588024400"), ("赵", "13819120723"),
            ("孙", "13862111871"), ("周", "18989479865"), ("吴", "13805701560"),
            ("郑", "13396506951")
        ]
        
    def create_xinye_samples(self, count: int = 3) -> List[Dict[str, Any]]:
        """创建新叶古村的增强样本"""
        samples = []
        
        templates = [
            "{agency}，{date}，{people}，新叶古村，\n导游：{guide_name} {phone}",
            "{date} {agency}\n新叶古村门票+导游服务\n人数：{people}\n导游联系：{guide_name} {phone}",
            "{agency}团队，新叶古村一日游\n日期：{date}\n人数：{people}\n领队：{guide_name}（{phone}）"
        ]
        
        for i in range(count):
            template = random.choice(templates)
            agency = random.choice(self.travel_agencies)
            date = random.choice(self.dates)
            people = random.choice(self.people_counts)
            guide_name, phone = random.choice(self.guide_names)
            
            input_text = template.format(
                agency=agency, date=date, people=people, 
                guide_name=guide_name, phone=phone
            )
            
            sample = {
                "instruction": "请从OCR文本中抽取旅行订单中的所有资源名称。\n\n可识别的资源名称包括：七里扬帆-七里扬帆草莓采摘入园票、七里扬帆-七里扬帆小火车、七里扬帆-七里扬帆葫芦山庄餐饮、七里扬帆-七里扬帆门票、七里扬帆-七里扬帆游船、七里扬帆-三江口游线（游船）、七里扬帆-七里扬帆停车场、七里扬帆-七里扬帆葫芦峡漂流、千岛湖好运岛-千岛湖好运岛门票、千岛湖好运岛-千岛湖好运岛游船、灵栖洞-灵栖洞门票、灵栖洞-灵栖洞手划船、灵栖洞-灵栖洞西游魔毯、新安江-新安江游船、宿江公司-江清月近人实景演艺门票、大慈岩-大慈岩门票、大慈岩-大慈岩索道、大慈岩-大慈岩中餐、大慈岩-大慈岩餐饮、大慈岩-大慈岩停车场、新叶古村-新叶古村门票、导服中心-灵栖洞景区导服。",
                "input": input_text,
                "output": '{"resource_names": ["新叶古村-新叶古村门票"]}'
            }
            samples.append(sample)
            
        return samples
    
    def create_daciyan_samples(self, count: int = 2) -> List[Dict[str, Any]]:
        """创建大慈岩索道+门票的增强样本"""
        samples = []
        
        templates = [
            "{agency}，日期：{date} 大慈岩（含上下索道），\n导游：{guide_name} {phone} 人数{people}",
            "{date} {agency}\n大慈岩索道+门票套票\n{people} 导游：{guide_name}（{phone}）",
        ]
        
        for i in range(count):
            template = random.choice(templates)
            agency = random.choice(self.travel_agencies)
            date = random.choice(self.dates)
            people = random.choice(self.people_counts)
            guide_name, phone = random.choice(self.guide_names)
            
            input_text = template.format(
                agency=agency, date=date, people=people,
                guide_name=guide_name, phone=phone
            )
            
            sample = {
                "instruction": "请从OCR文本中抽取旅行订单中的所有资源名称。\n\n可识别的资源名称包括：七里扬帆-七里扬帆草莓采摘入园票、七里扬帆-七里扬帆小火车、七里扬帆-七里扬帆葫芦山庄餐饮、七里扬帆-七里扬帆门票、七里扬帆-七里扬帆游船、七里扬帆-三江口游线（游船）、七里扬帆-七里扬帆停车场、七里扬帆-七里扬帆葫芦峡漂流、千岛湖好运岛-千岛湖好运岛门票、千岛湖好运岛-千岛湖好运岛游船、灵栖洞-灵栖洞门票、灵栖洞-灵栖洞手划船、灵栖洞-灵栖洞西游魔毯、新安江-新安江游船、宿江公司-江清月近人实景演艺门票、大慈岩-大慈岩门票、大慈岩-大慈岩索道、大慈岩-大慈岩中餐、大慈岩-大慈岩餐饮、大慈岩-大慈岩停车场、新叶古村-新叶古村门票、导服中心-灵栖洞景区导服。",
                "input": input_text,
                "output": '{"resource_names": ["大慈岩-大慈岩索道", "大慈岩-大慈岩门票"]}'
            }
            samples.append(sample)
            
        return samples
    
    def create_jiangqing_samples(self, count: int = 4) -> List[Dict[str, Any]]:
        """创建江清月近人相关的增强样本"""
        samples = []
        
        # 纯江清月近人样本
        for i in range(2):
            agency = random.choice(self.travel_agencies)
            date = random.choice(self.dates)
            people = random.choice(self.people_counts)
            guide_name, phone = random.choice(self.guide_names)
            
            input_text = f"{agency}，{date}，{people}，江清月近人演出，导游：{guide_name} {phone}"
            
            sample = {
                "instruction": "请从OCR文本中抽取旅行订单中的所有资源名称。\n\n可识别的资源名称包括：七里扬帆-七里扬帆草莓采摘入园票、七里扬帆-七里扬帆小火车、七里扬帆-七里扬帆葫芦山庄餐饮、七里扬帆-七里扬帆门票、七里扬帆-七里扬帆游船、七里扬帆-三江口游线（游船）、七里扬帆-七里扬帆停车场、七里扬帆-七里扬帆葫芦峡漂流、千岛湖好运岛-千岛湖好运岛门票、千岛湖好运岛-千岛湖好运岛游船、灵栖洞-灵栖洞门票、灵栖洞-灵栖洞手划船、灵栖洞-灵栖洞西游魔毯、新安江-新安江游船、宿江公司-江清月近人实景演艺门票、大慈岩-大慈岩门票、大慈岩-大慈岩索道、大慈岩-大慈岩中餐、大慈岩-大慈岩餐饮、大慈岩-大慈岩停车场、新叶古村-新叶古村门票、导服中心-灵栖洞景区导服。",
                "input": input_text,
                "output": '{"resource_names": ["宿江公司-江清月近人实景演艺门票"]}'
            }
            samples.append(sample)
        
        # 江清月近人+新安江游船组合样本
        for i in range(2):
            agency = random.choice(self.travel_agencies)
            date = random.choice(self.dates)
            people = random.choice(self.people_counts)
            guide_name, phone = random.choice(self.guide_names)
            
            input_text = f"{agency}，{date}，{people}，江清月近人+新安江游船，导游：{guide_name} {phone}"
            
            sample = {
                "instruction": "请从OCR文本中抽取旅行订单中的所有资源名称。\n\n可识别的资源名称包括：七里扬帆-七里扬帆草莓采摘入园票、七里扬帆-七里扬帆小火车、七里扬帆-七里扬帆葫芦山庄餐饮、七里扬帆-七里扬帆门票、七里扬帆-七里扬帆游船、七里扬帆-三江口游线（游船）、七里扬帆-七里扬帆停车场、七里扬帆-七里扬帆葫芦峡漂流、千岛湖好运岛-千岛湖好运岛门票、千岛湖好运岛-千岛湖好运岛游船、灵栖洞-灵栖洞门票、灵栖洞-灵栖洞手划船、灵栖洞-灵栖洞西游魔毯、新安江-新安江游船、宿江公司-江清月近人实景演艺门票、大慈岩-大慈岩门票、大慈岩-大慈岩索道、大慈岩-大慈岩中餐、大慈岩-大慈岩餐饮、大慈岩-大慈岩停车场、新叶古村-新叶古村门票、导服中心-灵栖洞景区导服。",
                "input": input_text,
                "output": '{"resource_names": ["宿江公司-江清月近人实景演艺门票", "新安江-新安江游船"]}'
            }
            samples.append(sample)
            
        return samples
        
    def create_mixed_samples(self, count: int = 3) -> List[Dict[str, Any]]:
        """创建复杂混合资源的样本"""
        samples = []
        
        # 大慈岩+新叶古村组合
        agency = random.choice(self.travel_agencies)
        date1 = random.choice(self.dates)
        date2 = random.choice(self.dates)
        people = random.choice(self.people_counts)
        guide_name, phone = random.choice(self.guide_names)
        
        input_text = f"{agency}，日期：{date1}大慈岩（含上下索道），{date2}新叶古村\n导游：{guide_name} {phone} 人数{people}"
        
        sample = {
            "instruction": "请从OCR文本中抽取旅行订单中的所有资源名称。\n\n可识别的资源名称包括：七里扬帆-七里扬帆草莓采摘入园票、七里扬帆-七里扬帆小火车、七里扬帆-七里扬帆葫芦山庄餐饮、七里扬帆-七里扬帆门票、七里扬帆-七里扬帆游船、七里扬帆-三江口游线（游船）、七里扬帆-七里扬帆停车场、七里扬帆-七里扬帆葫芦峡漂流、千岛湖好运岛-千岛湖好运岛门票、千岛湖好运岛-千岛湖好运岛游船、灵栖洞-灵栖洞门票、灵栖洞-灵栖洞手划船、灵栖洞-灵栖洞西游魔毯、新安江-新安江游船、宿江公司-江清月近人实景演艺门票、大慈岩-大慈岩门票、大慈岩-大慈岩索道、大慈岩-大慈岩中餐、大慈岩-大慈岩餐饮、大慈岩-大慈岩停车场、新叶古村-新叶古村门票、导服中心-灵栖洞景区导服。",
            "input": input_text,
            "output": '{"resource_names": ["大慈岩-大慈岩索道", "大慈岩-大慈岩门票", "新叶古村-新叶古村门票"]}'
        }
        samples.append(sample)
        
        return samples
    
    def generate_all_samples(self) -> List[Dict[str, Any]]:
        """生成所有增强样本"""
        all_samples = []
        
        print("🔄 正在生成增强样本...")
        
        # 新叶古村样本
        xinye_samples = self.create_xinye_samples(3)
        all_samples.extend(xinye_samples)
        print(f"✅ 生成新叶古村样本: {len(xinye_samples)}个")
        
        # 大慈岩索道样本  
        daciyan_samples = self.create_daciyan_samples(2)
        all_samples.extend(daciyan_samples)
        print(f"✅ 生成大慈岩索道样本: {len(daciyan_samples)}个")
        
        # 江清月近人样本
        jiangqing_samples = self.create_jiangqing_samples(4)
        all_samples.extend(jiangqing_samples)
        print(f"✅ 生成江清月近人样本: {len(jiangqing_samples)}个")
        
        # 混合样本
        mixed_samples = self.create_mixed_samples(1)
        all_samples.extend(mixed_samples)
        print(f"✅ 生成混合资源样本: {len(mixed_samples)}个")
        
        print(f"\n📊 总计生成增强样本: {len(all_samples)}个")
        
        return all_samples

def main():
    augmenter = AdvancedDataAugmenter()
    
    print("🚀 开始高级数据增强...")
    print("=" * 50)
    
    # 生成增强样本
    enhanced_samples = augmenter.generate_all_samples()
    
    # 保存增强样本
    output_file = "enhanced_samples.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(enhanced_samples, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 增强样本已保存到: {output_file}")
    print("🎉 数据增强完成！")
    
    # 显示样本预览
    print("\n📋 样本预览:")
    print("-" * 30)
    for i, sample in enumerate(enhanced_samples[:3]):
        print(f"\n样本 {i+1}:")
        print(f"输入: {sample['input'][:100]}...")
        print(f"输出: {sample['output']}")

if __name__ == "__main__":
    main()
