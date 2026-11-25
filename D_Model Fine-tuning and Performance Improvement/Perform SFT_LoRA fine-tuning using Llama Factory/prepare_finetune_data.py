#!/usr/bin/env python3
"""
数据准备脚本：将历史工单和诊断案例转化为指令微调JSON格式
包含数据清洗、去重和格式转换功能
"""

import json
import re
from typing import List, Dict
import hashlib

def clean_text(text: str) -> str:
    """清理文本数据"""
    # 去除多余空格和换行
    text = re.sub(r'\s+', ' ', text).strip()
    # 去除特殊字符
    text = re.sub(r'[^一-鿿0-9a-zA-Z\s,.?!:;()\[\]{}]', '', text)
    return text

def deduplicate_data(data: List[Dict]) -> List[Dict]:
    """对数据进行去重处理"""
    seen = set()
    unique_data = []
    for item in data:
        # 创建唯一哈希键
        key = hashlib.md5(json.dumps(item, sort_keys=True).encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique_data.append(item)
    return unique_data

def convert_to_json_format(history_data: List[Dict]) -> List[Dict]:
    """将历史工单数据转换为指令微调JSON格式"""
    formatted_data = []
    
    for item in history_data:
        instruction = f"根据以下故障现象进行诊断：{item['fault_symptom']}"
        input_text = f"【历史案例】{item['history_case']}【手册】{item['manual_info']}"
        output_text = f"根据症状和资料，最可能的原因是{item['diagnosis']}。建议步骤：{'; '.join(item['suggestions'])}"
        
        formatted_item = {
            "instruction": clean_text(instruction),
            "input": clean_text(input_text),
            "output": clean_text(output_text)
        }
        
        formatted_data.append(formatted_item)
    
    return formatted_data

def main():
    # 示例历史工单数据
    sample_history_data = [
        {
            "fault_symptom": "主轴异响，加工精度下降",
            "history_case": "去年类似情况为轴承磨损",
            "manual_info": "主轴轴承寿命通常为5000小时",
            "diagnosis": "主轴轴承磨损",
            "suggestions": ["停机检查轴承游隙", "使用听诊器定位异响源", "如确认磨损，更换同型号轴承并校准"]
        },
        {
            "fault_symptom": "液压系统压力不稳",
            "history_case": "上个月更换液压油后压力恢复正常",
            "manual_info": "液压油每6个月需要更换一次",
            "diagnosis": "液压油污染",
            "suggestions": ["检查液压油颜色和杂质", "更换液压油和滤芯", "排空系统空气"]
        }
    ]
    
    # 转换为JSON格式
    formatted_data = convert_to_json_format(sample_history_data)
    
    # 数据去重
    unique_data = deduplicate_data(formatted_data)
    
    # 保存到文件
    output_file = "diagnosis_finetune_data.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(unique_data, f, ensure_ascii=False, indent=2)
    
    print(f"数据准备完成，已保存到 {output_file}")
    print(f"原始数据: {len(sample_history_data)} 条")
    print(f"去重后数据: {len(unique_data)} 条")

if __name__ == "__main__":
    main()