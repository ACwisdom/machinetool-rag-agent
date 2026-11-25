import json

# 修正：用合法的字符串替换...，补充完整的字典/列表格式
work_orders = [
    {
        "故障现象": "主轴异响，运转时伴随不规则噪音",
        "根本原因": "主轴轴承磨损",
        "解决方案": "更换主轴轴承并重新校准"
    },
    {
        "故障现象": "刀库换刀卡顿",
        "根本原因": "换刀机构润滑油不足",
        "解决方案": "添加专用润滑油并清洁导轨"
    }
]

# 写入JSON文件（这部分逻辑完全正确）
with open("历史工单数据.json", "w", encoding="utf-8") as f:
    json.dump(work_orders, f, ensure_ascii=False, indent=2)

print("JSON文件已成功写入！")