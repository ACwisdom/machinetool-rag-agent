import re
# 1. 导入步骤3的PDF提取函数（确保两个.py文件在同一文件夹）
from Step3_PDF_Document_Content_Extraction import extract_pdf_text

# 2. 重新提取PDF文本（不依赖外部变量）
manual_text = extract_pdf_text("机床用户手册.pdf")

# 3. 原关键信息抽取逻辑（正则按文档2修正，避免图片中显示错误）
def extract_fault_info(text):
    fault_codes = re.findall(r"故障代码:(\w+).+?(?:\n|$)", text)
    components = re.findall(r"(主轴|轴承|丝杠|导轨)", text)
    symptoms = re.findall(r"(异响|振动|精度超差|过热)", text)
    return {
        "故障代码": list(set(fault_codes)),
        "部件": list(set(components)),
        "症状": list(set(symptoms))
    }

# 4. 执行函数
fault_info = extract_fault_info(manual_text)
print(fault_info)  # 可选：验证结果