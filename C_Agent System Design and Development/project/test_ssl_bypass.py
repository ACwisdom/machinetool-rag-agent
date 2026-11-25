import requests
import urllib3

# 禁用 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

url = "https://api.ppio.com/openai/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer sk_RKTMSLGtCDQWJj9SDV1T-Mp9LnGxad3RHfx169D9MO8"
}
data = {
    "model": "qwen/qwen2.5-7b-instruct",
    "messages": [{"role": "user", "content": "你好"}],
    "temperature": 0.1
}

print("正在测试（禁用 SSL 验证）...")
try:
    response = requests.post(url, json=data, headers=headers, timeout=30, verify=False)
    print(f"✅ 状态码: {response.status_code}")
    print(f"响应内容: {response.text[:300]}")
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()