import os

import requests

API_KEY = os.environ["ARK_API_KEY"]
URL = "https://api.ppio.com/openai/v1/chat/completions"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

data = {
    "model": "qwen/qwen2.5-7b-instruct",
    "messages": [{"role": "user", "content": "你好"}]
}

try:
    # 把 timeout 从 10 秒改成 30 秒
    response = requests.post(URL, headers=headers, json=data, timeout=30)
    print(f"状态码: {response.status_code}")
    print(f"响应内容: {response.text}")
except Exception as e:
    print(f"请求异常: {e}")