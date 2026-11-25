# test_ppio_api.py - PPIO API 连接测试（最终调试版）

from langchain_openai import ChatOpenAI
import os
import requests

# ==================== 加载配置 ====================
env_file_path = r"D:\online\program\C_Agent System Design and Development\project\PPIO.env"


def load_env_file(file_path):
    """加载 .env 文件中的环境变量"""
    env_vars = {}
    if not os.path.exists(file_path):
        print(f"❌ 配置文件不存在：{file_path}")
        return env_vars

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                env_vars[key.strip()] = value.strip()

    print(f"📋 .env 文件中找到的变量：{list(env_vars.keys())}")
    return env_vars


# 加载环境变量
env_vars = load_env_file(env_file_path)

# ✅ 改为 PPIO_ 前缀
PPIO_API_KEY = env_vars.get("PPIO_API_KEY")
PPIO_BASE_URL = env_vars.get("PPIO_BASE_URL")
PPIO_MODEL_NAME = env_vars.get("PPIO_MODEL_NAME")

# ==================== 配置检查 ====================
print("=" * 60)
print("=== PPIO API 配置信息检查 ===")
print("=" * 60)
print(f"PPIO_API_KEY 是否存在：{'✅' if PPIO_API_KEY else '❌'}")
print(f"PPIO_BASE_URL: {PPIO_BASE_URL}")
print(f"PPIO_MODEL_NAME: {PPIO_MODEL_NAME}")
print(f"PPIO_MODEL_NAME 长度：{len(PPIO_MODEL_NAME) if PPIO_MODEL_NAME else 0}")
print(f"PPIO_MODEL_NAME repr: {repr(PPIO_MODEL_NAME)}")
print("=" * 60)


# ==================== 网络连通性测试 ====================
def test_api_connectivity(base_url, api_key, model_name):
    """测试 API 地址是否可达"""
    if not base_url or not api_key or not model_name:
        return False, "❌ 配置不完整"

    # ✅ 强制去除空格
    model_name = model_name.strip()

    print(f"\n🔍 发送的 model_name: '{model_name}'")

    try:
        base_url = base_url.rstrip('/')
        # 若 base_url 已含 /v1，只拼 /chat/completions，避免重复路径
        api_url = f"{base_url}/chat/completions" if base_url.endswith("/v1") else f"{base_url}/v1/chat/completions"
        print(f"📡 请求 URL: {api_url}")

        response = requests.post(
            api_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": "你好"}],
                "temperature": 0.1,
                "max_tokens": 50
            },
            timeout=60
        )

        if response.status_code == 200:
            return True, f"✅ 网络连通成功，状态码：{response.status_code}"
        else:
            return False, f"❌ 网络请求失败，状态码：{response.status_code}，响应：{response.text[:200]}"

    except requests.exceptions.Timeout:
        return False, "❌ 请求超时"
    except requests.exceptions.ConnectionError as e:
        return False, f"❌ 连接错误：{str(e)}"
    except Exception as e:
        return False, f"❌ 测试失败：{type(e).__name__}: {str(e)}"


# ==================== 执行连通性测试 ====================
if PPIO_BASE_URL and PPIO_API_KEY and PPIO_MODEL_NAME:
    print("\n=== 网络连通性测试 ===")
    success, message = test_api_connectivity(PPIO_BASE_URL, PPIO_API_KEY, PPIO_MODEL_NAME)
    print(message)
else:
    print("\n❌ 配置不完整，跳过连通性测试")

# ==================== LangChain API 调用测试 ====================
print("\n=== LangChain API 调用测试 ===")
try:
    llm = ChatOpenAI(
        model=PPIO_MODEL_NAME.strip(),  # ✅ 强制去除空格
        api_key=PPIO_API_KEY,
        base_url=PPIO_BASE_URL,
        timeout=60.0,
        max_retries=2,
        temperature=0.1
    )

    print("正在发送请求...")
    response = llm.invoke("你好，请用一句话介绍自己")
    print("✅ API 调用成功：", response.content)

except Exception as e:
    print("❌ API 调用失败：", str(e))
    import traceback

    traceback.print_exc()

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)