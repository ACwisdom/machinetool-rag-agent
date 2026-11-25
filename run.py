# -*- coding: utf-8 -*-
"""
一键运行脚本：从项目根目录启动 Agent 演示或 API 服务。
用法：
  python run.py           # 运行多智能体故障诊断演示（C 模块）
  python run.py --api     # 启动 FastAPI 诊断服务（E 模块，默认 http://0.0.0.0:8000）
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

# 项目根目录（run.py 所在目录）
ROOT = Path(__file__).resolve().parent

# 统一配置：加载根目录 .env（若存在）
try:
    from config import load_dotenv
    load_dotenv()
except Exception:
    pass

# 各模块路径
C_PROJECT = ROOT / "C_Agent System Design and Development" / "project"
E_MODULE = ROOT / "E_System Integration, Evaluation and Deployment"
AGENT_SCRIPT = C_PROJECT / "Multi-agent Fault Diagnosis System.py"


def run_agent_demo():
    """运行 C 模块：多智能体故障诊断演示（单次调用并打印结果）"""
    if not AGENT_SCRIPT.exists():
        print(f"未找到 Agent 脚本：{AGENT_SCRIPT}")
        sys.exit(1)
    os.chdir(C_PROJECT)
    env = os.environ.copy()
    env.setdefault("PROGRAM_ROOT", str(ROOT))
    return subprocess.run(
        [sys.executable, str(AGENT_SCRIPT)],
        cwd=str(C_PROJECT),
        env=env,
    ).returncode


def run_api(host="0.0.0.0", port=8000):
    """运行 E 模块：FastAPI 诊断 API"""
    app_server = E_MODULE / "app_server.py"
    if not app_server.exists():
        print(f"未找到 API 入口：{app_server}")
        sys.exit(1)
    os.chdir(E_MODULE)
    # 让 E 模块能解析到项目根目录（用于加载 C 的 Agent）
    env = os.environ.copy()
    env["PROGRAM_ROOT"] = str(ROOT)
    return subprocess.run(
        [sys.executable, "-m", "uvicorn", "app_server:app", "--host", host, "--port", str(port)],
        cwd=str(E_MODULE),
        env=env,
    ).returncode


def main():
    parser = argparse.ArgumentParser(description="一键运行：Agent 演示或 API 服务")
    parser.add_argument("--api", action="store_true", help="启动 FastAPI 诊断服务")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"), help="API 监听地址")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")), help="API 端口")
    args = parser.parse_args()

    if args.api:
        sys.exit(run_api(host=args.host, port=args.port))
    else:
        sys.exit(run_agent_demo())


if __name__ == "__main__":
    main()
