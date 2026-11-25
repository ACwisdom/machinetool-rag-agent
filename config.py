# -*- coding: utf-8 -*-
"""
统一配置：从项目根目录 .env 加载环境变量，便于复现与部署。
用法：在 run.py 或需要读取配置的脚本开头执行 load_dotenv()，再通过 os.environ 或 get() 读取。
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROGRAM_ROOT = str(ROOT)


def load_dotenv(env_file: str = None) -> None:
    """从项目根目录 .env 加载变量到 os.environ（若文件存在）。"""
    if env_file is None:
        env_file = ROOT / ".env"
    path = Path(env_file)
    if not path.is_file():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip()
                if k and (k not in os.environ or os.environ.get(k) == ""):
                    os.environ.setdefault(k, v)


def get(key: str, default: str = "") -> str:
    """获取环境变量，优先已加载的 .env。"""
    return os.environ.get(key, default)
