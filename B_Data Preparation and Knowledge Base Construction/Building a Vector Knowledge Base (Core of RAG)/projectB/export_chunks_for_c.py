# -*- coding: utf-8 -*-
"""
从 B 模块的 Chroma 向量库导出全部文档文本，供 C 模块用 Qdrant 加载（方案2：B 只当数据源）。
运行前请先在本目录或上级执行主脚本构建好 Chroma 向量库。
用法：python export_chunks_for_c.py
输出：本目录下 exported_chunks.json（C 启动时会读取该文件灌入 Qdrant）
"""
import os
import json

# 本脚本所在目录（projectB）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Chroma 库目录：优先环境变量，否则尝试 B 模块下 chroma_db 或与主脚本一致的旧路径（无 B_ 前缀）
_B_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
CHROMA_DB_DIR = os.environ.get(
    "B_CHROMA_DB_DIR",
    os.path.join(_B_ROOT, "chroma_db")
)
if not os.path.isdir(CHROMA_DB_DIR):
    _alt = os.path.join(os.path.dirname(_B_ROOT), "Data Preparation and Knowledge Base Construction", "chroma_db")
    if os.path.isdir(_alt):
        CHROMA_DB_DIR = _alt
OUTPUT_JSON = os.path.join(SCRIPT_DIR, "exported_chunks.json")


def export_chunks():
    try:
        import chromadb
    except ImportError:
        print("请先安装 chromadb: pip install chromadb")
        return False

    if not os.path.isdir(CHROMA_DB_DIR):
        print(f"Chroma 目录不存在：{CHROMA_DB_DIR}")
        print("请先运行「文档解析与分块_向量化与存储_检索器测试.py」生成向量库，或设置环境变量 B_CHROMA_DB_DIR")
        return False

    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    colls = client.list_collections()
    if not colls:
        print("Chroma 中暂无 collection，请先构建向量库")
        return False

    # LangChain Chroma 默认 collection 名为 "langchain"，若有多个则取第一个
    coll = colls[0]
    data = coll.get(include=["documents"])
    documents = data.get("documents") or []
    texts = [t for t in documents if t and isinstance(t, str)]

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=0)

    print(f"已导出 {len(texts)} 条文本到 {OUTPUT_JSON}")
    return True


if __name__ == "__main__":
    export_chunks()
