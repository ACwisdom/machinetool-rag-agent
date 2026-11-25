# -*- coding: utf-8 -*-
"""
评估脚本：对比「默认 / 接 B / 接 D」的答复与耗时。
用法：
  python eval_agent.py                    # 默认（仅 INIT_DOCS，PPIO）
  python eval_agent.py --mode b           # 使用 B 导出知识库（需先运行 B 导出）
  python eval_agent.py --mode d           # 使用 D 微调模型（需设置 merged_model）
  python eval_agent.py --mode default    # 同默认，显式跳过 B 导出
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
try:
    from config import load_dotenv
    load_dotenv()
except Exception:
    pass
os.environ.setdefault("PROGRAM_ROOT", str(ROOT))

C_PROJECT = ROOT / "C_Agent System Design and Development" / "project"
AGENT_SCRIPT = C_PROJECT / "Multi-agent Fault Diagnosis System.py"
QUESTIONS_JSON = ROOT / "eval_questions.json"


def load_agent():
    sys.path.insert(0, str(C_PROJECT))
    import importlib.util
    spec = importlib.util.spec_from_file_location("agent_workflow", str(AGENT_SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.app


def run_eval(mode: str):
    if mode == "default":
        os.environ["SKIP_B_EXPORT"] = "1"
        os.environ.pop("USE_FINETUNED_MODEL", None)
    elif mode == "b":
        os.environ.pop("SKIP_B_EXPORT", None)
        os.environ.pop("USE_FINETUNED_MODEL", None)
    else:  # d
        os.environ.pop("SKIP_B_EXPORT", None)
        os.environ["USE_FINETUNED_MODEL"] = "1"

    if not QUESTIONS_JSON.exists():
        print(f"未找到 {QUESTIONS_JSON}，请在该文件内配置 question / expected_keywords")
        return
    with open(QUESTIONS_JSON, "r", encoding="utf-8") as f:
        items = json.load(f)

    app = load_agent()
    correct = 0
    total = len(items)
    times = []

    for i, item in enumerate(items):
        q = item.get("question", "")
        keywords = item.get("expected_keywords") or []
        t0 = time.perf_counter()
        result = app.invoke({
            "question": q,
            "analysis": None,
            "retrieved_docs": [],
            "similar_cases": [],
            "final_answer": None,
        })
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        answer = (result.get("final_answer") or "").strip()
        hit = any(k in answer for k in keywords) if keywords else True
        if hit:
            correct += 1
        print(f"[{mode}] Q{i+1}: {q[:20]}... -> {answer[:50]}... | {elapsed:.2f}s | {'✓' if hit else '✗'}")

    acc = correct / total * 100 if total else 0
    avg_time = sum(times) / len(times) if times else 0
    print(f"\n[{mode}] 准确率: {correct}/{total} = {acc:.1f}% | 平均耗时: {avg_time:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="评估 Agent：默认 / B / D")
    parser.add_argument("--mode", default="default", choices=["default", "b", "d"], help="default=仅PPIO+默认文档, b=B导出, d=微调模型")
    args = parser.parse_args()
    run_eval(args.mode)


if __name__ == "__main__":
    main()
