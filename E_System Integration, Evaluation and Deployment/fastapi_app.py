# -*- coding: utf-8 -*-
"""
FastAPI 诊断接口（单进程，直接加载 C 的 Agent）。
推荐入口：根目录 run.py --api 使用 app_server（子进程更稳定）。若需本文件：
  uvicorn fastapi_app:app --host 0.0.0.0 --port 8000
  （需设置环境变量 PROGRAM_ROOT 为项目根目录）
"""
import os
import sys
import importlib.util
from pathlib import Path

ROOT = Path(os.environ.get("PROGRAM_ROOT", Path(__file__).resolve().parent.parent))
C_PROJECT = ROOT / "C_Agent System Design and Development" / "project"
AGENT_SCRIPT = C_PROJECT / "Multi-agent Fault Diagnosis System.py"

def _load_agent():
    if not AGENT_SCRIPT.exists():
        raise FileNotFoundError(f"未找到 Agent 脚本：{AGENT_SCRIPT}，请设置 PROGRAM_ROOT")
    sys.path.insert(0, str(C_PROJECT))
    spec = importlib.util.spec_from_file_location("agent_workflow", str(AGENT_SCRIPT))
    agent_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = agent_module
    spec.loader.exec_module(agent_module)
    return agent_module.app

agent_app = _load_agent()

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="机床故障诊断AI助手 API")

class DiagnosisRequest(BaseModel):
    question: str

class DiagnosisResponse(BaseModel):
    final_answer: str
    retrieved_docs: list = []

@app.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose(request: DiagnosisRequest):
    result = agent_app.invoke({
        "question": request.question,
        "analysis": None,
        "retrieved_docs": [],
        "similar_cases": [],
        "final_answer": None,
    })
    docs = []
    for d in result.get("retrieved_docs", []):
        content = getattr(d, "page_content", str(d))[:200]
        meta = getattr(d, "metadata", {})
        docs.append({"content": content, "metadata": meta})
    return DiagnosisResponse(
        final_answer=result.get("final_answer", ""),
        retrieved_docs=docs,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
