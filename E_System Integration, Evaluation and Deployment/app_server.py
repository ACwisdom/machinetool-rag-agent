# -*- coding: utf-8 -*-
"""
诊断 API 入口：Agent 在独立子进程中运行，通过队列通信，避免 Chroma/SQLite 在 uvicorn 进程内崩掉。
可通过根目录 run.py --api 或本目录下 uvicorn app_server:app 启动。
"""
import os
import sys
import asyncio
import importlib.util
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process, Queue

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 项目根目录（兼容 run.py 设置的 PROGRAM_ROOT 或按路径推导）
ROOT = Path(os.environ.get("PROGRAM_ROOT", Path(__file__).resolve().parent.parent))
C_PROJECT = ROOT / "C_Agent System Design and Development" / "project"
AGENT_SCRIPT = C_PROJECT / "Multi-agent Fault Diagnosis System.py"

# 确保子进程能拿到项目根
os.environ.setdefault("PROGRAM_ROOT", str(ROOT))

# 请求/结果队列与 worker 进程（在 startup 里初始化）
request_queue: Queue = None
result_queue: Queue = None
agent_process: Process = None
_blocking_executor = ThreadPoolExecutor(max_workers=2)


def _worker_main(req_q: Queue, res_q: Queue) -> None:
    """在独立进程中加载 Agent 并循环处理请求。"""
    sys.path.insert(0, str(C_PROJECT))
    spec = importlib.util.spec_from_file_location("agent_workflow", str(AGENT_SCRIPT))
    agent_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = agent_module
    spec.loader.exec_module(agent_module)
    app_impl = agent_module.app

    while True:
        question = req_q.get()
        if question is None:
            break
        try:
            result = app_impl.invoke({
                "question": question,
                "analysis": None,
                "retrieved_docs": [],
                "similar_cases": [],
                "final_answer": None,
            })
            docs = []
            for d in result.get("retrieved_docs", []):
                docs.append({
                    "content": getattr(d, "page_content", str(d))[:200],
                    "metadata": getattr(d, "metadata", {}),
                })
            res_q.put({
                "final_answer": result.get("final_answer", ""),
                "retrieved_docs": docs,
            })
        except Exception as e:
            res_q.put({"error": str(e)})


app = FastAPI(title="机床故障诊断AI助手 API")


@app.on_event("startup")
async def startup():
    global request_queue, result_queue, agent_process
    request_queue = Queue()
    result_queue = Queue()
    agent_process = Process(target=_worker_main, args=(request_queue, result_queue))
    agent_process.start()


@app.on_event("shutdown")
async def shutdown():
    global request_queue, agent_process
    if request_queue is not None and agent_process is not None:
        request_queue.put(None)
        agent_process.join(timeout=5)
        if agent_process.is_alive():
            agent_process.terminate()


class DiagnosisRequest(BaseModel):
    question: str


class DiagnosisResponse(BaseModel):
    final_answer: str
    retrieved_docs: list = []


@app.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose(request: DiagnosisRequest):
    if request_queue is None or result_queue is None:
        raise HTTPException(status_code=503, detail="Agent worker not ready")
    request_queue.put(request.question)
    loop = asyncio.get_event_loop()

    def get_result():
        return result_queue.get(timeout=60)

    try:
        result = await loop.run_in_executor(_blocking_executor, get_result)
    except Exception as e:
        raise HTTPException(status_code=504, detail=f"Agent timeout or error: {e}")

    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])
    return DiagnosisResponse(
        final_answer=result.get("final_answer", ""),
        retrieved_docs=result.get("retrieved_docs", []),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
