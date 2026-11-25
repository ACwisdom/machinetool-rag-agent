# -*- coding: utf-8 -*-
"""
Streamlit 诊断界面。
推荐入口：根目录 run.py --api 使用 app_server（子进程 Agent）；若需 Streamlit 请在本目录执行：
  streamlit run app.py
  （需设置环境变量 PROGRAM_ROOT 为项目根目录，或在本目录的上一级为 program 时自动推导）
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
        raise FileNotFoundError(f"未找到 Agent 脚本：{AGENT_SCRIPT}，请设置 PROGRAM_ROOT 或从 program 根目录运行 run.py --api")
    sys.path.insert(0, str(C_PROJECT))
    spec = importlib.util.spec_from_file_location("agent_workflow", str(AGENT_SCRIPT))
    agent_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = agent_module
    spec.loader.exec_module(agent_module)
    return agent_module.app

import streamlit as st
agent_app = _load_agent()

st.title("🏭 机床故障诊断AI助手")

question = st.text_area("请输入故障现象或问题：", height=100)

if st.button("开始诊断"):
    with st.spinner("AI专家正在会诊..."):
        result = agent_app.invoke({"question": question, "analysis": None, "retrieved_docs": [], "similar_cases": [], "final_answer": None})
    st.markdown("### 诊断意见")
    st.write(result["final_answer"])
    with st.expander("查看检索到的参考资料"):
        for doc in result.get("retrieved_docs", []):
            content = getattr(doc, "page_content", str(doc))
            st.caption(content[:300])
