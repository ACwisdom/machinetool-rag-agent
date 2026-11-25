# 1. 导入必要库（替换Windows不兼容的signal超时，改用threading实现超时）
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
import os
import warnings
import sys
import time
import logging
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from functools import wraps
import threading
from queue import Queue, Empty

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("fault_diagnosis")

# 彻底消除所有警告
warnings.filterwarnings('ignore')


# 兼容Windows的超时装饰器（用threading替代signal）
def timeout(seconds=30, error_message="Function call timed out"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result_queue = Queue()

            # 定义执行函数的线程
            def worker():
                try:
                    result = func(*args, **kwargs)
                    result_queue.put(('success', result))
                except Exception as e:
                    result_queue.put(('error', e))

            # 启动线程
            thread = threading.Thread(target=worker)
            thread.daemon = True
            thread.start()

            # 等待线程完成或超时
            thread.join(timeout=seconds)

            if thread.is_alive():
                # 超时：返回超时错误
                raise TimeoutError(error_message)
            else:
                # 获取线程执行结果
                try:
                    status, result = result_queue.get_nowait()
                    if status == 'success':
                        return result
                    else:
                        raise result
                except Empty:
                    raise RuntimeError("Function execution returned no result")

        return wrapper

    return decorator


# 加载配置 → 手动读取.env文件
env_file_path = r"D:\online\program\C_Agent System Design and Development\project\PPIO.env"


# 手动解析.env文件
def load_env_file(file_path):
    env_vars = {}
    if not os.path.exists(file_path):
        logger.error(f".env文件不存在：{file_path}")
        sys.exit(1)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                env_vars[key.strip()] = value.strip()
    return env_vars


# 执行手动加载
env_vars = load_env_file(env_file_path)

# 读取环境变量
OPAI_API_KEY = env_vars.get("OPAI_API_KEY")
OPAI_BASE_URL = env_vars.get("OPAI_BASE_URL")
OPAI_MODEL_NAME = env_vars.get("OPAI_MODEL_NAME")

# 校验环境变量
if not OPAI_API_KEY:
    logger.error("未读取到OPAI_API_KEY！请检查.env文件")
    sys.exit(1)
if not OPAI_BASE_URL:
    logger.error("未读取到OPAI_BASE_URL！请检查.env文件")
    sys.exit(1)
if not OPAI_MODEL_NAME:
    logger.error("未读取到OPAI_MODEL_NAME！请检查.env文件")
    sys.exit(1)
logger.info("✅ 环境变量加载成功！")


# 2. 定义状态
class AgentState(TypedDict):
    question: str
    analysis: Optional[str] = None
    retrieved_docs: List = []
    similar_cases: List = []
    final_answer: Optional[str] = None


# 3. 优化本地模型嵌入类（批量编码+缓存+日志）
class LocalText2VecEmbeddings:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        self.embedding_cache = {}  # 缓存已编码的文本，避免重复计算
        if not os.path.exists(model_path):
            raise ValueError(f"模型路径不存在：{model_path}")

        # 加载模型时添加日志
        logger.info(f"开始加载本地模型：{model_path}")
        start_time = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
        logger.info(f"模型加载完成，耗时：{time.time() - start_time:.2f}秒")

    def _encode(self, text: str) -> np.ndarray:
        """单文本编码（带缓存）"""
        # 优先查缓存
        if text in self.embedding_cache:
            return self.embedding_cache[text]

        text = text.strip()
        if not text:
            return np.zeros(768)

        # 分词
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # 模型推理（加速：关闭梯度+推理模式）
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()[0]

        # 归一化
        embeddings = embeddings / np.linalg.norm(embeddings)
        # 存入缓存
        self.embedding_cache[text] = embeddings
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        logger.info(f"开始编码查询文本：{text[:20]}...")
        start_time = time.time()
        result = self._encode(text).tolist()
        logger.info(f"查询文本编码完成，耗时：{time.time() - start_time:.2f}秒")
        return result

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量编码（优化性能）"""
        logger.info(f"开始批量编码{len(texts)}个文档")
        start_time = time.time()
        results = []
        for text in texts:
            results.append(self._encode(text).tolist())
        logger.info(f"批量编码完成，耗时：{time.time() - start_time:.2f}秒")
        return results


# 4. 初始化嵌入模型
try:
    local_model_path = r"D:\online\program\C_Agent System Design and Development\project\models\text2vec-base-chinese"
    embeddings = LocalText2VecEmbeddings(
        model_path=local_model_path,
        device="cpu"
    )
    logger.info("✅ 本地text2vec模型加载成功！")
except Exception as e:
    logger.error(f"本地模型加载失败：{str(e)}，切换到在线模型")
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

# 5. 初始化向量库（添加日志）
logger.info("初始化向量库...")
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_db_lowcost"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
logger.info("✅ 向量库初始化完成！")

# 6. 低成本大模型初始化（添加超时+日志）
try:
    logger.info(f"初始化大模型：{OPAI_MODEL_NAME}")
    llm_lowcost = ChatOpenAI(
        model_name=OPAI_MODEL_NAME,
        temperature=0,
        max_tokens=1000,
        openai_api_key=OPAI_API_KEY,
        openai_api_base=OPAI_BASE_URL,
        timeout=20,  # 关键：API调用超时时间（20秒）
        max_retries=1  # 减少重试次数，避免等待更久
    )
    logger.info("✅ 大模型初始化成功！")
except Exception as e:
    logger.error(f"大模型初始化失败：{str(e)}")
    sys.exit(1)


# 7. 案例库（不变）
def filter_cases_by_symptom(question: str) -> list:
    case_base = [
        "案例1：主轴振动-轴承磨损-换轴承",
        "案例2：主轴异响-润滑不足-加润滑油",
        "案例3：主轴高温-冷却故障-洗管路",
        "案例4：振动+异响-联轴器松动-紧固"
    ]
    keywords = ["振动", "异响", "主轴"]
    matched_cases = [case for case in case_base if any(k in question for k in keywords)]
    return matched_cases[:2]


# 8. 智能体节点（添加日志+兼容Windows的超时控制）
def coordinator_node(state: AgentState):
    logger.info("进入协调器节点，分析用户问题...")
    if any(word in state['question'] for word in ["历史", "类似", "之前"]):
        logger.info("判定需要查询案例库")
        return {"analysis": "need_case"}
    else:
        logger.info("判定需要检索文档")
        return {"analysis": "need_retrieval"}


@timeout(15, "向量检索超时（15秒）")  # 兼容Windows的超时控制
def retrieval_agent_node(state: AgentState):
    logger.info(f"开始检索相关文档，问题：{state['question'][:20]}...")
    start_time = time.time()
    docs = retriever.invoke(state['question'])
    logger.info(f"检索完成，找到{len(docs)}个文档，耗时：{time.time() - start_time:.2f}秒")
    return {"retrieved_docs": docs}


def case_agent_node(state: AgentState):
    logger.info("开始匹配相似案例...")
    start_time = time.time()
    cases = filter_cases_by_symptom(state['question'])
    logger.info(f"案例匹配完成，找到{len(cases)}个案例，耗时：{time.time() - start_time:.2f}秒")
    return {"similar_cases": cases}


@timeout(20, "大模型回答超时（20秒）")  # 兼容Windows的超时控制
def synthesis_node(state: AgentState):
    logger.info("进入合成节点，生成最终回答...")
    start_time = time.time()

    # 拼接上下文
    context = ""
    if state['retrieved_docs']:
        context += "文档：" + "；".join([d.page_content[:50] for d in state['retrieved_docs']])
    if state['similar_cases']:
        if context:
            context += "；"
        context += "案例：" + "；".join(state['similar_cases'])
    if not context:
        context = "无相关文档和案例信息"

    # 构建prompt并调用大模型
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一名严谨的机床维修工程师。请按以下步骤思考：1. 理解症状；2. 列举可能原因；3. 根据提供资料验证原因；4. 给出诊断建议。\n"),
        ("system", "示例1：\n用户问题：主轴振动严重\n回答：根据资料，主轴振动可能由轴承磨损或联轴器松动导致。建议检查轴承磨损情况，必要时更换轴承；同时检查联轴器是否松动，若松动需紧固。\n"),
        ("system", "示例2：\n用户问题：主轴异响\n回答：根据资料，主轴异响可能由润滑不足或轴承损坏导致。建议检查润滑油量，若不足及时添加；同时检查轴承是否损坏，必要时更换轴承。\n"),
        ("system", "必须且仅能依据提供的背景信息回答。如果信息中没有明确依据，请回答‘根据现有资料无法确定，建议检查XXX部位或提供更多信息’。\n"),
        ("system", "背景信息：{context}\n"),
        ("user", "{question}")
    ])
    chain = prompt | llm_lowcost | StrOutputParser()

    try:
        final_answer = chain.invoke({"context": context, "question": state['question']})
        logger.info(f"最终回答生成完成，耗时：{time.time() - start_time:.2f}秒")
        return {"final_answer": final_answer}
    except Exception as e:
        logger.error(f"大模型调用失败：{str(e)}")
        return {"final_answer": f"生成回答失败：{str(e)}"}


# 9. 工作流
logger.info("构建工作流...")
workflow = StateGraph(AgentState)
workflow.add_node("coordinator", coordinator_node)
workflow.add_node("retriever", retrieval_agent_node)
workflow.add_node("case_agent", case_agent_node)
workflow.add_node("synthesizer", synthesis_node)
workflow.set_entry_point("coordinator")


def route_after_coordinator(state):
    return "case_agent" if state['analysis'] == "need_case" else "retriever"


workflow.add_conditional_edges("coordinator", route_after_coordinator, {
    "case_agent": "case_agent",
    "retriever": "retriever"
})
workflow.add_edge("retriever", "synthesizer")
workflow.add_edge("case_agent", "synthesizer")
workflow.add_edge("synthesizer", END)

app = workflow.compile()
logger.info("✅ 工作流构建完成！")


# 10. 初始化向量库（仅一次）
def init_vector_db():
    logger.info("初始化向量库数据...")
    docs = [
        "主轴振动：轴承磨损、联轴器松",
        "主轴异响：润滑不足、轴承坏",
        "主轴维护：换轴承、加润滑油"
    ]
    vectorstore.add_texts(docs)
    vectorstore.persist()
    logger.info("✅ 向量库数据初始化完成！")


# 11. 运行测试（完整异常捕获+详细日志）
if __name__ == "__main__":
    # 首次运行取消注释
    init_vector_db()

    try:
        logger.info("🚀 开始运行故障诊断系统...")
        start_total = time.time()

        # 调用工作流
        result = app.invoke({
            "question": "主轴高速振动+异响，原因？",
            "analysis": None,
            "retrieved_docs": [],
            "similar_cases": [],
            "final_answer": None
        })

        # 输出结果
        logger.info("\n=== 故障诊断结果 ===")
        print(result["final_answer"])
        logger.info(f"✅ 系统运行完成，总耗时：{time.time() - start_total:.2f}秒")

    except TimeoutError as e:
        logger.error(f"\n❌ 系统运行超时：{str(e)}")
        logger.info("💡 建议：检查网络/派欧云API响应速度，或缩短超时时间")
    except Exception as e:
        logger.error(f"\n❌ 系统运行失败：{str(e)}")
        logger.info("\n💡 排查步骤：")
        logger.info("1. 确认派欧云API密钥有效且网络可访问")
        logger.info("2. 检查本地text2vec模型文件是否完整")
        logger.info("3. 确认向量库已初始化（取消init_vector_db()注释运行一次）")
    finally:
        sys.exit(0)