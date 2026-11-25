# 1. 导入必要库（替换Windows不兼容的signal超时，改用threading实现超时）
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional, Any
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.local.qdrant_local import QdrantLocal
from langchain_community.vectorstores import Qdrant


class _QdrantClientWithSearch(QdrantClient):
    """内存模式下 QdrantClient 对外没有 search()，LangChain 需要 search()，子类把 search 委托给内部的 QdrantLocal。"""

    def search(self, *args, **kwargs):
        if isinstance(self._client, QdrantLocal):
            return self._client.search(*args, **kwargs)
        raise NotImplementedError("仅支持 :memory: 本地模式")


class _FinetunedChatModel(SimpleChatModel):
    """D 模块微调模型的 LangChain 封装，供 synthesis 链使用。"""

    model_path: str
    _integrator: Optional[Any] = None

    @property
    def _llm_type(self) -> str:
        return "finetuned_local"

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if self._integrator is None:
            raise RuntimeError("微调模型未加载，请先调用 load_finetuned_model()")
        instruction = "你是一位专业的故障诊断专家，请根据提供的信息进行准确诊断。"
        input_parts = []
        for m in messages:
            if isinstance(m, SystemMessage):
                instruction = m.content if isinstance(m.content, str) else str(m.content)
            elif isinstance(m, HumanMessage):
                input_parts.append(m.content if isinstance(m.content, str) else str(m.content))
        input_text = "\n\n".join(input_parts) if input_parts else ""
        return self._integrator.call_llm_api(instruction=instruction, input_text=input_text)


def _load_finetuned_model(model_path: str, program_root: str):
    """懒加载 D 模块 ModelIntegrator 并返回 LangChain 封装。"""
    import sys
    d_module = os.path.join(
        program_root,
        "D_Model Fine-tuning and Performance Improvement",
        "Perform SFT_LoRA fine-tuning using Llama Factory",
    )
    if not os.path.isdir(d_module):
        raise FileNotFoundError(f"D 模块路径不存在: {d_module}")
    if d_module not in sys.path:
        sys.path.insert(0, d_module)
    from integrate_to_agent import ModelIntegrator
    integrator = ModelIntegrator(model_path=model_path)
    integrator.load_finetuned_model()
    wrapper = _FinetunedChatModel(model_path=model_path)
    wrapper._integrator = integrator
    return wrapper


from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
import os
import json
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


# =============================================================================
# 2. 加载配置 → 根目录 .env（可选）+ C 模块 PPIO.env
# =============================================================================
_c_script_dir = os.path.dirname(os.path.abspath(__file__))
_program_root = os.path.dirname(os.path.dirname(_c_script_dir))
env_file_path = os.path.join(_c_script_dir, "PPIO.env")


def load_env_file(file_path, required=True):
    env_vars = {}
    if not os.path.exists(file_path):
        if required:
            logger.error(f".env文件不存在：{file_path}")
            sys.exit(1)
        return env_vars
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                env_vars[k.strip()] = v.strip()
    return env_vars


# 先加载根目录 .env（若存在），再加载 PPIO.env，后者覆盖
env_vars = {}
_root_env = os.path.join(_program_root, ".env")
if os.path.isfile(_root_env):
    env_vars = load_env_file(_root_env, required=False)
env_vars.update(load_env_file(env_file_path))
logger.info(f"已加载的环境变量：{list(env_vars.keys())}")

# ✅ 读取环境变量（改为 PPIO_ 前缀，与.env文件一致）
PPIO_API_KEY = env_vars.get("PPIO_API_KEY")
PPIO_BASE_URL = env_vars.get("PPIO_BASE_URL")
PPIO_MODEL_NAME = env_vars.get("PPIO_MODEL_NAME")

# ✅ 添加调试输出
logger.info("=" * 50)
logger.info("环境变量检查：")
logger.info(f"  PPIO_API_KEY: {'✅ 已加载' if PPIO_API_KEY else '❌ 未找到'}")
logger.info(f"  PPIO_BASE_URL: {PPIO_BASE_URL if PPIO_BASE_URL else '❌ 未找到'}")
logger.info(f"  PPIO_MODEL_NAME: {PPIO_MODEL_NAME if PPIO_MODEL_NAME else '❌ 未找到'}")
logger.info("=" * 50)

# 校验环境变量
if not PPIO_API_KEY:
    logger.error("未读取到 PPIO_API_KEY！请检查.env文件中是否使用 PPIO_API_KEY")
    sys.exit(1)
if not PPIO_BASE_URL:
    logger.error("未读取到 PPIO_BASE_URL！请检查.env文件中是否使用 PPIO_BASE_URL")
    sys.exit(1)
if not PPIO_MODEL_NAME:
    logger.error("未读取到 PPIO_MODEL_NAME！请检查.env文件中是否使用 PPIO_MODEL_NAME")
    sys.exit(1)
logger.info("✅ 环境变量加载成功！")


# 3. 定义状态
class AgentState(TypedDict):
    question: str
    analysis: Optional[str] = None
    retrieved_docs: List = []
    similar_cases: List = []
    final_answer: Optional[str] = None


# 4. 优化本地模型嵌入类（批量编码 + 缓存 + 日志），继承 Embeddings 以便 Qdrant 使用 embed_documents
class LocalText2VecEmbeddings(Embeddings):
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

        # 模型推理（加速：关闭梯度 + 推理模式）
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


# 5. 初始化嵌入模型
try:
    local_model_path = r"D:\online\program\C_Agent System Design and Development\project\models\text2vec-base-chinese"
    embeddings = LocalText2VecEmbeddings(
        model_path=local_model_path,
        device="cpu"
    )
    logger.info("✅ 本地 text2vec 模型加载成功！")
except Exception as e:
    logger.error(f"本地模型加载失败：{str(e)}，切换到在线模型")
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

# 6. 向量库：Qdrant 内存模式 + 可选加载 B 模块导出（方案2：B 只当数据源）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROGRAM_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
INIT_DOCS = [
    "主轴振动：轴承磨损、联轴器松",
    "主轴异响：润滑不足、轴承坏",
    "主轴维护：换轴承、加润滑油",
]
B_EXPORT_PATH = os.path.join(
    PROGRAM_ROOT,
    "B_Data Preparation and Knowledge Base Construction",
    "Building a Vector Knowledge Base (Core of RAG)",
    "projectB",
    "exported_chunks.json",
)


def _load_b_export():
    """若存在 B 导出的 JSON 且未设置 SKIP_B_EXPORT，返回文本列表，否则返回 None。"""
    if os.environ.get("SKIP_B_EXPORT", "").strip().lower() in ("1", "true", "yes"):
        return None
    if not os.path.isfile(B_EXPORT_PATH):
        return None
    try:
        with open(B_EXPORT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and len(data) > 0:
            return [str(x) for x in data if x]
        return None
    except Exception as e:
        logger.warning("读取 B 导出文件失败，将使用默认 INIT_DOCS: %s", e)
        return None


logger.info("初始化向量库（Qdrant 内存）...")
docs_to_add = _load_b_export()
if docs_to_add is None:
    docs_to_add = INIT_DOCS
    logger.info("使用默认 INIT_DOCS（%d 条）", len(docs_to_add))
else:
    logger.info("已加载 B 导出文档（%d 条）", len(docs_to_add))
_vector_size = len(embeddings.embed_documents(docs_to_add[:1])[0])
_qdrant_client = _QdrantClientWithSearch(":memory:")
_qdrant_client.create_collection(
    collection_name="fault_diagnosis",
    vectors_config=VectorParams(size=_vector_size, distance=Distance.COSINE),
)
vectorstore = Qdrant(client=_qdrant_client, collection_name="fault_diagnosis", embeddings=embeddings)
vectorstore.add_texts(docs_to_add)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
logger.info("✅ 向量库初始化完成！")

# =============================================================================
# 7. 大模型初始化：可选 PPIO 或 D 模块微调模型（环境变量 USE_FINETUNED_MODEL=1 + FINETUNED_MODEL_PATH）
# =============================================================================
USE_FINETUNED_MODEL = os.environ.get("USE_FINETUNED_MODEL", "").strip().lower() in ("1", "true", "yes")
FINETUNED_MODEL_PATH = os.environ.get("FINETUNED_MODEL_PATH", "").strip()
if USE_FINETUNED_MODEL and not FINETUNED_MODEL_PATH:
    FINETUNED_MODEL_PATH = os.path.join(
        PROGRAM_ROOT,
        "D_Model Fine-tuning and Performance Improvement",
        "Perform SFT_LoRA fine-tuning using Llama Factory",
        "merged_model",
    )

try:
    if USE_FINETUNED_MODEL and os.path.isdir(FINETUNED_MODEL_PATH):
        logger.info("使用 D 模块微调模型：%s", FINETUNED_MODEL_PATH)
        llm_lowcost = _load_finetuned_model(FINETUNED_MODEL_PATH, PROGRAM_ROOT)
        logger.info("✅ 微调模型加载成功！")
    else:
        if USE_FINETUNED_MODEL:
            logger.warning("未找到微调模型路径 %s，回退到 PPIO", FINETUNED_MODEL_PATH)
        logger.info(f"初始化大模型：{PPIO_MODEL_NAME}")
        logger.info(f"Base URL: {PPIO_BASE_URL}")
        llm_lowcost = ChatOpenAI(
            model=PPIO_MODEL_NAME,
            api_key=PPIO_API_KEY,
            base_url=PPIO_BASE_URL,
            temperature=0,
            max_tokens=1000,
            timeout=60,
            max_retries=1,
        )
        logger.info("✅ 大模型初始化成功！")
        logger.info("正在测试 API 连接...")
        test_response = llm_lowcost.invoke("测试")
        logger.info(f"✅ API 测试成功！响应：{test_response.content[:50]}...")
except Exception as e:
    logger.error(f"大模型初始化/测试失败：{str(e)}")
    logger.error("\n💡 排查建议：")
    logger.error("1. PPIO 模式：确认 PPIO.env 中 PPIO_API_KEY, PPIO_BASE_URL, PPIO_MODEL_NAME")
    logger.error("2. 微调模式：设置 USE_FINETUNED_MODEL=1 和 FINETUNED_MODEL_PATH（或使用默认 merged_model）")
    sys.exit(1)


# 8. 案例库（从 case_base.json 加载，与 RAG/微调数据格式可打通；缺省用内置列表）
CASE_BASE_PATH = os.path.join(SCRIPT_DIR, "case_base.json")
_DEFAULT_CASES = [
    {"keywords": ["振动", "异响", "主轴"], "text": "案例 1：主轴振动 - 轴承磨损 - 换轴承"},
    {"keywords": ["异响", "主轴"], "text": "案例 2：主轴异响 - 润滑不足 - 加润滑油"},
    {"keywords": ["高温", "主轴"], "text": "案例 3：主轴高温 - 冷却故障 - 洗管路"},
    {"keywords": ["振动", "异响", "联轴器"], "text": "案例 4：振动 + 异响 - 联轴器松动 - 紧固"},
]


def _load_case_base():
    if os.path.isfile(CASE_BASE_PATH):
        try:
            with open(CASE_BASE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                return data
        except Exception as e:
            logger.warning("加载 case_base.json 失败，使用内置案例: %s", e)
    return _DEFAULT_CASES


def filter_cases_by_symptom(question: str) -> list:
    case_base = _load_case_base()
    matched = [c.get("text", str(c)) for c in case_base if any(k in question for k in c.get("keywords", []))]
    return matched[:2]


# 9. 智能体节点（添加日志 + 兼容 Windows 的超时控制）
def coordinator_node(state: AgentState):
    logger.info("进入协调器节点，分析用户问题...")
    if any(word in state['question'] for word in ["历史", "类似", "之前"]):
        logger.info("判定需要查询案例库")
        return {"analysis": "need_case"}
    else:
        logger.info("判定需要检索文档")
        return {"analysis": "need_retrieval"}


def retrieval_agent_node(state: AgentState):
    """检索节点：使用全局 Qdrant 内存向量库，无文件锁问题。"""
    logger.info(f"开始检索相关文档，问题：{state['question'][:20]}...")
    start_time = time.time()
    try:
        docs = retriever.invoke(state["question"])
    except Exception as e:
        logger.exception("向量检索异常: %s", e)
        docs = []
    logger.info(f"检索完成，找到{len(docs)}个文档，耗时：{time.time() - start_time:.2f}秒")
    return {"retrieved_docs": docs}


def case_agent_node(state: AgentState):
    logger.info("开始匹配相似案例...")
    start_time = time.time()
    cases = filter_cases_by_symptom(state['question'])
    logger.info(f"案例匹配完成，找到{len(cases)}个案例，耗时：{time.time() - start_time:.2f}秒")
    return {"similar_cases": cases}


def _strip_think_and_truncate(text: str, max_len: int = 80) -> str:
    """去掉模型输出中的 <think> 思考过程，并做长度截断。"""
    if not isinstance(text, str):
        return text
    start = text.find("<think>")
    end = text.find("</think>")
    if start != -1 and end != -1 and end > start:
        text = text[end + len("</think>") :].lstrip()
    text = text.strip()
    if len(text) > max_len:
        text = text[:max_len]
    return text


@timeout(60, "大模型回答超时（60 秒）")
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

    # 构建 prompt 并调用大模型
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "你是机床专家，仅基于{context}回答。"
            "只输出最终结论，不要输出任何思考过程或 <think> 标签。"
            "请简洁说明可能原因和排查步骤，控制在 50 字以内。",
        ),
        ("user", "{question}")
    ])
    chain = prompt | llm_lowcost | StrOutputParser()

    try:
        final_answer = chain.invoke({"context": context, "question": state['question']})
        final_answer = _strip_think_and_truncate(final_answer)
        logger.info(f"最终回答生成完成，耗时：{time.time() - start_time:.2f}秒")
        return {"final_answer": final_answer}
    except Exception as e:
        logger.error(f"大模型调用失败：{str(e)}")
        return {"final_answer": f"生成回答失败：{str(e)}"}


# 10. 工作流
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


# 11. 初始化向量库（Qdrant 内存模式：启动时已 add_texts，此处仅兼容旧调用）
def init_vector_db():
    logger.info("向量库（Qdrant 内存）已在启动时初始化，无需重复执行。")


# 12. 运行测试（完整异常捕获 + 详细日志）
if __name__ == "__main__":
    # 首次运行取消注释
    #init_vector_db()

    try:
        logger.info("🚀 开始运行故障诊断系统...")
        start_total = time.time()

        # 调用工作流
        result = app.invoke({
            "question": "主轴高速振动 + 异响，原因？",
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
        logger.info("💡 建议：检查网络/派欧云 API 响应速度，或缩短超时时间")
    except Exception as e:
        logger.error(f"\n❌ 系统运行失败：{str(e)}")
        logger.info("\n💡 排查步骤：")
        logger.info("1. 确认派欧云 API 密钥有效且网络可访问")
        logger.info("2. 检查本地 text2vec 模型文件是否完整")
        logger.info("3. 确认向量库已初始化（取消 init_vector_db() 注释运行一次）")
    finally:
        sys.exit(0)