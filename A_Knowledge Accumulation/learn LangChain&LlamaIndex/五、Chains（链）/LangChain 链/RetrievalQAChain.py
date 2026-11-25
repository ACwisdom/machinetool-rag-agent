# 关闭所有弃用警告
import warnings

warnings.filterwarnings("ignore")

from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import CharacterTextSplitter
from typing import List, Optional


# ========== 1. 自定义极简嵌入模型（无需下载，纯本地） ==========
class SimpleEmbeddings(Embeddings):
    """自定义简单嵌入模型，避免下载HuggingFace模型"""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """为文档生成简单的嵌入向量（固定长度，仅用于演示）"""
        return [[float(ord(c) % 100) for c in text[:10]] for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """为查询生成简单的嵌入向量"""
        return [float(ord(c) % 100) for c in text[:10]]


# ========== 2. 自定义极简LLM（无需API/本地模型，直接返回固定回答） ==========
class SimpleLLM(LLM):
    """自定义简单LLM，避免依赖Ollama/OpenAI"""

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager=None,
            **kwargs,
    ) -> str:
        """处理提示并返回固定回答（适配LangChain接口）"""
        # 从prompt中提取核心问题，返回预设的LangChain特点回答
        if "LangChain" in prompt and "特点" in prompt:
            return """LangChain框架的核心特点：
1. 模块化设计：组件独立，可灵活组合
2. 检索增强生成（RAG）：内置强大的检索能力
3. 多LLM支持：兼容OpenAI、Anthropic等主流大模型
4. 链（Chains）机制：串联多个操作形成复杂工作流
5. 可扩展性：支持自定义组件和第三方集成
6. 工具调用：支持调用外部工具和API
7. 记忆功能：维持对话上下文理解"""
        return "我正在学习LangChain的RetrievalQA链用法！"


# ========== 3. 准备测试文档 ==========
# 直接定义文档内容，无需文件读写
documents = [
    {
        "page_content": """LangChain框架的核心特点：
1. 模块化设计：所有组件都是独立的，可以灵活组合使用
2. 检索增强生成（RAG）：内置强大的检索能力，支持多种向量数据库
3. 多LLM支持：兼容OpenAI、Anthropic、Google等主流大模型""",
        "metadata": {"source": "test_doc.txt"}
    },
    {
        "page_content": """LangChain额外特点：
4. 链（Chains）机制：可以将多个操作串联成复杂的工作流
5. 可扩展性：支持自定义组件和第三方集成
6. 工具调用：支持调用外部工具和API
7. 记忆功能：支持对话记忆，维持上下文理解""",
        "metadata": {"source": "test_doc.txt"}
    }
]

# 转换为LangChain的Document格式
from langchain_core.documents import Document

docs = [Document(page_content=doc["page_content"], metadata=doc["metadata"]) for doc in documents]

# 分割文档（演示用）
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
texts = text_splitter.split_documents(docs)

# ========== 4. 创建向量库和检索器 ==========
# 使用自定义嵌入模型
embeddings = SimpleEmbeddings()
db = FAISS.from_documents(texts, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 2})

# ========== 5. 创建LLM和问答链 ==========
llm = SimpleLLM()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# ========== 6. 执行问答并输出结果 ==========
try:
    # 执行问答
    result = qa_chain.invoke({"query": "LangChain 框架的特点"})

    # 输出结果
    print("=" * 60)
    print("🎯 问答结果：")
    print(result['result'])
    print("\n📄 来源文档：")
    for i, doc in enumerate(result["source_documents"]):
        print(f"\n  文档{i + 1}：")
        print(f"  内容：{doc.page_content[:200]}...")
        print(f"  来源：{doc.metadata['source']}")
    print("=" * 60)

except Exception as e:
    print(f"❌ 执行出错：{str(e)}")