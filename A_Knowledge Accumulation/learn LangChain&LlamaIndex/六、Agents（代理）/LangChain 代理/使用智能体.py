# 关闭所有弃用警告
import warnings

warnings.filterwarnings("ignore")

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from typing import List, Optional, Dict, Any, Union
from langchain_core.outputs import ChatGeneration, ChatResult, GenerationChunk
from langchain_core.runnables import RunnableConfig

# ========== 第一步：先定义极简版的RetrievalQA链（复用之前的逻辑） ==========
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document


# 1. 自定义极简嵌入模型
class SimpleEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[float(ord(c) % 100) for c in text[:10]] for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return [float(ord(c) % 100) for c in text[:10]]


# 2. 自定义极简LLM（用于RetrievalQA）
class SimpleLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        if "LangChain" in prompt and "特点" in prompt:
            return """LangChain框架核心特点：
1. 模块化设计：所有组件都是独立的，可以灵活组合使用
2. 检索增强生成（RAG）：内置强大的检索能力，支持多种向量数据库
3. 多LLM支持：兼容OpenAI、Anthropic、Google等主流大模型
4. 链（Chains）机制：可以将多个操作串联成复杂的工作流
5. 可扩展性：支持自定义组件和第三方集成"""
        return "我是LangChain助手，专注解答LangChain相关问题！"


# 3. 创建RetrievalQA链（知识库问答核心）
documents = [
    Document(
        page_content="LangChain框架的核心特点包括模块化设计、RAG检索增强生成、多LLM支持、Chains链机制、高可扩展性",
        metadata={"source": "langchain_docs.txt"}
    )
]

text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = SimpleEmbeddings()
db = FAISS.from_documents(texts, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 1})

qa_chain = RetrievalQA.from_chain_type(
    llm=SimpleLLM(),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)


# ========== 第二步：修复增强版ChatLLM（适配OpenAI工具调用协议） ==========
class EnhancedChatLLM(BaseChatModel):
    """修复版ChatLLM，适配OpenAI工具调用协议"""

    @property
    def _llm_type(self) -> str:
        return "custom_chat_model"

    def _generate(
            self,
            messages: List[Union[HumanMessage, AIMessage, SystemMessage, ToolMessage]],
            stop: Optional[List[str]] = None,
            run_manager=None, **kwargs,
    ) -> ChatResult:
        """
        修复核心：
        1. 第一阶段：返回符合OpenAI格式的工具调用
        2. 第二阶段：接收ToolMessage，生成自然回答
        """
        # 检查是否有工具返回的结果（ToolMessage）
        has_tool_results = any(isinstance(msg, ToolMessage) for msg in messages)
        user_question = ""

        # 提取用户问题
        for msg in messages:
            if isinstance(msg, HumanMessage):
                user_question = msg.content

        # 第一阶段：无工具结果，返回标准OpenAI工具调用格式
        if not has_tool_results and "LangChain" in user_question and "特点" in user_question:
            # 构建符合OpenAI工具调用协议的响应
            ai_message = AIMessage(
                content="",  # 工具调用时content为空
                tool_calls=[{
                    "name": "知识库问答",
                    "args": {"query": "LangChain 框架的特点是什么？"},
                    "id": "call_001"  # 必须有唯一ID
                }]
            )
            generation = ChatGeneration(message=ai_message)
            return ChatResult(generations=[generation])

        # 第二阶段：有工具结果，生成自然回答
        else:
            # 提取工具返回的结果
            tool_results = ""
            for msg in messages:
                if isinstance(msg, ToolMessage):
                    # 清理工具返回的前缀，提取核心内容
                    tool_results = msg.content.replace("工具返回结果：", "").strip()

            # 生成最终自然语言回答
            final_answer = f"""您好！LangChain框架的主要特点如下：

{tool_results}

这些特点使得LangChain成为构建大语言模型应用的强大框架，尤其在检索增强生成（RAG）场景中表现突出。"""

            ai_message = AIMessage(content=final_answer)
            generation = ChatGeneration(message=ai_message)
            return ChatResult(generations=[generation])

    # 必须实现的抽象方法（修复缺失的方法）
    def _stream(
            self,
            messages: List[Union[HumanMessage, AIMessage, SystemMessage]],
            stop: Optional[List[str]] = None, **kwargs,
    ):
        yield GenerationChunk(message=AIMessage(content=""))

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {}


# ========== 第三步：工具包装器保持不变 ==========
def qa_tool_wrapper(input_dict):
    """工具包装器：调用QA链并格式化返回结果"""
    try:
        result = qa_chain.invoke({"query": input_dict.get("query", "")})
        return f"工具返回结果：{result['result']}"
    except Exception as e:
        return f"工具返回结果：{{'result': '获取信息失败：{str(e)}'}}"


# ========== 第四步：创建Agent并执行 ==========
# 1. 创建修复版ChatLLM
llm = EnhancedChatLLM()

# 2. 创建工具
tools = [
    Tool(
        name="知识库问答",
        func=qa_tool_wrapper,
        description="用于回答关于 LangChain 框架的问题，输入需要包含query参数"
    )
]

# 3. 提示模板保持不变（create_openai_tools_agent会自动处理工具调用）
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个专业的 LangChain 助手，使用提供的工具来回答问题。只能使用给定的工具，严格按照格式调用。调用工具后，请根据工具返回的结果生成自然、友好的回答。"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# 4. 创建代理和执行器
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=2
)

# 5. 执行代理任务
try:
    print("=" * 70)
    print("🚀 执行Agent代理任务...")
    print("=" * 70)

    result = agent_executor.invoke({"input": "LangChain 框架的特点是什么？"})

    print("\n" + "=" * 70)
    print("🎯 Agent最终回答：")
    print(result['output'])
    print("=" * 70)

except Exception as e:
    print(f"\n❌ 执行出错：{str(e)}")
    print("\n💡 简易版执行结果：")
    qa_result = qa_chain.invoke({"query": "LangChain 框架的特点是什么？"})
    print(f"回答：{qa_result['result']}")