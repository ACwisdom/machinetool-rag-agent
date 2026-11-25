from pathlib import Path

# 【终极最终版】彻底禁用LLM，只做纯向量检索，无任何OpenAI依赖
import os
import PyPDF2
# 核心：先禁用LlamaIndex的默认LLM配置
from llama_index.core import Settings
Settings.llm = None  # 关键！直接关掉LLM，不加载任何大模型

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# 改用最基础的检索工具，不依赖任何LLM
from llama_index.core.retrievers import VectorIndexRetriever

# ===================== 第一步：修复异常PDF =====================
def fix_broken_pdf(pdf_path, output_txt="pdf_content.txt"):
    """将异常PDF转为TXT文本"""
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"✅ PDF转TXT成功，保存为：{output_txt}")
        return output_txt
    except Exception as e:
        print(f"❌ PDF处理失败：{e}")
        raise

# 替换为你的PDF路径（必改！）
PDF_PATH = str((Path(__file__).resolve().parents[4] / 'B_Data Preparation and Knowledge Base Construction/data/manuals/机床用户手册.pdf'))
txt_file = fix_broken_pdf(PDF_PATH)

# ===================== 第二步：读取文档+构建向量索引 =====================
documents = SimpleDirectoryReader(input_files=[txt_file]).load_data()
print(f"✅ 读取文档成功，共{len(documents)}个文档对象")

# 本地嵌入模型（无API）
embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
Settings.embed_model = embed_model  # 全局指定嵌入模型，避免重复初始化

# 分割文档+构建向量索引
splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)
nodes = splitter.get_nodes_from_documents(documents)
index = VectorStoreIndex(nodes, show_progress=True)

# ===================== 第三步：纯向量检索（彻底绕开LLM） =====================
# 创建检索器
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=3  # 返回最相似的3段文本
)

# 直接调用检索器，不通过query_engine（核心！避开LLM检查）
question = "What maintenance procedures are described in the machine-tool user manual?"
retrieved_nodes = retriever.retrieve(question)

# ===================== 第四步：输出检索结果 =====================
print("\n========== 纯本地向量检索结果 ==========")
print(f"查询问题：{question}")
print("-" * 60)
for i, node in enumerate(retrieved_nodes):
    print(f"\n📌 匹配结果 {i+1}（相似度：{node.score:.2f}）：")
    # 只显示有效文本，过滤空内容
    clean_text = node.text.strip()
    if clean_text:
        print(clean_text[:600] + ("..." if len(clean_text) > 600 else ""))
    else:
        print("该片段无有效文本")

# 保存向量索引到本地
index.storage_context.persist(persist_dir="./final_llama_index_db")
print("\n✅ 向量索引已保存到本地：./final_llama_index_db")
print("✅ 所有操作完成，无任何OpenAI API依赖！")
