# 1. 导入必要库（适配LangChain v0.2+和兼容NumPy 1.x）
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
import sys

# 可选：配置代理（不需要则注释掉）
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

# 2. 配置基础路径（支持环境变量覆盖；默认使用 B_ 前缀与项目结构一致）
_script_dir = os.path.dirname(os.path.abspath(__file__))
_b_root = os.path.dirname(os.path.dirname(os.path.dirname(_script_dir)))
_program_root = os.path.dirname(_b_root)
pdf_dir = os.environ.get("B_PDF_DIR", os.path.join(_b_root, "data", "manuals"))
chroma_db_dir = os.environ.get("B_CHROMA_DB_DIR", os.path.join(_b_root, "chroma_db"))
local_model_path = os.environ.get("B_MODEL_PATH", os.path.join(_script_dir, "models", "bge-small-zh-v1.5"))

# 3. 确保目录存在
if not os.path.exists(pdf_dir):
    os.makedirs(pdf_dir)
    print(f"⚠️ 目录 {pdf_dir} 已创建，请将PDF文件放入该目录后重新运行！")
    sys.exit(1)

# 验证本地模型目录（关键）
if not os.path.exists(local_model_path):
    print(f"❌ 本地模型目录不存在：{local_model_path}")
    print("请检查路径是否正确，或确认模型文件已完整下载（包含config.json、pytorch_model.bin等）")
    sys.exit(1)

# 4. 加载并分块PDF文档
def load_and_split_documents():
    """加载PDF并进行文本分块"""
    try:
        loader = DirectoryLoader(
            pdf_dir,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        documents = loader.load()

        if len(documents) == 0:
            print(f"❌ 目录 {pdf_dir} 中未找到任何PDF文件！")
            sys.exit(1)

        # 中文优化的文本分块
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        print(f"✅ 成功加载 {len(documents)} 个PDF，分割为 {len(chunks)} 个文本块")
        return chunks
    except Exception as e:
        print(f"❌ 加载PDF失败：{str(e)}")
        sys.exit(1)

# 5. 初始化嵌入模型并构建向量库（修复model_name为local_model_path）
def build_vector_db(chunks):
    """基于分块文本构建Chroma向量库"""
    try:
        # 关键修复：使用本地模型路径（而非未定义的model_name）
        embedding_model = HuggingFaceEmbeddings(
            model_name=local_model_path,  # 正确引用本地模型路径
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # 构建向量库
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=chroma_db_dir
        )
        print(f"✅ 向量库构建完成！存储路径：{chroma_db_dir}")
        return vectorstore
    except Exception as e:
        print(f"❌ 构建向量库失败：{str(e)}")
        sys.exit(1)

# 6. 主执行逻辑
if __name__ == "__main__":
    try:
        # 先验证关键依赖（新增NumPy版本检查）
        import torch
        import sentence_transformers
        import numpy as np
        print(f"✅ PyTorch版本：{torch.__version__}（要求≥2.6.0）")
        print(f"✅ sentence_transformers版本：{sentence_transformers.__version__}")
        print(f"✅ NumPy版本：{np.__version__}（要求1.x，当前：{np.__version__}）")

        # 加载分块
        chunks = load_and_split_documents()
        # 构建向量库
        vectorstore = build_vector_db(chunks)

        # 测试检索
        query = "测试检索"
        docs = vectorstore.similarity_search(query, k=2)
        print(f"\n📝 检索示例（关键词：{query}）：")
        for i, doc in enumerate(docs):
            print(f"\n结果 {i + 1}：")
            print(f"来源：{doc.metadata.get('source', '未知')} 第{doc.metadata.get('page', '未知')}页")
            print(f"内容：{doc.page_content[:200]}...")

    except Exception as e:
        print(f"❌ 程序执行出错：{str(e)}")
        import traceback
        traceback.print_exc()