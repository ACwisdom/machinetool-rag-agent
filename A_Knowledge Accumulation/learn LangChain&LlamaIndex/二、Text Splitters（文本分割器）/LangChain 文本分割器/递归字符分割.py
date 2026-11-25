from pathlib import Path

# 1. 导入必要的工具：文件读取器 + 文本分割器
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 2. 读取文件，生成documents变量
loader = PyPDFLoader(str((Path(__file__).resolve().parents[4] / 'B_Data Preparation and Knowledge Base Construction/data/manuals/机床用户手册.pdf')))
documents = loader.load()  # 这一步会生成后续分割需要的documents

# 3. 初始化递归字符分割器
splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],  # 分割符
    chunk_size=1000,  # 每个文本块的字符数
    chunk_overlap=200  # 文本块之间的重叠字符数
)

# 4. 分割文档
split_docs = splitter.split_documents(documents)
print(f"分割后文档块数量：{len(split_docs)}")
