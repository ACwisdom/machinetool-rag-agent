from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

# 创建 PDF 加载器
loader = PyPDFLoader(str((Path(__file__).resolve().parents[3] / 'B_Data Preparation and Knowledge Base Construction/data/manuals/机床用户手册.pdf')))
# 加载文档
documents = loader.load()
# 打印文档信息
print(f"文档数量: {len(documents)}")
print(f"第一页内容: {documents[0].page_content[:200]}...")
print(f"第一页元数据: {documents[0].metadata}")
