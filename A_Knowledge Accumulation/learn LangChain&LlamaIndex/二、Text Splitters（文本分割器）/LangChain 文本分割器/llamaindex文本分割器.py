# 1. 从core导入SimpleDirectoryReader（兼容更多版本）
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

# 2. 读取当前目录下的PDF文件（用目录读取器指定单个文件）
documents = SimpleDirectoryReader(input_dir=".", required_exts=[".pdf"]).load_data()

# 3. 创建句子分割器
node_parser = SentenceSplitter(
    chunk_size=1024,
    chunk_overlap=200
)

# 4. 分割文档
nodes = node_parser.get_nodes_from_documents(documents)
print(f"分割后节点数量：{len(nodes)}")