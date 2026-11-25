from llama_index.core import SimpleDirectoryReader

# 加载本地目录下的所有文件
documents = SimpleDirectoryReader("C:\\Users\\Lenovo\\Desktop\\线上初试\\程序\\learn LangChain LlamaIndex").load_data()
print(f"加载文档数量：{len(documents)}")