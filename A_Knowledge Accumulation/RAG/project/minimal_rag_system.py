#!/usr/bin/env python3
"""
最小RAG系统实现
实现PDF加载、文本分割、向量编码、ChromaDB存储、检索和大模型生成全流程
"""

import fitz  # PyMuPDF
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import openai
import argparse
import os

# 配置OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")

def load_pdf(file_path: str) -> str:
    """加载PDF文件并提取文本"""
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        raise Exception(f"PDF加载失败: {str(e)}")

def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """将长文本分割为语义片段"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - chunk_overlap)
    return chunks

def init_chroma(persist_directory: Optional[str] = None) -> chromadb.Collection:
    """初始化ChromaDB向量数据库"""
    if persist_directory:
        client = chromadb.PersistentClient(path=persist_directory)
    else:
        client = chromadb.Client(Settings(allow_reset=True))
        client.reset()  # 重置内存数据库
    
    collection = client.get_or_create_collection("rag_collection")
    return collection

def encode_texts(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> List[List[float]]:
    """使用SentenceTransformer生成文本向量"""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts)
    return embeddings

def store_vectors(collection: chromadb.Collection, texts: List[str], embeddings: List[List[float]]):
    """将向量存储到ChromaDB"""
    ids = [f"id_{i}" for i in range(len(texts))]
    collection.add(
        embeddings=embeddings,
        documents=texts,
        ids=ids
    )

def retrieve_relevant(collection: chromadb.Collection, query: str, top_k: int = 3, model_name: str = "all-MiniLM-L6-v2") -> List[str]:
    """检索与查询相关的文本片段"""
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query])
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    return results["documents"][0]

def generate_answer(query: str, context: List[str], model: str = "openai") -> str:
    """结合上下文生成最终答案"""
    context_str = "\n\n".join(context)
    prompt = f"根据以下上下文回答问题：\n\n{context_str}\n\n问题：{query}\n\n答案："
    
    if model == "openai":
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API调用失败: {str(e)}")
    else:
        # 本地Qwen2.5-7B调用示例
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
            model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(**inputs, max_new_tokens=500)
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            raise Exception(f"本地模型调用失败: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="最小RAG系统实现")
    parser.add_argument("--pdf", required=True, help="PDF故障手册路径")
    parser.add_argument("--query", required=True, help="用户查询问题")
    parser.add_argument("--model", choices=["openai", "qwen"], default="openai", help="大模型选择")
    parser.add_argument("--chunk-size", type=int, default=1000, help="文本分割块大小")
    parser.add_argument("--top-k", type=int, default=3, help="检索返回结果数量")
    parser.add_argument("--persist", help="ChromaDB持久化目录")
    
    args = parser.parse_args()
    
    try:
        print("1. 加载PDF文档...")
        text = load_pdf(args.pdf)
        print(f"文档加载完成，总字符数: {len(text)}")
        
        print("2. 文本分割...")
        chunks = split_text(text, chunk_size=args.chunk_size)
        print(f"分割完成，共生成 {len(chunks)} 个片段")
        
        print("3. 初始化向量数据库...")
        collection = init_chroma(args.persist)
        
        print("4. 文本向量编码...")
        embeddings = encode_texts(chunks)
        print(f"向量生成完成，维度: {embeddings.shape}")
        
        print("5. 向量存储...")
        store_vectors(collection, chunks, embeddings)
        print("向量存储成功")
        
        print("6. 检索相关片段...")
        relevant_chunks = retrieve_relevant(collection, args.query, top_k=args.top_k)
        print(f"检索到 {len(relevant_chunks)} 个相关片段")
        
        print("\n检索到的相关片段：")
        for i, chunk in enumerate(relevant_chunks):
            print(f"\n片段{i+1}: {chunk[:300]}...")
        
        print("\n7. 生成最终答案...")
        answer = generate_answer(args.query, relevant_chunks, model=args.model)
        
        print("\n" + "="*50)
        print("最终答案：")
        print(answer)
        print("="*50)
        
    except Exception as e:
        print(f"错误: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()