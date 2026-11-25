# debug_chroma.py
import os, time
from langchain_community.vectorstores import Chroma

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(SCRIPT_DIR, "chroma_db_lowcost")
print("CHROMA_DIR =", CHROMA_DIR)
print("exists =", os.path.exists(CHROMA_DIR))
if os.path.exists(CHROMA_DIR):
    print("files =", os.listdir(CHROMA_DIR)[:20])

# 用你项目同款 embedding（text2vec）
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch

class LocalText2VecEmbeddings:
    def __init__(self, model_path, device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
    def _encode(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model(**inputs)
            emb = out.last_hidden_state[:, 0].cpu().numpy()[0]
        emb = emb / np.linalg.norm(emb)
        return emb
    def embed_query(self, text): return self._encode(text).tolist()
    def embed_documents(self, texts): return [self._encode(t).tolist() for t in texts]

model_path = os.path.join(SCRIPT_DIR, "models", "text2vec-base-chinese")
emb = LocalText2VecEmbeddings(model_path)

vs = Chroma(embedding_function=emb, persist_directory=CHROMA_DIR)
# 下面这一行能快速判断“库里到底有没有东西”
try:
    print("count =", vs._collection.count())
except Exception as e:
    print("count error:", repr(e))

q = "主轴高速振动且有异响，可能原因？"
t0 = time.time()
docs = vs.similarity_search(q, k=2)
print("search done, secs =", round(time.time() - t0, 2), "docs =", len(docs))
for i, d in enumerate(docs):
    print(i+1, d.page_content[:80])