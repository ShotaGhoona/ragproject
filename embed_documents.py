import os
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from glob import glob

load_dotenv()
docs_dir = Path("docs")
model = SentenceTransformer("all-MiniLM-L6-v2")

text_chunks = []

# 🔁 全 .txt を処理
for filepath in glob(str(docs_dir / "*.txt")):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
        filename = Path(filepath).name
        sentences = [s.strip() for s in content.split("。") if s.strip()]
        for sentence in sentences:
            text_chunks.append((filename, sentence))

# ベクトル化
sentences_only = [s for (_, s) in text_chunks]
embeddings = model.encode(sentences_only)

# 保存
with open("sentences.pkl", "wb") as f:
    pickle.dump(text_chunks, f)

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)
faiss.write_index(index, "faiss_index.bin")

print("✅ ベクトル化と保存が完了しました！")
