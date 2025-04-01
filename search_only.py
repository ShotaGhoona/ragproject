import pickle, faiss
from sentence_transformers import SentenceTransformer

text_chunks = pickle.load(open("sentences.pkl", "rb"))
index = faiss.read_index("faiss_index.bin")
model = SentenceTransformer("all-MiniLM-L6-v2")

question = input("質問をどうぞ：")
query_embedding = model.encode([question])
D, I = index.search(query_embedding, k=3)

print("\n📄 類似文（上位3件）:")
for i, idx in enumerate(I[0]):
    filename, sentence = text_chunks[idx]
    print(f"{i+1}. [{filename}] {sentence}（距離: {D[0][i]:.2f}）")
