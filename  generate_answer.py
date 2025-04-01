import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 検索用データ読み込み
text_chunks = pickle.load(open("sentences.pkl", "rb"))  # [(filename, sentence)]
index = faiss.read_index("faiss_index.bin")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 質問
question = input("💬 質問をどうぞ：")
query_embedding = model.encode([question])
D, I = index.search(query_embedding, k=3)

# 関連文を整形
relevant = [text_chunks[idx] for idx in I[0]]
context = "\n".join([f"[{filename}] {sentence}" for filename, sentence in relevant])

# プロンプト作成
prompt = f"""以下の情報に基づいて、質問に回答してください。

### 参考情報:
{context}

### 質問:
{question}

### 回答:"""

# LLMに問い合わせ
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}],
)

# 回答表示
answer = response.choices[0].message.content
print("\n🧠 回答:")
print(answer)
