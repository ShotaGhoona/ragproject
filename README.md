# 🧠 MyDoc-RAG

RAG（Retrieval-Augmented Generation）を使って、自分のメモ・文書に基づいた質問応答ができるチャットボットです。

## 🚀 機能

- `docs/` フォルダにある `.txt` ファイルをすべて読み込んでベクトル化
- 類似する文を検索（FAISS）
- 関連文 + 質問 を LLM に投げて回答生成（OpenAI API）
- ファイル名つきでどこから情報が来たかも分かる！

---

## 📦 インストール

```bash
git clone https://github.com/yourname/mydoc-rag.git
cd mydoc-rag
pip install -r requirements.txt
```

### ✅ 必須：`.env` ファイル

```env
OPENAI_API_KEY=sk-xxxxx...
```

---

## 🛠 セットアップ手順

### 1. 文書をベクトル化（初回 or 文書更新時）

```bash
python embed_documents.py
```

### 2. 質問 → 回答（RAG）

```bash
python generate_answer.py
```

### 3. 類似文だけ確認したいとき

```bash
python search_only.py
```

---

## 📁 文書例（`docs/` 配下）

```txt
memo.txt:
A社との契約予算は3億円で調整中。
...

note.txt:
新入社員歓迎会は4月15日（金）開催。
...

meeting.txt:
プレゼン提出期限は4月10日。
...
```

---

## 🤖 使用技術

- [FAISS](https://github.com/facebookresearch/faiss) – 高速ベクトル検索
- [sentence-transformers](https://www.sbert.net/) – テキストのベクトル化
- [OpenAI API](https://platform.openai.com/) – 回答生成
- [Python](https://www.python.org/) 3.8+

---

## 📝 ライセンス

MIT License
