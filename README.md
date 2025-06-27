# ARGUS: AI Research Guidance and Understanding System

**ARGUS** is a cloud-deployable research assistant that uses Retrieval-Augmented Generation (RAG) and document similarity to answer technical questions and recommend related papers. It supports PDF ingestion and arXiv metadata parsing.

---

## 🔍 Features

- **RAG Q&A**: Ask natural language questions over research papers (from PDF and arXiv JSON).
- **Recommendations**: Find similar papers using document embedding + FAISS.
- **Cohere embeddings**: Uses `embed-english-v3.0` for document vectorization.
- **Dockerized**: Easy to deploy on any cloud (AWS EC2, etc.).

---

## 📁 Project Structure

```
ARGUS/
│
├── Notebooks/                 # Jupyter notebooks for development
│   ├── step1_RAG_implementation.ipynb
│   ├── step2_recommendation_system.ipynb
│   └── utils_ingest.py        # Shared PDF/JSON processing utilities
│
├── Papers/                    # Local PDF papers
├── archive/                   # arXiv JSON dataset
├── vectorstore/               # FAISS indexes
│   ├── faiss_small/           # For factual Q&A (small chunks)
│   └── faiss_large/           # For reasoning & summaries (large chunks)
│
├── app/                       # FastAPI application
│   ├── main.py                # RAG & recommendation endpoints
│   └── requirements.txt       # API dependencies
│
├── Dockerfile                 # For containerized deployment
└── README.md                  # This file
```

---

## 📥 Ingesting Research Documents

Supports:
- PDF papers (split by page)
- arXiv `.json` snapshot (1 line = 1 paper)

Uses recursive chunking for:
- Small chunks (500 tokens) for **factual** retrieval
- Large chunks (1500 tokens) for **summarization-level** retrieval

---

## 🔠 Embedding & Indexing

Embeddings:
- Model: `embed-english-v3.0` from [Cohere](https://cohere.com/)
- Batching done with retry logic (batch size ≤ 96)

Vector store:
- FAISS
- Two indexes:
  - `faiss_small` for factual QA
  - `faiss_large` for summarization/recommendation

---

## 🧠 RAG Chains

Two RAG chains are created using LangChain:
- `rag_chain_small`: For fast, accurate questions
- `rag_chain_large`: For long-form answers, summaries

---

## 🔗 Recommendation System

Similarity search using FAISS with large embeddings:

```python
def recommend_related_papers(query_text, top_k=5):
    results = vector_store.similarity_search(query_text, k=top_k)
```

Input: Paper title or abstract  
Output: List of top-K similar documents

---

## 🐳 Docker Setup

**Build the image**:

```bash
sudo docker build -t argus-app .
```

**Run the container**:

```bash
sudo docker run -p 8000:8000 argus-app
```

App will be accessible at: `http://localhost:8000`

---

## 🧪 API Endpoints (FastAPI)

Once deployed, supports:

- `POST /rag/qa` → Ask a question over documents  
- `POST /recommend` → Get similar papers to a query

Example request:

```json
{
  "query": "What are the core contributions of ISAAC?"
}
```

---

## 🔐 API Keys

Store your keys in a `.env` file:

```
COHERE_API_KEY=your-api-key-here
```

---

## 📌 Future Work

- Support for real-time PDF upload
- LLM switching (OpenAI, Mistral, Claude)
- UI frontend with Streamlit or React

---

## 👨‍💻 Maintainer

- Sri Harsha  
- MS Computer Engineering, Iowa State University

---


