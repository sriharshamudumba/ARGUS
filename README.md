# ARGUS: AI Research Guidance and Understanding System

**ARGUS** is a cloud-deployable research assistant designed to help researchers, students, and engineers explore and understand scientific papers.  
It integrates **Retrieval-Augmented Generation (RAG)** and **semantic similarity search** to answer natural language questions about research documents and recommend related work.  
The system supports both **PDF ingestion** and **arXiv metadata parsing**, enabling flexible and large-scale research discovery.

---

## Overview

ARGUS combines modern language models, document embeddings, and retrieval systems to bridge the gap between human queries and scientific literature.  
It builds an intelligent interface over research documents, allowing users to query papers in natural language, generate summaries, and explore related publications—all through an efficient, containerized deployment.

---

## Key Features

### 1. Retrieval-Augmented Generation (RAG) Q&A
- Ask questions directly over a collection of research papers.  
- Works with both **PDF files** and **arXiv JSON snapshots**.  
- Uses two retrieval granularities:
  - Small chunks (~500 tokens) for factual precision.  
  - Large chunks (~1500 tokens) for reasoning and summarization.

### 2. Research Paper Recommendations
- Recommends semantically similar papers based on **document embeddings**.  
- Uses **FAISS** for efficient vector similarity search.  
- Allows queries by paper title, abstract, or free text.

### 3. Cohere Embeddings
- Uses **Cohere’s embed-english-v3.0** model for vectorization.  
- Embedding generation supports batching with retry logic for reliability.  
- Vectors are stored in two FAISS indexes for flexible retrieval:
  - `faiss_small`: factual and short-form QA.  
  - `faiss_large`: long-form reasoning and recommendation.

### 4. Cloud-Ready Deployment
- Fully containerized with Docker.  
- Can be deployed easily to AWS EC2, GCP, or other cloud services.  
- FastAPI backend provides RESTful endpoints for all major functionalities.

---

## Project Structure

```
ARGUS/
│
├── Notebooks/                  # Jupyter notebooks for experimentation
│   ├── step1_RAG_implementation.ipynb
│   ├── step2_recommendation_system.ipynb
│   └── utils_ingest.py         # Shared utilities for PDF/JSON ingestion
│
├── Papers/                     # Local PDF storage
├── archive/                    # arXiv JSON dataset
├── vectorstore/                # FAISS vector databases
│   ├── faiss_small/            # Small chunks for factual retrieval
│   └── faiss_large/            # Large chunks for summaries and recommendations
│
├── app/                        # FastAPI application
│   ├── main.py                 # API routes for RAG and recommendations
│   └── requirements.txt        # Dependencies for the backend
│
├── Dockerfile                  # For containerized deployment
└── README.md                   # Project documentation
```

---

## Document Ingestion

ARGUS supports ingestion of:
- **PDF papers**, parsed and chunked by page or section.  
- **arXiv JSON snapshots**, where each line corresponds to a single paper’s metadata.

Two levels of recursive chunking are used:
- **Small chunks (≈500 tokens):** Optimized for factual retrieval and quick lookups.  
- **Large chunks (≈1500 tokens):** Designed for summarization and reasoning.

---

## Embedding and Indexing

### Embeddings
- Model: `embed-english-v3.0` from **Cohere**.  
- Handles batching efficiently with retry logic (batch size ≤ 96).  

### Vector Store
- **FAISS** is used for high-speed similarity search.  
- Maintains two separate indexes:
  - `faiss_small` – short factual queries.  
  - `faiss_large` – contextual and long-form queries.

---

## RAG Chains

ARGUS creates two independent RAG pipelines using **LangChain**:

- **`rag_chain_small`** – optimized for short, fact-based questions.  
- **`rag_chain_large`** – optimized for deeper reasoning and summarization.

Each chain retrieves relevant chunks, feeds them into a language model, and produces a contextually grounded answer.

---

## Recommendation System

ARGUS includes a similarity-based recommendation module using FAISS.

Example function:

```python
def recommend_related_papers(query_text, top_k=5):
    results = vector_store.similarity_search(query_text, k=top_k)
    return results
```

**Input:** Title, abstract, or full text of a paper.  
**Output:** List of top-K semantically similar documents.

This feature enables automated discovery of related research, improving literature exploration and citation mapping.

---

## Docker Deployment

### Build the Image
```bash
sudo docker build -t argus-app .
```

### Run the Container
```bash
sudo docker run -p 8000:8000 argus-app
```

Once running, the API is accessible at:  
**http://localhost:8000**

---

## API Endpoints (FastAPI)

| Endpoint | Method | Description |
|-----------|--------|-------------|
| `/rag/qa` | POST | Ask a natural language question over stored papers |
| `/recommend` | POST | Retrieve a list of related papers |

### Example Request
```json
{
  "query": "What are the main contributions of the ISAAC architecture?"
}
```

### Example Response
- Short factual explanation or a paragraph-level summary drawn from the most relevant documents.

---

## API Keys

Create a `.env` file in your working directory and add:
```
COHERE_API_KEY=your-api-key-here
```

---

## Future Enhancements
- Real-time PDF uploads through the API or UI  
- Support for multiple embedding and LLM providers (OpenAI, Mistral, Claude)  
- Streamlit or React-based web interface for interactive exploration  

---

## Maintainer

**Sri Harsha Mudumba**  
M.S. Computer Engineering, Iowa State University  
Focus: AI Inference Optimization, RAG Systems, and Cloud-based Research Tools  

---

**ARGUS — Bridging human curiosity and machine understanding for research discovery.**
