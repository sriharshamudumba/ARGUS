# ARGUS: Multimodal LLM-Powered Memory Assistant

ARGUS is an intelligent assistant that enables long-term memory, multimodal document understanding (PDF + images), hybrid retrieval, and LangGraph-based reasoning. It is designed for research workflows, technical Q&A, and scientific exploration, and is deployable on AWS.

---

## Features

- PDF ingestion and semantic search using RAG
- Hybrid retrieval with FAISS (dense) + BM25 (sparse)
- Source-aware QA with citations
- Image embedding and caption-based memory (BLIP-2 / CLIP)
- LangGraph agent orchestration for tool selection and multi-step reasoning
- Early-exit inference based on entropy or task complexity
- AWS deployment with Lambda, S3, and ECS

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone git@github.com:yourusername/ARGUS.git
cd ARGUS

## Create and Activate Virtual Environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

## Installing
pip install -r requirements.txt

## Set Environment
cp .env.example .env

