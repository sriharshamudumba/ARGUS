from langchain.vectorstores import FAISS

def recommend_papers(query_text, top_k=5):
    results = vector_store.similarity_search(query_text, k=top_k)
    return [{"rank": i+1, "content": doc.page_content[:500]} for i, doc in enumerate(results)]
