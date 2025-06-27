from fastapi import FastAPI, Query
from rag_chain import answer_question
from recommender import recommend_papers

app = FastAPI()

@app.get("/rag")
def rag_query(query: str = Query(...)):
    return answer_question(query)

@app.get("/recommend")
def recommend(query: str = Query(...), top_k: int = 5):
    return recommend_papers(query, top_k)
