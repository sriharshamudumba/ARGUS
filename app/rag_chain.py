from langchain.chains import RetrievalQA
from langchain.llms import Cohere
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()
cohere_key = os.getenv("COHERE_API_KEY")

llm = Cohere(cohere_api_key=cohere_key)
retriever = FAISS.load_local("vectorstore/faiss_small", "small_index", embedding_model).as_retriever(search_kwargs={"k": 3})

rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

def answer_question(query):
    result = rag_chain.invoke(query)
    return {"answer": result['result'], "sources": [doc.page_content[:300] for doc in result['source_documents']]}
