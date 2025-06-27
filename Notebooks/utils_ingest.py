# utils_ingest.py
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

def load_pdf_as_documents(pdf_path):
    loader = PyPDFLoader(pdf_path)
    return loader.load()

def split_documents(documents):
    small_chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents(documents)
    large_chunks = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200).split_documents(documents)
    small_texts = [doc.page_content for doc in small_chunks]
    large_texts = [doc.page_content for doc in large_chunks]
    return small_chunks, small_texts, large_chunks, large_texts

def convert_to_docs(texts):
    return [Document(page_content=t) for t in texts]
