import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, UnstructuredURLLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from src.config import CHROMA_PERSIST_DIRECTORY, EMBEDDING_MODEL_NAME

def get_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)
    vector_store = Chroma(
        persist_directory=CHROMA_PERSIST_DIRECTORY,
        embedding_function=embeddings
    )
    return vector_store

def clear_vector_store():
    """Deletes all documents currently indexed in the vector store."""
    vector_store = get_vector_store()
    data = vector_store.get()
    ids = data.get("ids", [])
    if ids:
        vector_store.delete(ids=ids)
    return len(ids)

def ingest_document(source_type: str, path_or_url: str, reset_existing: bool = True):
    if source_type == "file":
        loader = TextLoader(path_or_url, encoding="utf-8")
    elif source_type == "url":
        loader = UnstructuredURLLoader(urls=[path_or_url])
    else:
        raise ValueError("Invalid source_type. Use 'file' or 'url'.")
    
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n## ", "\n### ", "\n", " "]
    )
    splits = text_splitter.split_documents(docs)
    
    vector_store = get_vector_store()

    # Automatically remove old chunks for this source so changed content updates cleanly
    if reset_existing:
        try:
            vector_store._collection.delete(where={"source": path_or_url})
        except Exception:
            pass

    vector_store.add_documents(splits)
    return len(splits)
