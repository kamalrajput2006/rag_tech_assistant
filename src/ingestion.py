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

def ingest_document(source_type: str, path_or_url: str):
    if source_type == "file":
        loader = TextLoader(path_or_url)
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
    vector_store.add_documents(splits)
    vector_store.persist()
    return len(splits)