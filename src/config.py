import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
CHROMA_PERSIST_DIRECTORY = "./chroma_db"


EMBEDDING_MODEL_NAME = "models/embedding-001"
LLM_MODEL_NAME = "gemini-1.5-flash"