import os
from dotenv import load_dotenv

load_dotenv()

# Disable Chroma telemetry to suppress "capture() takes 1 positional argument" errors
os.environ["ANONYMIZED_TELEMETRY"] = "False"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
CHROMA_PERSIST_DIRECTORY = "./chroma_db"

EMBEDDING_MODEL_NAME = "models/gemini-embedding-001"
LLM_MODEL_NAME = "gemini-1.5-flash"
