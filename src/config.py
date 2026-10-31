import os
from dotenv import load_dotenv

# Silence Chroma telemetry errors caused by posthog version mismatch
try:
    import posthog
    posthog.capture = lambda *args, **kwargs: None
except Exception:
    pass

os.environ["ANONYMIZED_TELEMETRY"] = "False"

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
CHROMA_PERSIST_DIRECTORY = "./chroma_db"

EMBEDDING_MODEL_NAME = "models/gemini-embedding-001"
LLM_MODEL_NAME = "gemini-3.5-flash-lite"
