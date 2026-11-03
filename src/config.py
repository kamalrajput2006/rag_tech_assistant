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

GOOGLE_API_KEY = (os.getenv("GOOGLE_API_KEY") or "").strip()
TAVILY_API_KEY = (os.getenv("TAVILY_API_KEY") or "").strip()
CHROMA_PERSIST_DIRECTORY = "./chroma_db"

EMBEDDING_MODEL_NAME = "models/gemini-embedding-001"
# Use a stable, widely supported Gemini model name.
LLM_MODEL_NAME = "gemini-3.5-flash-lite"


def get_google_api_key() -> str:
    key = GOOGLE_API_KEY
    if not key:
        raise RuntimeError(
            "GOOGLE_API_KEY is missing. Create a .env file from .env.example "
            "and add a valid Google AI Studio Gemini API key."
        )
    return key
