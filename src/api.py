from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.config import get_google_api_key
from src.models import QueryRequest, IngestRequest, FeedbackRequest
from src.ingestion import ingest_document, get_vector_store, clear_vector_store
from src.graph import build_graph, SESSION_MEMORY

app = FastAPI(title="RAG Technical Documentation Assistant", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _validate_google_api_key():
    try:
        get_google_api_key()
    except RuntimeError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@app.get("/")
def root():
    return {"status": "ok", "message": "RAG Technical Documentation Assistant API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


rag_app = build_graph()

@app.post("/query")
def query_endpoint(req: QueryRequest):
    _validate_google_api_key()
    session_id = req.session_id
    if session_id not in SESSION_MEMORY:
        SESSION_MEMORY[session_id] = []

    initial_state = {
        "question": req.question,
        "rewritten_question": "",
        "generation": "",
        "session_id": session_id,
        "chat_history": SESSION_MEMORY[session_id],
        "documents": [],
        "retry_count": 0,
        "web_search_used": False,
        "hallucination_passed": True,
        "sources": []
    }

    try:
        final_state = rag_app.invoke(initial_state)
        answer = final_state.get("generation", "No answer generated.")
        sources = final_state.get("sources", [])

        # Update session memory
        SESSION_MEMORY[session_id].append({"role": "user", "content": req.question})
        SESSION_MEMORY[session_id].append({"role": "assistant", "content": answer})

        return {
            "answer": answer,
            "sources": sources,
            "web_search_used": final_state.get("web_search_used", False),
            "session_id": session_id
        }
    except Exception as exc:
        import traceback
        traceback.print_exc()
        message = str(exc)
        lower_message = message.lower()
        if "api key" in lower_message or "unauthenticated" in lower_message:
            raise HTTPException(
                status_code=401,
                detail="GOOGLE_API_KEY is invalid or expired. Get a fresh Gemini API key from Google AI Studio and update your .env file."
            ) from exc
        raise HTTPException(status_code=500, detail=message) from exc

@app.post("/ingest")
def ingest_endpoint(req: IngestRequest):
    _validate_google_api_key()
    try:
        chunks_count = ingest_document(req.source_type, req.path_or_url)
        return {"status": "success", "chunks_indexed": chunks_count}
    except Exception as exc:
        message = str(exc)
        lower_message = message.lower()
        if "api key" in lower_message or "unauthenticated" in lower_message:
            raise HTTPException(
                status_code=401,
                detail="GOOGLE_API_KEY is invalid or expired. Get a fresh Gemini API key from Google AI Studio and update your .env file."
            ) from exc
        raise HTTPException(status_code=500, detail=message) from exc

@app.post("/clear")
def clear_endpoint():
    try:
        count = clear_vector_store()
        SESSION_MEMORY.clear()
        return {"status": "success", "cleared_chunks": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
def list_documents():
    try:
        vector_store = get_vector_store()
        data = vector_store.get()
        metas = data.get("metadatas", [])
        sources = sorted(list(set(m.get("source") for m in metas if isinstance(m, dict) and "source" in m)))
        return {"documents": sources, "total_chunks": len(data.get("ids", []))}
    except Exception as e:
        return {"documents": [], "total_chunks": 0, "error": str(e)}

@app.post("/feedback")
def feedback_endpoint(req: FeedbackRequest):
    # Log feedback for evaluation purposes
    return {
        "status": "received", 
        "session_id": req.session_id, 
        "rating": req.rating, 
        "comment": req.comment
    }
