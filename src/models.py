from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default_session"

class IngestRequest(BaseModel):
    source_type: str = Field(..., description="file or url")
    path_or_url: str

class FeedbackRequest(BaseModel):
    session_id: str
    rating: str = Field(..., description="thumbs_up or thumbs_down")
    comment: Optional[str] = None

class DocumentGrade(BaseModel):
    binary_score: str = Field(description="Relevance score: 'yes' or 'no'")

class HallucinationGrade(BaseModel):
    binary_score: str = Field(description="Answer supported by facts: 'yes' or 'no'")

# LangGraph State Schema
class GraphState(TypedDict):
    question: str
    rewritten_question: str
    generation: str
    session_id: str
    chat_history: List[Dict[str, str]]
    documents: List[Any]
    retry_count: int
    web_search_used: bool
    hallucination_passed: bool
    sources: List[str]