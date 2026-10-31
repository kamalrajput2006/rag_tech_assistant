from typing import Dict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.tools.tavily_search import TavilyAnswer
from langgraph.graph import StateGraph, END

from src.models import GraphState, DocumentGrade, HallucinationGrade
from src.ingestion import get_vector_store
from src.config import LLM_MODEL_NAME

SESSION_MEMORY: Dict[str, List[Dict[str, str]]] = {}

# Initialize Gemini Flash model via LangGraph integration
llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, temperature=0)

# (The rest of the nodes, routing logic, and build_graph function remain exactly the same as written previously)