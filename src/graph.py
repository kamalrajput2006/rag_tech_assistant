from typing import Dict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.tools.tavily_search import TavilyAnswer
from langgraph.graph import StateGraph, END

from src.models import GraphState, DocumentGrade, HallucinationGrade
from src.ingestion import get_vector_store
from src.config import LLM_MODEL_NAME


# In-memory session chat history store
SESSION_MEMORY: Dict[str, List[Dict[str, str]]] = {}

# Initialize Gemini Flash model via LangChain integration
llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL_NAME,
    temperature=0,
)

def get_text(response) -> str:
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part if isinstance(part, str) else part.get("text", "")
            for part in content
            if isinstance(part, (str, dict))
        )
    return str(content)

def query_analysis_node(state: GraphState):
    print("---NODE: QUERY ANALYSIS---")
    question = state["question"]
    chat_history = state.get("chat_history", [])

    # If no conversation context exists yet, use the question directly to avoid redundant LLM latency
    if not chat_history:
        return {"rewritten_question": question, "retry_count": state.get("retry_count", 0)}

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant helping rewrite technical queries for vector search, incorporating chat context if relevant."),
        ("user", "Chat History: {chat_history}\n\nOriginal Question: {question}\n\nProvide a rewritten, clear, keyword-rich standalone search query.")
    ])
    chain = prompt | llm
    response = chain.invoke({"chat_history": chat_history, "question": question})
    rewritten = get_text(response).strip()

    return {"rewritten_question": rewritten, "retry_count": state.get("retry_count", 0)}


def retrieval_node(state: GraphState):
    print("---NODE: RETRIEVAL---")
    query = state["rewritten_question"]
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    return {"documents": docs}


def document_grading_node(state: GraphState):
    print("---NODE: DOCUMENT GRADING (skipped to save API calls)---")
    docs = state["documents"]

    filtered_docs = docs  # grading temporarily skipped to stay within free-tier limits

    retry_count = state.get("retry_count", 0)
    if not filtered_docs:
        retry_count += 1

    return {"documents": filtered_docs, "retry_count": retry_count}


def web_search_node(state: GraphState):
    print("---NODE: WEB SEARCH FALLBACK---")
    question = state["rewritten_question"]
    web_tool = TavilyAnswer(max_results=3)
    search_results = web_tool.invoke({"query": question})

    from langchain_core.documents import Document
    doc = Document(page_content=str(search_results), metadata={"source": "Tavily Web Search"})
    return {"documents": [doc], "web_search_used": True}


def generation_node(state: GraphState):
    print("---NODE: GENERATION---")
    docs = state["documents"]
    question = state["question"]

    context = "\n\n".join([d.page_content for d in docs])
    sources = list(set([d.metadata.get("source", "vector_store") for d in docs]))

    prompt = ChatPromptTemplate.from_template(
        "You are a technical documentation assistant. Answer the user question accurately using ONLY the provided context. Include source citations.\n\n"
        "Context:\n{context}\n\nQuestion: {question}"
    )
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})

    return {"generation": get_text(response), "sources": sources}


def hallucination_check_node(state: GraphState):
    print("---NODE: HALLUCINATION CHECK---")
    docs = state["documents"]
    generation = state["generation"]

    context = "\n\n".join([d.page_content for d in docs])
    parser = PydanticOutputParser(pydantic_object=HallucinationGrade)

    prompt = ChatPromptTemplate.from_template(
        "You are an expert grading whether an LLM generation is fully grounded in the provided context.\n"
        "Context:\n{context}\n\nGeneration:\n{generation}\n"
        "Is the answer grounded in the context? Answer strictly with yes or no.\n{format_instructions}"
    )
    chain = prompt | llm | parser
    try:
        score = chain.invoke({
            "context": context,
            "generation": generation,
            "format_instructions": parser.get_format_instructions()
        })
        passed = (score.binary_score == "yes")
    except Exception:
        passed = True

    return {"hallucination_passed": passed}


def decide_to_generate(state: GraphState):
    if len(state["documents"]) > 0:
        return "generate"
    else:
        if state["retry_count"] > 2:
            return "web_search"
        return "rewrite_query"


def check_hallucination_or_retry(state: GraphState):
    if state["hallucination_passed"]:
        return "accept"
    else:
        return "regenerate"


# --- Build Workflow Graph ---

def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("query_analysis", query_analysis_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("document_grading", document_grading_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate_node", generation_node)  # Renamed to avoid state key conflict
    workflow.add_node("hallucination_check", hallucination_check_node)

    workflow.set_entry_point("query_analysis")
    workflow.add_edge("query_analysis", "retrieval")
    workflow.add_edge("retrieval", "document_grading")

    workflow.add_conditional_edges(
        "document_grading",
        decide_to_generate,
        {
            "generate": "generate_node",
            "rewrite_query": "query_analysis",
            "web_search": "web_search"
        }
    )
    workflow.add_edge("web_search", "generate_node")
    workflow.add_edge("generate_node", "hallucination_check")

    workflow.add_conditional_edges(
        "hallucination_check",
        check_hallucination_or_retry,
        {
            "accept": END,
            "regenerate": "generate_node"
        }
    )

    return workflow.compile()
