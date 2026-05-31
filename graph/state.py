from typing import TypedDict, List, Dict, Any, Optional

class GraphState(TypedDict):
    """Represents the global state of the SRE Copilot workflow."""
    question: str
    rewritten_question: Optional[str]
    hyde_document: Optional[str]
    retrieved_chunks: List[Dict[str, Any]]
    relevant_chunks: List[Dict[str, Any]]
    draft_answer: Optional[str]
    final_answer: Optional[str]
    sql_query: Optional[str]
    sql_results: Optional[Any]
    sql_approved: bool
    retry_count: int
    retrieval_strategy: str # "RAG" | "CRAG" | "Self-RAG" | "Text2SQL" | "Cache"
    web_search_results: Optional[str]
    cache_hit: bool
    logs: List[str]
