import logging
from graph.state import GraphState
from graph.llm_helper import llm_check_hallucination, llm_check_usefulness

logger = logging.getLogger("graph.edges")

def route_after_cache(state: GraphState) -> str:
    """Routes to END if cache hit, else to classify_intent."""
    if state.get("cache_hit", False):
        logger.info("Routing decision: CACHE HIT -> END")
        return "end"
    logger.info("Routing decision: CACHE MISS -> classify_intent")
    return "classify_intent"

def route_after_intent(state: GraphState) -> str:
    """Routes to text2sql or hyde based on intent classification."""
    strategy = state.get("retrieval_strategy", "RAG")
    if strategy == "Text2SQL":
        logger.info("Routing decision: SQL -> text2sql")
        return "text2sql"
    logger.info("Routing decision: RAG -> hyde")
    return "hyde"

def route_after_grading(state: GraphState) -> str:
    """Routes to web_search if relevant docs < 3, else to generate."""
    strategy = state.get("retrieval_strategy")
    if strategy == "CRAG" or len(state.get("relevant_chunks", [])) < 3:
        logger.info("Routing decision: Docs count low -> web_search")
        return "web_search"
    logger.info("Routing decision: Enough docs -> generate")
    return "generate"

def route_post_generation(state: GraphState) -> str:
    """Performs self-reflection checks (Self-RAG) and decides routing."""
    answer = state.get("draft_answer")
    if not answer:
        logger.warning("No draft answer found. Default routing to generate.")
        return "generate"
        
    contexts = [c["content"] for c in state.get("relevant_chunks", [])]
    if state.get("web_search_results"):
        contexts.append(state["web_search_results"])
        
    # 1. Hallucination Check
    hallucination_res = llm_check_hallucination(answer, contexts)
    is_hallucinating = hallucination_res.get("hallucination", False)
    
    if is_hallucinating:
        logger.warning("Self-RAG: Hallucination detected! Routing back to generate.")
        return "generate"
        
    # 2. Usefulness Check
    usefulness_res = llm_check_usefulness(state["question"], answer)
    is_useful = usefulness_res.get("useful", True)
    retry_count = state.get("retry_count", 0)
    
    if not is_useful and retry_count < 3:
        logger.warning(f"Self-RAG: Answer not useful. Retry attempt: {retry_count + 1}/3. Routing to rewrite_query.")
        return "rewrite_query"
        
    logger.info("Self-RAG: Answer passed hallucination and usefulness validation. Routing to cache_write.")
    return "cache_write"
