import logging
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from graph.state import GraphState
from graph.nodes import (
    cache_check_node, classify_intent_node, hyde_node, retrieve_node, 
    rerank_node, grade_docs_node, web_search_node, generate_node, 
    rewrite_query_node, text2sql_node, execute_sql_node, cache_write_node, 
    respond_node
)
from graph.edges import (
    route_after_cache, route_after_intent, route_after_grading, route_post_generation
)

logger = logging.getLogger("graph.workflow")

def build_workflow():
    workflow = StateGraph(GraphState)

    # 1. Register Nodes
    workflow.add_node("cache_check", cache_check_node)
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("hyde", hyde_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("grade_docs", grade_docs_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("text2sql", text2sql_node)
    workflow.add_node("execute_sql", execute_sql_node)
    workflow.add_node("cache_write", cache_write_node)
    workflow.add_node("respond", respond_node)

    # 2. Add Transitions
    workflow.add_edge(START, "cache_check")
    
    # Conditional edge after cache check
    workflow.add_conditional_edges(
        "cache_check",
        route_after_cache,
        {
            "end": END,
            "classify_intent": "classify_intent"
        }
    )
    
    # Conditional edge after intent classification
    workflow.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {
            "text2sql": "text2sql",
            "hyde": "hyde"
        }
    )
    
    workflow.add_edge("hyde", "retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "grade_docs")
    
    # Conditional edge after grading docs (CRAG check)
    workflow.add_conditional_edges(
        "grade_docs",
        route_after_grading,
        {
            "web_search": "web_search",
            "generate": "generate"
        }
    )
    
    workflow.add_edge("web_search", "generate")
    
    # Conditional edge after generation (Self-RAG check)
    workflow.add_conditional_edges(
        "generate",
        route_post_generation,
        {
            "generate": "generate",          # Hallucination correction loop
            "rewrite_query": "rewrite_query",  # Usefulness correction loop
            "cache_write": "cache_write"
        }
    )
    
    # Retry loop link back to retrieval
    workflow.add_edge("rewrite_query", "retrieve")
    
    # Text2SQL stream
    workflow.add_edge("text2sql", "execute_sql")
    workflow.add_edge("execute_sql", "cache_write")
    
    # Final endpoints
    workflow.add_edge("cache_write", "respond")
    workflow.add_edge("respond", END)

    # 3. Setup Checkpointer and compile
    # We set execute_sql as an interrupt node to implement human-in-the-loop approval
    memory = MemorySaver()
    app = workflow.compile(
        checkpointer=memory,
        interrupt_before=["execute_sql"]
    )
    
    logger.info("Compiled LangGraph SRE Copilot workflow with execution interrupts.")
    return app

# Singleton compiled app
graph_app = build_workflow()
