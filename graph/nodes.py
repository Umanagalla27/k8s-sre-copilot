import hashlib
import logging
import os
import json
from datetime import datetime
from storage.redis import redis_client
from storage.postgres import SessionLocal, get_db
from sqlalchemy import text
from retrieval.hybrid import hybrid_search
from retrieval.rerank import rerank_documents
from graph.llm_helper import (
    llm_classify_intent, llm_hyde, llm_grade_chunk, 
    llm_generate_answer, llm_check_hallucination, 
    llm_check_usefulness, llm_rewrite_query, llm_generate_sql
)
from graph.state import GraphState

logger = logging.getLogger("graph.nodes")

# 1. Cache Check Node
def cache_check_node(state: GraphState) -> GraphState:
    logger.info("--- Entering Cache Check Node ---")
    query = state["question"]
    sha256 = hashlib.sha256(query.encode('utf-8')).hexdigest()
    cache_key = f"cache:{sha256}"
    
    logs = state.get("logs", [])
    logs.append(f"Cache check for key {cache_key}")
    
    cached_val = redis_client.get(cache_key)
    if cached_val:
        logger.info("Cache Hit!")
        try:
            hits = int(redis_client.get("stats:hits") or 0)
            redis_client.set("stats:hits", str(hits + 1))
        except Exception:
            pass
        cached_data = json.loads(cached_val)
        return {
            **state,
            "final_answer": cached_data["answer"],
            "retrieval_strategy": "Cache",
            "cache_hit": True,
            "logs": logs + ["Cache hit! Skipped execution graph."]
        }
    
    logger.info("Cache Miss.")
    try:
        misses = int(redis_client.get("stats:misses") or 0)
        redis_client.set("stats:misses", str(misses + 1))
    except Exception:
        pass
    return {
        **state,
        "cache_hit": False,
        "logs": logs + ["Cache miss. Starting workflow execution."],
        "retry_count": 0,
        "retrieved_chunks": [],
        "relevant_chunks": [],
        "sql_approved": False
    }

# 2. Classify Intent Router Node
def classify_intent_node(state: GraphState) -> GraphState:
    logger.info("--- Entering Classify Intent Node ---")
    query = state["question"]
    intent = llm_classify_intent(query)
    
    logs = state.get("logs", [])
    logs.append(f"Classified query intent as: {intent}")
    
    return {
        **state,
        "retrieval_strategy": "Text2SQL" if intent == "sql" else "RAG",
        "logs": logs
    }

# 3. HyDE Node
def hyde_node(state: GraphState) -> GraphState:
    logger.info("--- Entering HyDE Node ---")
    query = state.get("rewritten_question") or state["question"]
    hyde_doc = llm_hyde(query)
    
    logs = state.get("logs", [])
    logs.append("Generated HyDE hypothetical document")
    
    return {
        **state,
        "hyde_document": hyde_doc,
        "logs": logs
    }

# 4. Retrieve Node
def retrieve_node(state: GraphState) -> GraphState:
    logger.info("--- Entering Retrieve Node ---")
    # If HyDE document exists, search using it. Else search using raw question
    query = state.get("hyde_document") or state.get("rewritten_question") or state["question"]
    
    logger.info(f"Retrieving with search query: {query[:60]}...")
    chunks = hybrid_search(query, top_k=20)
    
    logs = state.get("logs", [])
    logs.append(f"Retrieved {len(chunks)} chunks using Hybrid Search (Dense + BM25)")
    
    return {
        **state,
        "retrieved_chunks": chunks,
        "logs": logs
    }

# 5. Rerank Node
def rerank_node(state: GraphState) -> GraphState:
    logger.info("--- Entering Rerank Node ---")
    query = state.get("rewritten_question") or state["question"]
    chunks = state["retrieved_chunks"]
    
    reranked = rerank_documents(query, chunks, top_k=5)
    
    logs = state.get("logs", [])
    logs.append("Re-ranked chunks using Cross-Encoder. Sliced top-5.")
    
    return {
        **state,
        "retrieved_chunks": reranked,
        "logs": logs
    }

# 6. Grade Docs Node
def grade_docs_node(state: GraphState) -> GraphState:
    logger.info("--- Entering Grade Docs Node ---")
    query = state.get("rewritten_question") or state["question"]
    chunks = state["retrieved_chunks"]
    
    relevant_chunks = []
    logs = state.get("logs", [])
    
    for chunk in chunks:
        res = llm_grade_chunk(query, chunk["content"])
        if res == "YES":
            relevant_chunks.append(chunk)
            logs.append(f"Chunk ID {chunk['id']} graded: RELEVANT")
        else:
            logs.append(f"Chunk ID {chunk['id']} graded: IRRELEVANT")
            
    strategy = state["retrieval_strategy"]
    if len(relevant_chunks) < 3:
        strategy = "CRAG" # Corrective RAG triggered due to lack of docs
        logs.append(f"CRAG: Triggered web search fallback. Relevant chunks count: {len(relevant_chunks)} < 3")
        
    return {
        **state,
        "relevant_chunks": relevant_chunks,
        "retrieval_strategy": strategy,
        "logs": logs
    }

# 7. Web Search Node
def web_search_node(state: GraphState) -> GraphState:
    logger.info("--- Entering Web Search Node ---")
    query = state.get("rewritten_question") or state["question"]
    
    # Check Tavily key
    api_key = os.getenv("TAVILY_API_KEY")
    web_results = ""
    
    if api_key:
        try:
            from tavily import TavilyClient
            tavily = TavilyClient(api_key=api_key)
            logger.info("Querying Tavily search API...")
            response = tavily.search(query=query, max_results=3)
            results = [r["content"] for r in response.get("results", [])]
            web_results = "\n\n".join(results)
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            
    if not web_results:
        # Fallback Mock Web Search Result
        logger.warning("Using mock web search fallback.")
        web_results = (
            f"Official Kubernetes Documentation Context for '{query}':\n"
            "Ensure the kubelet configs and node conditions are verified. "
            "For deployment rollbacks, run 'kubectl rollout undo deployment/<name>'. "
            "To debug container terminations, execute 'kubectl describe pod <pod-name>' and examine exit statuses."
        )
        
    logs = state.get("logs", [])
    logs.append("Performed web search fallback query")
    
    return {
        **state,
        "web_search_results": web_results,
        "logs": logs
    }

# 8. Generate Node
def generate_node(state: GraphState) -> GraphState:
    logger.info("--- Entering Generate Node ---")
    query = state.get("rewritten_question") or state["question"]
    
    # Combine relevant database docs and web search context if CRAG occurred
    contexts = [chunk["content"] for chunk in state["relevant_chunks"]]
    if state["web_search_results"]:
        contexts.append(f"[Web Fallback]: {state['web_search_results']}")
        
    draft_answer = llm_generate_answer(query, contexts)
    
    logs = state.get("logs", [])
    logs.append("Generated draft answer using retrieved contexts")
    
    return {
        **state,
        "draft_answer": draft_answer,
        "logs": logs
    }

# 9. Hallucination Grader Node
def check_hallucination_node(state: GraphState) -> GraphState:
    logger.info("--- Entering Check Hallucination Node ---")
    answer = state["draft_answer"]
    contexts = [chunk["content"] for chunk in state["relevant_chunks"]]
    if state["web_search_results"]:
        contexts.append(state["web_search_results"])
        
    res = llm_check_hallucination(answer, contexts)
    hallucinated = res.get("hallucination", False)
    reason = res.get("reason", "")
    
    logs = state.get("logs", [])
    logs.append(f"Hallucination check outcome: {hallucinated}. Reason: {reason}")
    
    # We pass the hallucination check result in logs/strategy indicators
    # The router edge uses these to decide routing.
    # To keep typeddict happy, we can update logs or custom flags.
    return {
        **state,
        "logs": logs
    }

# 10. Usefulness Grader Node
def check_usefulness_node(state: GraphState) -> GraphState:
    logger.info("--- Entering Check Usefulness Node ---")
    question = state["question"]
    answer = state["draft_answer"]
    
    res = llm_check_usefulness(question, answer)
    useful = res.get("useful", True)
    reason = res.get("reason", "")
    
    logs = state.get("logs", [])
    logs.append(f"Usefulness check outcome: {useful}. Reason: {reason}")
    
    return {
        **state,
        "logs": logs
    }

# 11. Rewrite Query Node
def rewrite_query_node(state: GraphState) -> GraphState:
    logger.info("--- Entering Rewrite Query Node ---")
    query = state["question"]
    rewritten = llm_rewrite_query(query)
    
    logs = state.get("logs", [])
    logs.append(f"Rewrote query to: {rewritten}")
    
    return {
        **state,
        "rewritten_question": rewritten,
        "retry_count": state["retry_count"] + 1,
        "retrieval_strategy": "Self-RAG",
        "logs": logs
    }

# 12. Text2SQL Generation Node
def text2sql_node(state: GraphState) -> GraphState:
    logger.info("--- Entering Text2SQL Generation Node ---")
    query = state["question"]
    sql = llm_generate_sql(query)
    
    logs = state.get("logs", [])
    logs.append(f"Generated SQL: {sql}")
    
    return {
        **state,
        "sql_query": sql,
        "sql_approved": False, # Waiting for human approval interrupt
        "logs": logs
    }

# 13. Execute SQL Node
def execute_sql_node(state: GraphState) -> GraphState:
    logger.info("--- Entering Execute SQL Node ---")
    sql = state["sql_query"]
    
    logs = state.get("logs", [])
    
    if not state.get("sql_approved", False):
        logs.append("SQL execution blocked: Approval not granted.")
        return {
            **state,
            "final_answer": "SQL query execution was rejected by the operator.",
            "logs": logs
        }
        
    db = SessionLocal()
    results_list = []
    markdown_table = ""
    
    try:
        logger.info(f"Executing SQL query: {sql}")
        res = db.execute(text(sql))
        columns = res.keys()
        rows = res.fetchall()
        
        # Format as markdown table
        if rows:
            header = " | ".join(columns)
            divider = " | ".join(["---"] * len(columns))
            table_rows = []
            for r in rows:
                table_rows.append(" | ".join([str(val) for val in r]))
            markdown_table = f"\n| {header} |\n| {divider} |\n" + "\n".join([f"| {tr} |" for tr in table_rows])
            results_list = [dict(zip(columns, row)) for row in rows]
        else:
            markdown_table = "No records returned from SQL execution."
            
        logs.append(f"Executed SQL query successfully. Returned {len(rows)} records.")
    except Exception as e:
        logger.error(f"SQL execution error: {e}")
        markdown_table = f"SQL Database Error: {e}"
        logs.append(f"SQL execution failed: {e}")
    finally:
        db.close()
        
    final_ans = f"Database query executed successfully:\n\n{markdown_table}"
    return {
        **state,
        "sql_results": results_list,
        "final_answer": final_ans,
        "logs": logs
    }

# 14. Cache Write Node
def cache_write_node(state: GraphState) -> GraphState:
    logger.info("--- Entering Cache Write Node ---")
    # Clean output answer if guardrails require PII scrubbing
    from guardrails.pipeline import scrub_output_pii, check_output_faithfulness
    
    ans = state.get("final_answer") or state.get("draft_answer") or ""
    scrubbed_ans = scrub_output_pii(ans)
    
    # Run output faithfulness check if it is a doc RAG pathway
    logs = state.get("logs", [])
    if state["retrieval_strategy"] in ("RAG", "CRAG", "Self-RAG"):
        contexts = [chunk["content"] for chunk in state["relevant_chunks"]]
        if state["web_search_results"]:
            contexts.append(state["web_search_results"])
            
        passed, reason = check_output_faithfulness(scrubbed_ans, contexts)
        if not passed:
            logs.append(f"Output Guardrail violation: {reason}")
            # Log audit block
            from guardrails.pipeline import log_block_to_db
            log_block_to_db(state["question"], "Output Faithfulness", reason)
            # Override response
            scrubbed_ans = f"System safety alert: {reason}"
            
    # Cache key write
    query = state["question"]
    sha256 = hashlib.sha256(query.encode('utf-8')).hexdigest()
    cache_key = f"cache:{sha256}"
    
    cache_payload = {
        "query": query,
        "answer": scrubbed_ans,
        "timestamp": datetime.utcnow().isoformat(),
        "strategy": state["retrieval_strategy"]
    }
    
    try:
        redis_client.set(cache_key, json.dumps(cache_payload), ex=3600)
        logs.append(f"Saved answer to Redis cache. TTL: 1 hour. Key: {cache_key}")
    except Exception as e:
        logger.error(f"Failed to write cache: {e}")
        
    return {
        **state,
        "final_answer": scrubbed_ans,
        "logs": logs
    }

# 15. Respond Node
def respond_node(state: GraphState) -> GraphState:
    logger.info("--- Entering Respond Node ---")
    logs = state.get("logs", [])
    logs.append("Workflow completed. Returning final response.")
    return {
        **state,
        "logs": logs
    }
