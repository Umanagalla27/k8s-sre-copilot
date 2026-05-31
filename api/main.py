import os
import uuid
import logging
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from storage.postgres import init_db, SessionLocal, EvalRun
from storage.redis import redis_client
from guardrails.pipeline import run_input_guardrails
from graph.workflow import graph_app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api.main")

app = FastAPI(title="Kubernetes SRE Copilot API", version="1.0.0")

# Setup database on startup
@app.on_event("startup")
def on_startup():
    init_db()
    logger.info("Application startup database initializations complete.")

class QueryRequest(BaseModel):
    question: str
    user_id: str = "default_user"
    thread_id: Optional[str] = None

class ResumeRequest(BaseModel):
    thread_id: str
    approved: bool

class QueryResponse(BaseModel):
    status: str # "success" | "pending_approval" | "blocked"
    answer: Optional[str] = None
    strategy: Optional[str] = None
    thread_id: str
    sql_query: Optional[str] = None
    logs: List[str] = []

@app.post("/query", response_model=QueryResponse)
def handle_query(req: QueryRequest):
    # 1. Run Input Guardrails
    passed, reason = run_input_guardrails(req.question, req.user_id)
    if not passed:
        return QueryResponse(
            status="blocked",
            answer=f"Guardrail Alert: {reason}",
            thread_id=req.thread_id or str(uuid.uuid4()),
            logs=[f"Blocked by input guardrails: {reason}"]
        )
        
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    # 2. Invoke Graph
    initial_state = {
        "question": req.question,
        "retry_count": 0,
        "logs": ["Query submitted."],
        "sql_approved": False,
        "cache_hit": False,
        "retrieval_strategy": "RAG",
        "web_search_results": None,
        "retrieved_chunks": [],
        "relevant_chunks": [],
        "hyde_document": None
    }
    
    # Run the graph
    events = graph_app.stream(initial_state, config, stream_mode="values")
    final_val = None
    for event in events:
        final_val = event
        
    # Check if the graph is interrupted (paused before execute_sql)
    state_snapshot = graph_app.get_state(config)
    
    if state_snapshot.next:
        # We are interrupted
        next_node = state_snapshot.next[0]
        if "execute_sql" in next_node:
            sql_query = state_snapshot.values.get("sql_query")
            return QueryResponse(
                status="pending_approval",
                answer="SQL query generated and requires human operator approval before execution.",
                strategy="Text2SQL",
                thread_id=thread_id,
                sql_query=sql_query,
                logs=state_snapshot.values.get("logs", [])
            )
            
    # Regular path completion
    return QueryResponse(
        status="success",
        answer=final_val.get("final_answer") or final_val.get("draft_answer"),
        strategy=final_val.get("retrieval_strategy"),
        thread_id=thread_id,
        logs=final_val.get("logs", [])
    )

@app.post("/query/resume", response_model=QueryResponse)
def resume_query(req: ResumeRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    
    # Fetch state snapshot
    state_snapshot = graph_app.get_state(config)
    if not state_snapshot.next:
        raise HTTPException(status_code=400, detail="No active paused execution found for this thread ID.")
        
    # Update state based on user approval
    graph_app.update_state(
        config,
        {"sql_approved": req.approved},
        as_node="text2sql"
    )
    
    # Resume stream execution
    events = graph_app.stream(None, config, stream_mode="values")
    final_val = None
    for event in events:
        final_val = event
        
    return QueryResponse(
        status="success",
        answer=final_val.get("final_answer"),
        strategy=final_val.get("retrieval_strategy"),
        thread_id=req.thread_id,
        logs=final_val.get("logs", [])
    )

@app.get("/cache/stats")
def get_cache_stats():
    try:
        info = redis_client.info()
        keys_count = redis_client.dbsize()
        
        # Calculate hitting rate
        # We increment key hits and misses in redis logic if real, or estimate
        # Since it's a stats dashboard, let's keep track of simple metrics
        hits = int(redis_client.get("stats:hits") or 0)
        misses = int(redis_client.get("stats:misses") or 0)
        total = hits + misses
        hit_rate = (hits / total) if total > 0 else 0.0
        
        return {
            "hit_rate": hit_rate,
            "key_count": keys_count,
            "memory_usage": info.get("used_memory_human", "0B"),
            "used_memory_bytes": info.get("used_memory", 0)
        }
    except Exception as e:
        logger.error(f"Error fetching cache stats: {e}")
        return {
            "hit_rate": 0.0,
            "key_count": 0,
            "memory_usage": "N/A (Offline)",
            "used_memory_bytes": 0
        }

@app.get("/eval/metrics")
def get_eval_metrics():
    db = SessionLocal()
    try:
        runs = db.query(EvalRun).order_by(EvalRun.timestamp.desc()).limit(10).all()
        return [
            {
                "timestamp": run.timestamp.isoformat(),
                "graph_version": run.graph_version,
                "faithfulness": run.faithfulness,
                "answer_relevancy": run.answer_relevancy,
                "context_recall": run.context_recall,
                "context_precision": run.context_precision
            }
            for run in runs
        ]
    except Exception as e:
        logger.error(f"Error fetching eval metrics: {e}")
        return []
    finally:
        db.close()
