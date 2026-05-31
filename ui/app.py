import sys
import os
import streamlit as st
import pandas as pd
import requests
import uuid
import logging
from sqlalchemy import text

# Add root folder to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from storage.postgres import SessionLocal, Incident, Deployment, Alert, EvalRun
from storage.redis import redis_client

# Page config
st.set_page_config(
    page_title="Kubernetes SRE Copilot",
    page_icon="☸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Glassmorphism & Vibrant Dark Theme)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0c20 0%, #15102a 50%, #090714 100%);
        color: #e2e8f0;
    }
    
    /* Header Gradient styling */
    .title-gradient {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0b0918;
        border-right: 1px solid #2d264d;
    }
    
    /* Custom Card container */
    .sre-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }
    
    /* Strategy Badge indicator */
    .badge {
        padding: 0.35em 0.65em;
        font-size: 0.85em;
        font-weight: 600;
        border-radius: 30px;
        text-align: center;
        display: inline-block;
        margin-right: 0.5rem;
    }
    
    .badge-rag { background-color: #3b82f6; color: white; }
    .badge-crag { background-color: #f59e0b; color: white; }
    .badge-selfrag { background-color: #8b5cf6; color: white; }
    .badge-sql { background-color: #ec4899; color: white; }
    .badge-cache { background-color: #10b981; color: white; }
    
    /* Chat bubbles */
    .user-bubble {
        background-color: #2e1d52;
        border-radius: 15px 15px 0 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        max-width: 80%;
        float: right;
        clear: both;
        border: 1px solid #4a347d;
    }
    
    .bot-bubble {
        background-color: #15112e;
        border-radius: 15px 15px 15px 0;
        padding: 1rem;
        margin: 0.5rem 0;
        max-width: 80%;
        float: left;
        clear: both;
        border: 1px solid #292254;
    }
    
    .log-line {
        font-family: 'JetBrains Mono', monospace;
        color: #a78bfa;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

API_URL = "http://localhost:8000"

# Initialize Session State variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "pending_sql" not in st.session_state:
    st.session_state.pending_sql = None
if "pending_thread" not in st.session_state:
    st.session_state.pending_thread = None
if "last_strategy" not in st.session_state:
    st.session_state.last_strategy = "N/A"
if "last_chunks" not in st.session_state:
    st.session_state.last_chunks = []
if "last_cache_hit" not in st.session_state:
    st.session_state.last_cache_hit = False
if "last_logs" not in st.session_state:
    st.session_state.last_logs = []

def run_query_standalone(question: str, thread_id: str) -> dict:
    """Fallback local execution if FastAPI server is not running."""
    from guardrails.pipeline import run_input_guardrails
    from graph.workflow import graph_app
    # 1. Run Input Guardrails
    passed, reason = run_input_guardrails(question)
    if not passed:
        return {
            "status": "blocked",
            "answer": f"Guardrail Alert: {reason}",
            "strategy": "Blocked",
            "logs": [f"Blocked by input guardrails: {reason}"]
        }
        
    config = {"configurable": {"thread_id": thread_id}}
    
    initial_state = {
        "question": question,
        "retry_count": 0,
        "logs": ["Query submitted (Direct local invocation)."],
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
        
    state_snapshot = graph_app.get_state(config)
    
    if state_snapshot.next:
        next_node = state_snapshot.next[0]
        if "execute_sql" in next_node:
            sql_query = state_snapshot.values.get("sql_query")
            return {
                "status": "pending_approval",
                "answer": "SQL Query generated. Awaiting confirmation.",
                "strategy": "Text2SQL",
                "sql_query": sql_query,
                "chunks": [],
                "cache_hit": False,
                "logs": state_snapshot.values.get("logs", [])
            }
            
    # Complete execution
    return {
        "status": "success",
        "answer": final_val.get("final_answer") or final_val.get("draft_answer"),
        "strategy": final_val.get("retrieval_strategy"),
        "chunks": final_val.get("retrieved_chunks", []) or final_val.get("relevant_chunks", []),
        "cache_hit": final_val.get("cache_hit", False),
        "logs": final_val.get("logs", [])
    }

def resume_query_standalone(thread_id: str, approved: bool) -> dict:
    from graph.workflow import graph_app
    config = {"configurable": {"thread_id": thread_id}}
    
    # Update status
    graph_app.update_state(config, {"sql_approved": approved}, as_node="text2sql")
    
    # Resume
    events = graph_app.stream(None, config, stream_mode="values")
    final_val = None
    for event in events:
        final_val = event
        
    return {
        "status": "success",
        "answer": final_val.get("final_answer"),
        "strategy": final_val.get("retrieval_strategy"),
        "chunks": [],
        "cache_hit": False,
        "logs": final_val.get("logs", [])
    }

def submit_query_to_api(question: str):
    # Try calling API first, fallback to standalone local execution
    try:
        res = requests.post(f"{API_URL}/query", json={
            "question": question,
            "thread_id": st.session_state.thread_id
        }, timeout=5)
        
        if res.status_code == 200:
            data = res.json()
            return data
    except Exception as e:
        logging.warning(f"API server unreachable: {e}. Executing graph locally.")
        
    return run_query_standalone(question, st.session_state.thread_id)

def submit_approval_to_api(approved: bool):
    try:
        res = requests.post(f"{API_URL}/query/resume", json={
            "thread_id": st.session_state.pending_thread,
            "approved": approved
        }, timeout=5)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        logging.warning(f"API server unreachable: {e}. Executing resume locally.")
        
    return resume_query_standalone(st.session_state.pending_thread, approved)


# Sidebar layout with status indicators and search logs
with st.sidebar:
    st.markdown("### ☸️ SRE Session Metadata")
    st.text(f"Thread ID: {st.session_state.thread_id[:8]}...")
    
    # Strategy Badge display
    strategy_colors = {
        "RAG": "badge-rag",
        "CRAG": "badge-crag",
        "Self-RAG": "badge-selfrag",
        "Text2SQL": "badge-sql",
        "Cache": "badge-cache"
    }
    badge_cls = strategy_colors.get(st.session_state.last_strategy, "badge-rag")
    st.markdown(f"**Retrieval Strategy:** <span class='badge {badge_cls}'>{st.session_state.last_strategy}</span>", unsafe_allow_html=True)
    
    # Cache hit check
    cache_status = "💚 Hit" if st.session_state.last_cache_hit else "💔 Miss"
    st.markdown(f"**Cache Status:** `{cache_status}`")
    
    st.markdown("---")
    st.markdown("### 📊 Retrieved Context Chunks")
    if st.session_state.last_chunks:
        for idx, chunk in enumerate(st.session_state.last_chunks[:5], start=1):
            score_type = "RRF Score" if "rrf_score" in chunk else "Rerank Score"
            score = chunk.get("rrf_score") or chunk.get("rerank_score") or chunk.get("score", 0.0)
            
            with st.expander(f"Chunk {idx}: {chunk.get('title', 'Doc')} ({score:.4f})"):
                st.caption(f"Category: {chunk.get('category', 'Runbook')}")
                st.write(chunk.get("content", ""))
    else:
        st.info("No documents retrieved in the last run.")
        
    st.markdown("---")
    st.markdown("### 🪵 Trace Executions Logs")
    if st.session_state.last_logs:
        for log in st.session_state.last_logs:
            st.markdown(f"<span class='log-line'>▸ {log}</span>", unsafe_allow_html=True)
    else:
        st.caption("No logs recorded.")

# Tabs main interface
tab_chat, tab_db, tab_eval, tab_cache = st.tabs([
    "💬 Chat Playground", 
    "🗄️ Database Explorer", 
    "📈 Evaluation Analytics",
    "⚡ Cache Telemetry"
])

# TAB 1: Chat Playground
with tab_chat:
    st.markdown("<div class='title-gradient'>Kubernetes SRE Copilot</div>", unsafe_allow_html=True)
    st.markdown("##### Production-Grade Corrective & Self-Reflecting Ops Agent")
    
    # Message log display
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"<div class='user-bubble'>🧑‍💻 <b>Operator:</b><br>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-bubble'>☸️ <b>SRE Copilot:</b><br>{msg['content']}</div>", unsafe_allow_html=True)
            
    # Interrupt Dialog for SQL execution
    if st.session_state.pending_sql:
        st.markdown("<div style='clear:both;'></div>", unsafe_allow_html=True)
        st.warning("⚠️ **Human-In-The-Loop Approval Interruption**")
        st.code(st.session_state.pending_sql, language="sql")
        
        col1, col2 = st.columns([1, 10])
        with col1:
            if st.button("Run SQL", type="primary", key="approve_sql_btn"):
                # Resume execution
                with st.spinner("Executing SQL query..."):
                    result = submit_approval_to_api(approved=True)
                    st.session_state.messages.append({"role": "assistant", "content": result.get("answer", "")})
                    
                    st.session_state.last_strategy = result.get("strategy", "Text2SQL")
                    st.session_state.last_logs = result.get("logs", [])
                    st.session_state.last_chunks = []
                    
                    st.session_state.pending_sql = None
                    st.session_state.pending_thread = None
                    st.rerun()
        with col2:
            if st.button("Reject", key="reject_sql_btn"):
                result = submit_approval_to_api(approved=False)
                st.session_state.messages.append({"role": "assistant", "content": "SQL Query run was aborted by operator."})
                st.session_state.pending_sql = None
                st.session_state.pending_thread = None
                st.rerun()
                
    st.markdown("<div style='clear:both; margin-bottom: 2rem;'></div>", unsafe_allow_html=True)
    
    # Prompt submission
    if prompt := st.chat_input("Ask a Kubernetes runbook, cluster stats, or SQL query..."):
        # Add to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("Processing SRE prompt workflow..."):
            res = submit_query_to_api(prompt)
            
            if res.get("status") == "pending_approval":
                st.session_state.pending_sql = res.get("sql_query")
                st.session_state.pending_thread = res.get("thread_id")
                st.session_state.last_strategy = "Text2SQL"
                st.session_state.last_logs = res.get("logs", [])
                st.session_state.last_chunks = []
            else:
                st.session_state.messages.append({"role": "assistant", "content": res.get("answer", "")})
                st.session_state.last_strategy = res.get("strategy", "RAG")
                st.session_state.last_cache_hit = res.get("cache_hit", False)
                st.session_state.last_chunks = res.get("chunks", [])
                st.session_state.last_logs = res.get("logs", [])
                
        st.rerun()

# TAB 2: Database Explorer
with tab_db:
    st.markdown("### 🗄️ Kubernetes Ops PostgreSQL Database Schema")
    db = SessionLocal()
    try:
        st.markdown("#### `incidents` Table")
        incidents = db.query(Incident).all()
        if incidents:
            df_inc = pd.DataFrame([{
                "id": i.id, "title": i.title, "service": i.service, 
                "severity": i.severity, "status": i.status, 
                "created_at": i.created_at, "resolved_at": i.resolved_at
            } for i in incidents])
            st.dataframe(df_inc, use_container_width=True)
        else:
            st.info("No incident records seeded.")
            
        st.markdown("#### `deployments` Table")
        deployments = db.query(Deployment).all()
        if deployments:
            df_dep = pd.DataFrame([{
                "id": d.id, "service": d.service, "version": d.version, 
                "status": d.status, "created_at": d.created_at
            } for d in deployments])
            st.dataframe(df_dep, use_container_width=True)
        else:
            st.info("No deployment records seeded.")
            
        st.markdown("#### `alerts` Table")
        alerts = db.query(Alert).all()
        if alerts:
            df_al = pd.DataFrame([{
                "id": a.id, "alert_name": a.alert_name, 
                "severity": a.severity, "status": a.status, "created_at": a.created_at
            } for a in alerts])
            st.dataframe(df_al, use_container_width=True)
        else:
            st.info("No alert records seeded.")
    except Exception as e:
        st.error(f"Error querying local database tables: {e}")
    finally:
        db.close()

# TAB 3: Evaluation Analytics
with tab_db:
    # Adding a separate visual division
    pass
with tab_eval:
    st.markdown("### 📈 LLM Ragas Metrics Dashboard")
    
    # Query Postgres for historical runs
    db = SessionLocal()
    runs_data = []
    try:
        runs = db.query(EvalRun).order_by(EvalRun.timestamp.asc()).all()
        runs_data = [{
            "Timestamp": r.timestamp,
            "Version": r.graph_version,
            "Faithfulness": r.faithfulness,
            "Answer Relevancy": r.answer_relevancy,
            "Context Recall": r.context_recall,
            "Context Precision": r.context_precision
        } for r in runs]
    except Exception as e:
        st.error(f"Error querying Postgres for eval logs: {e}")
    finally:
        db.close()
        
    if runs_data:
        df_runs = pd.DataFrame(runs_data)
        st.markdown("#### Evaluation Metric Progress Trends")
        st.line_chart(df_runs, x="Timestamp", y=["Faithfulness", "Answer Relevancy", "Context Recall", "Context Precision"])
        
        st.markdown("#### Detailed Historical Runs")
        st.dataframe(df_runs.sort_values(by="Timestamp", ascending=False), use_container_width=True)
    else:
        st.info("No Ragas evaluation records found. Please trigger eval/run_eval.py to generate metrics.")

# TAB 4: Cache Telemetry
with tab_cache:
    st.markdown("### ⚡ Redis Query Cache Monitoring")
    
    # Stats logic fallback or REST query
    stats = {}
    try:
        res = requests.get(f"{API_URL}/cache/stats", timeout=2)
        if res.status_code == 200:
            stats = res.json()
    except Exception:
        # local direct stats
        try:
            info = redis_client.info()
            keys_count = redis_client.dbsize()
            hits = int(redis_client.get("stats:hits") or 0)
            misses = int(redis_client.get("stats:misses") or 0)
            total = hits + misses
            hit_rate = (hits / total) if total > 0 else 0.0
            stats = {
                "hit_rate": hit_rate,
                "key_count": keys_count,
                "memory_usage": info.get("used_memory_human", "0B")
            }
        except Exception as e:
            stats = {"hit_rate": 0.0, "key_count": 0, "memory_usage": f"Error: {e}"}
            
    col_rate, col_keys, col_mem = st.columns(3)
    with col_rate:
        st.metric(label="Cache Hit Rate", value=f"{stats.get('hit_rate', 0.0) * 100:.1f}%")
    with col_keys:
        st.metric(label="Total Cached Keys", value=stats.get("key_count", 0))
    with col_mem:
        st.metric(label="Redis Memory Overhead", value=stats.get("memory_usage", "0B"))
        
    st.markdown("---")
    st.markdown("##### Cache Validation Guidelines")
    st.info("Identical queries submitted within 1 hour bypass RAG execution flows completely, returning immediate cached results.")
