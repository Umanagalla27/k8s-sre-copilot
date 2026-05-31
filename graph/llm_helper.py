import os
import re
import json
import logging
from openai import OpenAI

logger = logging.getLogger("graph.llm_helper")

def get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return OpenAI(api_key=api_key)
    return None

def llm_classify_intent(question: str) -> str:
    """Classifies user intent into 'sql' (metrics/alerts counts) or 'rag' (runbooks/incidents content)."""
    client = get_client()
    if client:
        try:
            prompt = (
                "You are an routing agent. Classify the user query into one of two categories:\n"
                "- 'sql': if the user asks for metrics, counts, lists, statistics, SLA, active alert counts, "
                "or questions requiring database aggregation/lookups (e.g., 'how many pods crashed last week?', "
                "'list active alerts', 'average incident duration').\n"
                "- 'rag': if the user asks for conceptual explanations, troubleshooting procedures, runbooks, "
                "root cause analysis, or how-to documentation (e.g., 'how to resolve OOMKilled pods?', 'why is my pod crashing?').\n\n"
                f"Query: {question}\n\n"
                "Respond with exactly one word: 'sql' or 'rag'."
            )
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=5
            )
            val = response.choices[0].message.content.strip().lower()
            if "sql" in val:
                return "sql"
            return "rag"
        except Exception as e:
            logger.warning(f"LLM classify intent failed: {e}. Falling back to rule-based classification.")
            
    # Fallback rule-based
    q = question.lower()
    sql_keywords = {
        "how many", "count", "average", "sla", "timestamp", "list alerts", "active alerts",
        "incidents last week", "number of", "crashed pods count", "failed deployments"
    }
    for kw in sql_keywords:
        if kw in q:
            return "sql"
    return "rag"

def llm_hyde(question: str) -> str:
    """Generates a hypothetical runbook or incident response to expand queries."""
    client = get_client()
    if client:
        try:
            prompt = (
                "Write a short, highly technical, hypothetical runbook snippet, error log, or system guide "
                f"that would answer the following SRE question: '{question}'.\n"
                "Keep it concise, and write only the technical solution description."
            )
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"LLM HyDE generation failed: {e}. Using deterministic fallback.")
            
    return f"Document discussing Kubernetes troubleshooting, status, events, and root cause analysis for {question}."

def llm_grade_chunk(question: str, chunk_content: str) -> str:
    """Grades chunk relevance: YES or NO."""
    client = get_client()
    if client:
        try:
            prompt = (
                "Analyze if the following document chunk contains information relevant to answering the question.\n"
                f"Question: {question}\n"
                f"Document: {chunk_content}\n\n"
                "Answer YES if the chunk is relevant, or NO if it is not. Reply with exactly one word: 'YES' or 'NO'."
            )
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=5
            )
            val = response.choices[0].message.content.strip().upper()
            if "YES" in val:
                return "YES"
            return "NO"
        except Exception as e:
            logger.warning(f"LLM grade chunk failed: {e}. Falling back to overlap check.")
            
    # Fallback overlap
    q_words = set(question.lower().split())
    doc_words = set(chunk_content.lower().split())
    overlap = q_words.intersection(doc_words)
    if len(overlap) >= 2:
        return "YES"
    return "NO"

def llm_generate_answer(question: str, contexts: list[str]) -> str:
    """Generates final runbook resolution answer based on retrieved contexts."""
    client = get_client()
    formatted_contexts = "\n\n".join([f"Source {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])
    
    if client:
        try:
            prompt = (
                "You are an expert Kubernetes SRE Copilot. Generate a clear, highly technical, "
                "actionable troubleshooting guide or response answering the question using ONLY the provided contexts.\n"
                "Provide step-by-step commands and reference original terms where appropriate. Citations are mandatory.\n\n"
                f"Question: {question}\n\n"
                f"Contexts:\n{formatted_contexts}\n\n"
                "Response:"
            )
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=800
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"LLM generation failed: {e}. Using deterministic fallback.")
            
    # Fallback generation synthesis
    if not contexts:
        return f"Could not find any relevant documentation to answer: '{question}'."
    
    answer_parts = []
    answer_parts.append(f"Based on retrieved SRE documentation, here is the resolution for: '{question}'\n")
    for i, ctx in enumerate(contexts):
        first_sentence = ctx.split('.')[0] + "."
        answer_parts.append(f"- [Source {i+1}]: {first_sentence} Detailed reference: {ctx[:200]}...")
    return "\n".join(answer_parts)

def llm_check_hallucination(answer: str, contexts: list[str]) -> dict:
    """Checks if answer is supported by the contexts."""
    client = get_client()
    formatted_contexts = "\n\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(contexts)])
    
    if client:
        try:
            prompt = (
                "Determine if the draft answer contains statements or information NOT supported by the retrieved contexts (hallucination).\n"
                "Return a JSON object with exactly two fields:\n"
                "1. 'hallucination': boolean (true if answer contradicts or is unsupported by context, false if fully supported)\n"
                "2. 'reason': string explaining your rating.\n\n"
                f"Contexts:\n{formatted_contexts}\n\n"
                f"Draft Answer:\n{answer}\n\n"
                "JSON response:"
            )
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return json.loads(response.choices[0].message.content.strip())
        except Exception as e:
            logger.warning(f"LLM check hallucination failed: {e}. Using fallback.")
            
    return {"hallucination": False, "reason": "Fallback assumed faithful."}

def llm_check_usefulness(question: str, answer: str) -> dict:
    """Checks if answer actually resolves the question."""
    client = get_client()
    if client:
        try:
            prompt = (
                "Determine if the answer successfully resolves the user's question.\n"
                "Return a JSON object with exactly two fields:\n"
                "1. 'useful': boolean (true if answer provides a solution or answers the prompt, false if vague or unhelpful)\n"
                "2. 'reason': string explaining your rating.\n\n"
                f"Question: {question}\n"
                f"Answer:\n{answer}\n\n"
                "JSON response:"
            )
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return json.loads(response.choices[0].message.content.strip())
        except Exception as e:
            logger.warning(f"LLM check usefulness failed: {e}. Using fallback.")
            
    # Simple length-based check for usefulness
    if len(answer) > 50:
        return {"useful": True, "reason": "Answer length indicates sufficiency."}
    return {"useful": False, "reason": "Answer is too short to be useful."}

def llm_rewrite_query(question: str) -> str:
    """Rewrites query to improve retrieval precision."""
    client = get_client()
    if client:
        try:
            prompt = (
                "You are an SRE operator. Rewrite this user question to optimize vector search retrieval "
                "over technical Kubernetes runbooks. Keep it concise, focused on key concepts.\n"
                f"Original question: {question}\n"
                "Rewritten question:"
            )
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=40
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"LLM query rewrite failed: {e}. Using fallback.")
            
    return f"{question} troubleshooting"

def llm_generate_sql(question: str) -> str:
    """Generates read-only SQL queries against PostgreSQL tables: incidents, deployments, alerts."""
    client = get_client()
    schema_desc = (
        "Table schema info:\n"
        "1. Table 'incidents': columns: id (integer), title (varchar), service (varchar), severity (varchar), status (varchar), created_at (timestamp), resolved_at (timestamp)\n"
        "2. Table 'deployments': columns: id (integer), service (varchar), version (varchar), status (varchar), created_at (timestamp)\n"
        "3. Table 'alerts': columns: id (integer), alert_name (varchar), severity (varchar), status (varchar), created_at (timestamp)\n\n"
        "Generate only valid read-only SQL SELECT queries. Avoid modifications. Wrap SQL in ```sql blocks."
    )
    
    if client:
        try:
            prompt = (
                "You are a database engineer. Generate a single postgres SELECT statement to answer the SRE question.\n"
                f"{schema_desc}\n\n"
                f"Question: {question}\n\n"
                "SQL Query:"
            )
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=150
            )
            content = response.choices[0].message.content.strip()
            match = re.search(r'```sql\s*(.*?)\s*```', content, re.DOTALL)
            if match:
                return match.group(1).strip()
            return content
        except Exception as e:
            logger.warning(f"LLM generate SQL failed: {e}. Using fallback templates.")
            
    # Fallback templates
    q = question.lower()
    if "how many pods crashed" in q or "crashlooping" in q:
        return "SELECT count(*) FROM alerts WHERE alert_name = 'KubePodCrashLooping';"
    elif "how many incidents" in q or "incident count" in q:
        return "SELECT count(*) FROM incidents;"
    elif "list active alerts" in q or "active alerts" in q:
        return "SELECT * FROM alerts WHERE status = 'Firing';"
    elif "failed deployments" in q or "deployments failed" in q:
        return "SELECT * FROM deployments WHERE status = 'Failed';"
    elif "postgres" in q and "oomkilled" in q:
        return "SELECT * FROM incidents WHERE service = 'postgres' AND title LIKE '%OOMKilled%';"
    return "SELECT * FROM incidents ORDER BY created_at DESC LIMIT 5;"
