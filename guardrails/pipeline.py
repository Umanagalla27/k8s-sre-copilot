import os
import re
import time
import logging
from storage.redis import redis_client
from storage.postgres import SessionLocal, AuditLog

logger = logging.getLogger("guardrails.pipeline")

# Regex rules
PII_EMAIL_REGEX = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
PII_IP_REGEX = re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')
SECRET_KEY_REGEX = re.compile(r'(?i)(sk-[a-zA-Z0-9]{48}|AIza[yI][a-zA-Z0-9-_]{35}|password\s*=\s*[\'"][^\'"]+[\'"])')
PROMPT_INJECTION_REGEX = re.compile(
    r'(?i)(ignore previous instructions|system prompt|jailbreak|override policy|act as a|developer mode|dan mode|do anything now)'
)

TOXICITY_WORDS = {"abuse", "hate", "kill", "hack", "bypass", "exploit", "attack", "spam", "scam"}
K8S_TOPIC_KEYWORDS = {
    "kubernetes", "k8s", "docker", "pod", "container", "node", "cluster", "deployment",
    "service", "ingress", "runbook", "alert", "prometheus", "grafana", "helm", "yaml",
    "kubectl", "namespace", "pvc", "pv", "daemonset", "statefulset", "configmap", "secret",
    "postgres", "database", "sql", "metric", "cpu", "memory", "oomkilled", "crashloopbackoff",
    "coredns", "dns", "eviction", "replica", "scheduler", "kubelet", "incident"
}

def log_block_to_db(query: str, layer: str, reason: str):
    db = SessionLocal()
    try:
        log = AuditLog(query=query, blocked_by=layer, reason=reason)
        db.add(log)
        db.commit()
        logger.info(f"Blocked query logged to database. Layer: {layer}")
    except Exception as e:
        logger.error(f"Error logging block to DB: {e}")
        db.rollback()
    finally:
        db.close()

# Presidio Analyzer Initialization (optional fallback)
_analyzer = None
def get_presidio_analyzer():
    global _analyzer
    if _analyzer is None:
        try:
            from presidio_analyzer import AnalyzerEngine
            _analyzer = AnalyzerEngine()
        except Exception as e:
            logger.warning(f"presidio-analyzer not available ({type(e).__name__}: {e}). Falling back to regex-based PII detection.")
            _analyzer = "fallback"
    return _analyzer

def check_prompt_injection(query: str) -> tuple[bool, str]:
    if PROMPT_INJECTION_REGEX.search(query):
        return False, "Prompt injection pattern detected via regex."
    return True, ""

def check_pii(query: str) -> tuple[bool, str]:
    # Regex checks
    if PII_EMAIL_REGEX.search(query):
        return False, "PII email address detected."
    if PII_IP_REGEX.search(query):
        return False, "PII IP address detected."
    
    # Presidio check if available
    analyzer = get_presidio_analyzer()
    if analyzer and analyzer != "fallback":
        try:
            results = analyzer.analyze(text=query, language="en")
            # Filter for specific entity types
            blocked_types = {"PERSON", "EMAIL_ADDRESS", "IP_ADDRESS", "PHONE_NUMBER", "US_SSN"}
            for res in results:
                if res.entity_type in blocked_types and res.score > 0.6:
                    return False, f"PII entity '{res.entity_type}' detected."
        except Exception as e:
            logger.error(f"Presidio PII analyzer error: {e}")
            
    return True, ""

def check_toxicity(query: str) -> tuple[bool, str]:
    words = set(re.findall(r'\w+', query.lower()))
    toxic_hits = words.intersection(TOXICITY_WORDS)
    if toxic_hits:
        return False, f"Toxic keyword(s) detected: {list(toxic_hits)}."
    return True, ""

def check_topic_policy(query: str) -> tuple[bool, str]:
    words = re.findall(r'\w+', query.lower())
    has_match = False
    for word in words:
        if word in K8S_TOPIC_KEYWORDS:
            has_match = True
            break
        # Check plural form (e.g., pods, incidents, deployments, alerts)
        if word.endswith('s') and word[:-1] in K8S_TOPIC_KEYWORDS:
            has_match = True
            break
    if not has_match:
        return False, "Topic policy violation: Prompt does not appear related to Kubernetes or DevOps infrastructure."
    return True, ""

def check_input_length(query: str) -> tuple[bool, str]:
    # Standard estimate: 1 token ~= 4 characters or 0.75 words.
    # Exact check: split on spaces
    token_count = len(query.split())
    if token_count > 2000:
        return False, f"Input length exceeded. Query contains {token_count} tokens (max 2000)."
    return True, ""

def check_secrets(query: str) -> tuple[bool, str]:
    if SECRET_KEY_REGEX.search(query):
        return False, "Secret or credentials pattern detected in prompt."
    return True, ""

def check_rate_limiting(user_id: str) -> tuple[bool, str]:
    key = f"rate_limit:{user_id}"
    limit = 15 # limit: 15 queries per minute
    window = 60 # seconds
    
    now = time.time()
    try:
        redis_client.zremrangebyscore(key, 0, now - window)
        current_requests = redis_client.zcard(key)
        if current_requests >= limit:
            return False, f"Rate limit exceeded. Maximum {limit} requests per minute allowed."
        redis_client.zadd(key, {str(now): now})
        redis_client.expire(key, window * 2)
    except Exception as e:
        logger.error(f"Redis rate limiting error: {e}")
        # Soft fail: permit execution if Redis is down/faulty
    return True, ""

def check_output_faithfulness(answer: str, retrieved_contexts: list[str]) -> tuple[bool, str]:
    """Verification layer: Verify the answer makes references/citations to retrieved documents."""
    if not retrieved_contexts:
        return True, "" # Skip if web search or SQL was executed instead of doc RAG
    
    # Check if answer makes any reference to keywords in chunks
    lowercase_ans = answer.lower()
    found_evidence = False
    for context in retrieved_contexts:
        # Check if keywords overlap or sentences exist
        # We look for substantial words of length > 5
        context_words = [w for w in re.findall(r'\w+', context.lower()) if len(w) > 5]
        matches = [w for w in context_words if w in lowercase_ans]
        if len(matches) > 3:
            found_evidence = True
            break
            
    if not found_evidence and len(retrieved_contexts) > 0:
        # Warning: if answer doesn't reference the contexts, it might be unfaithful
        return False, "Output failed faithfulness check: Generated answer does not align with or cite retrieved contexts."
    return True, ""

def scrub_output_pii(answer: str) -> str:
    """Redacts IP addresses and email patterns from output to ensure data security."""
    scrubbed = PII_EMAIL_REGEX.sub("[EMAIL REDACTED]", answer)
    scrubbed = PII_IP_REGEX.sub("[IP REDACTED]", scrubbed)
    return scrubbed

def run_input_guardrails(query: str, user_id: str = "default_user") -> tuple[bool, str]:
    """Runs layers 1-6 + 9 sequentially. Logs failure and blocks query if violated."""
    
    layers = [
        ("Rate Limiter", lambda: check_rate_limiting(user_id)),
        ("Input Length", lambda: check_input_length(query)),
        ("Prompt Injection", lambda: check_prompt_injection(query)),
        ("PII Detector", lambda: check_pii(query)),
        ("Secret Scanner", lambda: check_secrets(query)),
        ("Toxicity Filter", lambda: check_toxicity(query)),
        ("Topic Policy", lambda: check_topic_policy(query)),
    ]
    
    for layer_name, check_func in layers:
        passed, reason = check_func()
        if not passed:
            log_block_to_db(query, layer_name, reason)
            return False, f"Blocked by {layer_name}: {reason}"
            
    return True, ""
