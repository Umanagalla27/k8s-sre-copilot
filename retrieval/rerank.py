import logging

logger = logging.getLogger("retrieval.rerank")

_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info("Initializing CrossEncoder 'cross-encoder/ms-marco-MiniLM-L-6-v2' for re-ranking...")
            # We set a limit on max length to be memory efficient
            _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
            logger.info("CrossEncoder re-ranker successfully loaded.")
        except Exception as e:
            logger.warning(f"Could not load CrossEncoder: {e}. Using TF-IDF/Overlap fallback re-ranker.")
            _reranker = "fallback"
    return _reranker

def fallback_rerank(query: str, chunks: list[dict]) -> list[dict]:
    """Fallback scoring using simple word overlap and title match boosts."""
    query_words = set(query.lower().split())
    scored_chunks = []
    
    for chunk in chunks:
        content_words = set(chunk["content"].lower().split())
        title_words = set(chunk["title"].lower().split())
        
        # Word overlap score
        overlap = len(query_words.intersection(content_words))
        title_overlap = len(query_words.intersection(title_words))
        
        # Calculate a pseudo relevance score
        score = overlap * 0.1 + title_overlap * 0.5
        
        # Retain original search context ranking indicator
        score += (chunk.get("rrf_score", 0.0) * 10)
        
        scored_chunks.append((score, chunk))
        
    # Sort descending
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored_chunks]

def rerank_documents(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """Re-ranks retrieved chunks using CrossEncoder or fallback similarity and returns top_k."""
    if not chunks:
        return []
        
    reranker = get_reranker()
    
    if reranker == "fallback":
        logger.info("Executing fallback re-ranking.")
        ranked_chunks = fallback_rerank(query, chunks)
    else:
        logger.info(f"Executing CrossEncoder re-ranking on {len(chunks)} chunks...")
        try:
            # Prepare pairs: [query, doc_text]
            pairs = [[query, chunk["content"]] for chunk in chunks]
            scores = reranker.predict(pairs)
            
            # Match scores back to chunks
            scored_chunks = []
            for score, chunk in zip(scores, chunks):
                chunk_copy = chunk.copy()
                chunk_copy["rerank_score"] = float(score)
                scored_chunks.append(chunk_copy)
                
            # Sort descending by re-rank score
            scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
            ranked_chunks = scored_chunks
        except Exception as e:
            logger.error(f"Error during CrossEncoder predict: {e}. Falling back...")
            ranked_chunks = fallback_rerank(query, chunks)

    # Log ranking changes
    logger.info("--- Re-ranking rank shifts ---")
    for i, doc in enumerate(ranked_chunks[:top_k], start=1):
        original_rank = doc.get("hybrid_rank", "N/A")
        logger.info(f"Top {i} Doc ID {doc['id']} ('{doc['title']}'): Hybrid Rank: {original_rank} -> Reranked Rank: {i}")
        
    return ranked_chunks[:top_k]
