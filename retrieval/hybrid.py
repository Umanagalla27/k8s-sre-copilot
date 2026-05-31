import logging
from storage.qdrant import get_qdrant_client, COLLECTION_NAME
from retrieval.embed import get_embedding
from retrieval.bm25 import get_bm25_retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("retrieval.hybrid")

def hybrid_search(query: str, top_k: int = 20) -> list[dict]:
    """Combines dense vector search and sparse BM25 search using Reciprocal Rank Fusion (RRF)."""
    # 1. Run Dense Vector Search
    dense_results = []
    try:
        query_vector = get_embedding(query)
        client = get_qdrant_client()
        qdrant_res = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k
        )
        for idx, hit in enumerate(qdrant_res.points, start=1):
            dense_results.append({
                "id": hit.id,
                "rank": idx,
                "title": hit.payload.get("title", ""),
                "category": hit.payload.get("category", ""),
                "content": hit.payload.get("content", ""),
                "score": hit.score
            })
    except Exception as e:
        logger.error(f"Error performing dense search: {e}")

    # 2. Run Sparse BM25 Search
    bm25_retriever = get_bm25_retriever()
    sparse_results = bm25_retriever.search(query, top_k=top_k)

    # 3. Reciprocal Rank Fusion (RRF)
    # RRF constant k
    k = 60
    rrf_scores = {}
    doc_details = {}

    # Accumulate dense ranks
    for item in dense_results:
        doc_id = item["id"]
        rank = item["rank"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
        doc_details[doc_id] = {
            "id": doc_id,
            "title": item["title"],
            "category": item["category"],
            "content": item["content"],
            "dense_rank": rank,
            "sparse_rank": None
        }

    # Accumulate sparse ranks
    for item in sparse_results:
        doc_id = item["id"]
        rank = item["rank"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
        if doc_id not in doc_details:
            doc_details[doc_id] = {
                "id": doc_id,
                "title": item["title"],
                "category": item["category"],
                "content": item["content"],
                "dense_rank": None,
                "sparse_rank": rank
            }
        else:
            doc_details[doc_id]["sparse_rank"] = rank

    # Sort documents by RRF score descending
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    hybrid_results = []
    for rank, (doc_id, rrf_score) in enumerate(sorted_docs[:top_k], start=1):
        details = doc_details[doc_id]
        hybrid_results.append({
            "id": doc_id,
            "title": details["title"],
            "category": details["category"],
            "content": details["content"],
            "rrf_score": rrf_score,
            "hybrid_rank": rank,
            "dense_rank": details["dense_rank"],
            "sparse_rank": details["sparse_rank"]
        })

    logger.info(f"Hybrid search returned {len(hybrid_results)} documents for query '{query}'")
    return hybrid_results
