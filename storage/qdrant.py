import os
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models

logger = logging.getLogger("storage.qdrant")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

COLLECTION_NAME = "k8s_runbooks"

_qdrant_client = None

def get_qdrant_client() -> QdrantClient:
    """Lazily initializes and returns the Qdrant client singleton.
    
    This avoids taking a file lock at module import time, which prevents
    concurrent processes (e.g. FastAPI + Streamlit) from conflicting on
    the local disk-based fallback client.
    """
    global _qdrant_client
    if _qdrant_client is not None:
        return _qdrant_client

    # Try remote Qdrant server first
    try:
        logger.info(f"Attempting to connect to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=3.0)
        client.get_collections()  # Ping to verify connectivity
        logger.info("Successfully connected to Qdrant Server.")
        _qdrant_client = client
        return _qdrant_client
    except Exception as e:
        logger.warning(f"Failed to connect to Qdrant Server: {e}. Falling back to local disk-based Qdrant client at './qdrant_db'.")

    # Fallback to file-based Qdrant client
    try:
        _qdrant_client = QdrantClient(path="./qdrant_db")
        logger.info("Local disk-based Qdrant client initialized.")
    except RuntimeError as e:
        if "already accessed" in str(e):
            logger.warning(f"Qdrant local storage locked by another process: {e}. Using in-memory client.")
            _qdrant_client = QdrantClient(location=":memory:")
            logger.info("In-memory Qdrant client initialized (read operations will return empty results).")
        else:
            raise e

    return _qdrant_client


# Backward-compatible module-level alias.
# Code that does `from storage.qdrant import qdrant_client` will get this proxy.
class _QdrantClientProxy:
    """Transparent proxy that lazily initializes the real client on first use."""

    def __getattr__(self, name):
        client = get_qdrant_client()
        return getattr(client, name)

qdrant_client = _QdrantClientProxy()


def init_qdrant():
    """Initializes Qdrant collection if not already existing."""
    client = get_qdrant_client()
    try:
        collections = client.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in collections)
        
        if not exists:
            logger.info(f"Creating Qdrant collection: {COLLECTION_NAME}...")
            # We use 1536 dimensions for text-embedding-3-small
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=1536,
                    distance=models.Distance.COSINE
                ),
                # We will also support sparse vectors for hybrid BM25 search
                sparse_vectors_config={
                    "bm25": models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=True
                        )
                    )
                }
            )
            logger.info(f"Collection {COLLECTION_NAME} created successfully.")
        else:
            logger.info(f"Qdrant collection {COLLECTION_NAME} already exists.")
    except Exception as e:
        logger.error(f"Error initializing Qdrant collection: {e}")
        raise e
