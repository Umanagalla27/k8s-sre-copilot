import os
import logging
import numpy as np
from openai import OpenAI

logger = logging.getLogger("retrieval.embed")

_client = None
_local_model = None

def get_openai_client():
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            _client = OpenAI(api_key=api_key)
        else:
            logger.warning("OPENAI_API_KEY environment variable not found. OpenAI client cannot be initialized.")
    return _client

def get_local_model():
    global _local_model
    if _local_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading local sentence-transformers model 'all-MiniLM-L6-v2' for fallback embeddings...")
            _local_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Local model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load sentence-transformers: {e}")
            raise e
    return _local_model

import math

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Generates 1536-dimensional embeddings for a list of texts.
    
    If OpenAI is available, it uses 'text-embedding-3-small'.
    If not, it uses a local model (384-d) and pads it with zeros to 1536-d.
    If local model loading fails, it uses a deterministic pseudo-random hash fallback.
    """
    client = get_openai_client()
    if client:
        try:
            logger.info(f"Generating embeddings using OpenAI for {len(texts)} texts...")
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.warning(f"OpenAI embedding generation failed: {e}. Trying fallback model...")
            
    # Fallback to SentenceTransformers
    try:
        model = get_local_model()
        embeddings_384 = model.encode(texts)
        
        # Pad to 1536 dimensions
        padded_embeddings = []
        for emb in embeddings_384:
            emb_norm = emb / np.linalg.norm(emb) if np.linalg.norm(emb) > 0 else emb
            padded = np.zeros(1536)
            padded[:384] = emb_norm
            padded_embeddings.append(padded.tolist())
            
        return padded_embeddings
    except Exception as e:
        logger.warning(f"SentenceTransformers fallback failed: {e}. Using hash-based deterministic embedding mock.")
        # Pure python deterministic generator
        import hashlib
        import random
        hashed_embeddings = []
        for txt in texts:
            # Seed random generator with SHA-256 of text
            seed = int(hashlib.sha256(txt.encode('utf-8')).hexdigest(), 16) % (2**32)
            rng = random.Random(seed)
            # Create a 1536 float list with values between -1 and 1
            emb = [rng.uniform(-1, 1) for _ in range(1536)]
            # Normalize
            norm = math.sqrt(sum(x*x for x in emb))
            if norm > 0:
                emb = [x/norm for x in emb]
            hashed_embeddings.append(emb)
        return hashed_embeddings

def get_embedding(text: str) -> list[float]:
    return get_embeddings([text])[0]
