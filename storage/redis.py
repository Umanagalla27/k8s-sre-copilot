import os
import logging
import time
import json
import threading

logger = logging.getLogger("storage.redis")

class InMemoryRedisMock:
    """Fallback in-memory Redis client with TTL and basic command support."""
    def __init__(self):
        self._store = {}
        self._ttls = {}
        self._lock = threading.Lock()
        logger.warning("Using in-memory Redis mock instead of a live Redis server.")

    def get(self, key):
        with self._lock:
            if key in self._store:
                # Check expiration
                if self._ttls[key] is not None and time.time() > self._ttls[key]:
                    del self._store[key]
                    del self._ttls[key]
                    return None
                return self._store[key]
            return None

    def set(self, key, value, ex=None):
        with self._lock:
            self._store[key] = value
            if ex:
                self._ttls[key] = time.time() + ex
            else:
                self._ttls[key] = None
            return True

    def info(self, section=None):
        # Mock Redis INFO payload
        return {
            "used_memory": len(self._store) * 100, # mock size estimation
            "used_memory_human": f"{len(self._store) * 100}B",
            "db0": {"keys": len(self._store)}
        }

    def dbsize(self):
        with self._lock:
            # clean expired keys first
            now = time.time()
            expired = [k for k, t in self._ttls.items() if t is not None and now > t]
            for k in expired:
                del self._store[k]
                del self._ttls[k]
            return len(self._store)

    # Sliding window rate limiter support using list/sorted-set mock
    def zadd(self, key, mapping):
        with self._lock:
            if key not in self._store:
                self._store[key] = []
            # mapping is {member: score}
            for member, score in mapping.items():
                self._store[key].append((score, member))
            self._store[key].sort()
            return len(mapping)

    def zremrangebyscore(self, key, min_score, max_score):
        with self._lock:
            if key not in self._store:
                return 0
            original_len = len(self._store[key])
            self._store[key] = [item for item in self._store[key] if not (min_score <= item[0] <= max_score)]
            removed = original_len - len(self._store[key])
            return removed

    def zcard(self, key):
        with self._lock:
            return len(self._store.get(key, []))

    def expire(self, key, seconds):
        # We ignore expire on sets for mock simplicity
        return True

# Redis connection setup
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

redis_client = None

try:
    import redis
    logger.info("Attempting to connect to Redis...")
    # Parse URL
    redis_client = redis.from_url(REDIS_URL, socket_timeout=2, decode_responses=True)
    # Test connection
    redis_client.ping()
    logger.info("Successfully connected to Redis.")
except Exception as e:
    logger.warning(f"Failed to connect to Redis: {e}. Falling back to InMemoryRedisMock.")
    redis_client = InMemoryRedisMock()
