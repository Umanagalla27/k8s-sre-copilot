import json
import math
import os
import re

CORPUS_PATH = "./storage/runbooks_corpus.json"

def tokenize(text: str) -> list[str]:
    # Lowercase and keep alphanumeric words
    return re.findall(r'\w+', text.lower())

class BM25Retriever:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.documents = []
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lens = []
        self.doc_freqs = {}
        self.idf = {}
        self.term_freqs = []
        
        self.load_corpus()

    def load_corpus(self):
        if not os.path.exists(CORPUS_PATH):
            return
        
        with open(CORPUS_PATH, "r", encoding="utf-8") as f:
            self.documents = json.load(f)
            
        self.corpus_size = len(self.documents)
        if self.corpus_size == 0:
            return
            
        # Tokenize all documents
        tokenized_docs = [tokenize(doc["content"]) for doc in self.documents]
        self.doc_lens = [len(doc) for doc in tokenized_docs]
        self.avg_doc_len = sum(self.doc_lens) / self.corpus_size
        
        # Calculate term & doc frequencies
        for doc in tokenized_docs:
            frequencies = {}
            for term in doc:
                frequencies[term] = frequencies.get(term, 0) + 1
            self.term_freqs.append(frequencies)
            
            # Document frequency
            for term in frequencies.keys():
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1
                
        # Calculate IDF
        for term, df in self.doc_freqs.items():
            # Standard BM25 IDF formula
            self.idf[term] = math.log(1 + (self.corpus_size - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Runs BM25 search over the corpus and returns top documents with scores."""
        if not self.documents:
            # Re-attempt loading in case corpus was just ingested
            self.load_corpus()
            if not self.documents:
                return []
                
        query_terms = tokenize(query)
        scores = []
        
        for idx, doc in enumerate(self.documents):
            score = 0.0
            doc_len = self.doc_lens[idx]
            tf_dict = self.term_freqs[idx]
            
            for term in query_terms:
                if term not in tf_dict:
                    continue
                
                tf = tf_dict[term]
                idf = self.idf.get(term, 0.0)
                
                # BM25 term weighting formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                score += idf * (numerator / denominator)
                
            scores.append((score, doc))
            
        # Sort descending by score
        scores.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for rank, (score, doc) in enumerate(scores[:top_k], start=1):
            results.append({
                "score": score,
                "rank": rank,
                "id": doc["id"],
                "title": doc["title"],
                "category": doc["category"],
                "content": doc["content"]
            })
            
        return results

# Shared singleton retriever instance
_retriever = None

def get_bm25_retriever() -> BM25Retriever:
    global _retriever
    if _retriever is None:
        _retriever = BM25Retriever()
    return _retriever
