import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

INDEX_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

class SemanticRetriever:
    """
    Hybrid retriever that prefers Sentence-Transformers embeddings (when available),
    with an automatic, fast TF-IDF cosine similarity fallback.
    Returns:
    {
      "results": [
        {
          "conversation_id": "...",
          "similarity": 0.84,
          "customer_problem": "...",
          "resolution": "...",
          "quality": "HIGH"
        }
      ],
      "top_similarity": 0.84,
      "method": "sentence_transformers" | "tfidf"
    }
    """
    def __init__(self, use_embeddings: bool = True):
        self.corpus: List[Dict[str, Any]] = []
        self.problems: List[str] = []
        self.tfidf_vectorizer: TfidfVectorizer = None
        self.tfidf_matrix = None
        self.encoder = None
        self.embedding_matrix = None
        self.method = "tfidf"
        self._load_corpus()
        self._build_index(use_embeddings)

    def _load_corpus(self):
        corpus_path = INDEX_DIR / "conversations_amazonhelp.json"
        if not corpus_path.exists():
            from pipeline.prepare_demo_data import prepare_demo_data
            prepare_demo_data()
            
        with open(corpus_path, "r", encoding="utf-8") as f:
            all_data = json.load(f)
            # Only keep HIGH and MEDIUM resolution records for retrieval corpus
            self.corpus = [
                d for d in all_data 
                if d.get("resolution_quality") in ("HIGH", "MEDIUM") and d.get("resolution_text")
            ]
            self.problems = [d["problem_text"] for d in self.corpus]

    def _build_index(self, use_embeddings: bool = True):
        # 1. Always build TF-IDF as base / fallback
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.problems)
        
        # 2. Try loading Sentence-Transformers if requested
        if use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
                self.embedding_matrix = self.encoder.encode(self.problems, convert_to_numpy=True, normalize_embeddings=True)
                self.method = "sentence_transformers"
            except Exception:
                # Graceful fallback to TF-IDF
                self.method = "tfidf"

    def search(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        if not self.corpus:
            return {"results": [], "top_similarity": 0.0, "method": self.method}

        if self.method == "sentence_transformers" and self.encoder is not None:
            q_emb = self.encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            similarities = np.dot(self.embedding_matrix, q_emb.T).flatten()
        else:
            q_vec = self.tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(q_vec, self.tfidf_matrix).flatten()

        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            item = self.corpus[idx]
            results.append({
                "conversation_id": item["conversation_id"],
                "similarity": round(score, 4),
                "customer_problem": item["problem_text"],
                "resolution": item["resolution_text"],
                "quality": item["resolution_quality"],
                "intent": item.get("intent", "UNKNOWN")
            })

        top_sim = results[0]["similarity"] if results else 0.0
        return {
            "results": results,
            "top_similarity": top_sim,
            "method": self.method
        }

_retriever_instance = None

def get_retriever() -> SemanticRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = SemanticRetriever()
    return _retriever_instance

if __name__ == "__main__":
    retriever = get_retriever()
    query = "Where is my package? It was supposed to be delivered yesterday!"
    out = retriever.search(query, top_k=3)
    print(f"Retrieval Method: {out['method']}")
    print(f"Top Similarity: {out['top_similarity']}")
    for r in out["results"]:
        print(f"\n[Similarity: {r['similarity']}] Conv: {r['conversation_id']}")
        print(f"Problem: {r['customer_problem']}")
        print(f"Resolution: {r['resolution']}")
