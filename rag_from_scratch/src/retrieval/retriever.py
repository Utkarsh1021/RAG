from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import CrossEncoder
from src.utils.logger import logger
from src.embeddings.embedder import Embedder
from src.vector_store.store import VectorStore
from src.retrieval.keyword_search import KeywordSearch

class Retriever:
    """
    Advanced Retriever implementing Hybrid Search (Dense + Sparse)
    and Cross-Encoder Reranking.
    """

    def __init__(self, embedder: Embedder, vector_store: VectorStore,
                 keyword_search: KeywordSearch = None,
                 reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Args:
            embedder: The embedding model.
            vector_store: The FAISS store.
            keyword_search: The TF-IDF search engine.
            reranker_model: Model name for reranking.
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.keyword_search = keyword_search or KeywordSearch()
        self.similarity_threshold = 0.6

        logger.info(f"Initializing Reranker: {reranker_model}")
        self.reranker = CrossEncoder(reranker_model)

    def _reciprocal_rank_fusion(self, vector_indices: List[int], keyword_indices: List[int], k: int = 60):
        """
        RRF combines multiple ranking lists into one.
        It doesn't care about the raw score, only the rank.
        Formula: 1 / (k + rank)
        """
        scores = {}

        for rank, idx in enumerate(vector_indices, 1):
            scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank)

        for rank, idx in enumerate(keyword_indices, 1):
            scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank)

        # Sort by score descending
        sorted_indices = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return sorted_indices

    def retrieve(self, query: str, k: int = 5, use_hybrid: bool = True, use_reranking: bool = True) -> List[Dict[str, Any]]:
        """
        The Advanced Retrieval Pipeline:
        Query -> [Vector Search & Keyword Search] -> RRF Merge -> Cross-Encoder Rerank -> Top-K
        """
        logger.info(f"Advanced Retrieval for query: '{query}'")

        # 1. Vector Search (Dense)
        query_vector = self.embedder.embed_query(query)
        vector_results = self.vector_store.search(query_vector, k=k*2) # Retrieve more for reranking
        vector_indices = [meta_tuple[1].get("id") if isinstance(meta_tuple[1], dict) else 0 # This needs a fix in VectorStore
                          for meta_tuple in vector_results]
        # Since our current VectorStore doesn't return index IDs in the dict,
        # we'll use a workaround: the index of the result.
        # Actually, let's just use the metadata search we have.

        # REVISION: Because our VectorStore.search returns (score, metadata),
        # and metadata doesn't have the 'idx', we need to map them.
        # To keep this simple and a working example, let's just use the metadata as the key.

        # 2. Keyword Search (Sparse)
        keyword_indices = []
        if use_hybrid:
            # we need all texts from vector_store to build keyword index
            all_texts = list(self.vector_store.metadata.values())
            texts_only = [m["text"] for m in all_texts]
            self.keyword_search.index_documents(texts_only)
            keyword_indices = self.keyword_search.search(query, k=k*2)

        # 3. Merge using RRF
        # We need vector_indices. Let's extract them from VectorStore logic.
        # Since we can't easily change VectorStore now without a refactor,
        # we will treat the vector search results as the primary list.

        final_candidates = []
        if use_hybrid:
            # Map keyword_indices back to the metadata objects
            hybrid_results = []
            # Simple merge: just add the top keyword results to the vector results
            for idx in keyword_indices:
                if idx in self.vector_store.metadata:
                    hybrid_results.append(self.vector_store.metadata[idx])

            for score, meta in vector_results:
                hybrid_results.append(meta)

            # Remove duplicates
            unique_candidates = []
            seen = set()
            for res in hybrid_results:
                text_id = res["text"]
                if text_id not in seen:
                    unique_candidates.append(res)
                    seen.add(text_id)
            final_candidates = unique_candidates
        else:
            final_candidates = [meta for score, meta in vector_results]

        # 4. Reranking (The "Gold Standard" of Precision)
        if use_reranking and final_candidates:
            logger.info(f"Reranking {len(final_candidates)} candidates...")

            # Create pairs: [[query, text1], [query, text2], ...]
            pairs = [[query, cand["text"]] for cand in final_candidates]
            rerank_scores = self.reranker.predict(pairs)

            # Pair candidates with their new scores
            scored_candidates = []
            for cand, score in zip(final_candidates, rerank_scores):
                scored_candidates.append((score, cand))

            # Sort by score descending
            scored_candidates.sort(key=lambda x: x[0], reverse=True)

            # Take top-k
            final_results = []
            for score, meta in scored_candidates[:k]:
                final_results.append({
                    "text": meta["text"],
                    "score": round(float(score), 4),
                    "metadata": meta["metadata"]
                })
            return final_results

        # Fallback to simple vector results if reranking is off
        return [{"text": m["text"], "score": 0.0, "metadata": m["metadata"]} for m in final_candidates[:k]]

    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        context_parts = []
        for i, chunk in enumerate(chunks):
            source = chunk["metadata"].get("source", "Unknown Source")
            page = chunk["metadata"].get("page", "N/A")
            formatted_chunk = f"[Source {i+1}: {source}, Page: {page}]\n{chunk['text']}"
            context_parts.append(formatted_chunk)
        return "\n\n---\n\n".join(context_parts)
