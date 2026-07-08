from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils.logger import logger

class KeywordSearch:
    """
    The KeywordSearch class implements 'Sparse Retrieval' using TF-IDF.
    This is essential for catching exact matches (IDs, names, specific terms)
    that dense vector search often misses.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self.documents = []

    def index_documents(self, documents: List[str]):
        """
        Builds the TF-IDF matrix for the provided documents.
        """
        if not documents:
            return

        self.documents = documents
        # fit_transform calculates the TF-IDF score for every word in every document
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        logger.info(f"Keyword index built for {len(documents)} chunks.")

    def search(self, query: str, k: int = 5) -> List[int]:
        """
        Returns the indices of the top-k documents most similar to the query.
        """
        if self.tfidf_matrix is None:
            return []

        # Transform query into the same TF-IDF space
        query_vec = self.vectorizer.transform([query])

        # Compute cosine similarity between query vector and all document vectors
        # similarity = (A . B) / (||A|| ||B||)
        similarities = (self.tfidf_matrix * query_vec.T).toarray().flatten()

        # Get indices of the top k scores
        top_indices = np.argsort(similarities)[::-1][:k]

        return top_indices.tolist()
