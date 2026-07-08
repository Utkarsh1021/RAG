import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
from src.utils.logger import logger

class Embedder:
    """
    The Embedder class transforms raw text into dense vector representations.
    It acts as the bridge between human-readable text and machine-readable
    mathematical space.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the embedding model.

        Args:
            model_name: The name of the Sentence-Transformer model.
                       'all-MiniLM-L6-v2' is a great balance of speed and performance.
        """
        logger.info(f"Initializing embedding model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Vector dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to load embedding model {model_name}: {e}")
            raise e

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Generates embeddings for a list of document chunks.

        Args:
            texts: A list of strings (the chunks).

        Returns:
            A numpy array of shape (len(texts), dimension).
        """
        if not texts:
            return np.array([])

        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        # encode() handles batching internally for efficiency
        embeddings = self.model.encode(texts, show_progress_bar=True)

        # Ensure the result is a numpy array for FAISS compatibility
        return np.array(embeddings).astype('float32')

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generates an embedding for a single user query.

        Args:
            query: The user's question.

        Returns:
            A numpy array of shape (1, dimension).
        """
        if not query:
            return np.array([]).reshape(1, self.dimension)

        # We wrap the query in a list because encode expects a list of texts
        embedding = self.model.encode([query])

        return np.array(embedding).astype('float32')

    def get_dimension(self) -> int:
        """Returns the dimensionality of the vectors produced by the model."""
        return self.dimension
