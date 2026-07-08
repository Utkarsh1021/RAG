import faiss
import numpy as np
import pickle
from typing import List, Tuple, Dict, Any
from pathlib import Path
from src.utils.logger import logger

class VectorStore:
    """
    The VectorStore class manages the storage and retrieval of embeddings.
    It uses FAISS for efficient similarity search and a local pickle file
    for metadata mapping.
    """

    def __init__(self, dimension: int, index_path: str = "vector_store/faiss_index.bin", metadata_path: str = "vector_store/metadata.pkl"):
        """
        Initializes the Vector Store.

        Args:
            dimension: The dimensionality of the vectors (must match the embedding model).
            index_path: Path to save/load the FAISS index.
            metadata_path: Path to save/load the text metadata.
        """
        self.dimension = dimension
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)

        # Ensure the directory exists
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # IndexFlatIP = Inner Product.
        # If vectors are normalized, Inner Product is equivalent to Cosine Similarity.
        self.index = faiss.IndexFlatIP(self.dimension)

        # Metadata map: {index_id: {"text": "...", "metadata": {...}}}
        self.metadata = {}

        # Load existing data if available
        self._load()

    def add(self, vectors: np.ndarray, texts: List[str], metadatas: List[Dict[str, Any]] = None):
        """
        Adds vectors and their corresponding metadata to the store.

        Args:
            vectors: Numpy array of embeddings (float32).
            texts: The original text chunks.
            metadatas: Optional metadata for each chunk (e.g., source file, page number).
        """
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension mismatch. Expected {self.dimension}, got {vectors.shape[1]}")

        # Normalize vectors for Cosine Similarity (Inner Product of normalized vectors = Cosine Sim)
        faiss.normalize_L2(vectors)

        # Add to FAISS index
        start_id = self.index.ntotal
        self.index.add(vectors)

        # Add to metadata map
        for i in range(len(texts)):
            idx = start_id + i
            self.metadata[idx] = {
                "text": texts[i],
                "metadata": metadatas[i] if metadatas else {}
            }

        logger.info(f"Added {len(texts)} vectors to the store. Total size: {self.index.ntotal}")

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Performs a similarity search for the nearest neighbors.

        Args:
            query_vector: The embedding of the user query.
            k: Number of top results to return.

        Returns:
            A list of tuples (score, metadata_dict).
        """
        if query_vector.shape[0] == 0:
            return []

        # Ensure query vector is normalized for Cosine Similarity
        query_vector = query_vector.copy()
        faiss.normalize_L2(query_vector)

        # FAISS search: returns distances (scores) and indices
        scores, indices = self.index.search(query_vector, k)

        # FAISS returns results for each query vector (in case of batch queries)
        # We only expect one query vector here.
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1: continue # FAISS returns -1 if not enough neighbors are found

            if idx in self.metadata:
                results.append((float(score), self.metadata[idx]))
            else:
                logger.warning(f"Index ID {idx} found but no metadata available.")

        return results

    def save(self):
        """Persists the FAISS index and metadata map to disk."""
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        logger.info(f"Vector store saved to {self.index_path} and {self.metadata_path}")

    def _load(self):
        """Loads the FAISS index and metadata map from disk."""
        if self.index_path.exists() and self.metadata_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                logger.info("Existing vector store loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading vector store: {e}")
                logger.info("Starting with a fresh index.")
