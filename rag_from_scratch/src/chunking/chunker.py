from typing import List
from src.utils.logger import logger

class TextChunker:
    """
    Responsible for splitting long documents into smaller,
    overlapping chunks to fit LLM context windows.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Args:
            chunk_size: Maximum number of characters per chunk.
            chunk_overlap: Number of characters to overlap between consecutive chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> List[str]:
        """
        Splits text into overlapping chunks using a sliding window approach.

        Returns:
            List[str]: A list of text chunks.
        """
        if not text:
            return []

        chunks = []
        start = 0

        # The sliding window approach
        # We move the start pointer forward by (chunk_size - chunk_overlap)
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)

            # Move window forward
            start += (self.chunk_size - self.chunk_overlap)

            # Avoid infinite loop if overlap >= size
            if self.chunk_overlap >= self.chunk_size:
                logger.error("chunk_overlap must be smaller than chunk_size")
                break

        logger.info(f"Text split into {len(chunks)} chunks.")
        return chunks
