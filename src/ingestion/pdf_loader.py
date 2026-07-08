import pdfplumber
from typing import List
from pathlib import Path
from src.utils.logger import logger

class PDFLoader:
    """
    Responsible for loading and extracting raw text from PDF files.
    """
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF file not found at: {file_path}")

    def load(self) -> str:
        """
        Extracts all text from the PDF pages.

        Returns:
            str: The concatenated text of the entire document.
        """
        full_text = []
        try:
            with pdfplumber.open(self.file_path) as pdf:
                logger.info(f"Loading PDF: {self.file_path.name} with {len(pdf.pages)} pages")
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text.append(text)

            return "\n".join(full_text)
        except Exception as e:
            logger.error(f"Error extracting text from {self.file_path}: {e}")
            raise e
