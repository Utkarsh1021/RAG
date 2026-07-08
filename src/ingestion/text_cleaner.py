import re
from src.utils.logger import logger

class TextCleaner:
    """
    Responsible for cleaning raw extracted text to remove noise
    and normalize formatting before chunking.
    """

    def __init__(self, custom_patterns: list = None):
        """
        Args:
            custom_patterns: Optional list of regex patterns to remove.
        """
        self.custom_patterns = custom_patterns or []

    def clean(self, text: str) -> str:
        """
        Cleans the input text using a series of normalization steps.

        Returns:
            str: The cleaned text.
        """
        if not text:
            return ""

        # 1. Remove custom patterns (e.g. specific header/footer text)
        for pattern in self.custom_patterns:
            text = re.sub(pattern, " ", text)

        # 2. Normalize whitespace: replace multiple spaces/tabs/newlines with a single space
        # We keep newlines for now to maintain basic structure, but trim them
        text = re.sub(r'[ \t]+', ' ', text)

        # 3. Remove repetitive patterns common in PDFs (e.g., "Page X of Y")
        text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)

        # 4. Strip leading/trailing whitespace from each line
        lines = [line.strip() for line in text.splitlines()]

        # 5. Rejoin lines and remove empty ones
        text = "\n".join([line for line in lines if line])

        logger.debug("Text cleaning completed.")
        return text
