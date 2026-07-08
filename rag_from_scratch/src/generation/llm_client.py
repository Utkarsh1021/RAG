from openai import AsyncOpenAI
from typing import Dict, Any
import os
from src.utils.logger import logger

class LLMClient:
    """
    The LLMClient provides a unified interface to interact with different
    Large Language Models. Updated to be ASYNC for production performance.
    """

    def __init__(self, provider: str = "openai", api_key: str = None, base_url: str = None, model_name: str = None):
        self.provider = provider.lower()
        self.model_name = model_name

        if self.provider == "openai":
            key = api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError("OpenAI API Key is required.")
            self.client = AsyncOpenAI(api_key=key, base_url=base_url)
            self.model_name = model_name or "gpt-4o"
        elif self.provider == "local":
            if not base_url:
                base_url = "http://localhost:11434/v1"
            self.client = AsyncOpenAI(api_key="ollama", base_url=base_url)
            self.model_name = model_name or "llama3"
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        logger.info(f"LLMClient (Async) initialized with provider: {self.provider}, model: {self.model_name}")

    async def generate(self, prompt: str, temperature: float = 0.1) -> str:
        """
        Asynchronously generates a response from the LLM.
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response from LLM: {e}")
            return f"Error generating response: {str(e)}"
