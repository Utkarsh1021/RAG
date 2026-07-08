from typing import List
from src.generation.llm_client import LLMClient
from src.utils.logger import logger

class QueryExpander:
    """
    Query Expansion is a technique to improve retrieval recall.
    Users often ask questions in a way that doesn't perfectly match
    the wording in the documents.

    The QueryExpander uses an LLM to generate multiple variations of the
    original query, which are then all used for retrieval.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def expand(self, query: str, num_variations: int = 3) -> List[str]:
        """
        Generates semantically similar variations of the user query.
        """
        logger.info(f"Expanding query: '{query}'")

        expansion_prompt = (
            f"You are an AI search expert. Your task is to help a RAG system retrieve "
            f"the most relevant documents by providing multiple ways to ask the same question.\n\n"
            f"Original Query: {query}\n\n"
            f"Please provide {num_variations} different variations of this query. "
            f"Focus on different keywords and phrasing while keeping the meaning identical.\n"
            f"Return only the variations, one per line, without numbers or bullets."
        )

        try:
            response = await self.llm_client.generate(expansion_prompt)
            variations = [v.strip() for v in response.splitlines() if v.strip()]

            # Always include the original query in the list
            results = [query] + variations[:num_variations-1]

            logger.info(f"Generated {len(results)} query variations.")
            return results
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return [query]
