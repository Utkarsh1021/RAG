from typing import List, Dict, Any

class PromptBuilder:
    """
    The PromptBuilder is responsible for constructing the final string
    that is sent to the LLM. It ensures the retrieved context is
    presented clearly and the LLM is given strict instructions to
    avoid hallucinations.
    """

    def __init__(self, system_prompt: str = None):
        """
        Args:
            system_prompt: A custom system persona. If None, a default
                           RAG-optimized prompt is used.
        """
        if system_prompt:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = (
                "You are a precise and helpful AI assistant. Your task is to answer the user's "
                "question based ONLY on the provided context. \n\n"
                "STRICT RULES:\n"
                "1. Use the provided context to answer the query.\n"
                "2. If the answer is not present in the context, explicitly state: "
                "'I am sorry, but the provided documents do not contain information to answer this question.'\n"
                "3. Do not use your internal knowledge to supplement the answer.\n"
                "4. Always cite your sources by mentioning the [Source X] label provided in the context."
            )

    def build_rag_prompt(self, query: str, context: str) -> str:
        """
        Combines the system prompt, retrieved context, and user query.

        Args:
            query: The raw user question.
            context: The formatted string of retrieved chunks.

        Returns:
            The complete prompt to be sent to the LLM.
        """
        prompt = (
            f"{self.system_prompt}\n\n"
            f"--- CONTEXT START ---\n"
            f"{context}\n"
            f"--- CONTEXT END ---\n\n"
            f"USER QUESTION: {query}\n\n"
            f"ANSWER:"
        )
        return prompt
