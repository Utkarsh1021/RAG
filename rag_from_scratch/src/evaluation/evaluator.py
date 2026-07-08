import json
import numpy as np
from typing import List, Dict, Any
from src.utils.logger import logger
from src.generation.llm_client import LLMClient

class RAGEvaluator:
    """
    The RAGEvaluator measures the quality of the RAG system.
    It focuses on the 'RAG Triad': Retrieval Relevance, Faithfulness, and Answer Relevance.
    """

    def __init__(self, rag_system, llm_client: LLMClient):
        """
        Args:
            rag_system: The initialized RAGSystem instance to evaluate.
            llm_client: A strong LLM (e.g., GPT-4o) used as a judge for qualitative metrics.
        """
        self.rag_system = rag_system
        self.llm_client = llm_client

    def load_golden_dataset(self, path: str) -> List[Dict[str, Any]]:
        with open(path, 'r') as f:
            return json.load(f)

    def compute_hit_rate(self, retrieved_texts: List[str], expected_texts: List[str]) -> float:
        """
        Hit Rate: Percentage of queries where the expected chunk is in the top-K.
        """
        hit = 0
        for exp in expected_texts:
            if any(exp.lower() in ret.lower() for ret in retrieved_texts):
                hit += 1
        return hit / len(expected_texts) if expected_texts else 0.0

    def compute_mrr(self, retrieved_texts: List[str], expected_texts: List[str]) -> float:
        """
        Mean Reciprocal Rank (MRR): Measures where the first relevant chunk appears.
        1/1 if it's the first result, 1/2 if second, etc.
        """
        total_rr = 0
        for exp in expected_texts:
            for rank, ret in enumerate(retrieved_texts, 1):
                if exp.lower() in ret.lower():
                    total_rr += 1 / rank
                    break
        return total_rr / len(expected_texts) if expected_texts else 0.0

    def judge_faithfulness(self, query: str, context: str, answer: str) -> float:
        """
        LLM-as-a-Judge: Evaluates if the answer is derived ONLY from the context.
        Returns a score from 1 to 5.
        """
        judge_prompt = (
            f"You are an expert evaluator. Rate the 'Faithfulness' of the AI answer based on the provided context.\n"
            f"Faithfulness means the answer is derived ONLY from the context and does not include outside info.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"ANSWER:\n{answer}\n\n"
            f"Score from 1 to 5:\n"
            f"1: Completely hallucinates or uses outside knowledge.\n"
            f"2: Mostly outside knowledge with slight context use.\n"
            f"3: Mixed use of context and outside knowledge.\n"
            f"4: Mostly faithful, minor unsupported claims.\n"
            f"5: Perfectly faithful. Every claim is supported by the context.\n\n"
            f"Return ONLY the digit (1-5)."
        )

        try:
            response = self.llm_client.generate(judge_prompt).strip()
            # Extract only the digit from the response
            digit = [char for char in response if char.isdigit()][0]
            return float(digit)
        except Exception as e:
            logger.error(f"Judging failed: {e}")
            return 0.0

    def run_evaluation(self, dataset_path: str):
        """
        Runs a full evaluation suite over the golden dataset.
        """
        dataset = self.load_golden_dataset(dataset_path)
        metrics = {
            "hit_rate": [],
            "mrr": [],
            "faithfulness": []
        }

        logger.info(f"Running evaluation on {len(dataset)} samples...")

        for item in dataset:
            query = item["query"]
            expected_chunks = item["expected_chunks"]

            # Get RAG output
            result = self.rag_system.query(query)

            if isinstance(result, str): # Case where RAG said "I don't know"
                answer = result
                retrieved_texts = []
            else:
                answer = result["answer"]
                retrieved_texts = [c["text"] for c in result["context"]]

            # 1. Retrieval Metrics
            metrics["hit_rate"].append(self.compute_hit_rate(retrieved_texts, expected_chunks))
            metrics["mrr"].append(self.compute_mrr(retrieved_texts, expected_chunks))

            # 2. Generation Metrics (Faithfulness)
            context_str = " ".join(retrieved_texts)
            metrics["faithfulness"].append(self.judge_faithfulness(query, context_str, answer))

        # Aggregate Results
        summary = {
            "Avg Hit Rate": np.mean(metrics["hit_rate"]),
            "Avg MRR": np.mean(metrics["mrr"]),
            "Avg Faithfulness": np.mean(metrics["faithfulness"])
        }

        return summary
