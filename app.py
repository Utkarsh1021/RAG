import os
import asyncio
from typing import List, Dict, Any
from src.utils.logger import logger

# Import our manual RAG components
from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.text_cleaner import TextCleaner
from src.chunking.chunker import TextChunker
from src.embeddings.embedder import Embedder
from src.vector_store.store import VectorStore
from src.retrieval.retriever import Retriever
from src.retrieval.query_expander import QueryExpander
from src.generation.prompt_builder import PromptBuilder
from src.generation.llm_client import LLMClient

class RAGSystem:
    """
    The Orchestrator class updated for Production Performance.
    Features: Async Processing, Query Expansion, and Hybrid Retrieval.
    """

    def __init__(self, llm_provider: str = "openai", model_name: str = None):
        logger.info("Initializing Production-Ready RAG System...")

        # 1. Setup Embedding & Storage
        self.embedder = Embedder()
        self.vector_store = VectorStore(dimension=self.embedder.get_dimension())

        # 2. Setup Generation first (needed for QueryExpander)
        self.llm_client = LLMClient(provider=llm_provider, model_name=model_name)
        self.prompt_builder = PromptBuilder()

        # 3. Setup Advanced Retrieval
        self.query_expander = QueryExpander(self.llm_client)
        self.retriever = Retriever(self.embedder, self.vector_store)

        # 4. Processing Tools
        self.cleaner = TextCleaner()
        self.chunker = TextChunker(chunk_size=500, chunk_overlap=50)

    def process_document(self, pdf_path: str):
        """
        Standard sync ingestion pipeline.
        """
        logger.info(f"Processing document: {pdf_path}")
        loader = PDFLoader(pdf_path)
        raw_text = loader.load()
        cleaned_text = self.cleaner.clean(raw_text)
        chunks = self.chunker.chunk(cleaned_text)
        embeddings = self.embedder.embed_documents(chunks)
        metadatas = [{"source": pdf_path, "page": "N/A"} for _ in chunks]
        self.vector_store.add(embeddings, chunks, metadatas)
        self.vector_store.save()
        logger.info(f"Document indexed. Added {len(chunks)} chunks.")

    async def query(self, user_query: str, use_expansion: bool = True):
        """
        Production Inference Pipeline:
        Query -> Expansion -> Parallel Retrieval -> Merge -> Prompt -> Async LLM Generation
        """
        logger.info(f"Querying system (Async): {user_query}")

        # Step 1: Query Expansion (Multi-Query)
        queries_to_run = [user_query]
        if use_expansion:
            queries_to_run = await self.query_expander.expand(user_query)

        # Step 2: Retrieve chunks for all variations
        # In a true production system, we would run these retrieval calls in parallel.
        all_retrieved_chunks = []
        for q in queries_to_run:
            chunks = self.retriever.retrieve(q, k=3)
            all_retrieved_chunks.extend(chunks)

        # Step 3: Deduplicate chunks (since different queries might find the same chunk)
        unique_chunks = []
        seen_texts = set()
        for chunk in all_retrieved_chunks:
            if chunk["text"] not in seen_texts:
                unique_chunks.append(chunk)
                seen_texts.add(chunk["text"])

        # Step 4: Top-K Filter from the combined set
        # We sort by score and take the top 5 most relevant overall
        unique_chunks.sort(key=lambda x: x["score"], reverse=True)
        final_chunks = unique_chunks[:5]

        if not final_chunks:
            return "I could not find any relevant information in the documents."

        # Step 5: Format context and build prompt
        context_string = self.retriever.format_context(final_chunks)
        final_prompt = self.prompt_builder.build_rag_prompt(user_query, context_string)

        # Step 6: Async Generation
        answer = await self.llm_client.generate(final_prompt)

        return {
            "answer": answer,
            "context": final_chunks,
            "queries_used": queries_to_run
        }

async def main():
    # --- CONFIGURATION ---
    LLM_PROVIDER = "openai"
    MODEL_NAME = "gpt-4o"
    os.environ["OPENAI_API_KEY"] = "your-api-key-here"
    # ---------------------

    try:
        rag = RAGSystem(llm_provider=LLM_PROVIDER, model_name=MODEL_NAME)
    except Exception as e:
        print(f"Failed to initialize RAG system: {e}")
        return

    print("\n--- Welcome to the Production RAG System ---")
    print("1. Index a PDF Document")
    print("2. Chat with Documents")
    print("3. Exit")

    while True:
        choice = input("\nChoose an option (1/2/3): ")

        if choice == "1":
            path = input("Enter the path to your PDF file: ")
            try:
                rag.process_document(path)
                print("Document indexed successfully!")
            except Exception as e:
                print(f"Error processing document: {e}")

        elif choice == "2":
            query = input("\nAsk a question: ")
            result = await rag.query(query)

            if isinstance(result, str):
                print(f"\nAI: {result}")
            else:
                print(f"\nAI: {result['answer']}")
                print(f"\n(Queries used for retrieval: {result['queries_used']})")
                print("\n--- SOURCES USED ---")
                for i, chunk in enumerate(result['context']):
                    print(f"Source {i+1} (Score: {chunk['score']}): {chunk['text'][:100]}...")

        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid option.")

if __name__ == "__main__":
    asyncio.run(main())
