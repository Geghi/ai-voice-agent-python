import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import logging

class ChromaDBManager:
    def __init__(self, path: str):
        self.vectorstore = Chroma(
            persist_directory=path,
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
        )

    def query_db(self, query_texts: str, n_results: int = 3) -> list[str]:
        top_docs_with_scores = self.vectorstore.similarity_search_with_score(query_texts, k=n_results)
        for doc, score in top_docs_with_scores:
            print(f"Source: {doc.metadata.get('source', 'N/A')}, Score: {score:.4f}")
        return top_docs_with_scores

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

    # Initialize the manager with a test path
    db_manager = ChromaDBManager(path="./chroma/curriculum-mantovani_chroma_db")

    # Query the database
    query = "What is Mavena?"
    print(f"\nQuerying for: '{query}'")
    query_results = db_manager.query_db(query_texts=query, n_results=5)
    
    print("\nQuery results:")
    if query_results:
        print(query_results[0])
    else:
        print("No results found.")
