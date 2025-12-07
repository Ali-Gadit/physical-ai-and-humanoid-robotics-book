import os
from qdrant_client import models
from src.rag.embedding import embed_text
from src.rag.vector_store import get_qdrant_client

COLLECTION_NAME = "physical-ai-robotics-textbook"

def retrieve_context(query: str, limit: int = 3) -> str:
    """
    Retrieves relevant context from the Qdrant vector store based on the query.
    Returns a formatted string containing the retrieved text chunks.
    """
    try:
        # 1. Generate embedding for the query
        query_embedding = embed_text([query])[0]
        
        # 2. Search Qdrant
        client = get_qdrant_client()
        
        # Use query_points if search is not available (newer clients)
        if hasattr(client, 'query_points'):
            search_result = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_embedding,
                limit=limit,
                with_payload=True
            ).points
        else:
            # Fallback to search (older clients or alias)
            search_result = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_embedding,
                limit=limit,
                with_payload=True
            )
        
        # 3. Format results
        if not search_result:
            return ""

        formatted_results = []
        for scored_point in search_result:
            payload = scored_point.payload
            text = payload.get("text", "")
            source = payload.get("source", "Unknown")
            formatted_results.append(f"Source: {source}\nContent: {text}")
        
        return "\n\n---\n\n".join(formatted_results)

    except Exception as e:
        print(f"Error during retrieval: {e}")
        return ""