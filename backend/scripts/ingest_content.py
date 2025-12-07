import os
import uuid
from qdrant_client import models, QdrantClient
from dotenv import load_dotenv

from src.services.content_service import load_markdown_docs, clean_text
from src.rag.processing import chunk_text
from src.rag.embedding import embed_text
from src.rag.vector_store import get_qdrant_client

load_dotenv()

COLLECTION_NAME = "physical-ai-robotics-textbook"
DOCS_PATH = "docusaurus-book/docs"

def ingest_content():
    client = get_qdrant_client()

    # 1. Determine Embedding Dimension dynamically
    # Generate a dummy embedding to check size
    dummy_text = "Test dimension"
    dummy_emb = embed_text([dummy_text])
    if not dummy_emb:
        print("Failed to generate dummy embedding. Exiting.")
        return
    
    embedding_size = len(dummy_emb[0])
    print(f"Detected embedding dimension: {embedding_size}")

    # 2. Re-create collection if needed
    if client.collection_exists(collection_name=COLLECTION_NAME):
        # Check if dimension matches
        # It's hard to check config easily without complex calls, so safer to recreate for this setup script
        # OR try to delete and recreate to be safe.
        print(f"Deleting existing collection '{COLLECTION_NAME}' to ensure schema match...")
        client.delete_collection(collection_name=COLLECTION_NAME)
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=embedding_size, distance=models.Distance.COSINE),
    )
    print(f"Collection '{COLLECTION_NAME}' created with size {embedding_size}.")

    print(f"Loading documents from {DOCS_PATH}...")
    documents_processed = 0
    for file_path, content in load_markdown_docs(DOCS_PATH):
        print(f"Processing: {file_path}")
        cleaned_content = clean_text(content)
        chunks = chunk_text(cleaned_content)

        if not chunks:
            print(f"No chunks generated for {file_path}. Skipping.")
            continue

        # Generate embeddings for all chunks in this file
        chunk_embeddings = embed_text(chunks)
        
        # Prepare points for upserting
        points = []
        for i, chunk in enumerate(chunks):
            if chunk_embeddings[i]:
                # Generate a UUID based on the file path and chunk index to ensure consistency/idempotency
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{file_path}_{i}"))
                
                points.append(
                    models.PointStruct(
                        id=point_id, 
                        vector=chunk_embeddings[i],
                        payload={"text": chunk, "source": file_path, "chunk_id": i}
                    )
                )
            else:
                print(f"Warning: No embedding generated for a chunk in {file_path}. Skipping chunk.")

        if points:
            client.upsert(
                collection_name=COLLECTION_NAME,
                wait=True,
                points=points
            )
            print(f"Upserted {len(points)} chunks from {file_path}")
            documents_processed += 1
        else:
            print(f"No valid points to upsert for {file_path}.")

    print(f"Content ingestion complete. Processed {documents_processed} documents.")

if __name__ == "__main__":
    # Ensure the script can find its modules when run directly
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
    
    ingest_content()
