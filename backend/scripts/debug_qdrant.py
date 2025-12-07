import os
import sys
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# Add backend to path to import modules if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

def check_qdrant():
    print(f"QDRANT_HOST: {os.getenv('QDRANT_HOST')}")
    
    try:
        client = QdrantClient(
            url=os.getenv("QDRANT_HOST"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        print("Client initialized.")
        print(f"Client type: {type(client)}")
        print(f"Has 'search' attribute: {hasattr(client, 'search')}")
        print(f"Has 'query' attribute: {hasattr(client, 'query')}")
        
        COLLECTION_NAME = "physical-ai-robotics-textbook"
        
        if client.collection_exists(COLLECTION_NAME):
            print(f"Collection '{COLLECTION_NAME}' exists.")
            count = client.count(COLLECTION_NAME)
            print(f"Number of points in collection: {count}")
        else:
            print(f"Collection '{COLLECTION_NAME}' DOES NOT EXIST. You need to run the ingestion script.")

    except Exception as e:
        print(f"Error checking Qdrant: {e}")

if __name__ == "__main__":
    check_qdrant()
