import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def embed_text(texts: list[str]) -> list[list[float]]:
    """
    Generates embeddings for a list of texts using a Gemini embedding model.
    """
    if not texts:
        return []
    
    try:
        # Using 'models/gemini-embedding-001' as explicitly requested by the user.
        # This is the full resource name for the model.
        model_name = "models/gemini-embedding-001" 
        
        response = genai.embed_content(
            model=model_name,
            content=texts,
            task_type="retrieval_document"
        )
        
        embeddings = response['embedding']
        
        if hasattr(embeddings, 'tolist'):
            return [embeddings.tolist()] if len(texts) == 1 else embeddings.tolist()
            
        if isinstance(embeddings, list):
             if embeddings and isinstance(embeddings[0], float):
                 return [embeddings]
             return embeddings
             
        return embeddings

    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return []

if __name__ == "__main__":
    sample_texts = ["Test sentence."]
    try:
        embeddings = embed_text(sample_texts)
        for i, emb in enumerate(embeddings):
            print(f"Embedding for text {i+1}: length: {len(emb)}")
    except Exception as e:
        print(f"Failed: {e}")
