import ollama
import httpx
from typing import List
from config.settings import settings

class Embedder:
    """Generate embeddings using Ollama nomic-embed-text"""

    def __init__(self):
        self.model = settings.ollama_embed_model
        self.host = settings.ollama_host

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        try:
            client = ollama.Client(host=self.host)
            response = client.embeddings(model=self.model, prompt=text)
            return response['embedding']
        except Exception as e:
            # Return zero vector if embedding fails
            print(f"⚠️  Embedding failed: {e}")
            return [0.0] * 768

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        return [self.embed(t) for t in texts]
