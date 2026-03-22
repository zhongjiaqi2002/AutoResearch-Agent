"""
Embedding Service Wrapper - Using Aliyun text-embedding-v4
"""
from typing import List, Union
from openai import OpenAI
from config import settings


class EmbeddingService:
    """Embedding Service Class"""

    def __init__(self):
        if not settings.DASHSCOPE_API_KEY:
            raise ValueError("DASHSCOPE_API_KEY is not set, please configure it in the .env file")

        self.client = OpenAI(
            api_key=settings.DASHSCOPE_API_KEY,
            base_url=settings.LLM_BASE_URL
        )
        self.model = settings.EMBEDDING_MODEL
        self.dimensions = settings.EMBEDDING_DIMENSIONS

    def embed(self, text: Union[str, List[str]]) -> List[List[float]]:
        """
        Generate text embeddings

        Args:
            text: Single text string or list of text strings

        Returns:
            List of embeddings
        """
        if isinstance(text, str):
            text = [text]

        response = self.client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=self.dimensions,
            encoding_format="float"
        )

        embeddings = [item.embedding for item in response.data]
        return embeddings

    def embed_single(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string

        Args:
            text: Input text string

        Returns:
            Text embedding vector
        """
        return self.embed(text)[0]


# Singleton pattern
_embedding_service = None


def get_embedding_service() -> EmbeddingService:
    """Get the Embedding service singleton instance"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
