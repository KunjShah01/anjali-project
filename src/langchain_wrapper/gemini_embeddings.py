"""
LangChain Embeddings wrapper for Google Gemini.
"""

from typing import List, Optional
import asyncio
from langchain.embeddings.base import Embeddings

from ..embeddings.embedder import EmbeddingService
from ..config import GeminiConfig
from ..utils.logger import LoggerMixin


class GeminiEmbeddings(Embeddings, LoggerMixin):
    """
    LangChain compatible wrapper for Google Gemini embeddings.
    """

    def __init__(self, config: Optional[GeminiConfig] = None):
        """Initialize Gemini embeddings."""
        super().__init__()
        LoggerMixin.__init__(self)

        self.config = config
        self._embedding_service = None

    @property
    def embedding_service(self) -> EmbeddingService:
        """Lazy initialization of embedding service."""
        if self._embedding_service is None:
            if self.config is None:
                raise ValueError("GeminiConfig is required")
            self._embedding_service = EmbeddingService(self.config)
        return self._embedding_service

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        try:
            # Run async method in event loop
            return asyncio.run(self.aembed_documents(texts))
        except Exception as e:
            self.log_error("Error embedding documents", error=e)
            # Return zero vectors as fallback
            dimension = self.embedding_service.get_dimension()
            return [[0.0] * dimension for _ in texts]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Asynchronously embed a list of documents.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        try:
            embeddings = await self.embedding_service.get_embeddings(
                texts, task_type="retrieval_document"
            )

            # Filter out None embeddings
            valid_embeddings = []
            for emb in embeddings:
                if emb and emb != [0.0] * self.embedding_service.get_dimension():
                    valid_embeddings.append(emb)
                else:
                    # Use zero vector for failed embeddings
                    valid_embeddings.append(
                        [0.0] * self.embedding_service.get_dimension()
                    )

            self.log_debug("Embedded documents", count=len(valid_embeddings))
            return valid_embeddings

        except Exception as e:
            self.log_error("Error in async embed documents", error=e)
            # Return zero vectors as fallback
            dimension = self.embedding_service.get_dimension()
            return [[0.0] * dimension for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        try:
            return asyncio.run(self.aembed_query(text))
        except Exception as e:
            self.log_error("Error embedding query", error=e)
            # Return zero vector as fallback
            return [0.0] * self.embedding_service.get_dimension()

    async def aembed_query(self, text: str) -> List[float]:
        """
        Asynchronously embed a single query.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        try:
            embedding = await self.embedding_service.embed_query(text)

            if (
                embedding
                and embedding != [0.0] * self.embedding_service.get_dimension()
            ):
                self.log_debug("Embedded query successfully")
                return embedding
            else:
                self.log_warning("Received invalid query embedding")
                return [0.0] * self.embedding_service.get_dimension()

        except Exception as e:
            self.log_error("Error in async embed query", error=e)
            return [0.0] * self.embedding_service.get_dimension()

    def get_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        return self.embedding_service.get_dimension()

    async def get_stats(self) -> dict:
        """Get embedding service statistics."""
        if self._embedding_service:
            return await self._embedding_service.get_stats()
        return {}

    def clear_cache(self):
        """Clear the embedding cache."""
        if self._embedding_service:
            self._embedding_service.clear_cache()
            self.log_info("Embedding cache cleared")


def create_gemini_embeddings(config: Optional[GeminiConfig] = None) -> GeminiEmbeddings:
    """
    Factory function to create GeminiEmbeddings instance.

    Args:
        config: Optional GeminiConfig. If None, will use default config.

    Returns:
        GeminiEmbeddings instance
    """
    if config is None:
        from ..config import get_config

        config = get_config().gemini

    return GeminiEmbeddings(config)
