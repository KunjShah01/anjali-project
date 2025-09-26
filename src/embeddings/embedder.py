"""
Embedding service and providers for the Real-time RAG system using Google Gemini.
"""

import asyncio
from typing import List, Dict, Any, Optional
import google.generativeai as genai
import numpy as np
from dataclasses import dataclass
import time
import hashlib

from ..config import GeminiConfig
from ..utils.logger import LoggerMixin
from .typing import Vector


@dataclass
class EmbeddingStats:
    """Statistics for embedding operations."""

    total_texts_processed: int = 0
    total_tokens_used: int = 0
    average_embedding_time: float = 0.0
    errors: int = 0
    cache_hits: int = 0
    cache_misses: int = 0


class GeminiEmbeddingProvider(LoggerMixin):
    """
    Google Gemini embedding provider using the embedding API.
    """

    def __init__(self, config: GeminiConfig):
        super().__init__()
        self.config = config
        
        # Configure Gemini
        genai.configure(api_key=config.api_key)
        
        # Model configuration
        self.model = config.embedding_model
        self.dimension = 768  # Gemini embedding dimension
        
        # Statistics
        self.stats = EmbeddingStats()
        
        # Rate limiting
        self._semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests
        self._last_request_time = 0
        self._min_request_interval = 0.1  # 100ms between requests
        
        # Simple cache for repeated texts
        self._cache: Dict[str, Vector] = {}
        self._cache_max_size = 1000
        
        self.log_info("Initialized Gemini embedding provider", 
                     model=self.model,
                     dimension=self.dimension)
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    async def _rate_limit(self):
        """Apply rate limiting to API requests."""
        current_time = time.time()
        elapsed = current_time - self._last_request_time
        
        if elapsed < self._min_request_interval:
            sleep_time = self._min_request_interval - elapsed
            await asyncio.sleep(sleep_time)
        
        self._last_request_time = time.time()
    
    async def get_embedding(self, text: str) -> Vector:
        """
        Get embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        if not text or not text.strip():
            return [0.0] * self.dimension
        
        # Check cache first
        cache_key = self._get_cache_key(text)
        if cache_key in self._cache:
            self.stats.cache_hits += 1
            return self._cache[cache_key]
        
        self.stats.cache_misses += 1
        
        async with self._semaphore:
            try:
                await self._rate_limit()
                start_time = time.time()
                
                # Generate embedding using Gemini
                result = genai.embed_content(
                    model=self.model,
                    content=text,
                    task_type="retrieval_document"
                )
                
                embedding = result['embedding']
                
                # Update statistics
                processing_time = time.time() - start_time
                self._update_stats(processing_time, len(text.split()))
                
                # Cache the result
                if len(self._cache) < self._cache_max_size:
                    self._cache[cache_key] = embedding
                
                self.log_debug("Generated embedding for text", 
                             text_length=len(text),
                             processing_time=processing_time)
                
                return embedding
                
            except Exception as e:
                self.stats.errors += 1
                self.log_error("Error generating embedding", error=e, text_preview=text[:100])
                return [0.0] * self.dimension
    
    async def get_embeddings(self, texts: List[str]) -> List[Vector]:
        """
        Get embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Filter empty texts
        filtered_texts = [text for text in texts if text and text.strip()]
        
        if not filtered_texts:
            return [[0.0] * self.dimension] * len(texts)
        
        self.log_info(f"Generating embeddings for {len(filtered_texts)} texts")
        
        # Process texts concurrently with semaphore limiting
        tasks = [self.get_embedding(text) for text in filtered_texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        embeddings = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.log_error(f"Error in batch embedding {i}", error=result)
                embeddings.append([0.0] * self.dimension)
            else:
                embeddings.append(result)
        
        successful_embeddings = sum(1 for emb in embeddings if emb != [0.0] * self.dimension)
        self.log_info(f"Generated {successful_embeddings}/{len(texts)} embeddings successfully")
        
        return embeddings
    
    async def embed_query(self, query: str) -> Vector:
        """
        Generate embedding for a query text.
        
        Args:
            query: Query text to embed
            
        Returns:
            Embedding vector
        """
        if not query or not query.strip():
            return [0.0] * self.dimension
        
        # Check cache first
        cache_key = self._get_cache_key(query)
        if cache_key in self._cache:
            self.stats.cache_hits += 1
            return self._cache[cache_key]
        
        self.stats.cache_misses += 1
        
        async with self._semaphore:
            try:
                await self._rate_limit()
                start_time = time.time()
                
                # Generate embedding using Gemini with query task type
                result = genai.embed_content(
                    model=self.model,
                    content=query,
                    task_type="retrieval_query"
                )
                
                embedding = result['embedding']
                
                # Update statistics
                processing_time = time.time() - start_time
                self._update_stats(processing_time, len(query.split()))
                
                # Cache the result
                if len(self._cache) < self._cache_max_size:
                    self._cache[cache_key] = embedding
                
                self.log_debug("Generated query embedding", 
                             query_length=len(query),
                             processing_time=processing_time)
                
                return embedding
                
            except Exception as e:
                self.stats.errors += 1
                self.log_error("Error generating query embedding", error=e, query_preview=query[:100])
                return [0.0] * self.dimension
    
    def _update_stats(self, processing_time: float, token_count: int):
        """Update embedding statistics."""
        self.stats.total_texts_processed += 1
        self.stats.total_tokens_used += token_count
        
        # Update average processing time using exponential moving average
        alpha = 0.1
        if self.stats.average_embedding_time == 0:
            self.stats.average_embedding_time = processing_time
        else:
            self.stats.average_embedding_time = (
                alpha * processing_time + (1 - alpha) * self.stats.average_embedding_time
            )
    
    def similarity(self, embedding1: Vector, embedding2: Vector) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            self.log_error("Error calculating similarity", error=e)
            return 0.0
    
    def batch_similarity(self, query_embedding: Vector, embeddings: List[Vector]) -> List[float]:
        """
        Calculate similarity between query and multiple embeddings.
        
        Args:
            query_embedding: Query embedding vector
            embeddings: List of embedding vectors to compare
            
        Returns:
            List of similarity scores
        """
        try:
            query_array = np.array(query_embedding)
            embeddings_matrix = np.array(embeddings)
            
            # Calculate batch cosine similarity
            dot_products = np.dot(embeddings_matrix, query_array)
            norms = np.linalg.norm(embeddings_matrix, axis=1) * np.linalg.norm(query_array)
            
            # Handle zero norms
            norms = np.where(norms == 0, 1, norms)
            similarities = dot_products / norms
            
            return similarities.tolist()
            
        except Exception as e:
            self.log_error("Error calculating batch similarity", error=e)
            return [0.0] * len(embeddings)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get embedding provider statistics."""
        return {
            "provider": "gemini",
            "model": self.model,
            "dimension": self.dimension,
            "cache_size": len(self._cache),
            "cache_max_size": self._cache_max_size,
            "total_texts_processed": self.stats.total_texts_processed,
            "total_tokens_used": self.stats.total_tokens_used,
            "average_embedding_time": self.stats.average_embedding_time,
            "errors": self.stats.errors,
            "cache_hits": self.stats.cache_hits,
            "cache_misses": self.stats.cache_misses,
        }
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()
        self.log_info("Cleared embedding cache")


class EmbeddingService(LoggerMixin):
    """
    Main embedding service that manages different providers.
    """
    
    def __init__(self, config: GeminiConfig, provider: str = "gemini"):
        super().__init__()
        self.config = config
        self.provider_name = provider
        
        # Initialize the appropriate provider
        if provider == "gemini":
            self.provider = GeminiEmbeddingProvider(config)
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
        
        self.log_info("Initialized embedding service", provider=provider)
    
    async def get_embedding(self, text: str) -> Vector:
        """Get embedding for text."""
        return await self.provider.get_embedding(text)
    
    async def get_embeddings(self, texts: List[str]) -> List[Vector]:
        """Get embeddings for multiple texts."""
        return await self.provider.get_embeddings(texts)
    
    async def embed_query(self, query: str) -> Vector:
        """Generate embedding for query."""
        return await self.provider.embed_query(query)
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.provider.get_dimension()
    
    def similarity(self, embedding1: Vector, embedding2: Vector) -> float:
        """Calculate similarity between embeddings."""
        return self.provider.similarity(embedding1, embedding2)
    
    def batch_similarity(self, query_embedding: Vector, embeddings: List[Vector]) -> List[float]:
        """Calculate batch similarity."""
        return self.provider.batch_similarity(query_embedding, embeddings)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get embedding statistics."""
        return self.provider.get_stats()
    
    def clear_cache(self):
        """Clear embedding cache."""
        self.provider.clear_cache()


def create_embedding_service(
    api_key: str, 
    model: str = "models/embedding-001"
) -> EmbeddingService:
    """
    Create embedding service with simple configuration.
    
    Args:
        api_key: Gemini API key
        model: Embedding model name
        
    Returns:
        Configured embedding service
    """
    config = GeminiConfig(api_key=api_key, embedding_model=model)
    return EmbeddingService(config, provider="gemini")


# Compatibility aliases
GeminiEmbedder = GeminiEmbeddingProvider
EmbeddingManager = EmbeddingService