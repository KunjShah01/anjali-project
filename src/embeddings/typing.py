"""
Type definitions for embeddings.
"""

from typing import List, Dict, Any, Protocol, runtime_checkable

# Type aliases
Vector = List[float]
Embedding = Vector
EmbeddingMatrix = List[Vector]
Metadata = Dict[str, Any]


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    async def get_embedding(self, text: str) -> Vector:
        """
        Get embedding for a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        ...

    async def get_embeddings(self, texts: List[str]) -> List[Vector]:
        """
        Get embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        ...

    def get_dimension(self) -> int:
        """Get the dimension of embeddings."""
        ...


class Document:
    """Represents a document with embedding."""

    def __init__(
        self, content: str, embedding: Vector = None, metadata: Metadata = None
    ):
        self.content = content
        self.embedding = embedding or []
        self.metadata = metadata or {}
        self.id = self.metadata.get("id", "")

    def __repr__(self) -> str:
        return f"Document(content='{self.content[:50]}...', embedding_dim={len(self.embedding)})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "id": self.id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """Create from dictionary."""
        return cls(
            content=data["content"],
            embedding=data.get("embedding", []),
            metadata=data.get("metadata", {}),
        )


class EmbeddingBatch:
    """Batch of documents with embeddings."""

    def __init__(self, documents: List[Document] = None):
        self.documents = documents or []

    def add(self, document: Document):
        """Add document to batch."""
        self.documents.append(document)

    def __len__(self) -> int:
        return len(self.documents)

    def __iter__(self):
        return iter(self.documents)

    def get_embeddings_matrix(self) -> EmbeddingMatrix:
        """Get embeddings as matrix."""
        return [doc.embedding for doc in self.documents if doc.embedding]

    def get_contents(self) -> List[str]:
        """Get all document contents."""
        return [doc.content for doc in self.documents]

    def get_metadatas(self) -> List[Metadata]:
        """Get all document metadatas."""
        return [doc.metadata for doc in self.documents]


class SimilarityResult:
    """Result from similarity search."""

    def __init__(self, document: Document, score: float, rank: int = 0):
        self.document = document
        self.score = score
        self.rank = rank

    def __repr__(self) -> str:
        return f"SimilarityResult(score={self.score:.3f}, rank={self.rank}, content='{self.document.content[:30]}...')"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document": self.document.to_dict(),
            "score": self.score,
            "rank": self.rank,
        }
