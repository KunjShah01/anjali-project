"""
FAISS vector store implementation for the Real-time RAG system.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from datetime import datetime

from ..config import VectorStoreConfig
from ..embeddings.typing import Vector, Document, SimilarityResult
from ..utils.logger import LoggerMixin
from ..errors import VectorStoreError, TransientError, retry


class FAISSVectorStore(LoggerMixin):
    """
    FAISS-based vector store for similarity search.
    """

    def __init__(self, config: VectorStoreConfig):
        super().__init__()
        self.config = config
        self.dimension = config.dimension

        # FAISS index
        self.index: Optional[faiss.Index] = None
        self.index_path = Path(config.index_path)
        self.metadata_path = self.index_path.with_suffix(".metadata.json")

        # Document storage
        self.documents: List[Document] = []
        self.id_to_index: Dict[str, int] = {}

        # Statistics
        self.stats = {
            "total_documents": 0,
            "index_size": 0,
            "last_save": None,
            "searches_performed": 0,
            "average_search_time": 0.0,
        }

        # Initialize or load index
        self._initialize_index()

    def _initialize_index(self):
        """Initialize or load FAISS index."""
        try:
            if self.index_path.exists():
                self.load_index()
                self.log_info(
                    "Loaded existing FAISS index",
                    documents=len(self.documents),
                    dimension=self.dimension,
                )
            else:
                self._create_new_index()
                self.log_info("Created new FAISS index", dimension=self.dimension)

        except Exception as e:
            self.log_error("Error initializing FAISS index", error=e)
            self._create_new_index()

    def _create_new_index(self):
        """Create a new FAISS index."""
        # Use IndexFlatIP for cosine similarity (inner product with normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents = []
        self.id_to_index = {}

        # Create directory if it doesn't exist
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

    async def add_document(
        self, content: str, embedding: Vector, metadata: Dict[str, Any]
    ) -> str:
        """
        Add a document to the vector store.

        Args:
            content: Document content
            embedding: Document embedding vector
            metadata: Document metadata

        Returns:
            Document ID
        """
        @retry((TransientError,), retries=2, backoff_factor=0.5)
        async def _do_add() -> str:
            # Generate document ID
            doc_id = metadata.get("id", f"doc_{len(self.documents)}")

            # Create document
            document = Document(
                content=content,
                embedding=embedding,
                metadata={
                    **metadata,
                    "id": doc_id,
                    "added_at": datetime.utcnow().isoformat(),
                },
            )

            # Normalize embedding for cosine similarity
            embedding_array = np.array(embedding, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(embedding_array)

            # Add to FAISS index
            self.index.add(embedding_array)

            # Add to document storage
            doc_index = len(self.documents)
            self.documents.append(document)
            self.id_to_index[doc_id] = doc_index

            # Update statistics
            self.stats["total_documents"] = len(self.documents)
            self.stats["index_size"] = self.index.ntotal

            self.log_debug("Added document to vector store", doc_id=doc_id)

            return doc_id

        try:
            return await _do_add()
        except TransientError as e:
            self.log_error("Transient error adding document to vector store", error=e)
            raise VectorStoreError(str(e))
        except Exception as e:
            self.log_error("Error adding document to vector store", error=e)
            raise VectorStoreError(str(e))

    async def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Add multiple documents to the vector store.

        Args:
            documents: List of document dictionaries with 'content', 'embedding', 'metadata'

        Returns:
            List of document IDs
        """
        doc_ids = []

        @retry((TransientError,), retries=2, backoff_factor=0.5)
        async def _do_add_batch() -> List[str]:
            # Prepare batch data
            embeddings = []
            doc_objects = []

            for doc_data in documents:
                content = doc_data["content"]
                embedding = doc_data["embedding"]
                metadata = doc_data.get("metadata", {})

                # Generate document ID
                doc_id = metadata.get(
                    "id", f"doc_{len(self.documents) + len(doc_objects)}"
                )

                # Create document object
                document = Document(
                    content=content,
                    embedding=embedding,
                    metadata={
                        **metadata,
                        "id": doc_id,
                        "added_at": datetime.utcnow().isoformat(),
                    },
                )

                doc_objects.append(document)
                embeddings.append(embedding)
                doc_ids.append(doc_id)

            # Normalize embeddings for cosine similarity
            embeddings_array = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(embeddings_array)

            # Add to FAISS index
            self.index.add(embeddings_array)

            # Add to document storage
            start_index = len(self.documents)
            self.documents.extend(doc_objects)

            # Update ID mapping
            for i, doc_id in enumerate(doc_ids):
                self.id_to_index[doc_id] = start_index + i

            # Update statistics
            self.stats["total_documents"] = len(self.documents)
            self.stats["index_size"] = self.index.ntotal

            self.log_info(f"Added {len(documents)} documents to vector store")

            return doc_ids

        try:
            return await _do_add_batch()
        except TransientError as e:
            self.log_error("Transient error adding documents to vector store", error=e)
            raise VectorStoreError(str(e))
        except Exception as e:
            self.log_error("Error adding documents to vector store", error=e)
            raise VectorStoreError(str(e))

    async def similarity_search(
        self, query_embedding: Vector, k: int = 5, threshold: float = 0.0
    ) -> List[SimilarityResult]:
        """
        Perform similarity search.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of similarity results
        """
        try:
            if self.index.ntotal == 0:
                self.log_warning("No documents in vector store for search")
                return []

            # Normalize query embedding
            query_array = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(query_array)

            # Perform search
            search_k = min(k, self.index.ntotal)
            scores, indices = self.index.search(query_array, search_k)

            # Convert to similarity results
            results = []
            for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # Invalid index
                    continue

                if score >= threshold:
                    document = self.documents[idx]
                    result = SimilarityResult(
                        document=document, score=float(score), rank=rank
                    )
                    results.append(result)

            # Update statistics
            self.stats["searches_performed"] += 1

            self.log_debug(
                "Similarity search completed",
                results_count=len(results),
                search_k=search_k,
            )

            return results

        except Exception as e:
            self.log_error("Error performing similarity search", error=e)
            # Map to typed error if needed by callers; here we return empty results
            return []

    async def get_document(self, doc_id: str) -> Optional[Document]:
        """
        Get document by ID.

        Args:
            doc_id: Document ID

        Returns:
            Document or None if not found
        """
        try:
            if doc_id in self.id_to_index:
                index = self.id_to_index[doc_id]
                return self.documents[index]
            return None

        except Exception as e:
            self.log_error("Error getting document", error=e, doc_id=doc_id)
            return None

    async def delete_document(self, doc_id: str) -> bool:
        """
        Delete document by ID.
        Note: FAISS doesn't support deletion, so this rebuilds the index.

        Args:
            doc_id: Document ID

        Returns:
            True if deleted successfully
        """
        try:
            if doc_id not in self.id_to_index:
                self.log_warning(f"Document not found for deletion: {doc_id}")
                return False

            # Remove from documents list
            index_to_remove = self.id_to_index[doc_id]
            del self.documents[index_to_remove]
            del self.id_to_index[doc_id]

            # Update indices for remaining documents
            for doc_id_key, doc_index in self.id_to_index.items():
                if doc_index > index_to_remove:
                    self.id_to_index[doc_id_key] = doc_index - 1

            # Rebuild FAISS index
            await self._rebuild_index()

            self.log_info(f"Deleted document: {doc_id}")
            return True

        except Exception as e:
            self.log_error("Error deleting document", error=e, doc_id=doc_id)
            return False

    async def _rebuild_index(self):
        """Rebuild FAISS index from current documents."""
        @retry((TransientError,), retries=2, backoff_factor=0.5)
        async def _do_rebuild():
            self._create_new_index()

            if self.documents:
                embeddings = [doc.embedding for doc in self.documents]
                embeddings_array = np.array(embeddings, dtype=np.float32)
                faiss.normalize_L2(embeddings_array)
                self.index.add(embeddings_array)

            # Update statistics
            self.stats["index_size"] = self.index.ntotal

            self.log_info("Rebuilt FAISS index", documents=len(self.documents))

        try:
            return await _do_rebuild()
        except TransientError as e:
            self.log_error("Transient error rebuilding index", error=e)
            raise VectorStoreError(str(e))
        except Exception as e:
            self.log_error("Error rebuilding index", error=e)
            raise VectorStoreError(str(e))

    def save_index(self):
        """Save FAISS index and metadata to disk."""
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_path))

            # Save metadata and documents
            metadata = {
                "documents": [doc.to_dict() for doc in self.documents],
                "id_to_index": self.id_to_index,
                "stats": self.stats,
                "config": {"dimension": self.dimension, "index_type": "IndexFlatIP"},
            }

            with open(self.metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            self.stats["last_save"] = datetime.utcnow().isoformat()
            self.log_info("Saved FAISS index and metadata")

        except Exception as e:
            self.log_error("Error saving FAISS index", error=e)
            # Do not raise - saving should not crash the application
            return

    def load_index(self):
        """Load FAISS index and metadata from disk."""
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_path))

            # Load metadata
            with open(self.metadata_path, "r") as f:
                metadata = json.load(f)

            # Restore documents
            self.documents = [
                Document.from_dict(doc_data) for doc_data in metadata["documents"]
            ]

            # Restore mappings and stats
            self.id_to_index = metadata["id_to_index"]
            self.stats.update(metadata.get("stats", {}))

            self.log_info(
                "Loaded FAISS index and metadata", documents=len(self.documents)
            )

        except Exception as e:
            self.log_error("Error loading FAISS index", error=e)
            raise VectorStoreError(str(e))

    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return {
            **self.stats,
            "index_path": str(self.index_path),
            "dimension": self.dimension,
            "index_type": "FAISS IndexFlatIP",
        }

    def __del__(self):
        """Save index when object is destroyed."""
        try:
            if self.index is not None:
                self.save_index()
        except Exception:
            pass


def create_faiss_store(index_path: str, dimension: int = 1536) -> FAISSVectorStore:
    """
    Create FAISS vector store with simple configuration.

    Args:
        index_path: Path to save/load index
        dimension: Embedding dimension

    Returns:
        Configured FAISS vector store
    """
    config = VectorStoreConfig(
        store_type="faiss", dimension=dimension, index_path=index_path
    )
    return FAISSVectorStore(config)
