"""
LangChain VectorStore implementation for Pathway integration.
"""

from typing import List, Dict, Any, Optional, Tuple, Iterable
import asyncio
from langchain.vectorstores.base import VectorStore
from langchain.docstore.document import Document as LangChainDocument
from langchain.embeddings.base import Embeddings

from ..vectorstores.pathway_client import (
    PathwayVectorStoreClient,
    create_pathway_client,
)
from ..embeddings.typing import Document, SimilarityResult
from ..config import VectorStoreConfig
from ..utils.logger import LoggerMixin


class PathwayVectorStore(VectorStore, LoggerMixin):
    """
    LangChain VectorStore implementation using Pathway for real-time vector storage.
    """

    def __init__(
        self,
        pathway_client: Optional[PathwayVectorStoreClient] = None,
        embedding: Optional[Embeddings] = None,
        **kwargs: Any,
    ):
        super().__init__()
        LoggerMixin.__init__(self)

        self.pathway_client = pathway_client or create_pathway_client()
        self.embedding = embedding

        # LangChain compatibility
        self.embeddings = embedding

    async def __aenter__(self):
        """Async context manager entry."""
        await self.pathway_client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.pathway_client.__aexit__(exc_type, exc_val, exc_tb)

    def _langchain_doc_to_internal(self, doc: LangChainDocument) -> Dict[str, Any]:
        """Convert LangChain Document to internal format."""
        return {"content": doc.page_content, "metadata": doc.metadata}

    def _internal_to_langchain_doc(self, doc: Document) -> LangChainDocument:
        """Convert internal Document to LangChain Document."""
        return LangChainDocument(page_content=doc.content, metadata=doc.metadata)

    def _similarity_result_to_langchain(
        self, result: SimilarityResult
    ) -> Tuple[LangChainDocument, float]:
        """Convert SimilarityResult to LangChain format."""
        langchain_doc = self._internal_to_langchain_doc(result.document)
        return (langchain_doc, result.score)

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Add texts to vector store asynchronously.

        Args:
            texts: Iterable of texts to add
            metadatas: Optional list of metadata for each text
            **kwargs: Additional arguments

        Returns:
            List of document IDs
        """
        try:
            if not self.embedding:
                raise ValueError("Embedding function not provided")

            text_list = list(texts)
            metadatas = metadatas or [{}] * len(text_list)

            # Generate embeddings
            embeddings = await self.embedding.aembed_documents(text_list)

            # Prepare documents for batch insertion
            documents = []
            for i, (text, metadata) in enumerate(zip(text_list, metadatas)):
                doc_data = {
                    "content": text,
                    "embedding": embeddings[i],
                    "metadata": metadata,
                }
                documents.append(doc_data)

            # Add to Pathway
            doc_ids = await self.pathway_client.add_documents(documents)

            self.log_info("Added texts to Pathway vector store", count=len(doc_ids))
            return doc_ids

        except Exception as e:
            self.log_error("Error adding texts to vector store", error=e)
            return []

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Add texts to vector store (sync wrapper).

        Args:
            texts: Iterable of texts to add
            metadatas: Optional list of metadata for each text
            **kwargs: Additional arguments

        Returns:
            List of document IDs
        """
        return asyncio.run(self.aadd_texts(texts, metadatas, **kwargs))

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[LangChainDocument, float]]:
        """
        Similarity search with scores asynchronously.

        Args:
            query: Query text
            k: Number of results to return
            filter: Optional metadata filters
            **kwargs: Additional arguments

        Returns:
            List of (Document, score) tuples
        """
        try:
            if not self.embedding:
                raise ValueError("Embedding function not provided")

            # Generate query embedding
            query_embedding = await self.embedding.aembed_query(query)

            # Perform search
            threshold = kwargs.get("score_threshold", 0.0)
            results = await self.pathway_client.similarity_search(
                query_embedding=query_embedding,
                k=k,
                threshold=threshold,
                filters=filter,
            )

            # Convert to LangChain format
            langchain_results = [
                self._similarity_result_to_langchain(result) for result in results
            ]

            self.log_debug(
                "Similarity search completed",
                query=query[:50] + "..." if len(query) > 50 else query,
                results_count=len(langchain_results),
            )

            return langchain_results

        except Exception as e:
            self.log_error("Error performing similarity search", error=e)
            return []

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[LangChainDocument, float]]:
        """
        Similarity search with scores (sync wrapper).

        Args:
            query: Query text
            k: Number of results to return
            filter: Optional metadata filters
            **kwargs: Additional arguments

        Returns:
            List of (Document, score) tuples
        """
        return asyncio.run(
            self.asimilarity_search_with_score(query, k, filter, **kwargs)
        )

    async def asimilarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[LangChainDocument]:
        """
        Similarity search asynchronously.

        Args:
            query: Query text
            k: Number of results to return
            filter: Optional metadata filters
            **kwargs: Additional arguments

        Returns:
            List of Documents
        """
        results = await self.asimilarity_search_with_score(query, k, filter, **kwargs)
        return [doc for doc, score in results]

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[LangChainDocument]:
        """
        Similarity search (sync wrapper).

        Args:
            query: Query text
            k: Number of results to return
            filter: Optional metadata filters
            **kwargs: Additional arguments

        Returns:
            List of Documents
        """
        return asyncio.run(self.asimilarity_search(query, k, filter, **kwargs))

    async def adelete(
        self, ids: Optional[List[str]] = None, **kwargs: Any
    ) -> Optional[bool]:
        """
        Delete documents by IDs asynchronously.

        Args:
            ids: List of document IDs to delete
            **kwargs: Additional arguments

        Returns:
            True if successful
        """
        if not ids:
            return False

        try:
            success_count = 0
            for doc_id in ids:
                if await self.pathway_client.delete_document(doc_id):
                    success_count += 1

            self.log_info(
                "Deleted documents from vector store",
                successful=success_count,
                total=len(ids),
            )

            return success_count == len(ids)

        except Exception as e:
            self.log_error("Error deleting documents", error=e)
            return False

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """
        Delete documents by IDs (sync wrapper).

        Args:
            ids: List of document IDs to delete
            **kwargs: Additional arguments

        Returns:
            True if successful
        """
        return asyncio.run(self.adelete(ids, **kwargs))

    async def aget_document(self, doc_id: str) -> Optional[LangChainDocument]:
        """
        Get document by ID asynchronously.

        Args:
            doc_id: Document ID

        Returns:
            LangChain Document or None
        """
        try:
            internal_doc = await self.pathway_client.get_document(doc_id)
            if internal_doc:
                return self._internal_to_langchain_doc(internal_doc)
            return None

        except Exception as e:
            self.log_error("Error getting document", error=e, doc_id=doc_id)
            return None

    def get_document(self, doc_id: str) -> Optional[LangChainDocument]:
        """
        Get document by ID (sync wrapper).

        Args:
            doc_id: Document ID

        Returns:
            LangChain Document or None
        """
        return asyncio.run(self.aget_document(doc_id))

    async def alist_documents(
        self, limit: int = 100, offset: int = 0, filter: Optional[Dict[str, Any]] = None
    ) -> List[LangChainDocument]:
        """
        List documents asynchronously.

        Args:
            limit: Maximum number of documents
            offset: Number to skip
            filter: Optional metadata filters

        Returns:
            List of LangChain Documents
        """
        try:
            internal_docs = await self.pathway_client.list_documents(
                limit=limit, offset=offset, filters=filter
            )

            return [self._internal_to_langchain_doc(doc) for doc in internal_docs]

        except Exception as e:
            self.log_error("Error listing documents", error=e)
            return []

    def list_documents(
        self, limit: int = 100, offset: int = 0, filter: Optional[Dict[str, Any]] = None
    ) -> List[LangChainDocument]:
        """
        List documents (sync wrapper).

        Args:
            limit: Maximum number of documents
            offset: Number to skip
            filter: Optional metadata filters

        Returns:
            List of LangChain Documents
        """
        return asyncio.run(self.alist_documents(limit, offset, filter))

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[LangChainDocument]:
        """
        Maximum marginal relevance search asynchronously.

        Args:
            query: Query text
            k: Number of results to return
            fetch_k: Number of results to fetch initially
            lambda_mult: Diversity factor (0=max diversity, 1=min diversity)
            filter: Optional metadata filters
            **kwargs: Additional arguments

        Returns:
            List of Documents
        """
        # Fetch more documents than needed for MMR
        results = await self.asimilarity_search_with_score(
            query, fetch_k, filter, **kwargs
        )

        if not results:
            return []

        # Simple MMR implementation
        # In a full implementation, you'd calculate MMR based on document embeddings
        # For now, we'll just return the top-k results
        selected_docs = [doc for doc, score in results[:k]]

        self.log_debug(
            "MMR search completed",
            query=query[:50] + "..." if len(query) > 50 else query,
            results_count=len(selected_docs),
        )

        return selected_docs

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[LangChainDocument]:
        """
        Maximum marginal relevance search (sync wrapper).

        Args:
            query: Query text
            k: Number of results to return
            fetch_k: Number of results to fetch initially
            lambda_mult: Diversity factor
            filter: Optional metadata filters
            **kwargs: Additional arguments

        Returns:
            List of Documents
        """
        return asyncio.run(
            self.amax_marginal_relevance_search(
                query, k, fetch_k, lambda_mult, filter, **kwargs
            )
        )

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> "PathwayVectorStore":
        """
        Create PathwayVectorStore from texts.

        Args:
            texts: List of texts
            embedding: Embedding function
            metadatas: Optional metadata list
            **kwargs: Additional arguments

        Returns:
            Configured PathwayVectorStore
        """
        # Create vector store
        vector_store = cls(embedding=embedding, **kwargs)

        # Add texts
        vector_store.add_texts(texts, metadatas)

        return vector_store

    @classmethod
    async def afrom_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> "PathwayVectorStore":
        """
        Create PathwayVectorStore from texts asynchronously.

        Args:
            texts: List of texts
            embedding: Embedding function
            metadatas: Optional metadata list
            **kwargs: Additional arguments

        Returns:
            Configured PathwayVectorStore
        """
        # Create vector store
        vector_store = cls(embedding=embedding, **kwargs)

        # Add texts
        await vector_store.aadd_texts(texts, metadatas)

        return vector_store

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics.

        Returns:
            Statistics dictionary
        """
        return await self.pathway_client.get_stats()


# Compatibility alias for easier imports
PathwayVectorstore = PathwayVectorStore
