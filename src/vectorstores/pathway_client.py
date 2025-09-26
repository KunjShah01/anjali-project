"""
Pathway vector store client for real-time streaming data.
"""

import asyncio
import json
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import aiohttp
from datetime import datetime, timedelta

from ..config import VectorStoreConfig
from ..embeddings.typing import Vector, Document, SimilarityResult
from ..utils.logger import LoggerMixin


class PathwayVectorStoreClient(LoggerMixin):
    """
    Client for Pathway vector store API for real-time vector operations.
    """

    def __init__(self, config: VectorStoreConfig):
        super().__init__()
        self.config = config
        self.base_url = f"http://{config.pathway_host}:{config.pathway_port}"
        self.session: Optional[aiohttp.ClientSession] = None

        # Connection settings
        self.timeout = aiohttp.ClientTimeout(total=30)
        self.max_retries = 3
        self.retry_delay = 1.0

        # Statistics
        self.stats = {
            "requests_made": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "documents_added": 0,
            "searches_performed": 0,
            "average_response_time": 0.0,
            "last_connection": None,
        }

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def _ensure_session(self):
        """Ensure aiohttp session is created."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
            self.log_debug("Created new aiohttp session")

    async def close(self):
        """Close the HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.log_debug("Closed aiohttp session")

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Make HTTP request to Pathway server with retries.

        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request JSON data
            params: URL parameters

        Returns:
            Response JSON or None if failed
        """
        await self._ensure_session()
        url = f"{self.base_url}{endpoint}"

        for attempt in range(self.max_retries):
            start_time = datetime.utcnow()

            try:
                self.stats["requests_made"] += 1

                async with self.session.request(
                    method, url, json=data, params=params
                ) as response:
                    # Calculate response time
                    response_time = (datetime.utcnow() - start_time).total_seconds()
                    self._update_response_time(response_time)

                    if response.status == 200:
                        self.stats["successful_requests"] += 1
                        self.stats["last_connection"] = datetime.utcnow().isoformat()

                        result = await response.json()
                        self.log_debug(
                            "Request successful",
                            method=method,
                            endpoint=endpoint,
                            response_time=response_time,
                        )
                        return result

                    else:
                        error_text = await response.text()
                        self.log_warning(
                            f"Request failed with status {response.status}",
                            method=method,
                            endpoint=endpoint,
                            error=error_text,
                        )

                        if response.status < 500 or attempt == self.max_retries - 1:
                            self.stats["failed_requests"] += 1
                            return None

            except asyncio.TimeoutError:
                self.log_warning(
                    f"Request timeout (attempt {attempt + 1})",
                    method=method,
                    endpoint=endpoint,
                )

            except Exception as e:
                self.log_error(
                    f"Request error (attempt {attempt + 1})",
                    method=method,
                    endpoint=endpoint,
                    error=e,
                )

            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay * (2**attempt))

        self.stats["failed_requests"] += 1
        return None

    def _update_response_time(self, response_time: float):
        """Update average response time statistics."""
        current_avg = self.stats["average_response_time"]
        total_requests = self.stats["successful_requests"]

        if total_requests == 1:
            self.stats["average_response_time"] = response_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.stats["average_response_time"] = (
                alpha * response_time + (1 - alpha) * current_avg
            )

    async def health_check(self) -> bool:
        """
        Check if Pathway server is healthy.

        Returns:
            True if server is healthy
        """
        try:
            result = await self._make_request("GET", "/health")
            return result is not None and result.get("status") == "healthy"

        except Exception as e:
            self.log_error("Health check failed", error=e)
            return False

    async def add_document(
        self, content: str, embedding: Vector, metadata: Dict[str, Any]
    ) -> Optional[str]:
        """
        Add a document to Pathway vector store.

        Args:
            content: Document content
            embedding: Document embedding vector
            metadata: Document metadata

        Returns:
            Document ID if successful
        """
        try:
            doc_id = metadata.get("id", f"doc_{datetime.utcnow().timestamp()}")

            payload = {
                "id": doc_id,
                "content": content,
                "embedding": embedding,
                "metadata": {**metadata, "added_at": datetime.utcnow().isoformat()},
            }

            result = await self._make_request("POST", "/documents", data=payload)

            if result and result.get("success"):
                self.stats["documents_added"] += 1
                self.log_debug("Added document to Pathway", doc_id=doc_id)
                return doc_id
            else:
                self.log_error("Failed to add document to Pathway", doc_id=doc_id)
                return None

        except Exception as e:
            self.log_error("Error adding document to Pathway", error=e)
            return None

    async def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Add multiple documents to Pathway vector store.

        Args:
            documents: List of document dictionaries

        Returns:
            List of successful document IDs
        """
        try:
            # Prepare batch payload
            batch_docs = []
            for doc_data in documents:
                doc_id = doc_data.get("metadata", {}).get(
                    "id", f"doc_{datetime.utcnow().timestamp()}"
                )

                doc_payload = {
                    "id": doc_id,
                    "content": doc_data["content"],
                    "embedding": doc_data["embedding"],
                    "metadata": {
                        **doc_data.get("metadata", {}),
                        "added_at": datetime.utcnow().isoformat(),
                    },
                }
                batch_docs.append(doc_payload)

            payload = {"documents": batch_docs}
            result = await self._make_request("POST", "/documents/batch", data=payload)

            if result and result.get("success"):
                successful_ids = result.get("document_ids", [])
                self.stats["documents_added"] += len(successful_ids)

                self.log_info(
                    "Added documents to Pathway",
                    count=len(successful_ids),
                    total=len(documents),
                )

                return successful_ids
            else:
                self.log_error("Failed to add documents batch to Pathway")
                return []

        except Exception as e:
            self.log_error("Error adding documents batch to Pathway", error=e)
            return []

    async def similarity_search(
        self,
        query_embedding: Vector,
        k: int = 5,
        threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SimilarityResult]:
        """
        Perform similarity search in Pathway vector store.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            threshold: Minimum similarity threshold
            filters: Optional metadata filters

        Returns:
            List of similarity results
        """
        try:
            payload = {"embedding": query_embedding, "k": k, "threshold": threshold}

            if filters:
                payload["filters"] = filters

            result = await self._make_request("POST", "/search", data=payload)

            if result and result.get("success"):
                self.stats["searches_performed"] += 1

                # Convert response to SimilarityResult objects
                results = []
                for rank, hit in enumerate(result.get("results", [])):
                    document = Document(
                        content=hit["content"],
                        embedding=hit.get("embedding", []),
                        metadata=hit.get("metadata", {}),
                    )

                    similarity_result = SimilarityResult(
                        document=document, score=hit["score"], rank=rank
                    )
                    results.append(similarity_result)

                self.log_debug(
                    "Similarity search completed",
                    results_count=len(results),
                    search_k=k,
                )

                return results
            else:
                self.log_error("Similarity search failed in Pathway")
                return []

        except Exception as e:
            self.log_error("Error performing similarity search in Pathway", error=e)
            return []

    async def get_document(self, doc_id: str) -> Optional[Document]:
        """
        Get document by ID from Pathway vector store.

        Args:
            doc_id: Document ID

        Returns:
            Document or None if not found
        """
        try:
            result = await self._make_request("GET", f"/documents/{doc_id}")

            if result and result.get("success"):
                doc_data = result["document"]
                return Document(
                    content=doc_data["content"],
                    embedding=doc_data.get("embedding", []),
                    metadata=doc_data.get("metadata", {}),
                )
            else:
                self.log_warning("Document not found in Pathway", doc_id=doc_id)
                return None

        except Exception as e:
            self.log_error(
                "Error getting document from Pathway", error=e, doc_id=doc_id
            )
            return None

    async def delete_document(self, doc_id: str) -> bool:
        """
        Delete document by ID from Pathway vector store.

        Args:
            doc_id: Document ID

        Returns:
            True if deleted successfully
        """
        try:
            result = await self._make_request("DELETE", f"/documents/{doc_id}")

            if result and result.get("success"):
                self.log_info("Deleted document from Pathway", doc_id=doc_id)
                return True
            else:
                self.log_warning(
                    "Failed to delete document from Pathway", doc_id=doc_id
                )
                return False

        except Exception as e:
            self.log_error(
                "Error deleting document from Pathway", error=e, doc_id=doc_id
            )
            return False

    async def list_documents(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        List documents in Pathway vector store.

        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            filters: Optional metadata filters

        Returns:
            List of documents
        """
        try:
            params = {"limit": limit, "offset": offset}
            if filters:
                params["filters"] = json.dumps(filters)

            result = await self._make_request("GET", "/documents", params=params)

            if result and result.get("success"):
                documents = []
                for doc_data in result.get("documents", []):
                    document = Document(
                        content=doc_data["content"],
                        embedding=doc_data.get("embedding", []),
                        metadata=doc_data.get("metadata", {}),
                    )
                    documents.append(document)

                self.log_debug("Listed documents from Pathway", count=len(documents))
                return documents
            else:
                self.log_error("Failed to list documents from Pathway")
                return []

        except Exception as e:
            self.log_error("Error listing documents from Pathway", error=e)
            return []

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics from Pathway server.

        Returns:
            Statistics dictionary
        """
        try:
            result = await self._make_request("GET", "/stats")

            if result and result.get("success"):
                server_stats = result.get("stats", {})

                # Combine client and server stats
                combined_stats = {
                    **self.stats,
                    "server_stats": server_stats,
                    "pathway_host": self.config.pathway_host,
                    "pathway_port": self.config.pathway_port,
                    "store_type": "pathway",
                }

                return combined_stats
            else:
                return self.stats

        except Exception as e:
            self.log_error("Error getting stats from Pathway", error=e)
            return self.stats


def create_pathway_client(
    host: str = "localhost", port: int = 8080
) -> PathwayVectorStoreClient:
    """
    Create Pathway vector store client with simple configuration.

    Args:
        host: Pathway server host
        port: Pathway server port

    Returns:
        Configured Pathway client
    """
    config = VectorStoreConfig(
        store_type="pathway", pathway_host=host, pathway_port=port
    )
    return PathwayVectorStoreClient(config)
