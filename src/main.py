"""
Main entry point for the Real-time RAG system.
"""

import asyncio
import signal
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import get_config
from .utils.logger import setup_logger, get_logger
from .ingestion.rss_ingest import RSSIngestor
from .ingestion.drive_ingest import DriveIngestor
from .ingestion.filewatch_ingest import FileWatchIngestor
from .embeddings.embedder import EmbeddingService
from .vectorstores.faiss_store import FAISSVectorStore
from .vectorstores.pathway_client import PathwayVectorStore


# Global variables for services
rss_ingestor = None
drive_ingestor = None
filewatch_ingestor = None
embedding_service = None
vector_store = None
logger = None


class QueryRequest(BaseModel):
    """Request model for queries."""

    query: str
    k: int = 5
    threshold: float = 0.0


class QueryResponse(BaseModel):
    """Response model for queries."""

    results: list
    total_results: int
    query: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan."""
    global \
        rss_ingestor, \
        drive_ingestor, \
        filewatch_ingestor, \
        embedding_service, \
        vector_store, \
        logger

    config = get_config()

    # Setup logging
    setup_logger(config.logging)
    logger = get_logger(__name__)

    try:
        # Validate configuration
        config.validate()
        logger.info("Configuration validated successfully")

        # Initialize services
        embedding_service = EmbeddingService(config.openai)
        logger.info("Embedding service initialized")

        # Initialize vector store
        if config.vectorstore.store_type == "faiss":
            vector_store = FAISSVectorStore(config.vectorstore)
        else:
            vector_store = PathwayVectorStore(config.vectorstore)
        logger.info(f"Vector store initialized: {config.vectorstore.store_type}")

        # Initialize ingestors
        if config.rss.feeds:
            rss_ingestor = RSSIngestor(config.rss, embedding_service, vector_store)
            logger.info(f"RSS ingestor initialized with {len(config.rss.feeds)} feeds")

        if config.drive.folder_id and config.drive.credentials_path:
            drive_ingestor = DriveIngestor(
                config.drive, embedding_service, vector_store
            )
            logger.info("Google Drive ingestor initialized")

        if config.filewatch.watch_directory:
            filewatch_ingestor = FileWatchIngestor(
                config.filewatch, embedding_service, vector_store
            )
            logger.info(
                f"File watch ingestor initialized for: {config.filewatch.watch_directory}"
            )

        # Start background tasks
        await start_background_tasks()

        logger.info("Real-time RAG system started successfully")

    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise

    yield

    # Cleanup
    logger.info("Shutting down Real-time RAG system...")
    await stop_background_tasks()


async def start_background_tasks():
    """Start all background ingestion tasks."""
    tasks = []

    if rss_ingestor:
        tasks.append(asyncio.create_task(rss_ingestor.start_monitoring()))

    if drive_ingestor:
        tasks.append(asyncio.create_task(drive_ingestor.start_monitoring()))

    if filewatch_ingestor:
        tasks.append(asyncio.create_task(filewatch_ingestor.start_monitoring()))

    if tasks:
        logger.info(f"Started {len(tasks)} background ingestion tasks")


async def stop_background_tasks():
    """Stop all background tasks gracefully."""
    if rss_ingestor:
        await rss_ingestor.stop_monitoring()

    if drive_ingestor:
        await drive_ingestor.stop_monitoring()

    if filewatch_ingestor:
        await filewatch_ingestor.stop_monitoring()


# Create FastAPI application
app = FastAPI(
    title="Real-time RAG Playground",
    description="A real-time Retrieval-Augmented Generation system",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Real-time RAG Playground API", "status": "running"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    status = {
        "status": "healthy",
        "services": {
            "embedding_service": embedding_service is not None,
            "vector_store": vector_store is not None,
            "rss_ingestor": rss_ingestor is not None,
            "drive_ingestor": drive_ingestor is not None,
            "filewatch_ingestor": filewatch_ingestor is not None,
        },
    }
    return status


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the vector store."""
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    try:
        # Get query embedding
        query_embedding = await embedding_service.get_embedding(request.query)

        # Search vector store
        results = await vector_store.similarity_search(
            query_embedding, k=request.k, threshold=request.threshold
        )

        return QueryResponse(
            results=results, total_results=len(results), query=request.query
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/stats")
async def stats():
    """Get system statistics."""
    stats_data = {
        "vector_store_stats": await vector_store.get_stats() if vector_store else {},
        "ingestion_stats": {},
    }

    if rss_ingestor:
        stats_data["ingestion_stats"]["rss"] = await rss_ingestor.get_stats()

    if drive_ingestor:
        stats_data["ingestion_stats"]["drive"] = await drive_ingestor.get_stats()

    if filewatch_ingestor:
        stats_data["ingestion_stats"][
            "filewatch"
        ] = await filewatch_ingestor.get_stats()

    return stats_data


@app.post("/ingest/trigger")
async def trigger_ingestion(background_tasks: BackgroundTasks):
    """Manually trigger ingestion from all sources."""
    triggered = []

    if rss_ingestor:
        background_tasks.add_task(rss_ingestor.force_refresh)
        triggered.append("rss")

    if drive_ingestor:
        background_tasks.add_task(drive_ingestor.force_refresh)
        triggered.append("drive")

    if filewatch_ingestor:
        background_tasks.add_task(filewatch_ingestor.force_refresh)
        triggered.append("filewatch")

    return {"message": f"Triggered ingestion for: {', '.join(triggered)}"}


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)


async def main():
    """Main function to run the application."""
    config = get_config()

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run the server
    config_dict = {
        "app": "main:app",
        "host": config.server.host,
        "port": config.server.port,
        "reload": config.server.reload,
        "log_level": config.logging.level.lower(),
    }

    await uvicorn.run(**config_dict)


if __name__ == "__main__":
    asyncio.run(main())
