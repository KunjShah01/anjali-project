"""
File system monitoring and ingestion for the Real-time RAG system.
"""

import asyncio
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, AsyncGenerator
import aiofiles
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent

from ..config import FileWatchConfig
from ..preprocessing.cleaner import DocumentProcessor
from ..utils.logger import LoggerMixin, ContextLogger


class FileChangeHandler(FileSystemEventHandler, LoggerMixin):
    """Handle file system events."""

    def __init__(self, ingestor: "FileWatchIngestor"):
        super().__init__()
        self.ingestor = ingestor

    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory:
            self.ingestor.queue_file_for_processing(event.src_path, "modified")

    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory:
            self.ingestor.queue_file_for_processing(event.src_path, "created")


class FileWatchIngestor(LoggerMixin):
    """
    File system monitoring and ingestion service.
    """

    def __init__(
        self, config: FileWatchConfig, embedding_service=None, vector_store=None
    ):
        super().__init__()
        self.config = config
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.processor = DocumentProcessor()

        # File monitoring
        self.observer = None
        self.event_handler = FileChangeHandler(self)
        self._stop_monitoring = False

        # Track processed files
        self.file_hashes: Dict[str, str] = {}
        self.file_queue: asyncio.Queue = asyncio.Queue()

        # Supported extensions
        self.supported_extensions = set(
            ext.lower() for ext in self.config.watch_extensions
        )

        # Statistics
        self.stats = {
            "total_files_processed": 0,
            "files_by_extension": {},
            "last_update": None,
            "errors": 0,
            "queue_size": 0,
        }

    def is_supported_file(self, file_path: str) -> bool:
        """
        Check if file is supported based on extension.

        Args:
            file_path: Path to the file

        Returns:
            True if file is supported
        """
        path = Path(file_path)
        return path.suffix.lower() in self.supported_extensions

    def queue_file_for_processing(self, file_path: str, event_type: str):
        """
        Queue file for processing.

        Args:
            file_path: Path to the file
            event_type: Type of event ('created', 'modified')
        """
        if self.is_supported_file(file_path):
            try:
                self.file_queue.put_nowait(
                    {
                        "path": file_path,
                        "event_type": event_type,
                        "timestamp": datetime.utcnow(),
                    }
                )
                self.log_debug(f"Queued file for processing: {file_path}")
            except asyncio.QueueFull:
                self.log_warning(f"File queue full, dropping: {file_path}")

    async def calculate_file_hash(self, file_path: str) -> Optional[str]:
        """
        Calculate MD5 hash of file content.

        Args:
            file_path: Path to the file

        Returns:
            MD5 hash string or None if failed
        """
        try:
            hash_md5 = hashlib.md5()
            async with aiofiles.open(file_path, "rb") as f:
                async for chunk in f:
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()

        except Exception as e:
            self.log_error(f"Error calculating file hash", error=e, file_path=file_path)
            return None

    async def read_file_content(self, file_path: str) -> Optional[str]:
        """
        Read file content based on extension.

        Args:
            file_path: Path to the file

        Returns:
            File content as string or None if failed
        """
        try:
            path = Path(file_path)
            extension = path.suffix.lower()

            if extension == ".pdf":
                return await self._read_pdf_file(file_path)
            elif extension in [".txt", ".md", ".rst", ".log"]:
                return await self._read_text_file(file_path)
            elif extension in [".html", ".htm"]:
                return await self._read_html_file(file_path)
            else:
                # Try to read as text
                return await self._read_text_file(file_path)

        except Exception as e:
            self.log_error("Error reading file", error=e, file_path=file_path)
            return None

    async def _read_text_file(self, file_path: str) -> str:
        """Read plain text file."""
        async with aiofiles.open(
            file_path, "r", encoding="utf-8", errors="ignore"
        ) as f:
            return await f.read()

    async def _read_pdf_file(self, file_path: str) -> str:
        """Read PDF file (placeholder - would need async PDF library)."""
        # For now, return empty string - would need async PDF processing
        self.log_warning(f"PDF processing not implemented for: {file_path}")
        return ""

    async def _read_html_file(self, file_path: str) -> str:
        """Read and clean HTML file."""
        async with aiofiles.open(
            file_path, "r", encoding="utf-8", errors="ignore"
        ) as f:
            html_content = await f.read()
            return self.processor.cleaner.clean_text(html_content)

    async def process_file(self, file_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a single file.

        Args:
            file_info: File information from queue

        Returns:
            List of processed document chunks
        """
        file_path = file_info["path"]

        try:
            with ContextLogger("file_processing", file_path=file_path):
                # Check if file exists
                if not Path(file_path).exists():
                    self.log_warning(f"File no longer exists: {file_path}")
                    return []

                # Calculate file hash
                current_hash = await self.calculate_file_hash(file_path)
                if not current_hash:
                    return []

                # Check if file has changed
                if (
                    file_path in self.file_hashes
                    and self.file_hashes[file_path] == current_hash
                ):
                    self.log_debug(f"File unchanged, skipping: {file_path}")
                    return []

                # Read file content
                content = await self.read_file_content(file_path)
                if not content:
                    return []

                # Create metadata
                path_obj = Path(file_path)
                metadata = {
                    "source": "file_watch",
                    "file_path": file_path,
                    "file_name": path_obj.name,
                    "file_extension": path_obj.suffix,
                    "file_size": path_obj.stat().st_size,
                    "modified_time": datetime.fromtimestamp(
                        path_obj.stat().st_mtime
                    ).isoformat(),
                    "event_type": file_info["event_type"],
                    "ingested_at": datetime.utcnow().isoformat(),
                }

                # Process into chunks
                documents = self.processor.process_document(content, metadata)

                # Store in vector database if available
                if self.embedding_service and self.vector_store:
                    for doc in documents:
                        try:
                            embedding = await self.embedding_service.get_embedding(
                                doc["content"]
                            )
                            await self.vector_store.add_document(
                                content=doc["content"],
                                embedding=embedding,
                                metadata=doc["metadata"],
                            )
                        except Exception as e:
                            self.log_error("Error storing document", error=e)

                # Update tracking
                self.file_hashes[file_path] = current_hash

                # Update statistics
                extension = path_obj.suffix.lower()
                self.stats["files_by_extension"][extension] = (
                    self.stats["files_by_extension"].get(extension, 0) + 1
                )

                self.log_info(f"Processed file: {path_obj.name}")
                return documents

        except Exception as e:
            self.log_error("Error processing file", error=e, file_path=file_path)
            self.stats["errors"] += 1
            return []

    async def scan_directory(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Scan directory for existing files.

        Yields:
            Processed document chunks
        """
        try:
            watch_path = Path(self.config.watch_directory)

            if not watch_path.exists():
                self.log_warning(f"Watch directory does not exist: {watch_path}")
                return

            # Get all supported files
            pattern = "**/*" if self.config.recursive else "*"

            for file_path in watch_path.glob(pattern):
                if file_path.is_file() and self.is_supported_file(str(file_path)):
                    file_info = {
                        "path": str(file_path),
                        "event_type": "scan",
                        "timestamp": datetime.utcnow(),
                    }

                    documents = await self.process_file(file_info)
                    for doc in documents:
                        yield doc

                    # Rate limiting
                    await asyncio.sleep(0.1)

        except Exception as e:
            self.log_error("Error scanning directory", error=e)

    async def process_queue(self):
        """Process files from the queue."""
        while not self._stop_monitoring:
            try:
                # Wait for files in queue
                file_info = await asyncio.wait_for(self.file_queue.get(), timeout=1.0)

                documents = await self.process_file(file_info)
                self.stats["total_files_processed"] += len(documents)

                # Update queue size stat
                self.stats["queue_size"] = self.file_queue.qsize()

            except asyncio.TimeoutError:
                # Update queue size even when no files
                self.stats["queue_size"] = self.file_queue.qsize()
                continue
            except Exception as e:
                self.log_error("Error processing file queue", error=e)
                await asyncio.sleep(1)

    async def start_monitoring(self):
        """Start file system monitoring."""
        try:
            # Initial directory scan
            self.log_info(
                "Starting initial directory scan", directory=self.config.watch_directory
            )

            scan_count = 0
            async for document in self.scan_directory():
                scan_count += 1

            self.log_info(f"Initial scan completed: {scan_count} documents processed")

            # Start file system observer
            self.observer = Observer()
            self.observer.schedule(
                self.event_handler,
                self.config.watch_directory,
                recursive=self.config.recursive,
            )
            self.observer.start()

            self.log_info(
                "File system monitoring started",
                directory=self.config.watch_directory,
                recursive=self.config.recursive,
            )

            # Start queue processing
            await self.process_queue()

        except Exception as e:
            self.log_error("Error starting file monitoring", error=e)
            raise

    async def stop_monitoring(self):
        """Stop file system monitoring."""
        self._stop_monitoring = True

        if self.observer:
            self.observer.stop()
            self.observer.join()

        self.log_info("File system monitoring stopped")

    async def force_refresh(self):
        """Force refresh all files in directory."""
        self.log_info("Force refreshing watched directory")

        # Clear file hashes to force reprocessing
        self.file_hashes.clear()

        file_count = 0
        async for document in self.scan_directory():
            file_count += 1

        self.log_info(f"Force refresh completed: {file_count} files processed")

    async def get_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics."""
        return {
            **self.stats,
            "watch_directory": self.config.watch_directory,
            "supported_extensions": list(self.supported_extensions),
            "recursive": self.config.recursive,
            "tracked_files_count": len(self.file_hashes),
        }


def create_filewatch_ingestor(
    watch_directory: str, extensions: List[str] = None, recursive: bool = True
) -> FileWatchIngestor:
    """
    Create file watch ingestor with simple configuration.

    Args:
        watch_directory: Directory to monitor
        extensions: List of file extensions to watch
        recursive: Whether to monitor subdirectories

    Returns:
        Configured file watch ingestor
    """
    extensions = extensions or [".txt", ".md", ".pdf"]

    config = FileWatchConfig(
        watch_directory=watch_directory,
        watch_extensions=extensions,
        recursive=recursive,
    )
    return FileWatchIngestor(config)
