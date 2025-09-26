"""
Google Drive ingestion for the Real-time RAG system.
"""

import asyncio
import io
import mimetypes
from datetime import datetime
from typing import List, Dict, Any, Optional, AsyncGenerator
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import PyPDF2

from ..config import DriveConfig
from ..preprocessing.cleaner import DocumentProcessor
from ..utils.logger import LoggerMixin, ContextLogger


class DriveIngestor(LoggerMixin):
    """
    Google Drive file ingestor with real-time monitoring.
    """

    def __init__(self, config: DriveConfig, embedding_service=None, vector_store=None):
        super().__init__()
        self.config = config
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.processor = DocumentProcessor()

        # Google Drive service
        self.service = None
        self.credentials = None

        # Track processed files
        self.processed_files: Dict[str, datetime] = {}
        self._stop_monitoring = False

        # Supported file types
        self.supported_mime_types = {
            "text/plain": self._process_text_file,
            "application/pdf": self._process_pdf_file,
            "application/vnd.google-apps.document": self._process_google_doc,
            "text/html": self._process_html_file,
            "text/markdown": self._process_text_file,
        }

        # Statistics
        self.stats = {
            "total_files_processed": 0,
            "files_by_type": {},
            "last_update": None,
            "errors": 0,
        }

    async def initialize(self):
        """Initialize Google Drive API connection."""
        try:
            self.credentials = self._get_credentials()
            self.service = build("drive", "v3", credentials=self.credentials)
            self.log_info("Google Drive service initialized successfully")

        except Exception as e:
            self.log_error("Failed to initialize Google Drive service", error=e)
            raise

    def _get_credentials(self) -> Credentials:
        """Get Google Drive API credentials."""
        creds = None

        # Token file for storing access and refresh tokens
        token_path = Path(self.config.credentials_path).parent / "token.json"

        if token_path.exists():
            creds = Credentials.from_authorized_user_file(
                str(token_path), self.config.scopes
            )

        # If there are no valid credentials, let the user log in
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.config.credentials_path, self.config.scopes
                )
                creds = flow.run_local_server(port=0)

            # Save credentials for next run
            with open(token_path, "w") as token:
                token.write(creds.to_json())

        return creds

    async def list_files(self, folder_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List files in Google Drive folder.

        Args:
            folder_id: Google Drive folder ID, uses config default if None

        Returns:
            List of file metadata
        """
        try:
            folder_id = folder_id or self.config.folder_id

            # Query for files in the folder
            query = f"'{folder_id}' in parents and trashed=false"

            results = (
                self.service.files()
                .list(
                    q=query,
                    fields="nextPageToken, files(id, name, mimeType, modifiedTime, size, webViewLink)",
                    pageSize=100,
                )
                .execute()
            )

            files = results.get("files", [])

            # Filter supported file types
            supported_files = []
            for file in files:
                if file["mimeType"] in self.supported_mime_types:
                    supported_files.append(file)

            self.log_info(
                f"Found {len(supported_files)} supported files out of {len(files)} total"
            )
            return supported_files

        except Exception as e:
            self.log_error("Error listing Drive files", error=e)
            return []

    async def download_file(self, file_id: str) -> Optional[bytes]:
        """
        Download file content from Google Drive.

        Args:
            file_id: Google Drive file ID

        Returns:
            File content as bytes or None if failed
        """
        try:
            request = self.service.files().get_media(fileId=file_id)
            file_content = io.BytesIO()

            downloader = MediaIoBaseDownload(file_content, request)
            done = False

            while not done:
                status, done = downloader.next_chunk()

            return file_content.getvalue()

        except Exception as e:
            self.log_error("Error downloading file", error=e, file_id=file_id)
            return None

    async def _process_text_file(
        self, file_content: bytes, metadata: Dict[str, Any]
    ) -> str:
        """Process plain text file."""
        try:
            return file_content.decode("utf-8", errors="ignore")
        except Exception as e:
            self.log_error("Error processing text file", error=e)
            return ""

    async def _process_pdf_file(
        self, file_content: bytes, metadata: Dict[str, Any]
    ) -> str:
        """Process PDF file."""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            text = ""

            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"

            return text

        except Exception as e:
            self.log_error("Error processing PDF file", error=e)
            return ""

    async def _process_google_doc(self, file_id: str, metadata: Dict[str, Any]) -> str:
        """Process Google Document."""
        try:
            # Export Google Doc as plain text
            request = self.service.files().export_media(
                fileId=file_id, mimeType="text/plain"
            )

            file_content = io.BytesIO()
            downloader = MediaIoBaseDownload(file_content, request)
            done = False

            while not done:
                status, done = downloader.next_chunk()

            return file_content.getvalue().decode("utf-8", errors="ignore")

        except Exception as e:
            self.log_error("Error processing Google Doc", error=e)
            return ""

    async def _process_html_file(
        self, file_content: bytes, metadata: Dict[str, Any]
    ) -> str:
        """Process HTML file."""
        try:
            html_content = file_content.decode("utf-8", errors="ignore")
            # Use the document processor's HTML cleaning
            return self.processor.cleaner.clean_text(html_content)

        except Exception as e:
            self.log_error("Error processing HTML file", error=e)
            return ""

    async def process_file(self, file_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a single Drive file.

        Args:
            file_info: File metadata from Drive API

        Returns:
            List of processed document chunks
        """
        try:
            file_id = file_info["id"]
            file_name = file_info["name"]
            mime_type = file_info["mimeType"]

            with ContextLogger(
                "drive_file_processing", file_name=file_name, file_id=file_id
            ):
                # Get the appropriate processor
                processor_func = self.supported_mime_types.get(mime_type)
                if not processor_func:
                    self.log_warning(f"Unsupported file type: {mime_type}")
                    return []

                # Process based on file type
                if mime_type == "application/vnd.google-apps.document":
                    content = await processor_func(file_id, file_info)
                else:
                    file_content = await self.download_file(file_id)
                    if not file_content:
                        return []
                    content = await processor_func(file_content, file_info)

                if not content:
                    self.log_warning(f"No content extracted from file: {file_name}")
                    return []

                # Create metadata
                metadata = {
                    "source": "google_drive",
                    "file_id": file_id,
                    "file_name": file_name,
                    "mime_type": mime_type,
                    "file_url": file_info.get("webViewLink", ""),
                    "modified_time": file_info.get("modifiedTime"),
                    "file_size": file_info.get("size"),
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

                # Update statistics
                self.stats["files_by_type"][mime_type] = (
                    self.stats["files_by_type"].get(mime_type, 0) + 1
                )

                return documents

        except Exception as e:
            self.log_error("Error processing Drive file", error=e, file_info=file_info)
            self.stats["errors"] += 1
            return []

    async def ingest_new_files(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Ingest new or modified files from Google Drive.

        Yields:
            Processed document chunks
        """
        try:
            files = await self.list_files()

            for file_info in files:
                file_id = file_info["id"]
                modified_time = datetime.fromisoformat(
                    file_info["modifiedTime"].replace("Z", "+00:00")
                )

                # Check if file is new or modified
                if (
                    file_id not in self.processed_files
                    or self.processed_files[file_id] < modified_time
                ):
                    documents = await self.process_file(file_info)

                    for doc in documents:
                        yield doc

                    # Mark as processed
                    self.processed_files[file_id] = modified_time
                    self.stats["total_files_processed"] += 1

        except Exception as e:
            self.log_error("Error ingesting Drive files", error=e)

    async def start_monitoring(self):
        """Start continuous Google Drive monitoring."""
        if not self.service:
            await self.initialize()

        self.log_info(
            "Starting Google Drive monitoring", folder_id=self.config.folder_id
        )

        while not self._stop_monitoring:
            try:
                start_time = datetime.utcnow()
                file_count = 0

                # Process new/modified files
                async for document in self.ingest_new_files():
                    file_count += 1

                    # Rate limiting
                    if file_count % 5 == 0:
                        await asyncio.sleep(2)

                # Update stats
                self.stats["last_update"] = start_time.isoformat()

                if file_count > 0:
                    self.log_info(f"Processed {file_count} Drive files")

                # Wait before next check (longer interval for Drive)
                await asyncio.sleep(300)  # 5 minutes

            except Exception as e:
                self.log_error("Error in Drive monitoring loop", error=e)
                await asyncio.sleep(120)  # Wait before retrying

    async def stop_monitoring(self):
        """Stop Google Drive monitoring."""
        self._stop_monitoring = True
        self.log_info("Google Drive monitoring stopped")

    async def force_refresh(self):
        """Force refresh all Drive files immediately."""
        self.log_info("Force refreshing Google Drive files")

        # Clear processed files to force reprocessing
        self.processed_files.clear()

        file_count = 0
        async for document in self.ingest_new_files():
            file_count += 1

        self.log_info(f"Force refresh completed: {file_count} files processed")

    async def get_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics."""
        return {
            **self.stats,
            "folder_id": self.config.folder_id,
            "supported_types": list(self.supported_mime_types.keys()),
            "processed_files_count": len(self.processed_files),
        }


def create_drive_ingestor(folder_id: str, credentials_path: str) -> DriveIngestor:
    """
    Create Drive ingestor with simple configuration.

    Args:
        folder_id: Google Drive folder ID
        credentials_path: Path to credentials JSON file

    Returns:
        Configured Drive ingestor
    """
    config = DriveConfig(folder_id=folder_id, credentials_path=credentials_path)
    return DriveIngestor(config)
