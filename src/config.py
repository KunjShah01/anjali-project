"""
Configuration management for the Real-time RAG system.
"""

import os
from typing import List, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class GeminiConfig:
    """Google Gemini API configuration."""

    api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "models/embedding-001")
    )
    chat_model: str = field(
        default_factory=lambda: os.getenv("CHAT_MODEL", "gemini-2.0-flash-exp")
    )
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("GEMINI_MAX_TOKENS", "8192"))
    )
    temperature: float = field(
        default_factory=lambda: float(os.getenv("GEMINI_TEMPERATURE", "0.1"))
    )
    # Image generation model
    image_model: str = field(
        default_factory=lambda: os.getenv("IMAGE_MODEL", "imagen-3.0-generate-001")
    )


@dataclass
class RSSConfig:
    """RSS feed configuration."""

    feeds: List[str] = field(
        default_factory=lambda: os.getenv("RSS_FEEDS", "").split(",")
        if os.getenv("RSS_FEEDS")
        else []
    )
    refresh_interval: int = field(
        default_factory=lambda: int(os.getenv("RSS_REFRESH_INTERVAL", "300"))
    )
    max_articles_per_feed: int = field(
        default_factory=lambda: int(os.getenv("RSS_MAX_ARTICLES", "50"))
    )


@dataclass
class DriveConfig:
    """Google Drive configuration."""

    folder_id: Optional[str] = field(
        default_factory=lambda: os.getenv("GOOGLE_DRIVE_FOLDER_ID")
    )
    credentials_path: Optional[str] = field(
        default_factory=lambda: os.getenv("GOOGLE_CREDENTIALS_PATH")
    )
    scopes: List[str] = field(
        default_factory=lambda: ["https://www.googleapis.com/auth/drive.readonly"]
    )


@dataclass
class FileWatchConfig:
    """File watching configuration."""

    watch_directory: str = field(
        default_factory=lambda: os.getenv("WATCH_DIRECTORY", "./data")
    )
    watch_extensions: List[str] = field(
        default_factory=lambda: os.getenv("WATCH_EXTENSIONS", ".txt,.md,.pdf").split(
            ","
        )
    )
    recursive: bool = field(
        default_factory=lambda: os.getenv("WATCH_RECURSIVE", "True").lower() == "true"
    )


@dataclass
class VectorStoreConfig:
    """Vector store configuration."""

    store_type: str = field(
        default_factory=lambda: os.getenv("VECTOR_STORE_TYPE", "faiss")
    )
    dimension: int = field(
        default_factory=lambda: int(os.getenv("VECTOR_DIMENSION", "768"))
    )
    index_path: str = field(
        default_factory=lambda: os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")
    )

    # Pathway-specific settings
    pathway_host: str = field(
        default_factory=lambda: os.getenv("PATHWAY_HOST", "localhost")
    )
    pathway_port: int = field(
        default_factory=lambda: int(os.getenv("PATHWAY_PORT", "8754"))
    )


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    format: str = field(
        default_factory=lambda: os.getenv(
            "LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    )
    file_path: Optional[str] = field(default_factory=lambda: os.getenv("LOG_FILE_PATH"))


@dataclass
class ServerConfig:
    """Server configuration."""

    host: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "8000")))
    debug: bool = field(
        default_factory=lambda: os.getenv("DEBUG", "False").lower() == "true"
    )
    reload: bool = field(
        default_factory=lambda: os.getenv("RELOAD", "False").lower() == "true"
    )


@dataclass
class Config:
    """Main configuration class."""

    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    rss: RSSConfig = field(default_factory=RSSConfig)
    drive: DriveConfig = field(default_factory=DriveConfig)
    filewatch: FileWatchConfig = field(default_factory=FileWatchConfig)
    vectorstore: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    def validate(self) -> bool:
        """Validate configuration settings."""
        errors = []

        if not self.gemini.api_key:
            errors.append("Gemini API key is required")

        if self.vectorstore.store_type not in ["faiss", "pathway"]:
            errors.append("Vector store type must be 'faiss' or 'pathway'")

        if (
            not self.rss.feeds
            and not self.drive.folder_id
            and not os.path.exists(self.filewatch.watch_directory)
        ):
            errors.append(
                "At least one data source must be configured (RSS feeds, Drive folder, or watch directory)"
            )

        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")

        return True


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def reload_config() -> Config:
    """Reload configuration from environment variables."""
    global config
    load_dotenv(override=True)
    config = Config()
    return config
