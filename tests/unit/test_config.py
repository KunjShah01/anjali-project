"""
Unit tests for configuration management.
"""
import os
import pytest
from unittest.mock import patch

from src.config import (
    Config,
    GeminiConfig,
    RSSConfig,
    DriveConfig,
    FileWatchConfig,
    VectorStoreConfig,
    LoggingConfig,
    ServerConfig,
    get_config,
    reload_config,
)


class TestGeminiConfig:
    """Test Gemini configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        # Clear environment to test defaults
        with patch.dict(os.environ, {}, clear=True):
            config = GeminiConfig()
            assert config.api_key == ""
            assert config.embedding_model == "models/embedding-001"
            assert config.chat_model == "gemini-2.0-flash-exp"
            assert config.max_tokens == 8192
            assert config.temperature == 0.1
            assert config.image_model == "imagen-3.0-generate-001"

    def test_environment_variables(self, mock_env_vars):
        """Test configuration from environment variables."""
        with patch.dict(os.environ, {
            "GEMINI_API_KEY": "test_key",
            "EMBEDDING_MODEL": "custom-embedding",
            "CHAT_MODEL": "custom-chat",
            "GEMINI_MAX_TOKENS": "4096",
            "GEMINI_TEMPERATURE": "0.5",
            "IMAGE_MODEL": "custom-image",
        }):
            config = GeminiConfig()
            assert config.api_key == "test_key"
            assert config.embedding_model == "custom-embedding"
            assert config.chat_model == "custom-chat"
            assert config.max_tokens == 4096
            assert config.temperature == 0.5
            assert config.image_model == "custom-image"


class TestRSSConfig:
    """Test RSS configuration."""

    def test_default_values(self):
        """Test default RSS configuration."""
        # Clear environment to test defaults
        with patch.dict(os.environ, {}, clear=True):
            config = RSSConfig()
            assert config.feeds == []
            assert config.refresh_interval == 300
            assert config.max_articles_per_feed == 50

    def test_environment_variables(self, mock_env_vars):
        """Test RSS config from environment."""
        with patch.dict(os.environ, {
            "RSS_FEEDS": "feed1.xml,feed2.xml",
            "RSS_REFRESH_INTERVAL": "600",
            "RSS_MAX_ARTICLES": "100",
        }):
            config = RSSConfig()
            assert config.feeds == ["feed1.xml", "feed2.xml"]
            assert config.refresh_interval == 600
            assert config.max_articles_per_feed == 100

    def test_empty_feeds(self):
        """Test handling of empty RSS_FEEDS."""
        with patch.dict(os.environ, {"RSS_FEEDS": ""}):
            config = RSSConfig()
            assert config.feeds == []


class TestDriveConfig:
    """Test Google Drive configuration."""

    def test_default_values(self):
        """Test default Drive configuration."""
        config = DriveConfig()
        assert config.folder_id is None
        assert config.credentials_path is None
        assert config.scopes == ["https://www.googleapis.com/auth/drive.readonly"]

    def test_environment_variables(self):
        """Test Drive config from environment."""
        with patch.dict(os.environ, {
            "GOOGLE_DRIVE_FOLDER_ID": "test_folder_id",
            "GOOGLE_CREDENTIALS_PATH": "/path/to/creds.json",
        }):
            config = DriveConfig()
            assert config.folder_id == "test_folder_id"
            assert config.credentials_path == "/path/to/creds.json"


class TestFileWatchConfig:
    """Test file watching configuration."""

    def test_default_values(self):
        """Test default file watch configuration."""
        config = FileWatchConfig()
        assert config.watch_directory == "./data"
        assert config.watch_extensions == [".txt", ".md", ".pdf"]
        assert config.recursive is True

    def test_environment_variables(self):
        """Test file watch config from environment."""
        with patch.dict(os.environ, {
            "WATCH_DIRECTORY": "/custom/path",
            "WATCH_EXTENSIONS": ".doc,.docx",
            "WATCH_RECURSIVE": "false",
        }):
            config = FileWatchConfig()
            assert config.watch_directory == "/custom/path"
            assert config.watch_extensions == [".doc", ".docx"]
            assert config.recursive is False


class TestVectorStoreConfig:
    """Test vector store configuration."""

    def test_default_values(self):
        """Test default vector store configuration."""
        config = VectorStoreConfig()
        assert config.store_type == "faiss"
        assert config.dimension == 768
        assert config.index_path == "./data/faiss_index"
        assert config.pathway_host == "localhost"
        assert config.pathway_port == 8754

    def test_environment_variables(self):
        """Test vector store config from environment."""
        with patch.dict(os.environ, {
            "VECTOR_STORE_TYPE": "pathway",
            "VECTOR_DIMENSION": "512",
            "FAISS_INDEX_PATH": "/custom/index",
            "PATHWAY_HOST": "remote-host",
            "PATHWAY_PORT": "9000",
        }):
            config = VectorStoreConfig()
            assert config.store_type == "pathway"
            assert config.dimension == 512
            assert config.index_path == "/custom/index"
            assert config.pathway_host == "remote-host"
            assert config.pathway_port == 9000


class TestLoggingConfig:
    """Test logging configuration."""

    def test_default_values(self):
        """Test default logging configuration."""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert config.format == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        assert config.file_path is None

    def test_environment_variables(self):
        """Test logging config from environment."""
        with patch.dict(os.environ, {
            "LOG_LEVEL": "DEBUG",
            "LOG_FORMAT": "custom format",
            "LOG_FILE_PATH": "/var/log/app.log",
        }):
            config = LoggingConfig()
            assert config.level == "DEBUG"
            assert config.format == "custom format"
            assert config.file_path == "/var/log/app.log"


class TestServerConfig:
    """Test server configuration."""

    def test_default_values(self):
        """Test default server configuration."""
        config = ServerConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.debug is False
        assert config.reload is False

    def test_environment_variables(self):
        """Test server config from environment."""
        with patch.dict(os.environ, {
            "HOST": "127.0.0.1",
            "PORT": "3000",
            "DEBUG": "true",
            "RELOAD": "true",
        }):
            config = ServerConfig()
            assert config.host == "127.0.0.1"
            assert config.port == 3000
            assert config.debug is True
            assert config.reload is True


class TestConfig:
    """Test main configuration class."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = Config()
        assert isinstance(config.gemini, GeminiConfig)
        assert isinstance(config.rss, RSSConfig)
        assert isinstance(config.drive, DriveConfig)
        assert isinstance(config.filewatch, FileWatchConfig)
        assert isinstance(config.vectorstore, VectorStoreConfig)
        assert isinstance(config.logging, LoggingConfig)
        assert isinstance(config.server, ServerConfig)

    def test_config_validation_success(self, mock_env_vars):
        """Test successful configuration validation."""
        with patch.dict(os.environ, {
            "GEMINI_API_KEY": "test_key",
            "VECTOR_STORE_TYPE": "faiss",
            "RSS_FEEDS": "test.xml",
        }):
            config = Config()
            assert config.validate() is True

    def test_config_validation_missing_api_key(self):
        """Test validation failure for missing API key."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            with pytest.raises(ValueError, match="Gemini API key is required"):
                config.validate()

    def test_config_validation_invalid_store_type(self, mock_env_vars):
        """Test validation failure for invalid vector store type."""
        with patch.dict(os.environ, {
            "GEMINI_API_KEY": "test_key",
            "VECTOR_STORE_TYPE": "invalid",
            "RSS_FEEDS": "test.xml",
        }):
            config = Config()
            with pytest.raises(ValueError, match="Vector store type must be"):
                config.validate()

    def test_config_validation_no_data_sources(self, mock_env_vars):
        """Test validation failure when no data sources are configured."""
        with patch.dict(os.environ, {}, clear=True):
            with patch.dict(os.environ, {
                "GEMINI_API_KEY": "test_key",
                "VECTOR_STORE_TYPE": "faiss",
                "WATCH_DIRECTORY": "/nonexistent/path",  # Ensure no data sources are available
            }):
                config = Config()
                with pytest.raises(ValueError, match="At least one data source"):
                    config.validate()


class TestGlobalConfig:
    """Test global configuration functions."""

    def test_get_config(self):
        """Test getting global configuration."""
        config = get_config()
        assert isinstance(config, Config)

    def test_reload_config(self):
        """Test configuration reloading."""
        with patch("src.config.load_dotenv") as mock_load:
            original_config = get_config()
            reloaded_config = reload_config()

            # Should be different instances after reload
            assert reloaded_config is not original_config
            mock_load.assert_called_with(override=True)