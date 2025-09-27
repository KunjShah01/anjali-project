"""
Tests for RSS ingestion functionality.
"""

import pytest
from src.ingestion.rss_ingest import RSSIngestor
from src.config import RSSConfig


@pytest.fixture
def sample_rss_response():
    """Mock RSS feed response."""
    return {
        "feed": {
            "title": "Test Feed",
            "description": "A test RSS feed", 
            "link": "https://example.com",
        },
        "entries": [
            {
                "title": "Test Article 1",
                "description": "First test article content",
                "link": "https://example.com/article1", 
                "published": "Mon, 01 Jan 2024 12:00:00 +0000",
                "id": "article1",
            },
            {
                "title": "Test Article 2", 
                "description": "Second test article content",
                "link": "https://example.com/article2",
                "published": "Mon, 01 Jan 2024 13:00:00 +0000", 
                "id": "article2",
            },
        ],
    }


@pytest.fixture
def mock_rss_config():
    """Create mock RSS config."""
    return RSSConfig(
        feeds=["https://example.com/feed.rss"],
        refresh_interval=300
    )


@pytest.fixture
def rss_ingestor(mock_rss_config):
    """Create RSS ingestor instance."""
    return RSSIngestor(config=mock_rss_config)


class TestRSSIngestor:
    """Test cases for RSS ingestion."""

    def test_init(self, mock_rss_config):
        """Test RSS ingestor initialization."""
        ingestor = RSSIngestor(config=mock_rss_config)

        assert ingestor.config == mock_rss_config
        assert ingestor.processor is not None
        assert isinstance(ingestor.seen_articles, dict)
        assert isinstance(ingestor.last_updated, dict)

    def test_parse_entries(self, rss_ingestor, sample_rss_response):
        """Test parsing RSS entries."""
        entries = rss_ingestor.parse_entries(
            sample_rss_response["entries"], 
            "https://example.com/feed.rss"
        )

        assert len(entries) == 2
        assert entries[0].title == "Test Article 1"
        assert entries[0].content == "First test article content."
        assert entries[0].url == "https://example.com/article1"
        assert entries[0].article_id == "article1"
