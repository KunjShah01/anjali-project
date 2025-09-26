"""
Tests for RSS ingestion functionality.
"""

import pytest
from unittest.mock import Mock, patch
import feedparser
from src.ingestion.rss_ingest import RSSIngestor


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
def rss_ingestor():
    """Create RSS ingestor instance."""
    feeds = ["https://example.com/feed.rss"]
    return RSSIngestor(feeds=feeds, refresh_interval=300)


class TestRSSIngestor:
    """Test cases for RSS ingestion."""

    def test_init(self):
        """Test RSS ingestor initialization."""
        feeds = ["https://example.com/feed1.rss", "https://example.com/feed2.rss"]
        ingestor = RSSIngestor(feeds=feeds, refresh_interval=600)

        assert ingestor.feeds == feeds
        assert ingestor.refresh_interval == 600
        assert ingestor.last_updated == {}

    @patch("feedparser.parse")
    def test_fetch_feed_success(self, mock_parse, rss_ingestor, sample_rss_response):
        """Test successful RSS feed fetching."""
        mock_parse.return_value = sample_rss_response

        result = rss_ingestor.fetch_feed("https://example.com/feed.rss")

        mock_parse.assert_called_once_with("https://example.com/feed.rss")
        assert result == sample_rss_response

    @patch("feedparser.parse")
    def test_fetch_feed_failure(self, mock_parse, rss_ingestor):
        """Test RSS feed fetching failure."""
        mock_parse.side_effect = Exception("Network error")

        result = rss_ingestor.fetch_feed("https://example.com/feed.rss")

        assert result is None

    def test_parse_entries(self, rss_ingestor, sample_rss_response):
        """Test parsing RSS entries."""
        entries = rss_ingestor.parse_entries(sample_rss_response["entries"])

        assert len(entries) == 2
        assert entries[0]["title"] == "Test Article 1"
        assert entries[0]["content"] == "First test article content"
        assert entries[0]["url"] == "https://example.com/article1"
        assert "published_date" in entries[0]
        assert "article_id" in entries[0]

    def test_is_new_entry(self, rss_ingestor):
        """Test checking if entry is new."""
        feed_url = "https://example.com/feed.rss"
        article_id = "article1"

        # First time should be new
        assert rss_ingestor.is_new_entry(feed_url, article_id) is True

        # Mark as seen
        rss_ingestor.mark_as_seen(feed_url, article_id)

        # Second time should not be new
        assert rss_ingestor.is_new_entry(feed_url, article_id) is False

    @patch("src.ingestion.rss_ingest.RSSIngestor.fetch_feed")
    def test_ingest_new_articles(self, mock_fetch, rss_ingestor, sample_rss_response):
        """Test ingesting new articles."""
        mock_fetch.return_value = sample_rss_response

        articles = list(rss_ingestor.ingest_new_articles())

        assert len(articles) == 2
        assert all("title" in article for article in articles)
        assert all("content" in article for article in articles)

    @patch("src.ingestion.rss_ingest.RSSIngestor.fetch_feed")
    def test_ingest_no_duplicates(self, mock_fetch, rss_ingestor, sample_rss_response):
        """Test that duplicate articles are not ingested."""
        mock_fetch.return_value = sample_rss_response

        # First ingestion
        articles1 = list(rss_ingestor.ingest_new_articles())
        assert len(articles1) == 2

        # Second ingestion should return no new articles
        articles2 = list(rss_ingestor.ingest_new_articles())
        assert len(articles2) == 0

    @patch("time.sleep")
    @patch("src.ingestion.rss_ingest.RSSIngestor.ingest_new_articles")
    def test_start_monitoring(self, mock_ingest, mock_sleep, rss_ingestor):
        """Test starting RSS monitoring."""
        mock_ingest.return_value = iter([{"title": "Test Article"}])

        # Mock to stop after first iteration
        def side_effect(*args):
            rss_ingestor._stop_monitoring = True

        mock_sleep.side_effect = side_effect

        articles = []
        for article in rss_ingestor.start_monitoring():
            articles.append(article)
            break

        assert len(articles) == 1
        mock_sleep.assert_called_once_with(300)  # refresh_interval

    def test_stop_monitoring(self, rss_ingestor):
        """Test stopping RSS monitoring."""
        rss_ingestor.stop_monitoring()
        assert rss_ingestor._stop_monitoring is True


@pytest.mark.asyncio
class TestAsyncRSSIngestor:
    """Test cases for async RSS operations."""

    @patch("aiohttp.ClientSession.get")
    async def test_async_fetch_feed(self, mock_get, rss_ingestor):
        """Test async RSS feed fetching."""
        mock_response = Mock()
        mock_response.text.return_value = """<?xml version="1.0"?>
        <rss version="2.0">
            <channel>
                <title>Test Feed</title>
                <item>
                    <title>Test Article</title>
                    <description>Test content</description>
                    <link>https://example.com/article</link>
                </item>
            </channel>
        </rss>"""
        mock_get.return_value.__aenter__.return_value = mock_response

        # This test would require implementing async methods
        # For now, it's a placeholder to show the test structure
        pass


if __name__ == "__main__":
    pytest.main([__file__])
