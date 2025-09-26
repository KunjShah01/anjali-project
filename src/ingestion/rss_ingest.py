"""
RSS feed ingestion for the Real-time RAG system with comprehensive news sources.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set, AsyncGenerator
import aiohttp
import feedparser
from dataclasses import dataclass

from ..config import RSSConfig
from ..preprocessing.cleaner import DocumentProcessor, clean_rss_content
from ..utils.logger import LoggerMixin, ContextLogger


@dataclass
class RSSArticle:
    """Represents an RSS article with enhanced metadata."""

    title: str
    content: str
    url: str
    published_date: Optional[datetime]
    article_id: str
    feed_url: str
    feed_name: str
    category: str
    metadata: Dict[str, Any]


class RSSIngestor(LoggerMixin):
    """
    RSS feed ingestor with real-time monitoring capabilities for news sources.
    """

    def __init__(self, config: RSSConfig, embedding_service=None, vector_store=None):
        super().__init__()
        self.config = config
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.processor = DocumentProcessor()

        # Track processed articles
        self.seen_articles: Dict[str, Set[str]] = {}
        self.last_updated: Dict[str, datetime] = {}
        self._stop_monitoring = False

        # Feed metadata for better categorization
        self.feed_metadata = self._initialize_feed_metadata()

        # Statistics
        self.stats = {
            "total_articles_processed": 0,
            "articles_per_feed": {},
            "articles_per_category": {},
            "last_update": None,
            "errors": 0,
            "active_feeds": len(self.config.feeds),
        }

    def _initialize_feed_metadata(self) -> Dict[str, Dict[str, str]]:
        """Initialize metadata for known news feeds."""
        return {
            "rss.cnn.com": {"name": "CNN", "category": "General News"},
            "feeds.bbci.co.uk": {"name": "BBC News", "category": "General News"},
            "feeds.reuters.com": {"name": "Reuters", "category": "General News"},
            "feeds.feedburner.com/ndtvnews": {"name": "NDTV", "category": "General News"},
            "feeds.a.dj.com": {"name": "Wall Street Journal", "category": "Business"},
            "aljazeera.com": {"name": "Al Jazeera", "category": "International"},
            "feeds.guardian.co.uk": {"name": "The Guardian", "category": "International"},
            "feeds.npr.org": {"name": "NPR", "category": "General News"},
            "feeds.foxnews.com": {"name": "Fox News", "category": "General News"},
            "feeds.feedburner.com/TechCrunch": {"name": "TechCrunch", "category": "Technology"},
            "feeds.bloomberg.com": {"name": "Bloomberg", "category": "Business"},
            # Add more as needed
        }

    def _get_feed_info(self, feed_url: str) -> Dict[str, str]:
        """Get feed name and category from URL."""
        for domain, info in self.feed_metadata.items():
            if domain in feed_url:
                return info
        
        # Default fallback
        return {"name": "Unknown Feed", "category": "General"}

    async def fetch_feed(self, feed_url: str) -> Optional[feedparser.FeedParserDict]:
        """
        Fetch RSS feed asynchronously.

        Args:
            feed_url: URL of the RSS feed

        Returns:
            Parsed feed data or None if failed
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(feed_url, timeout=30) as response:
                    if response.status == 200:
                        content = await response.text()
                        return feedparser.parse(content)

        except Exception as e:
            self.log_error(f"Failed to fetch RSS feed: {feed_url}", error=e)
            self.stats["errors"] += 1
            return None

    def parse_entries(
        self, entries: List[Dict[str, Any]], feed_url: str
    ) -> List[RSSArticle]:
        """
        Parse RSS entries into RSSArticle objects.

        Args:
            entries: Raw RSS entries
            feed_url: URL of the source feed

        Returns:
            List of parsed RSS articles
        """
        articles = []

        for entry in entries:
            try:
                # Extract basic information
                title = entry.get("title", "Untitled")
                content = entry.get("description", "") or entry.get("summary", "")
                url = entry.get("link", "")
                article_id = entry.get("id") or entry.get("guid", url)

                # Parse published date
                published_date = None
                if "published_parsed" in entry and entry["published_parsed"]:
                    published_date = datetime(*entry["published_parsed"][:6])
                elif "published" in entry:
                    try:
                        published_date = datetime.fromisoformat(entry["published"])
                    except Exception:
                        pass

                # Clean content
                cleaned_content = clean_rss_content(content)
                if not cleaned_content:
                    continue

                # Get feed metadata
                feed_info = self._get_feed_info(feed_url)
                
                # Create metadata
                metadata = {
                    "source": "rss",
                    "feed_url": feed_url,
                    "feed_name": feed_info["name"],
                    "category": feed_info["category"],
                    "original_title": title,
                    "published_date": published_date.isoformat()
                    if published_date
                    else None,
                    "author": entry.get("author", ""),
                    "tags": entry.get("tags", []),
                    "ingested_at": datetime.utcnow().isoformat(),
                }

                article = RSSArticle(
                    title=title,
                    content=cleaned_content,
                    url=url,
                    published_date=published_date,
                    article_id=article_id,
                    feed_url=feed_url,
                    feed_name=feed_info["name"],
                    category=feed_info["category"],
                    metadata=metadata,
                )

                articles.append(article)

            except Exception as e:
                self.log_error(
                    "Error parsing RSS entry",
                    error=e,
                    entry_id=entry.get("id", "unknown"),
                )

        return articles

    def is_new_article(self, feed_url: str, article_id: str) -> bool:
        """
        Check if an article is new (not previously processed).

        Args:
            feed_url: URL of the RSS feed
            article_id: Unique identifier of the article

        Returns:
            True if the article is new
        """
        if feed_url not in self.seen_articles:
            self.seen_articles[feed_url] = set()

        return article_id not in self.seen_articles[feed_url]

    def mark_as_processed(self, feed_url: str, article_id: str):
        """Mark an article as processed."""
        if feed_url not in self.seen_articles:
            self.seen_articles[feed_url] = set()

        self.seen_articles[feed_url].add(article_id)

    async def ingest_new_articles(self) -> AsyncGenerator[RSSArticle, None]:
        """
        Ingest new articles from all configured feeds.

        Yields:
            New RSS articles
        """
        for feed_url in self.config.feeds:
            with ContextLogger("rss_ingestion", feed_url=feed_url):
                try:
                    # Fetch feed
                    feed_data = await self.fetch_feed(feed_url)
                    if not feed_data:
                        continue

                    # Parse entries
                    entries = feed_data.get("entries", [])
                    articles = self.parse_entries(entries, feed_url)

                    # Filter new articles
                    new_articles = []
                    for article in articles:
                        if self.is_new_article(feed_url, article.article_id):
                            new_articles.append(article)
                            self.mark_as_processed(feed_url, article.article_id)

                    # Update statistics
                    self.stats["articles_per_feed"][feed_url] = len(articles)
                    self.stats["total_articles_processed"] += len(new_articles)
                    self.last_updated[feed_url] = datetime.utcnow()

                    # Yield new articles
                    for article in new_articles:
                        yield article

                except Exception as e:
                    self.log_error("Error processing feed", error=e, feed_url=feed_url)
                    self.stats["errors"] += 1

    async def process_article(self, article: RSSArticle) -> List[Dict[str, Any]]:
        """
        Process an article into chunks and embeddings.

        Args:
            article: RSS article to process

        Returns:
            List of processed document chunks
        """
        try:
            # Process article into chunks
            full_content = f"{article.title}\n\n{article.content}"
            documents = self.processor.process_document(full_content, article.metadata)

            if self.embedding_service and self.vector_store:
                # Generate embeddings and store
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

            return documents

        except Exception as e:
            self.log_error("Error processing article", error=e, article_url=article.url)
            return []

    async def start_monitoring(self):
        """Start continuous RSS feed monitoring."""
        self.log_info(
            "Starting RSS feed monitoring",
            feeds_count=len(self.config.feeds),
            refresh_interval=self.config.refresh_interval,
        )

        while not self._stop_monitoring:
            try:
                start_time = datetime.utcnow()
                article_count = 0

                # Process new articles
                async for article in self.ingest_new_articles():
                    await self.process_article(article)
                    article_count += 1

                    # Respect rate limits
                    if article_count % 10 == 0:
                        await asyncio.sleep(1)

                # Update stats
                self.stats["last_update"] = start_time.isoformat()

                if article_count > 0:
                    self.log_info(f"Processed {article_count} new articles")

                # Wait for next refresh cycle
                await asyncio.sleep(self.config.refresh_interval)

            except Exception as e:
                self.log_error("Error in RSS monitoring loop", error=e)
                await asyncio.sleep(60)  # Wait before retrying

    async def stop_monitoring(self):
        """Stop RSS feed monitoring."""
        self._stop_monitoring = True
        self.log_info("RSS feed monitoring stopped")

    async def force_refresh(self):
        """Force refresh all feeds immediately."""
        self.log_info("Force refreshing all RSS feeds")

        article_count = 0
        async for article in self.ingest_new_articles():
            await self.process_article(article)
            article_count += 1

        self.log_info(f"Force refresh completed: {article_count} articles processed")

    async def get_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics."""
        return {
            **self.stats,
            "configured_feeds": len(self.config.feeds),
            "monitored_feeds": len(self.seen_articles),
            "refresh_interval": self.config.refresh_interval,
        }


def create_rss_ingestor(feeds: List[str], refresh_interval: int = 300) -> RSSIngestor:
    """
    Create RSS ingestor with simple configuration.

    Args:
        feeds: List of RSS feed URLs
        refresh_interval: Refresh interval in seconds

    Returns:
        Configured RSS ingestor
    """
    config = RSSConfig(feeds=feeds, refresh_interval=refresh_interval)
    return RSSIngestor(config)
