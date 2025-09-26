import feedparser
from typing import List,Dict
from ..preprocessing.cleaner import document_from_rss_items
from ..utils.logger import get_logger
from ..config import RSS_FEEDS
import time

logger=get_logger("rss-ingest")

def fetch_feed(url: str) -> List[Dict]:
    logger.info(f"Fetching RSS feed: {url}")
    feed = feedparser.parse(url)
    items = feed.get("entries", [])
    docs = [document_from_rss_items(item) for item in items]
    logger.info(f"Fetched {len(docs)} items from {url}")
    return docs

def fetch_all_feeds() -> List[Dict]:
    docs = []
    for url in RSS_FEEDS:
        try:
            docs.extend(fetch_feed(url))
            time.sleep(0.2)
        except Exception as e:
            logger.error(f"Error fetching feed {url}: {e}")
    return docs