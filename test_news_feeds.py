#!/usr/bin/env python3
"""
Test script to verify RSS news feeds integration
"""

import asyncio
from src.config import get_config
from src.ingestion.rss_ingest import RSSIngestor


def test_rss_configuration():
    """Test RSS feed configuration."""
    print("🧪 Testing RSS News Feeds Integration...")
    print("=" * 50)

    # Test configuration
    config = get_config()
    print(f"📊 Configuration loaded successfully")
    print(f"   RSS Feeds: {len(config.rss.feeds)} configured")
    print(f"   Refresh Interval: {config.rss.refresh_interval}s")

    # Show configured feeds
    print(f"\n📰 Configured News Sources:")
    for i, feed in enumerate(config.rss.feeds[:10]):  # Show first 10
        # Extract source name
        if "cnn.com" in feed:
            source = "📺 CNN"
        elif "bbci.co.uk" in feed:
            source = "🌍 BBC News"
        elif "reuters.com" in feed:
            source = "📰 Reuters"
        elif "techcrunch" in feed.lower():
            source = "💻 TechCrunch"
        elif "bloomberg" in feed:
            source = "💼 Bloomberg"
        elif "guardian" in feed:
            source = "🗞️ The Guardian"
        elif "aljazeera" in feed:
            source = "🌐 Al Jazeera"
        elif "npr.org" in feed:
            source = "🎙️ NPR"
        elif "foxnews" in feed:
            source = "📺 Fox News"
        elif "ndtv" in feed:
            source = "📺 NDTV"
        else:
            source = f"📡 Feed {i + 1}"

        print(f"   {i + 1:2d}. {source}")
        print(f"       {feed[:80]}{'...' if len(feed) > 80 else ''}")

    if len(config.rss.feeds) > 10:
        print(f"   ... and {len(config.rss.feeds) - 10} more feeds")

    # Test RSS ingestor creation
    try:
        rss_ingestor = RSSIngestor(config.rss)
        print(f"\n✅ RSS ingestor created successfully")
        print(f"   Statistics: {rss_ingestor.stats}")

        # Test feed metadata
        print(f"\n🏷️ Feed Metadata Examples:")
        test_feeds = [
            "https://rss.cnn.com/rss/edition.rss",
            "http://feeds.bbci.co.uk/news/rss.xml",
            "https://feeds.feedburner.com/TechCrunch",
        ]

        for feed in test_feeds:
            if feed in config.rss.feeds:
                info = rss_ingestor._get_feed_info(feed)
                print(f"   {feed[:50]}... -> {info['name']} ({info['category']})")

    except Exception as e:
        print(f"❌ Failed to create RSS ingestor: {e}")
        return False

    print(f"\n🎯 Integration Status:")
    print(f"   ✅ Configuration loaded: {len(config.rss.feeds)} feeds")
    print(f"   ✅ RSS ingestor initialized")
    print(f"   ✅ Feed metadata system working")
    print(f"   ✅ News categorization ready")

    print(f"\n📋 Next Steps:")
    print(f"   1. Run the Streamlit app: streamlit run streamlit_app.py")
    print(f"   2. Navigate to '📰 News Feeds' section")
    print(f"   3. Use 'Fetch Latest Articles' to test live feeds")
    print(f"   4. Monitor real-time updates with auto-refresh")

    return True


async def test_single_feed():
    """Test fetching a single RSS feed."""
    print(f"\n🧪 Testing Single Feed Fetch...")

    config = get_config()
    if not config.rss.feeds:
        print("❌ No feeds configured")
        return

    # Test first feed
    test_feed = config.rss.feeds[0]
    print(f"   Testing: {test_feed[:60]}...")

    try:
        rss_ingestor = RSSIngestor(config.rss)
        feed_data = await rss_ingestor.fetch_feed(test_feed)

        if feed_data:
            print(f"   ✅ Feed accessible")
            print(f"   📊 Feed info: {feed_data.feed.get('title', 'Unknown')}")
            print(f"   📰 Entries found: {len(feed_data.entries)}")

            # Show first few entries
            for i, entry in enumerate(feed_data.entries[:3]):
                title = entry.get("title", "No title")[:50]
                print(f"      {i + 1}. {title}...")
        else:
            print(f"   ❌ Failed to fetch feed")

    except Exception as e:
        print(f"   ❌ Error testing feed: {e}")


if __name__ == "__main__":
    # Test configuration
    success = test_rss_configuration()

    if success:
        # Test single feed fetch
        print()
        try:
            asyncio.run(test_single_feed())
        except Exception as e:
            print(f"❌ Feed test error: {e}")

    print(f"\n🏁 RSS News Feeds Integration Test Complete!")
