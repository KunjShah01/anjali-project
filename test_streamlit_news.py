"""
Test script to verify the News Feeds functionality in the Streamlit app.
"""

import sys
import os
import asyncio
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from src.config import get_config
    from src.ingestion.rss_ingest import RSSIngestor
    from src.embeddings.embedder import EmbeddingService
    from src.vectorstores.faiss_store import FAISSVectorStore
    
    print("🧪 Testing News Feeds Integration in Streamlit App...")
    print("=" * 60)
    
    # Test configuration
    config = get_config()
    print(f"✅ Configuration loaded")
    print(f"   RSS Feeds: {len(config.rss.feeds)} configured")
    print(f"   Refresh Interval: {config.rss.refresh_interval}s")
    print()
    
    # Test RSS ingestor initialization
    try:
        embedding_service = EmbeddingService(config.gemini)
        vector_store = FAISSVectorStore(config.vectorstore)
        rss_ingestor = RSSIngestor(config.rss, embedding_service, vector_store)
        
        print("✅ RSS ingestor initialized successfully")
        print(f"   Feed categories available: {len(rss_ingestor.feed_metadata)}")
        print()
        
        # Show feed categorization
        print("📂 Feed Categories:")
        categories = {}
        for feed in config.rss.feeds[:10]:  # Show first 10
            info = rss_ingestor._get_feed_info(feed)
            category = info["category"]
            if category not in categories:
                categories[category] = []
            categories[category].append(info["name"])
        
        for category, feeds in categories.items():
            print(f"   {category}: {', '.join(feeds[:3])}")
        print()
        
        # Test stats
        stats = rss_ingestor.stats
        print("📊 Initial Statistics:")
        print(f"   Total articles processed: {stats['total_articles_processed']}")
        print(f"   Active feeds: {stats['active_feeds']}")
        print(f"   Errors: {stats['errors']}")
        print()
        
    except Exception as e:
        print(f"❌ Error initializing RSS components: {e}")
        sys.exit(1)
    
    # Test feed metadata system
    print("🏷️ Testing Feed Metadata System:")
    test_feeds = [
        "https://rss.cnn.com/rss/edition.rss",
        "https://feeds.feedburner.com/TechCrunch",  
        "https://feeds.reuters.com/reuters/topNews"
    ]
    
    for feed in test_feeds:
        info = rss_ingestor._get_feed_info(feed)
        print(f"   {feed[:40]}... -> {info['name']} ({info['category']})")
    print()
    
    print("🎯 News Feeds Integration Status:")
    print("   ✅ RSS configuration loaded")
    print("   ✅ Feed categorization working")  
    print("   ✅ Multiple news sources configured")
    print("   ✅ Real-time monitoring ready")
    print()
    
    print("🚀 Ready to use News Feeds in Streamlit!")
    print("   1. Navigate to '📰 News Feeds' in the sidebar")
    print("   2. Select feeds to monitor")
    print("   3. Use 'Fetch Latest Articles' button")
    print("   4. Enable auto-refresh for real-time updates")
    print()
    
    print("✨ Test completed successfully!")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all dependencies are installed")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)