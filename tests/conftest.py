"""
Pytest configuration and shared fixtures for Real-time RAG testing.
"""
import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import responses
from dotenv import load_dotenv

# Load test environment variables
load_dotenv(".env.test", override=True)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    original_env = dict(os.environ)

    # Set test environment variables
    test_env = {
        "GEMINI_API_KEY": "test_gemini_key",
        "RSS_FEEDS": "https://example.com/feed1.xml,https://example.com/feed2.xml",
        "RSS_REFRESH_INTERVAL": "300",
        "VECTOR_STORE_TYPE": "faiss",
        "EMBEDDING_MODEL": "models/embedding-001",
        "CHAT_MODEL": "gemini-1.5-flash",
        "VECTOR_DIMENSION": "768",
        "LOG_LEVEL": "DEBUG",
        "STREAMLIT_PORT": "8501",
        "DEBUG": "True",
    }

    os.environ.update(test_env)
    yield test_env

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def sample_documents():
    """Provide sample documents for testing."""
    return {
        "text_doc": {
            "content": "This is a sample text document for testing purposes.",
            "filename": "sample.txt",
            "metadata": {"source": "test", "type": "text"}
        },
        "markdown_doc": {
            "content": "# Sample Markdown\n\nThis is a **sample** markdown document.",
            "filename": "sample.md",
            "metadata": {"source": "test", "type": "markdown"}
        },
        "empty_doc": {
            "content": "",
            "filename": "empty.txt",
            "metadata": {"source": "test", "type": "text"}
        }
    }


@pytest.fixture
def mock_rss_feed():
    """Mock RSS feed response."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
<channel>
<title>Test Feed</title>
<description>Test RSS Feed</description>
<link>https://example.com</link>
<item>
<title>Test Article 1</title>
<description>This is a test article description.</description>
<link>https://example.com/article1</link>
<pubDate>Mon, 27 Sep 2025 10:00:00 GMT</pubDate>
</item>
<item>
<title>Test Article 2</title>
<description>Another test article description.</description>
<link>https://example.com/article2</link>
<pubDate>Tue, 28 Sep 2025 10:00:00 GMT</pubDate>
</item>
</channel>
</rss>"""


@pytest.fixture
def mock_gemini_response():
    """Mock Gemini API response."""
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": "This is a mock response from Gemini API."
                        }
                    ]
                }
            }
        ]
    }


@pytest.fixture
def mock_drive_files():
    """Mock Google Drive files response."""
    return [
        {
            "id": "file1",
            "name": "document1.pdf",
            "mimeType": "application/pdf",
            "modifiedTime": "2025-09-27T10:00:00.000Z"
        },
        {
            "id": "file2",
            "name": "document2.txt",
            "mimeType": "text/plain",
            "modifiedTime": "2025-09-27T11:00:00.000Z"
        }
    ]


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    mock_store = MagicMock()
    mock_store.add_documents.return_value = ["doc1", "doc2"]
    mock_store.search.return_value = [
        {"content": "Test result 1", "score": 0.9},
        {"content": "Test result 2", "score": 0.8}
    ]
    mock_store.delete.return_value = True
    return mock_store


@pytest.fixture
def mock_embedder():
    """Mock embedder for testing."""
    mock_emb = MagicMock()
    mock_emb.embed_text.return_value = [0.1, 0.2, 0.3] * 256  # 768-dim vector
    mock_emb.embed_batch.return_value = [[0.1, 0.2, 0.3] * 256] * 5
    return mock_emb


@pytest.fixture(autouse=True)
def mock_external_apis():
    """Mock external API calls."""
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        # Mock Gemini API
        rsps.add(
            responses.POST,
            "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent",
            json={"candidates": [{"content": {"parts": [{"text": "Mock response"}]}}]},
            status=200
        )

        # Mock RSS feeds
        rsps.add(
            responses.GET,
            "https://example.com/feed1.xml",
            body="<?xml version='1.0'?><rss><channel><item><title>Test</title></item></channel></rss>",
            status=200
        )

        yield rsps


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "slow: Slow running tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on path."""
    for item in items:
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)