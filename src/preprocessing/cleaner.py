"""
Text cleaning and preprocessing utilities for the Real-time RAG system.
"""

import re
import html
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse, urljoin
from dataclasses import dataclass

from bs4 import BeautifulSoup
import chardet

from ..utils.logger import LoggerMixin


@dataclass
class CleaningConfig:
    """Configuration for text cleaning operations."""

    remove_html: bool = True
    remove_urls: bool = False
    remove_emails: bool = False
    remove_phone_numbers: bool = False
    normalize_whitespace: bool = True
    remove_special_chars: bool = False
    min_length: int = 10
    max_length: int = 10000
    preserve_newlines: bool = False
    remove_duplicate_sentences: bool = True


class TextCleaner(LoggerMixin):
    """
    Advanced text cleaning and preprocessing utilities.
    """

    def __init__(self, config: Optional[CleaningConfig] = None):
        super().__init__()
        self.config = config or CleaningConfig()

        # Compiled regex patterns for performance
        self._url_pattern = re.compile(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )
        self._email_pattern = re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        )
        self._phone_pattern = re.compile(
            r"(\+?1[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}"
        )
        self._whitespace_pattern = re.compile(r"\s+")
        self._special_char_pattern = re.compile(r"[^\w\s.-]")

    def clean_text(self, text: str) -> str:
        """
        Apply comprehensive text cleaning.

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""

        try:
            # Detect and handle encoding issues
            text = self._fix_encoding(text)

            # Remove HTML if present
            if self.config.remove_html:
                text = self._remove_html(text)

            # Unescape HTML entities
            text = html.unescape(text)

            # Remove URLs
            if self.config.remove_urls:
                text = self._url_pattern.sub("", text)

            # Remove emails
            if self.config.remove_emails:
                text = self._email_pattern.sub("", text)

            # Remove phone numbers
            if self.config.remove_phone_numbers:
                text = self._phone_pattern.sub("", text)

            # Remove special characters
            if self.config.remove_special_chars:
                text = self._special_char_pattern.sub(" ", text)

            # Normalize whitespace
            if self.config.normalize_whitespace:
                if self.config.preserve_newlines:
                    # Preserve single newlines, collapse other whitespace
                    text = re.sub(r"[ \t\r\f\v]+", " ", text)
                    text = re.sub(r"\n+", "\n", text)
                else:
                    text = self._whitespace_pattern.sub(" ", text)

            # Remove duplicate sentences
            if self.config.remove_duplicate_sentences:
                text = self._remove_duplicate_sentences(text)

            # Final cleanup
            text = text.strip()

            # Length filtering
            if len(text) < self.config.min_length or len(text) > self.config.max_length:
                self.log_debug(f"Text filtered by length: {len(text)} characters")
                return ""

            return text

        except Exception as e:
            self.log_error("Error during text cleaning", error=e)
            return ""

    def _fix_encoding(self, text: str) -> str:
        """Fix common encoding issues."""
        try:
            # Try to detect encoding if text appears to be bytes
            if isinstance(text, bytes):
                detected = chardet.detect(text)
                encoding = detected.get("encoding", "utf-8")
                text = text.decode(encoding, errors="ignore")

            # Fix common encoding artifacts
            replacements = {
                "â€™": "'",
                "â€œ": '"',
                "â€": '"',
                "â€¦": "...",
                'â€"': "—",
                'â€"': "–",
                "Ã¡": "á",
                "Ã©": "é",
                "Ã­": "í",
                "Ã³": "ó",
                "Ãº": "ú",
            }

            for artifact, replacement in replacements.items():
                text = text.replace(artifact, replacement)

            return text

        except Exception:
            return str(text)

    def _remove_html(self, text: str) -> str:
        """Remove HTML tags and convert to plain text."""
        try:
            soup = BeautifulSoup(text, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text and normalize spaces
            text = soup.get_text()

            return text

        except Exception:
            # Fallback: simple regex removal
            return re.sub(r"<[^>]+>", "", text)

    def _remove_duplicate_sentences(self, text: str) -> str:
        """Remove duplicate sentences while preserving order."""
        try:
            sentences = re.split(r"[.!?]+", text)
            seen = set()
            unique_sentences = []

            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and sentence not in seen:
                    seen.add(sentence)
                    unique_sentences.append(sentence)

            return ". ".join(unique_sentences) + "." if unique_sentences else ""

        except Exception:
            return text


class DocumentProcessor(LoggerMixin):
    """
    Process and chunk documents for embedding and storage.
    """

    def __init__(self, cleaner: Optional[TextCleaner] = None):
        super().__init__()
        self.cleaner = cleaner or TextCleaner()

    def process_document(
        self, content: str, metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Process a document into chunks with metadata.

        Args:
            content: Raw document content
            metadata: Document metadata

        Returns:
            List of document chunks with metadata
        """
        try:
            # Clean the content
            cleaned_content = self.cleaner.clean_text(content)

            if not cleaned_content:
                self.log_warning("Document content empty after cleaning")
                return []

            # Chunk the document
            chunks = self.chunk_text(cleaned_content)

            # Create chunk documents
            documents = []
            for i, chunk in enumerate(chunks):
                doc_metadata = metadata.copy()
                doc_metadata.update(
                    {
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "chunk_length": len(chunk),
                        "processed_at": self._get_timestamp(),
                    }
                )

                documents.append({"content": chunk, "metadata": doc_metadata})

            self.log_info(f"Processed document into {len(documents)} chunks")
            return documents

        except Exception as e:
            self.log_error("Error processing document", error=e)
            return []

    def chunk_text(
        self, text: str, chunk_size: int = 1000, overlap: int = 100
    ) -> List[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to chunk
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks

        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            if end < len(text):
                # Try to break at sentence boundary
                sentence_break = text.rfind(".", start, end)
                if sentence_break > start:
                    end = sentence_break + 1
                else:
                    # Try to break at word boundary
                    word_break = text.rfind(" ", start, end)
                    if word_break > start:
                        end = word_break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position with overlap
            start = max(start + 1, end - overlap)

        return chunks

    def extract_metadata(self, content: str, source_url: str = "") -> Dict[str, Any]:
        """
        Extract metadata from document content.

        Args:
            content: Document content
            source_url: Source URL if available

        Returns:
            Extracted metadata
        """
        metadata = {
            "length": len(content),
            "word_count": len(content.split()),
            "source_url": source_url,
            "domain": self._extract_domain(source_url),
            "extracted_at": self._get_timestamp(),
        }

        # Try to extract title from content
        title = self._extract_title(content)
        if title:
            metadata["title"] = title

        # Extract language (simple heuristic)
        metadata["language"] = self._detect_language(content)

        return metadata

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            return urlparse(url).netloc
        except:
            return ""

    def _extract_title(self, content: str) -> str:
        """Extract title from content."""
        try:
            # Look for title in HTML
            soup = BeautifulSoup(content, "html.parser")
            title_tag = soup.find("title")
            if title_tag:
                return title_tag.get_text().strip()

            # Fallback: use first line or sentence
            lines = content.split("\n")
            for line in lines:
                line = line.strip()
                if line and len(line) < 200:
                    return line

            return ""

        except:
            return ""

    def _detect_language(self, content: str) -> str:
        """Simple language detection."""
        # This is a placeholder - in production you might use langdetect
        common_english_words = {
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
        }
        words = set(content.lower().split()[:100])
        english_word_count = len(words.intersection(common_english_words))

        if english_word_count > 5:
            return "en"
        return "unknown"

    def _get_timestamp(self) -> str:
        """Get current timestamp as ISO string."""
        from datetime import datetime

        return datetime.utcnow().isoformat()


def create_cleaner(
    remove_html: bool = True,
    normalize_whitespace: bool = True,
    min_length: int = 10,
    max_length: int = 10000,
) -> TextCleaner:
    """
    Create a text cleaner with common settings.

    Args:
        remove_html: Whether to remove HTML tags
        normalize_whitespace: Whether to normalize whitespace
        min_length: Minimum text length
        max_length: Maximum text length

    Returns:
        Configured TextCleaner
    """
    config = CleaningConfig(
        remove_html=remove_html,
        normalize_whitespace=normalize_whitespace,
        min_length=min_length,
        max_length=max_length,
    )
    return TextCleaner(config)


def clean_rss_content(content: str) -> str:
    """
    Quick cleaning function for RSS content.

    Args:
        content: Raw RSS content

    Returns:
        Cleaned content
    """
    cleaner = create_cleaner()
    return cleaner.clean_text(content)
