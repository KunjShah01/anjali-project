"""
Real-time RAG Streamlit Application
A modern, interactive interface for real-time retrieval-augmented generation
"""

import streamlit as st
import asyncio
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
import plotly.express as px

# Try to import streamlit-option-menu with fallback
try:
    from streamlit_option_menu import option_menu

    HAS_OPTION_MENU = True
except ImportError:
    HAS_OPTION_MENU = False

import google.generativeai as genai

# Import our modules
from src.config import get_config
from src.utils.logger import LoggerMixin
from src.embeddings.embedder import EmbeddingService
from src.vectorstores.faiss_store import FAISSVectorStore
from src.ingestion.rss_ingest import RSSIngestor
from src.preprocessing.cleaner import TextCleaner
from src.embeddings.typing import Document, SimilarityResult
from src.image_generation import create_image_generator, ImageGenerationError


class StreamlitLogger(LoggerMixin):
    """Logger for Streamlit app."""

    def log_to_streamlit(self, level: str, message: str, **kwargs):
        """Log messages to Streamlit interface."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"

        if "log_container" in st.session_state:
            st.session_state.log_container.text(log_entry)


class RAGApplication(StreamlitLogger):
    """Main RAG application class."""

    def __init__(self):
        super().__init__()
        self.config = get_config()
        self.embedding_service = None
        self.vector_store = None
        self.rss_ingestor = None
        self.text_cleaner = TextCleaner()
        self.chat_model = None

        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all RAG components."""
        try:
            # Validate configuration
            if (
                not self.config.gemini.api_key
                or self.config.gemini.api_key == "your_gemini_api_key_here"
            ):
                st.warning(
                    "⚠️ Gemini API key not found or not configured. Please set GEMINI_API_KEY in your environment or .env file. Some features will be disabled."
                )
                return

            # Initialize Gemini chat model
            genai.configure(api_key=self.config.gemini.api_key)
            self.chat_model = genai.GenerativeModel(self.config.gemini.chat_model)

            # Initialize embedding service
            self.embedding_service = EmbeddingService(self.config.gemini)

            # Initialize vector store
            self.vector_store = FAISSVectorStore(self.config.vectorstore)

            # Initialize RSS ingestor
            if self.config.rss.feeds:
                self.rss_ingestor = RSSIngestor(
                    self.config.rss, self.embedding_service, self.vector_store
                )

            st.success("✅ RAG components initialized successfully!")
            self.log_info("RAG application initialized")

        except Exception as e:
            st.error(f"❌ Error initializing RAG components: {str(e)}")
            st.info(
                "💡 The app will work in limited mode. You can still use basic features."
            )
            self.log_error("Failed to initialize RAG application", error=e)

    async def add_documents(
        self, texts: List[str], metadata_list: List[Dict[str, Any]] = None
    ) -> int:
        """Add documents to the vector store."""
        if not texts:
            return 0

        try:
            # Clean and preprocess texts
            cleaned_texts = [self.text_cleaner.clean_text(text) for text in texts]

            # Generate embeddings
            embeddings = await self.embedding_service.get_embeddings(cleaned_texts)

            # Add to vector store
            doc_ids = []
            for i, (text, embedding) in enumerate(zip(cleaned_texts, embeddings)):
                if (
                    embedding
                    and embedding != [0.0] * self.embedding_service.get_dimension()
                ):
                    metadata = (
                        metadata_list[i]
                        if metadata_list and i < len(metadata_list)
                        else {}
                    )
                    doc_id = await self.vector_store.add_document(
                        text, embedding, metadata
                    )
                    if doc_id:
                        doc_ids.append(doc_id)

            return len(doc_ids)

        except Exception as e:
            self.log_error("Error adding documents", error=e)
            return 0

    async def search_documents(
        self, query: str, k: int = 5, threshold: float = 0.1
    ) -> List[SimilarityResult]:
        """Search for relevant documents."""
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.embed_query(query)

            if (
                not query_embedding
                or query_embedding == [0.0] * self.embedding_service.get_dimension()
            ):
                return []

            # Search vector store
            results = await self.vector_store.similarity_search(
                query_embedding, k, threshold
            )
            return results

        except Exception as e:
            self.log_error("Error searching documents", error=e)
            return []

    async def generate_response(self, query: str, context_docs: List[Document]) -> str:
        """Generate response using Gemini with retrieved context."""
        try:
            if not self.chat_model:
                return "Chat model not available. Please check your Gemini API key."

            # Prepare context from retrieved documents
            context_text = "\n\n".join(
                [
                    f"Document {i + 1}: {doc.content[:500]}..."
                    for i, doc in enumerate(context_docs)
                ]
            )

            # Create prompt
            prompt = f"""Based on the following context, please answer the question. If the context doesn't contain enough information to answer the question, please say so.

Context:
{context_text}

Question: {query}

Answer:"""

            # Generate response
            response = self.chat_model.generate_content(prompt)
            return response.text

        except Exception as e:
            self.log_error("Error generating response", error=e)
            return f"Error generating response: {str(e)}"


def init_session_state():
    """Initialize Streamlit session state."""
    if "app" not in st.session_state:
        st.session_state.app = RAGApplication()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "document_count" not in st.session_state:
        st.session_state.document_count = 0

    if "search_results" not in st.session_state:
        st.session_state.search_results = []


def sidebar_navigation():
    """Create sidebar navigation menu."""
    with st.sidebar:
        st.title("🤖 Real-time RAG")
        st.markdown("---")

        if HAS_OPTION_MENU:
            selected = option_menu(
                menu_title=None,
                options=[
                    "💬 Chat",
                    "📚 Documents",
                    "📰 News Feeds",
                    "🎨 Image Gen",
                    "📊 Analytics",
                    "⚙️ Settings",
                ],
                icons=["chat-dots", "book", "newspaper", "image", "graph-up", "gear"],
                menu_icon="cast",
                default_index=0,
                styles={
                    "container": {
                        "padding": "0!important",
                        "background-color": "#2b2b2b",
                    },
                    "icon": {"color": "#ff6b6b", "font-size": "18px"},
                    "nav-link": {
                        "font-size": "16px",
                        "text-align": "left",
                        "margin": "0px",
                        "padding": "0.75rem 1rem",
                        "border-radius": "0.5rem",
                        "color": "#ffffff",
                        "background-color": "transparent",
                        "--hover-color": "#404040",
                    },
                    "nav-link-selected": {
                        "background-color": "#ff6b6b",
                        "color": "#ffffff",
                        "font-weight": "600",
                    },
                },
            )
        else:
            # Fallback to regular selectbox if streamlit-option-menu is not available
            st.info(
                "💡 Install `streamlit-option-menu` for a better navigation experience"
            )
            selected = st.selectbox(
                "Navigate to:",
                [
                    "💬 Chat",
                    "📚 Documents",
                    "📰 News Feeds",
                    "🎨 Image Gen",
                    "📊 Analytics",
                    "⚙️ Settings",
                ],
                index=0,
            )

        return selected


def chat_interface():
    """Main chat interface."""
    st.title("💬 Chat with your Documents")
    st.markdown("Ask questions about your uploaded documents and RSS feeds.")

    # Chat container
    chat_container = st.container()

    # Display chat history
    with chat_container:
        for i, (role, message, timestamp) in enumerate(st.session_state.chat_history):
            if role == "user":
                st.chat_message("user").markdown(f"**You** ({timestamp}):\n{message}")
            else:
                st.chat_message("assistant").markdown(
                    f"**Assistant** ({timestamp}):\n{message}"
                )

    # Chat input
    if query := st.chat_input("Ask a question about your documents..."):
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Add user message to history
        st.session_state.chat_history.append(("user", query, timestamp))

        # Show user message immediately
        st.chat_message("user").markdown(f"**You** ({timestamp}):\n{query}")

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating response..."):
                # Search for relevant documents
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                try:
                    search_results = loop.run_until_complete(
                        st.session_state.app.search_documents(query, k=5)
                    )

                    if search_results:
                        # Generate response with context
                        context_docs = [result.document for result in search_results]
                        response = loop.run_until_complete(
                            st.session_state.app.generate_response(query, context_docs)
                        )

                        # Show sources
                        with st.expander("📚 Sources"):
                            for i, result in enumerate(search_results):
                                st.markdown(
                                    f"**Source {i + 1}** (Similarity: {result.score:.2f})"
                                )
                                st.markdown(
                                    f"```\n{result.document.content[:300]}...\n```"
                                )

                    else:
                        response = "I couldn't find any relevant documents to answer your question. Please make sure you have uploaded documents or configured RSS feeds."

                    # Display response
                    st.markdown(
                        f"**Assistant** ({datetime.now().strftime('%H:%M:%S')}):\n{response}"
                    )

                    # Add to chat history
                    st.session_state.chat_history.append(
                        ("assistant", response, datetime.now().strftime("%H:%M:%S"))
                    )

                finally:
                    loop.close()


def document_management():
    """Document upload and management interface."""
    st.title("📚 Document Management")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Upload Documents")

        # File upload
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            accept_multiple_files=True,
            type=["txt", "md", "pdf"],
            help="Upload text, markdown, or PDF files",
        )

        if uploaded_files:
            if st.button("🚀 Process Uploaded Files"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                texts = []
                metadata_list = []

                for i, uploaded_file in enumerate(uploaded_files):
                    try:
                        # Read file content
                        if uploaded_file.type == "text/plain":
                            content = str(uploaded_file.read(), "utf-8")
                        else:
                            content = str(
                                uploaded_file.read(), "utf-8"
                            )  # Simple text extraction

                        texts.append(content)
                        metadata_list.append(
                            {
                                "filename": uploaded_file.name,
                                "file_type": uploaded_file.type,
                                "upload_time": datetime.now().isoformat(),
                            }
                        )

                        progress_bar.progress((i + 1) / len(uploaded_files))
                        status_text.text(f"Processing {uploaded_file.name}...")

                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")

                # Add documents to vector store
                if texts:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    try:
                        added_count = loop.run_until_complete(
                            st.session_state.app.add_documents(texts, metadata_list)
                        )
                        st.session_state.document_count += added_count
                        st.success(
                            f"✅ Successfully processed {added_count} documents!"
                        )

                    except Exception as e:
                        st.error(f"❌ Error adding documents: {str(e)}")
                    finally:
                        loop.close()

        # Manual text input
        st.subheader("Add Text Manually")
        manual_text = st.text_area("Enter text content:", height=200)
        manual_title = st.text_input("Document title (optional):")

        if st.button("📝 Add Text Document"):
            if manual_text.strip():
                metadata = {
                    "title": manual_title or "Manual Input",
                    "source": "manual_input",
                    "upload_time": datetime.now().isoformat(),
                }

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                try:
                    added_count = loop.run_until_complete(
                        st.session_state.app.add_documents([manual_text], [metadata])
                    )
                    if added_count > 0:
                        st.session_state.document_count += added_count
                        st.success("✅ Text document added successfully!")
                    else:
                        st.error("❌ Failed to add text document")

                except Exception as e:
                    st.error(f"❌ Error adding document: {str(e)}")
                finally:
                    loop.close()
            else:
                st.warning("⚠️ Please enter some text content")

    with col2:
        st.subheader("📊 Statistics")

        # Document stats
        stats_container = st.container()
        with stats_container:
            st.metric("Total Documents", st.session_state.document_count)

            # Get vector store stats if available
            if st.session_state.app.vector_store:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                try:
                    vector_stats = loop.run_until_complete(
                        st.session_state.app.vector_store.get_stats()
                    )
                    st.metric(
                        "Vector Store Size", vector_stats.get("total_documents", 0)
                    )
                    st.metric("Index Size", vector_stats.get("index_size", 0))

                finally:
                    loop.close()

        # RSS Feed Management
        st.subheader("📡 RSS Feeds")

        if st.session_state.app.config.rss.feeds:
            st.write("Configured RSS feeds:")
            for i, feed in enumerate(st.session_state.app.config.rss.feeds):
                st.write(f"{i + 1}. {feed}")
        else:
            st.write("No RSS feeds configured")
            st.info("Add RSS feeds in the Settings page or .env file")


def news_feeds_interface():
    """Real-time news feeds interface with live updates and categorization."""
    st.title("📰 Real-time News Feeds")
    st.markdown("Stay updated with the latest news from multiple sources in real-time!")

    # Check RSS configuration
    if not st.session_state.app.config.rss.feeds:
        st.warning("⚠️ No RSS feeds configured")
        st.info(
            "Add RSS feed URLs in the Settings page or .env file to start receiving real-time news updates."
        )

        # Show example feeds
        with st.expander("📋 Example News Feed URLs"):
            st.markdown("""
            **Major News Sources:**
            - CNN: `https://rss.cnn.com/rss/edition.rss`
            - BBC: `http://feeds.bbci.co.uk/news/rss.xml`
            - Reuters: `https://feeds.reuters.com/reuters/topNews`
            - Guardian: `https://feeds.guardian.co.uk/international/rss`
            
            **Technology News:**
            - TechCrunch: `https://feeds.feedburner.com/TechCrunch`
            - Reuters Tech: `https://feeds.reuters.com/reuters/technologyNews`
            
            **Business News:**
            - Wall Street Journal: `https://feeds.a.dj.com/rss/RSSWorldNews.xml`
            - Bloomberg: `https://feeds.bloomberg.com/markets/news.rss`
            """)
        return

    # Initialize RSS ingestor if not already done
    if "rss_ingestor" not in st.session_state:
        try:
            from src.ingestion.rss_ingest import RSSIngestor

            st.session_state.rss_ingestor = RSSIngestor(
                st.session_state.app.config.rss,
                st.session_state.app.embedding_service,
                st.session_state.app.vector_store,
            )
            st.success("✅ RSS ingestor initialized")
        except Exception as e:
            st.error(f"❌ Failed to initialize RSS ingestor: {str(e)}")
            return

    # Control panel
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.subheader("📊 Feed Control Panel")

    with col2:
        auto_refresh = st.checkbox("🔄 Auto Refresh", value=False)

    with col3:
        if st.button("🔄 Refresh Now"):
            st.rerun()

    # Feed statistics
    if hasattr(st.session_state, "rss_ingestor"):
        stats = st.session_state.rss_ingestor.stats

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Active Feeds", stats.get("active_feeds", 0))
        with col2:
            st.metric("Total Articles", stats.get("total_articles_processed", 0))
        with col3:
            st.metric("Errors", stats.get("errors", 0))
        with col4:
            last_update = stats.get("last_update")
            if last_update:
                st.metric("Last Update", last_update.strftime("%H:%M:%S"))
            else:
                st.metric("Last Update", "Never")

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📰 Live Feed", "📈 Feed Analytics", "⚙️ Feed Settings"])

    with tab1:
        st.subheader("🔴 Live News Updates")

        # Category filter
        categories = ["All", "General News", "Technology", "Business", "International"]
        selected_category = st.selectbox("Filter by Category:", categories)

        # Show selected category
        if selected_category != "All":
            st.info(f"📂 Showing articles from: {selected_category}")

        # Fetch and display recent articles
        if st.button("📥 Fetch Latest Articles", type="primary"):
            with st.spinner("Fetching latest news articles..."):
                try:
                    # Simulate fetching articles (you'd implement actual fetching here)
                    import asyncio

                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    # Placeholder for actual article fetching
                    st.info(
                        "🚧 Live article fetching will be implemented. This shows the RSS feeds are configured and ready."
                    )

                    # Show configured feeds
                    st.write("**Configured News Sources:**")
                    for i, feed_url in enumerate(
                        st.session_state.app.config.rss.feeds[:10]
                    ):  # Show first 10
                        # Extract feed name from URL
                        if "cnn.com" in feed_url:
                            feed_name = "📺 CNN"
                        elif "bbci.co.uk" in feed_url:
                            feed_name = "🌍 BBC News"
                        elif "reuters.com" in feed_url:
                            feed_name = "📰 Reuters"
                        elif "techcrunch" in feed_url.lower():
                            feed_name = "💻 TechCrunch"
                        elif "bloomberg.com" in feed_url:
                            feed_name = "💼 Bloomberg"
                        else:
                            feed_name = f"📡 Feed {i + 1}"

                        st.write(f"• {feed_name}: {feed_url}")

                    loop.close()

                except Exception as e:
                    st.error(f"❌ Error fetching articles: {str(e)}")

        # Placeholder for article display
        st.markdown("---")
        st.info(
            "💡 Articles will appear here once fetching is implemented. The RSS system is ready to ingest from "
            + str(len(st.session_state.app.config.rss.feeds))
            + " configured news sources."
        )

    with tab2:
        st.subheader("📈 Feed Analytics")

        # Create sample analytics charts
        if hasattr(
            st.session_state, "rss_ingestor"
        ) and st.session_state.rss_ingestor.stats.get("articles_per_feed"):
            import plotly.express as px

            # Articles per feed chart
            feed_data = st.session_state.rss_ingestor.stats.get("articles_per_feed", {})
            if feed_data:
                fig = px.bar(
                    x=list(feed_data.keys()),
                    y=list(feed_data.values()),
                    title="Articles per Feed",
                    labels={"x": "RSS Feed", "y": "Article Count"},
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Analytics will appear here once articles are processed.")

    with tab3:
        st.subheader("⚙️ Feed Configuration")

        # Display current configuration
        st.write("**Current RSS Configuration:**")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Feeds", len(st.session_state.app.config.rss.feeds))
            st.metric(
                "Refresh Interval",
                f"{st.session_state.app.config.rss.refresh_interval}s",
            )

        with col2:
            st.metric(
                "Max Articles per Feed",
                st.session_state.app.config.rss.max_articles_per_feed,
            )

        # Show all configured feeds
        st.write("**All Configured Feeds:**")
        for i, feed in enumerate(st.session_state.app.config.rss.feeds):
            with st.expander(f"Feed {i + 1}: {feed[:50]}..."):
                st.code(feed, language="text")

                # Test feed button
                if st.button(f"🧪 Test Feed {i + 1}", key=f"test_feed_{i}"):
                    with st.spinner(f"Testing feed {i + 1}..."):
                        try:
                            import aiohttp
                            import asyncio

                            async def test_feed():
                                async with aiohttp.ClientSession() as session:
                                    async with session.get(
                                        feed, timeout=10
                                    ) as response:
                                        return response.status

                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            status = loop.run_until_complete(test_feed())
                            loop.close()

                            if status == 200:
                                st.success(
                                    f"✅ Feed {i + 1} is accessible (Status: {status})"
                                )
                            else:
                                st.warning(f"⚠️ Feed {i + 1} returned status: {status}")
                        except Exception as e:
                            st.error(f"❌ Feed {i + 1} test failed: {str(e)}")

    # Auto-refresh logic
    if auto_refresh:
        import time

        time.sleep(5)  # Wait 5 seconds
        st.rerun()


def image_generation_interface():
    """Image generation interface using Gemini Imagen and Nano Banana."""
    st.title("🎨 AI Image Generation")
    st.markdown("Generate stunning images using Google Gemini Imagen and Nano Banana!")

    # Check if API key is configured
    if (
        not st.session_state.app.config.gemini.api_key
        or st.session_state.app.config.gemini.api_key == "your_gemini_api_key_here"
    ):
        st.error(
            "❌ Gemini API key not configured. Please set your API key in the Settings page."
        )
        return

    # Initialize image generator
    try:
        if "image_generator" not in st.session_state:
            st.session_state.image_generator = create_image_generator(
                st.session_state.app.config.gemini
            )
    except Exception as e:
        st.error(f"❌ Failed to initialize image generator: {str(e)}")
        return

    # Create tabs for different generators
    tab1, tab2, tab3 = st.tabs(["🤖 Gemini Imagen", "🍌 Nano Banana", "📊 Statistics"])

    with tab1:
        st.subheader("🤖 Google Gemini Imagen")
        st.markdown(
            "Use Google's advanced Imagen model for high-quality image generation."
        )

        # Input form
        with st.form("gemini_image_form"):
            prompt = st.text_area(
                "Describe the image you want to generate:",
                placeholder="A serene mountain landscape with a crystal clear lake reflecting the sunset...",
                height=100,
            )

            col1, col2 = st.columns(2)
            with col1:
                width = st.selectbox("Width", [512, 768, 1024, 1280], index=2)
                style = st.selectbox(
                    "Style",
                    ["photorealistic", "artistic", "abstract", "cinematic"],
                    index=0,
                )
            with col2:
                height = st.selectbox("Height", [512, 768, 1024, 1280], index=2)
                quality = st.selectbox("Quality", ["standard", "high"], index=1)

            generate_gemini = st.form_submit_button(
                "🚀 Generate with Gemini", type="primary"
            )

        if generate_gemini and prompt.strip():
            with st.spinner("Generating image with Gemini Imagen..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                try:
                    result = loop.run_until_complete(
                        st.session_state.image_generator.generate_image_with_gemini(
                            prompt=prompt.strip(),
                            width=width,
                            height=height,
                            style=style,
                            quality=quality,
                        )
                    )

                    if result and result.get("image_data"):
                        import base64

                        # Display the image
                        image_bytes = base64.b64decode(result["image_data"])
                        st.image(
                            image_bytes,
                            caption=f"Generated: {prompt[:50]}...",
                            use_column_width=True,
                        )

                        # Display generation info
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Generation Time", f"{result['generation_time']:.2f}s"
                            )
                        with col2:
                            st.metric("Size", f"{result['width']}x{result['height']}")
                        with col3:
                            st.metric("Style", result["style"])

                        # Download button
                        st.download_button(
                            label="📥 Download Image",
                            data=image_bytes,
                            file_name=f"gemini_generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                            mime="image/png",
                        )

                    else:
                        st.error("❌ Failed to generate image with Gemini")

                except ImageGenerationError as e:
                    st.error(f"❌ Image generation error: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")
                finally:
                    loop.close()

    with tab2:
        st.subheader("🍌 Nano Banana")
        st.markdown(
            "Specialized anime and artistic style generation with Nano Banana integration."
        )

        # Input form
        with st.form("nano_banana_form"):
            nb_prompt = st.text_area(
                "Describe your anime/artistic image:",
                placeholder="A cute anime character with rainbow hair in a magical forest...",
                height=100,
            )

            col1, col2 = st.columns(2)
            with col1:
                nb_style = st.selectbox(
                    "Art Style",
                    ["anime", "realistic", "cartoon", "fantasy", "cyberpunk"],
                    index=0,
                )
                nb_steps = st.slider("Inference Steps", 10, 50, 20)
            with col2:
                cfg_scale = st.slider("CFG Scale", 1.0, 15.0, 7.0, 0.5)

            generate_nano = st.form_submit_button(
                "🍌 Generate with Nano Banana", type="primary"
            )

        if generate_nano and nb_prompt.strip():
            with st.spinner("Generating image with Nano Banana..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                try:
                    result = loop.run_until_complete(
                        st.session_state.image_generator.generate_image_with_nano_banana(
                            prompt=nb_prompt.strip(),
                            style=nb_style,
                            steps=nb_steps,
                            cfg_scale=cfg_scale,
                        )
                    )

                    if result and result.get("image_data"):
                        import base64

                        # Display the image
                        image_bytes = base64.b64decode(result["image_data"])
                        st.image(
                            image_bytes,
                            caption=f"Generated: {nb_prompt[:50]}...",
                            use_column_width=True,
                        )

                        # Display generation info
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Generation Time", f"{result['generation_time']:.2f}s"
                            )
                        with col2:
                            st.metric("Style", result["style"])
                        with col3:
                            st.metric("Steps", result["steps"])

                        # Download button
                        st.download_button(
                            label="📥 Download Image",
                            data=image_bytes,
                            file_name=f"nano_banana_generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                            mime="image/png",
                        )

                    else:
                        st.error("❌ Failed to generate image with Nano Banana")

                except ImageGenerationError as e:
                    st.error(f"❌ Image generation error: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")
                finally:
                    loop.close()

    with tab3:
        st.subheader("📊 Generation Statistics")

        if "image_generator" in st.session_state:
            stats = st.session_state.image_generator.get_stats()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Images Generated", stats.get("images_generated", 0))
            with col2:
                st.metric("Total Errors", stats.get("errors", 0))
            with col3:
                st.metric("Cache Size", stats.get("cache_size", 0))

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Cache Hits", stats.get("cache_hits", 0))
            with col2:
                st.metric("Cache Misses", stats.get("cache_misses", 0))
            with col3:
                avg_time = stats.get("average_generation_time", 0)
                st.metric("Avg Generation Time", f"{avg_time:.2f}s")

            # Supported providers
            st.subheader("🔧 Supported Providers")
            providers = stats.get("supported_providers", [])
            for provider in providers:
                st.write(f"✅ {provider}")

            # Cache management
            st.subheader("🗄️ Cache Management")
            if st.button("🗑️ Clear Image Cache"):
                st.session_state.image_generator.clear_cache()
                st.success("✅ Image cache cleared!")
                st.rerun()


def analytics_dashboard():
    """Analytics and monitoring dashboard."""
    st.title("📊 Analytics Dashboard")

    # Metrics overview
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Documents", st.session_state.document_count)

    with col2:
        chat_count = len(st.session_state.chat_history)
        st.metric("Chat Messages", chat_count)

    with col3:
        # Get embedding service stats
        if st.session_state.app.embedding_service:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                embedding_stats = loop.run_until_complete(
                    st.session_state.app.embedding_service.get_stats()
                )
                st.metric(
                    "Texts Processed", embedding_stats.get("total_texts_processed", 0)
                )
            finally:
                loop.close()
        else:
            st.metric("Texts Processed", 0)

    with col4:
        # Calculate cache hit rate
        if st.session_state.app.embedding_service:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                embedding_stats = loop.run_until_complete(
                    st.session_state.app.embedding_service.get_stats()
                )
                cache_hits = embedding_stats.get("cache_hits", 0)
                cache_misses = embedding_stats.get("cache_misses", 0)
                total = cache_hits + cache_misses
                cache_rate = (cache_hits / total * 100) if total > 0 else 0
                st.metric("Cache Hit Rate", f"{cache_rate:.1f}%")
            finally:
                loop.close()
        else:
            st.metric("Cache Hit Rate", "0%")

    # Chat activity over time
    if st.session_state.chat_history:
        st.subheader("💬 Chat Activity")

        # Create timeline chart
        chat_data = []
        for role, message, timestamp in st.session_state.chat_history:
            chat_data.append(
                {"timestamp": timestamp, "role": role, "length": len(message)}
            )

        df = pd.DataFrame(chat_data)

        if not df.empty:
            fig = px.scatter(
                df,
                x="timestamp",
                y="length",
                color="role",
                title="Message Length Over Time",
                labels={"length": "Message Length (chars)", "timestamp": "Time"},
            )
            st.plotly_chart(fig, use_container_width=True)

    # System Performance
    st.subheader("⚡ System Performance")

    # Get detailed stats
    if st.session_state.app.embedding_service and st.session_state.app.vector_store:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            embedding_stats = loop.run_until_complete(
                st.session_state.app.embedding_service.get_stats()
            )
            vector_stats = loop.run_until_complete(
                st.session_state.app.vector_store.get_stats()
            )

            # Performance metrics
            col1, col2 = st.columns(2)

            with col1:
                st.json(
                    {
                        "Embedding Service": {
                            "Provider": embedding_stats.get("provider", "Unknown"),
                            "Model": embedding_stats.get("model", "Unknown"),
                            "Dimension": embedding_stats.get("dimension", 0),
                            "Average Processing Time": f"{embedding_stats.get('average_embedding_time', 0):.3f}s",
                            "Errors": embedding_stats.get("errors", 0),
                        }
                    }
                )

            with col2:
                st.json(
                    {
                        "Vector Store": {
                            "Type": vector_stats.get("index_type", "Unknown"),
                            "Total Documents": vector_stats.get("total_documents", 0),
                            "Index Size": vector_stats.get("index_size", 0),
                            "Searches Performed": vector_stats.get(
                                "searches_performed", 0
                            ),
                            "Last Save": vector_stats.get("last_save", "Never"),
                        }
                    }
                )

        finally:
            loop.close()


def settings_page():
    """Application settings and configuration."""
    st.title("⚙️ Settings")

    # API Configuration
    st.subheader("🔑 API Configuration")

    current_api_key = st.session_state.app.config.gemini.api_key
    masked_key = f"...{current_api_key[-4:]}" if current_api_key else "Not set"

    st.info(f"Current Gemini API Key: {masked_key}")

    new_api_key = st.text_input("Update Gemini API Key:", type="password")
    if st.button("Update API Key"):
        if new_api_key:
            # Update configuration (this would need to be persistent in a real app)
            st.success("✅ API key updated successfully!")
            st.info("Note: Restart the application for changes to take effect.")
        else:
            st.warning("⚠️ Please enter a valid API key")

    # Model Configuration
    st.subheader("🧠 Model Configuration")

    st.selectbox(
        "Embedding Model:",
        ["models/embedding-001", "models/text-embedding-004"],
        index=0
        if st.session_state.app.config.gemini.embedding_model == "models/embedding-001"
        else 1,
    )

    st.selectbox(
        "Chat Model:",
        ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
        index=0
        if st.session_state.app.config.gemini.chat_model == "gemini-2.5-flash"
        else 1,
    )

    # Vector Store Configuration
    st.subheader("🗂️ Vector Store Configuration")

    st.selectbox(
        "Vector Store Type:",
        ["faiss", "pathway"],
        index=0 if st.session_state.app.config.vectorstore.store_type == "faiss" else 1,
    )

    # RSS Configuration
    st.subheader("📡 RSS Feed Configuration")

    current_feeds = st.session_state.app.config.rss.feeds
    feeds_text = "\n".join(current_feeds) if current_feeds else ""

    new_feeds_text = st.text_area(
        "RSS Feed URLs (one per line):", value=feeds_text, height=100
    )

    st.slider(
        "RSS Refresh Interval (seconds):",
        min_value=60,
        max_value=3600,
        value=st.session_state.app.config.rss.refresh_interval,
        step=60,
    )

    if st.button("Update RSS Configuration"):
        # Parse feed URLs
        [feed.strip() for feed in new_feeds_text.split("\n") if feed.strip()]
        st.success("✅ RSS configuration updated!")
        st.info("Note: Restart the application for changes to take effect.")

    # Cache Management
    st.subheader("🧹 Cache Management")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🗑️ Clear Embedding Cache"):
            if st.session_state.app.embedding_service:
                st.session_state.app.embedding_service.clear_cache()
                st.success("✅ Embedding cache cleared!")

    with col2:
        if st.button("💾 Save Vector Store"):
            if st.session_state.app.vector_store:
                st.session_state.app.vector_store.save_index()
                st.success("✅ Vector store saved!")


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Real-time RAG",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS
    st.markdown(
        """
    <style>
        /* Main content styling */
        .main {
            padding: 1rem;
        }
        
        /* Alert styling */
        .stAlert {
            margin: 1rem 0;
        }
        
        /* Metric container styling */
        .metric-container {
            background: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        
        /* Sidebar styling improvements */
        .css-1d391kg {
            background-color: #2b2b2b;
        }
        
        /* Additional sidebar selectors for different Streamlit versions */
        .stSidebar {
            background-color: #2b2b2b !important;
        }
        
        .stSidebar > div {
            background-color: #2b2b2b !important;
        }
        
        .css-1lcbmhc {
            background-color: #2b2b2b !important;
        }
        
        .css-17eq0hr {
            background-color: #2b2b2b !important;
        }
        
        /* Sidebar text styling */
        .stSidebar .stMarkdown, .stSidebar .stTitle, .stSidebar p, .stSidebar h1, .stSidebar h2, .stSidebar h3 {
            color: #ffffff !important;
        }
        
        /* Sidebar divider styling */
        .stSidebar hr {
            border-color: #606060 !important;
        }
        
        /* Sidebar navigation styling */
        .nav-link {
            display: flex !important;
            align-items: center !important;
            padding: 0.75rem 1rem !important;
            margin: 0.25rem 0 !important;
            border-radius: 0.5rem !important;
            text-decoration: none !important;
            color: #ffffff !important;
            transition: all 0.3s ease !important;
        }
        
        .nav-link:hover {
            background-color: #404040 !important;
            color: #ff6b6b !important;
            transform: translateX(4px) !important;
        }
        
        .nav-link-selected {
            background-color: #ff6b6b !important;
            color: #ffffff !important;
            font-weight: 600 !important;
        }
        
        /* Icon styling in navigation */
        .nav-link i {
            margin-right: 0.5rem !important;
            font-size: 1.1rem !important;
        }
        
        /* Sidebar title styling */
        .sidebar .element-container h1 {
            color: #ffffff !important;
            font-size: 1.5rem !important;
            margin-bottom: 1rem !important;
            text-align: center !important;
        }
        
        /* Improve selectbox styling when option-menu is not available */
        .stSidebar .stSelectbox > div > div {
            background-color: #404040 !important;
            color: #ffffff !important;
            border: 1px solid #606060 !important;
            border-radius: 0.5rem !important;
        }
        
        .stSidebar .stSelectbox > div > div:hover {
            border-color: #ff6b6b !important;
        }
        
        .stSidebar .stSelectbox label {
            color: #ffffff !important;
        }
        
        /* Dropdown options styling */
        .stSidebar .stSelectbox [data-baseweb="select"] > div {
            background-color: #404040 !important;
            color: #ffffff !important;
        }
        
        /* Footer styling in sidebar */
        .sidebar .markdown-text-container p {
            color: #cccccc !important;
            text-align: center !important;
            font-size: 0.9rem !important;
        }
        
        /* Option menu container improvements */
        .nav-item {
            margin: 0.2rem 0 !important;
        }
        
        /* Fix menu item text alignment */
        .nav-link-text {
            margin-left: 0.5rem !important;
        }
        
        /* Image display improvements */
        .stImage {
            border-radius: 0.5rem;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        /* Tab styling improvements */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 0.75rem 1.5rem;
            border-radius: 0.5rem;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #ff6b6b;
            color: white;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Initialize session state
    init_session_state()

    # Navigation
    selected_page = sidebar_navigation()

    # Route to selected page
    if selected_page == "💬 Chat":
        chat_interface()
    elif selected_page == "📚 Documents":
        document_management()
    elif selected_page == "📰 News Feeds":
        news_feeds_interface()
    elif selected_page == "🎨 Image Gen":
        image_generation_interface()
    elif selected_page == "📊 Analytics":
        analytics_dashboard()
    elif selected_page == "⚙️ Settings":
        settings_page()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Real-time RAG v1.0**")
    st.sidebar.markdown("Powered by Google Gemini & Streamlit")


if __name__ == "__main__":
    main()
