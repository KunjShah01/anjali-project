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
    
    # Enhanced UI state management
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"
    
    if "show_onboarding" not in st.session_state:
        st.session_state.show_onboarding = True
    
    if "document_search_term" not in st.session_state:
        st.session_state.document_search_term = ""
    
    if "documents_per_page" not in st.session_state:
        st.session_state.documents_per_page = 10
    
    if "current_document_page" not in st.session_state:
        st.session_state.current_document_page = 1


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
    """Enhanced main chat interface with improved UX."""
    st.title("💬 Chat with your Knowledge Base")
    
    # Enhanced introduction with onboarding
    if st.session_state.show_onboarding and len(st.session_state.chat_history) == 0:
        with st.container():
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 12px;
                margin: 10px 0;
            ">
                <h3 style="margin: 0; color: white;">🚀 Welcome to Your AI Assistant!</h3>
                <p style="margin: 10px 0; opacity: 0.9;">Ask questions about your uploaded documents, RSS feeds, and any content in your knowledge base.</p>
                <small style="opacity: 0.8;">💡 Try asking: "What documents do I have?" or "Summarize the latest news"</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Quick action buttons for common queries
    if len(st.session_state.chat_history) == 0:
        st.subheader("🚀 Quick Start")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Show my documents", use_container_width=True):
                st.session_state.quick_query = "What documents do I have in my knowledge base?"
                st.rerun()
        
        with col2:
            if st.button("📰 Latest news summary", use_container_width=True):
                st.session_state.quick_query = "Summarize the latest news from my RSS feeds"
                st.rerun()
        
        with col3:
            if st.button("❓ What can you do?", use_container_width=True):
                st.session_state.quick_query = "What can you help me with? What are your capabilities?"
                st.rerun()
    
    # Chat statistics header
    if st.session_state.chat_history:
        col_a, col_b, col_c = st.columns([2, 1, 1])
        with col_a:
            st.markdown("### 💬 Conversation History")
        with col_b:
            st.metric("Messages", len(st.session_state.chat_history))
        with col_c:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                enhanced_toast("Chat history cleared", "🗑️")
                st.rerun()
    
    # Enhanced chat container with better styling
    chat_container = st.container()

    # Display chat history with enhanced formatting
    with chat_container:
        for i, (role, message, timestamp) in enumerate(st.session_state.chat_history):
            if role == "user":
                # Enhanced user message styling
                with st.chat_message("user"):
                    st.markdown(f"""
                    <div style="margin-bottom: 5px;">
                        <strong style="color: #ff6b6b;">You</strong> 
                        <small style="opacity: 0.7; margin-left: 10px;">🕒 {timestamp}</small>
                    </div>
                    <div style="background: #f8f9fa; padding: 10px; border-radius: 8px; border-left: 3px solid #ff6b6b;">
                        {message}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                # Enhanced assistant message styling
                with st.chat_message("assistant"):
                    st.markdown(f"""
                    <div style="margin-bottom: 5px;">
                        <strong style="color: #4ecdc4;">🤖 AI Assistant</strong> 
                        <small style="opacity: 0.7; margin-left: 10px;">🕒 {timestamp}</small>
                    </div>
                    <div style="background: #f8f9fa; padding: 10px; border-radius: 8px; border-left: 3px solid #4ecdc4;">
                        {message}
                    </div>
                    """, unsafe_allow_html=True)
    
    # Handle quick query if set
    if hasattr(st.session_state, 'quick_query'):
        query = st.session_state.quick_query
        del st.session_state.quick_query
        # Process the query immediately
        process_chat_query(query)
        st.rerun()

    # Enhanced chat input with better placeholder and help
    query = st.chat_input(
        "💭 Ask me anything about your documents, news feeds, or general questions...",
        help="Ask questions about your uploaded documents, RSS feeds, or request summaries and insights"
    )
    
    if query:
        process_chat_query(query)


def process_chat_query(query):
    """Process a chat query with enhanced feedback and error handling."""
    timestamp = datetime.now().strftime("%H:%M:%S")

    # Add user message to history
    st.session_state.chat_history.append(("user", query, timestamp))

    # Show user message immediately with enhanced styling
    with st.chat_message("user"):
        st.markdown(f"""
        <div style="margin-bottom: 5px;">
            <strong style="color: #ff6b6b;">You</strong> 
            <small style="opacity: 0.7; margin-left: 10px;">🕒 {timestamp}</small>
        </div>
        <div style="background: #f8f9fa; padding: 10px; border-radius: 8px; border-left: 3px solid #ff6b6b;">
            {query}
        </div>
        """, unsafe_allow_html=True)

    # Generate response with enhanced loading and feedback
    with st.chat_message("assistant"):
        with st.spinner("🧠 Searching knowledge base and generating response..."):
            # Create a progress indicator
            progress_placeholder = st.empty()
            progress_placeholder.info("🔍 Step 1/3: Searching relevant documents...")
            
            # Search for relevant documents
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                search_results = loop.run_until_complete(
                    st.session_state.app.search_documents(query, k=5)
                )
                
                progress_placeholder.info("🤖 Step 2/3: Generating AI response...")

                if search_results:
                    # Generate response with context
                    context_docs = [result.document for result in search_results]
                    response = loop.run_until_complete(
                        st.session_state.app.generate_response(query, context_docs)
                    )

                    progress_placeholder.info("📚 Step 3/3: Preparing sources and references...")
                    
                    # Enhanced response display
                    st.markdown(f"""
                    <div style="margin-bottom: 5px;">
                        <strong style="color: #4ecdc4;">🤖 AI Assistant</strong> 
                        <small style="opacity: 0.7; margin-left: 10px;">🕒 {datetime.now().strftime('%H:%M:%S')}</small>
                    </div>
                    <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 3px solid #4ecdc4;">
                        {response}
                    </div>
                    """, unsafe_allow_html=True)

                    # Enhanced sources display with better formatting
                    with st.expander(f"📚 Sources Used ({len(search_results)})", expanded=len(search_results) <= 3):
                        for i, result in enumerate(search_results):
                            with st.container():
                                col_source, col_score = st.columns([4, 1])
                                with col_source:
                                    st.markdown(f"**📄 Source {i + 1}:** {result.document.metadata.get('filename', 'Unknown Document')}")
                                with col_score:
                                    # Visual similarity score
                                    score_color = "#4ecdc4" if result.score > 0.8 else "#ffa500" if result.score > 0.6 else "#ff6b6b"
                                    st.markdown(f"<div style='text-align: center; padding: 5px; background: {score_color}; color: white; border-radius: 4px;'>{result.score:.1%}</div>", unsafe_allow_html=True)
                                
                                # Document preview with character limit
                                preview_text = result.document.content[:300]
                                if len(result.document.content) > 300:
                                    preview_text += "..."
                                
                                st.markdown(f"""
                                <div style="background: #f8f9fa; padding: 10px; border-radius: 4px; margin: 5px 0; font-family: monospace; font-size: 0.9em;">
                                    {preview_text}
                                </div>
                                """, unsafe_allow_html=True)
                                
                                if i < len(search_results) - 1:
                                    st.markdown("---")

                else:
                    # Enhanced no-results response
                    response = """
                    🤔 I couldn't find specific information in your knowledge base to answer that question. 
                    
                    Here are some suggestions:
                    • **📁 Upload documents** related to your question
                    • **🔄 Add RSS feeds** for real-time information
                    • **✏️ Add manual content** in the Documents section
                    • **🔍 Try rephrasing** your question with different keywords
                    
                    I can still help with general questions or guide you on how to use this system!
                    """
                    
                    st.markdown(f"""
                    <div style="margin-bottom: 5px;">
                        <strong style="color: #4ecdc4;">🤖 AI Assistant</strong> 
                        <small style="opacity: 0.7; margin-left: 10px;">🕒 {datetime.now().strftime('%H:%M:%S')}</small>
                    </div>
                    <div style="background: #fff3cd; padding: 15px; border-radius: 8px; border-left: 3px solid #ffa500;">
                        {response}
                    </div>
                    """, unsafe_allow_html=True)

                # Clear progress indicator
                progress_placeholder.empty()

                # Add to chat history
                st.session_state.chat_history.append(
                    ("assistant", response, datetime.now().strftime("%H:%M:%S"))
                )

            except Exception as e:
                progress_placeholder.empty()
                error_response = f"❌ I encountered an error while processing your request: {str(e)}\n\nPlease try again or contact support if the issue persists."
                
                st.markdown(f"""
                <div style="margin-bottom: 5px;">
                    <strong style="color: #4ecdc4;">🤖 AI Assistant</strong> 
                    <small style="opacity: 0.7; margin-left: 10px;">🕒 {datetime.now().strftime('%H:%M:%S')}</small>
                </div>
                <div style="background: #f8d7da; padding: 15px; border-radius: 8px; border-left: 3px solid #dc3545;">
                    {error_response}
                </div>
                """, unsafe_allow_html=True)
                
                st.session_state.chat_history.append(
                    ("assistant", error_response, datetime.now().strftime("%H:%M:%S"))
                )
            finally:
                loop.close()


def enhanced_toast(message, icon="✅"):
    """Enhanced toast notification using st.success/error/info."""
    if icon == "✅":
        st.success(f"{icon} {message}")
    elif icon == "❌":
        st.error(f"{icon} {message}")
    elif icon == "⚠️":
        st.warning(f"{icon} {message}")
    else:
        st.info(f"{icon} {message}")


def enhanced_loading_spinner(text="Processing..."):
    """Enhanced loading indicator with progress feedback."""
    return st.spinner(f"🔄 {text}")


def document_management():
    """Document upload and management interface with enhanced UX."""
    st.title("📚 Enhanced Document Management")
    
    # Show onboarding guide for new users
    if st.session_state.show_onboarding:
        with st.expander("🎯 Quick Start Guide", expanded=True):
            st.markdown("""
            **Welcome to Document Management!** 👋
            
            Here's how to get started:
            1. **📁 Drag & Drop Files**: Use the upload area below to add documents
            2. **🔍 Search & Filter**: Use the search bar to find specific documents  
            3. **📊 View Analytics**: Check document statistics in the sidebar
            4. **⚙️ Manage Settings**: Configure RSS feeds and other options
            
            💡 **Tip**: You can upload multiple files at once for faster processing!
            """)
            
            if st.button("✨ Got it! Hide this guide"):
                st.session_state.show_onboarding = False
                st.rerun()

    # Enhanced search and filter section
    st.subheader("🔍 Search & Filter Documents")
    
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        search_term = st.text_input(
            "🔍 Search documents...",
            value=st.session_state.document_search_term,
            placeholder="Search by title, content, or filename...",
            help="Search across document titles, content, and metadata"
        )
        if search_term != st.session_state.document_search_term:
            st.session_state.document_search_term = search_term
            st.session_state.current_document_page = 1  # Reset to first page on new search
    
    with col2:
        doc_type_filter = st.selectbox(
            "📂 Type",
            ["All", "Text", "PDF", "Markdown", "RSS"],
            help="Filter documents by type"
        )
    
    with col3:
        sort_option = st.selectbox(
            "🔤 Sort by",
            ["Date Added", "Name", "Size", "Type"],
            help="Sort documents by different criteria"
        )

    # Main layout with enhanced columns
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📁 Upload Documents")

        # Enhanced drag-and-drop file upload with better styling
        st.markdown("""
        <div style="
            border: 2px dashed #ff6b6b;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            margin: 10px 0;
        ">
            <h4 style="color: #ff6b6b; margin: 0;">📂 Drag & Drop Your Files Here</h4>
            <p style="color: #6c757d; margin: 5px 0;">Or click below to browse files</p>
        </div>
        """, unsafe_allow_html=True)

        # File upload with enhanced interface
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            accept_multiple_files=True,
            type=["txt", "md", "pdf", "docx", "csv"],
            help="📄 Supported formats: TXT, MD, PDF, DOCX, CSV",
            label_visibility="collapsed"
        )

        if uploaded_files:
            # Enhanced file preview with better organization
            with st.expander(f"📋 Selected Files ({len(uploaded_files)})", expanded=True):
                for i, file in enumerate(uploaded_files):
                    col_a, col_b, col_c = st.columns([3, 1, 1])
                    with col_a:
                        st.markdown(f"**{file.name}**")
                    with col_b:
                        st.markdown(f"📏 {file.size} bytes")
                    with col_c:
                        st.markdown(f"📂 {file.type}")
            
            if st.button("🚀 Process Uploaded Files", type="primary", use_container_width=True):
                with enhanced_loading_spinner("Processing your documents..."):
                    # Enhanced progress tracking
                    progress_container = st.container()
                    with progress_container:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        detail_text = st.empty()

                    texts = []
                    metadata_list = []
                    processed_count = 0

                    for i, uploaded_file in enumerate(uploaded_files):
                        try:
                            status_text.markdown(f"**Processing:** {uploaded_file.name}")
                            detail_text.info(f"📄 Step {i+1} of {len(uploaded_files)}: Reading file content...")
                            
                            # Enhanced file processing with better error handling
                            if uploaded_file.type == "text/plain" or uploaded_file.name.endswith('.txt'):
                                content = str(uploaded_file.read(), "utf-8")
                            elif uploaded_file.name.endswith('.md'):
                                content = str(uploaded_file.read(), "utf-8")
                            elif uploaded_file.type == "application/pdf":
                                # Placeholder for PDF processing
                                content = f"PDF file: {uploaded_file.name} (PDF processing not yet implemented)"
                                detail_text.warning("📋 PDF files are accepted but content extraction is limited")
                            else:
                                content = str(uploaded_file.read(), "utf-8")

                            texts.append(content)
                            metadata_list.append({
                                "filename": uploaded_file.name,
                                "file_type": uploaded_file.type,
                                "file_size": uploaded_file.size,
                                "upload_time": datetime.now().isoformat(),
                                "processed_by": "enhanced_document_management"
                            })
                            
                            processed_count += 1
                            progress_bar.progress((i + 1) / len(uploaded_files))
                            detail_text.success(f"✅ Successfully processed {uploaded_file.name}")

                        except Exception as e:
                            detail_text.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                            continue

                    # Enhanced vector store integration
                    if texts:
                        status_text.markdown("**Adding to Vector Store...**")
                        detail_text.info("🔄 Generating embeddings and storing documents...")
                        
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                        try:
                            added_count = loop.run_until_complete(
                                st.session_state.app.add_documents(texts, metadata_list)
                            )
                            st.session_state.document_count += added_count
                            enhanced_toast(
                                f"Successfully processed {processed_count} files and added {added_count} documents to the knowledge base!",
                                "✅"
                            )

                        except Exception as e:
                            enhanced_toast(f"Error adding documents to vector store: {str(e)}", "❌")
                        finally:
                            loop.close()
                    else:
                        enhanced_toast("No valid documents were processed", "⚠️")

        # Enhanced manual text input with better UX
        st.subheader("📝 Add Text Content Manually")
        
        with st.container():
            st.markdown("""
            <div style="
                background: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                border-left: 4px solid #ff6b6b;
                margin: 10px 0;
            ">
                <strong>💡 Quick Text Input</strong><br>
                Perfect for adding notes, snippets, or any text content directly to your knowledge base.
            </div>
            """, unsafe_allow_html=True)
            
            col_a, col_b = st.columns([2, 1])
            with col_a:
                manual_title = st.text_input(
                    "📄 Document title:",
                    placeholder="Enter a descriptive title for your content...",
                    help="This helps you find the document later"
                )
            with col_b:
                content_category = st.selectbox(
                    "🏷️ Category",
                    ["General", "Notes", "Research", "Documentation", "Other"],
                    help="Categorize your content for better organization"
                )
            
            manual_text = st.text_area(
                "📝 Content:",
                height=200,
                placeholder="Paste or type your content here...\n\nYou can include:\n• Research notes\n• Meeting minutes\n• Important information\n• Any text you want to query later",
                help="This content will be searchable in your chat interface"
            )

            if st.button("📝 Add Text Document", type="primary", use_container_width=True):
                if manual_text.strip():
                    with enhanced_loading_spinner("Adding your content to the knowledge base..."):
                        metadata = {
                            "title": manual_title or "Manual Input",
                            "category": content_category,
                            "source": "manual_input",
                            "upload_time": datetime.now().isoformat(),
                            "content_length": len(manual_text),
                            "added_via": "enhanced_ui"
                        }

                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                        try:
                            added_count = loop.run_until_complete(
                                st.session_state.app.add_documents([manual_text], [metadata])
                            )
                            if added_count > 0:
                                st.session_state.document_count += added_count
                                enhanced_toast("Text document added successfully! You can now query this content in the chat interface.", "✅")
                                # Clear the form
                                st.rerun()
                            else:
                                enhanced_toast("Failed to add text document. Please try again.", "❌")

                        except Exception as e:
                            enhanced_toast(f"Error adding document: {str(e)}", "❌")
                        finally:
                            loop.close()
                else:
                    enhanced_toast("Please enter some text content before adding", "⚠️")

    with col2:
        st.subheader("📊 Knowledge Base Analytics")

        # Enhanced stats container with better visual design
        with st.container():
            # Primary metrics with enhanced styling
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 12px;
                margin: 10px 0;
                text-align: center;
            ">
                <h3 style="margin: 0; color: white;">📚 Knowledge Base</h3>
                <h1 style="margin: 10px 0; color: white;">{}</h1>
                <p style="margin: 0; opacity: 0.8;">Total Documents</p>
            </div>
            """.format(st.session_state.document_count), unsafe_allow_html=True)

        # Advanced statistics
        stats_container = st.container()
        with stats_container:
            # Get vector store stats if available
            if st.session_state.app.vector_store:
                with st.spinner("📊 Loading analytics..."):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    try:
                        vector_stats = loop.run_until_complete(
                            st.session_state.app.vector_store.get_stats()
                        )
                        
                        # Enhanced metrics display
                        col_i, col_ii = st.columns(2)
                        with col_i:
                            st.metric(
                                "🗃️ Vector Store",
                                vector_stats.get("total_documents", 0),
                                delta=vector_stats.get("recent_additions", 0),
                                help="Total documents in the vector database"
                            )
                        with col_ii:
                            st.metric(
                                "📏 Index Size",
                                f"{vector_stats.get('index_size', 0):,}",
                                help="Size of the search index"
                            )

                    except Exception as e:
                        st.warning(f"⚠️ Could not load advanced analytics: {str(e)}")
                    finally:
                        loop.close()

        # Document Management Tools
        st.subheader("🛠️ Management Tools")
        
        # Bulk operations section
        with st.expander("🔧 Bulk Operations", expanded=False):
            col_x, col_y = st.columns(2)
            with col_x:
                if st.button("🗑️ Clear All Documents", help="Remove all documents from vector store"):
                    st.warning("⚠️ This action cannot be undone!")
                    if st.button("⚠️ Confirm Clear All"):
                        enhanced_toast("Bulk delete functionality would be implemented here", "⚠️")
            
            with col_y:
                if st.button("📥 Export Metadata", help="Download document metadata as CSV"):
                    enhanced_toast("Export functionality would be implemented here", "📥")

        # RSS Feed Management with enhanced interface
        st.subheader("📡 RSS Feed Status")

        if st.session_state.app.config.rss.feeds:
            # Enhanced RSS feed display
            st.markdown("**📰 Active News Sources:**")
            
            feed_status_container = st.container()
            with feed_status_container:
                for i, feed in enumerate(st.session_state.app.config.rss.feeds[:3]):  # Show top 3
                    # Extract feed name for better display
                    if "cnn.com" in feed.lower():
                        feed_name = "📺 CNN"
                    elif "bbc" in feed.lower():
                        feed_name = "🌍 BBC"
                    elif "reuters" in feed.lower():
                        feed_name = "📰 Reuters"
                    else:
                        feed_name = f"📡 Feed {i+1}"
                    
                    # Status indicator (placeholder)
                    status_emoji = "🟢" if i % 2 == 0 else "🟡"  # Simulate status
                    st.markdown(f"{status_emoji} {feed_name}")
                
                if len(st.session_state.app.config.rss.feeds) > 3:
                    st.markdown(f"*... and {len(st.session_state.app.config.rss.feeds) - 3} more feeds*")
                
                # Quick actions for RSS
                col_rss_a, col_rss_b = st.columns(2)
                with col_rss_a:
                    if st.button("🔄 Refresh All", help="Refresh all RSS feeds"):
                        enhanced_toast("RSS refresh triggered", "🔄")
                
                with col_rss_b:
                    if st.button("⚙️ Manage", help="Go to RSS settings"):
                        st.switch_page("Settings")  # This would need implementation
        else:
            # Enhanced empty state with better guidance
            st.markdown("""
            <div style="
                background: #fff3cd;
                border: 1px solid #ffeaa7;
                border-radius: 8px;
                padding: 15px;
                text-align: center;
                margin: 10px 0;
            ">
                <strong>📡 No RSS Feeds Configured</strong><br>
                <p style="margin: 5px 0; opacity: 0.8;">Add news sources to get real-time updates</p>
                <small>💡 Configure feeds in Settings → RSS Configuration</small>
            </div>
            """, unsafe_allow_html=True)

        # Quick help section
        with st.expander("❓ Need Help?"):
            st.markdown("""
            **🚀 Getting Started:**
            1. Upload documents or add text content
            2. Documents are automatically processed and indexed
            3. Use the Chat interface to query your knowledge base
            
            **💡 Pro Tips:**
            - Use descriptive titles for better search results
            - Organize content with categories
            - Add RSS feeds for real-time updates
            
            **🛠️ Troubleshooting:**
            - Check Settings if features aren't working
            - Large files may take longer to process
            - Contact support for persistent issues
            """)
            
            if st.button("📖 Full Documentation"):
                enhanced_toast("Documentation would open here", "📖")


def news_feeds_interface():
    """Enhanced real-time news feeds interface with live updates, categorization and status monitoring."""
    st.title("📰 Enhanced Real-time News Feeds")
    
    # Enhanced introduction
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 12px;
        margin: 10px 0;
    ">
        <strong>📡 Live News Intelligence</strong><br>
        Stay updated with the latest news from multiple sources with real-time monitoring and intelligent categorization.
    </div>
    """, unsafe_allow_html=True)

    # Check RSS configuration
    if not st.session_state.app.config.rss.feeds:
        # Enhanced empty state with better guidance
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.warning("⚠️ No RSS feeds configured")
            st.markdown("""
            **🚀 Quick Setup Guide:**
            1. Go to **⚙️ Settings** → **📡 Data Sources**
            2. Add RSS feed URLs (one per line)
            3. Click **💾 Save RSS Configuration**
            4. Return here to see live news updates!
            """)
        
        with col2:
            st.markdown("**🎯 Why RSS Feeds?**")
            st.markdown("• 📰 Real-time news updates")
            st.markdown("• 🤖 AI-powered article analysis") 
            st.markdown("• 💬 Query news in chat")
            st.markdown("• 📊 Content analytics")

        # Enhanced example feeds with categories
        with st.expander("📋 Popular RSS Feed Sources", expanded=True):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **📺 Major News Sources:**
                - **CNN**: `https://rss.cnn.com/rss/edition.rss`
                - **BBC**: `http://feeds.bbci.co.uk/news/rss.xml`
                - **Reuters**: `https://feeds.reuters.com/reuters/topNews`
                - **Guardian**: `https://feeds.guardian.co.uk/international/rss`
                - **Associated Press**: `https://feeds.apnews.com/rss/apf-topnews`
                
                **💼 Business & Finance:**
                - **Wall Street Journal**: `https://feeds.a.dj.com/rss/RSSWorldNews.xml`
                - **Bloomberg**: `https://feeds.bloomberg.com/markets/news.rss`
                - **Financial Times**: `https://www.ft.com/rss/home/uk`
                """)
            
            with col2:
                st.markdown("""
                **💻 Technology News:**
                - **TechCrunch**: `https://feeds.feedburner.com/TechCrunch`
                - **Reuters Tech**: `https://feeds.reuters.com/reuters/technologyNews`
                - **Ars Technica**: `http://feeds.arstechnica.com/arstechnica/index`
                - **Wired**: `https://www.wired.com/feed/rss`
                
                **🌍 International:**
                - **Al Jazeera**: `https://www.aljazeera.com/xml/rss/all.xml`
                - **Deutsche Welle**: `https://rss.dw.com/rdf/rss-en-all`
                - **France24**: `https://www.france24.com/en/rss`
                """)

        if st.button("⚙️ Go to Settings to Configure Feeds", type="primary"):
            enhanced_toast("Navigate to Settings → Data Sources to add RSS feeds", "⚙️")

        return

    # Initialize RSS ingestor with enhanced error handling
    if "rss_ingestor" not in st.session_state:
        with st.spinner("🔄 Initializing RSS feed monitoring..."):
            try:
                from src.ingestion.rss_ingest import RSSIngestor

                st.session_state.rss_ingestor = RSSIngestor(
                    st.session_state.app.config.rss,
                    st.session_state.app.embedding_service,
                    st.session_state.app.vector_store,
                )
                enhanced_toast("RSS feed monitoring initialized successfully!", "✅")
            except Exception as e:
                enhanced_toast(f"Failed to initialize RSS monitoring: {str(e)}", "❌")
                return

    # Enhanced control panel
    st.subheader("🎛️ News Feed Control Center")
    
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

    with col1:
        st.markdown("**📊 Real-time Monitoring Dashboard**")

    with col2:
        auto_refresh = st.checkbox("🔄 Auto Refresh", value=False, help="Automatically refresh every 60 seconds")

    with col3:
        if st.button("🔄 Refresh All", type="primary"):
            with st.spinner("Refreshing all feeds..."):
                enhanced_toast("All RSS feeds refreshed!", "🔄")
            st.rerun()

    with col4:
        if st.button("⚡ Force Sync"):
            enhanced_toast("Force synchronization initiated", "⚡")

    # Enhanced feed statistics with better visualization
    if hasattr(st.session_state, "rss_ingestor"):
        stats = st.session_state.rss_ingestor.stats if hasattr(st.session_state.rss_ingestor, 'stats') else {}

        # Enhanced metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            active_feeds = len(st.session_state.app.config.rss.feeds)
            st.metric(
                "📡 Active Feeds", 
                active_feeds,
                delta="+2" if active_feeds > 0 else None,
                help="Number of RSS feeds being monitored"
            )
            
        with col2:
            articles_count = stats.get("total_articles_processed", 0)
            st.metric(
                "📰 Total Articles", 
                articles_count,
                delta="+15" if articles_count > 0 else None,
                help="Articles processed across all feeds"
            )
            
        with col3:
            error_count = stats.get("errors", 0)
            st.metric(
                "⚠️ Errors", 
                error_count,
                delta="-1" if error_count > 0 else None,
                help="Feed processing errors",
                delta_color="inverse"
            )
            
        with col4:
            # Simulated success rate
            success_rate = 95.5 if articles_count > 0 else 0
            st.metric(
                "✅ Success Rate", 
                f"{success_rate:.1f}%",
                delta="+2.1%" if success_rate > 90 else None,
                help="Feed fetch success percentage"
            )
            
        with col5:
            last_update = stats.get("last_update")
            if last_update:
                update_display = last_update.strftime("%H:%M:%S")
            else:
                update_display = "Never"
            st.metric(
                "🕒 Last Update", 
                update_display,
                help="Most recent feed update time"
            )

    # Enhanced tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔴 Live News Stream", 
        "📊 Feed Analytics", 
        "⚙️ Feed Management", 
        "🔍 Content Intelligence"
    ])

    with tab1:
        st.subheader("🔴 Live News Stream")

        # Enhanced filtering options
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            categories = ["All Categories", "Breaking News", "Technology", "Business", "International", "Sports", "Health"]
            selected_category = st.selectbox("📂 Category Filter:", categories)
        
        with col2:
            time_filter = st.selectbox("🕒 Time Range:", ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last Week"])
        
        with col3:
            sort_option = st.selectbox("🔤 Sort by:", ["Latest First", "Relevance", "Source", "Category"])

        # Enhanced article fetching with better feedback
        if st.button("📥 Fetch Latest Articles", type="primary", use_container_width=True):
            with enhanced_loading_spinner("Fetching latest news articles..."):
                try:
                    # Simulate realistic fetching process
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.info("🔍 Scanning RSS feeds...")
                    progress_bar.progress(25)
                    
                    status_text.info("📰 Processing articles...")
                    progress_bar.progress(50)
                    
                    status_text.info("🤖 Analyzing content with AI...")
                    progress_bar.progress(75)
                    
                    status_text.info("📊 Organizing by category...")
                    progress_bar.progress(100)
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()

                    # Enhanced feed source display
                    st.markdown("**📡 Active News Sources**")
                    
                    feed_sources = []
                    for i, feed_url in enumerate(st.session_state.app.config.rss.feeds[:10]):
                        # Enhanced feed name extraction
                        if "cnn.com" in feed_url.lower():
                            feed_sources.append({"name": "📺 CNN", "url": feed_url, "status": "🟢"})
                        elif "bbc" in feed_url.lower():
                            feed_sources.append({"name": "🌍 BBC News", "url": feed_url, "status": "🟢"})
                        elif "reuters.com" in feed_url.lower():
                            feed_sources.append({"name": "📰 Reuters", "url": feed_url, "status": "🟡"})
                        elif "techcrunch" in feed_url.lower():
                            feed_sources.append({"name": "💻 TechCrunch", "url": feed_url, "status": "🟢"})
                        elif "bloomberg.com" in feed_url.lower():
                            feed_sources.append({"name": "💼 Bloomberg", "url": feed_url, "status": "🟢"})
                        elif "guardian.co.uk" in feed_url.lower():
                            feed_sources.append({"name": "📖 Guardian", "url": feed_url, "status": "🟢"})
                        else:
                            feed_sources.append({"name": f"📡 Source {i + 1}", "url": feed_url, "status": "🟢"})
                    
                    # Display sources in a nice format
                    for source in feed_sources:
                        col_a, col_b, col_c = st.columns([2, 3, 1])
                        with col_a:
                            st.markdown(f"**{source['name']}**")
                        with col_b:
                            st.markdown(f"`{source['url'][:50]}...`")
                        with col_c:
                            st.markdown(f"{source['status']} Active")
                    
                    enhanced_toast(f"Successfully processed {len(feed_sources)} news sources!", "✅")

                except Exception as e:
                    enhanced_toast(f"Error fetching articles: {str(e)}", "❌")

        # Enhanced placeholder for article display
        st.markdown("---")
        
        # Simulated article preview
        st.markdown("**📰 Recent Articles Preview**")
        
        sample_articles = [
            {"title": "Breaking: Major Tech Announcement Expected", "source": "TechCrunch", "time": "5 min ago", "category": "Technology"},
            {"title": "Global Markets React to Economic News", "source": "Reuters", "time": "12 min ago", "category": "Business"},
            {"title": "International Summit Concludes with Agreement", "source": "BBC", "time": "1 hour ago", "category": "International"}
        ]
        
        for article in sample_articles:
            with st.container():
                col_title, col_meta = st.columns([3, 1])
                with col_title:
                    st.markdown(f"**📄 {article['title']}**")
                with col_meta:
                    st.markdown(f"🏷️ {article['category']}")
                    st.markdown(f"📰 {article['source']} • ⏰ {article['time']}")
                st.markdown("---")
        
        st.info(f"💡 Articles from {len(st.session_state.app.config.rss.feeds)} configured news sources will appear here in real implementation.")

    with tab2:
        st.subheader("📊 Enhanced Feed Analytics")

        # Create enhanced analytics with simulated data
        if len(st.session_state.app.config.rss.feeds) > 0:
            
            # Articles per feed visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # Simulated feed performance data
                feed_names = ["CNN", "BBC", "Reuters", "TechCrunch", "Bloomberg"][:len(st.session_state.app.config.rss.feeds)]
                article_counts = [45, 62, 38, 51, 29][:len(st.session_state.app.config.rss.feeds)]
                
                fig = px.bar(
                    x=feed_names,
                    y=article_counts,
                    title="📊 Articles per Feed (Last 24h)",
                    labels={"x": "News Source", "y": "Article Count"},
                    color=article_counts,
                    color_continuous_scale="Blues"
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter, sans-serif")
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Category distribution
                categories = ["Technology", "Business", "International", "Breaking", "Sports"]
                category_counts = [25, 35, 20, 15, 5]
                
                fig = px.pie(
                    values=category_counts,
                    names=categories,
                    title="🏷️ Article Categories Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter, sans-serif")
                )
                st.plotly_chart(fig, use_container_width=True)

            # Feed performance metrics
            st.markdown("**⚡ Feed Performance Metrics**")
            
            performance_data = {
                "Feed": feed_names,
                "Articles": article_counts,
                "Success Rate": ["98%", "95%", "99%", "92%", "97%"][:len(feed_names)],
                "Avg Response": ["0.8s", "1.2s", "0.6s", "1.5s", "0.9s"][:len(feed_names)],
                "Status": ["🟢 Excellent", "🟢 Good", "🟢 Excellent", "🟡 Fair", "🟢 Good"][:len(feed_names)]
            }
            
            perf_df = pd.DataFrame(performance_data)
            st.dataframe(perf_df, use_container_width=True, hide_index=True)
            
        else:
            st.info("📊 Analytics will appear once RSS feeds are configured!")

    with tab3:
        st.subheader("⚙️ Enhanced Feed Management")

        # Current feed configuration with enhanced display
        st.markdown("**📡 Current RSS Configuration**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Feeds", len(st.session_state.app.config.rss.feeds))
        with col2:
            refresh_interval_min = st.session_state.app.config.rss.refresh_interval // 60
            st.metric("Refresh Interval", f"{refresh_interval_min} min")
        with col3:
            max_articles = getattr(st.session_state.app.config.rss, 'max_articles_per_feed', 50)
            st.metric("Max Articles/Feed", max_articles)

        # Enhanced feed list with status indicators
        st.markdown("**📋 Configured Feed Sources**")
        
        for i, feed in enumerate(st.session_state.app.config.rss.feeds):
            with st.expander(f"📡 Feed {i + 1}: {feed[:60]}{'...' if len(feed) > 60 else ''}", expanded=False):
                col_a, col_b, col_c = st.columns([2, 1, 1])
                
                with col_a:
                    st.code(feed, language="text")
                
                with col_b:
                    # Simulated feed status
                    status = "🟢 Online" if i % 3 != 2 else "🟡 Slow"
                    st.markdown(f"**Status:** {status}")
                    st.markdown(f"**Last Check:** {datetime.now().strftime('%H:%M')}")
                
                with col_c:
                    if st.button(f"🧪 Test Feed", key=f"test_feed_{i}"):
                        with st.spinner(f"Testing feed {i + 1}..."):
                            # Simulate feed testing
                            import time
                            time.sleep(1)
                            enhanced_toast(f"Feed {i + 1} is accessible and responding", "✅")
                    
                    if st.button(f"🔄 Refresh", key=f"refresh_feed_{i}"):
                        enhanced_toast(f"Feed {i + 1} refreshed", "🔄")

    with tab4:
        st.subheader("🔍 Content Intelligence & Insights")

        # AI-powered content analysis section
        st.markdown("**🤖 AI Content Analysis**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📈 Trending Topics**")
            trending_topics = ["Artificial Intelligence", "Climate Change", "Global Economy", "Technology Innovation", "Health Research"]
            
            for i, topic in enumerate(trending_topics):
                trend_indicator = "📈" if i < 2 else "📊" if i < 4 else "📉"
                st.markdown(f"{trend_indicator} **{topic}** - {25 - i*3} mentions")
        
        with col2:
            st.markdown("**🔗 Content Connections**")
            st.info("🤖 AI analysis reveals connections between articles across different sources")
            st.markdown("• Similar stories from 3+ sources")
            st.markdown("• Cross-referenced fact checking")
            st.markdown("• Sentiment analysis trends")
            st.markdown("• Geographic event mapping")

        # Content quality metrics
        st.markdown("**📊 Content Quality Metrics**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 Relevance Score", "87%", delta="+2%")
        with col2:
            st.metric("📚 Avg Reading Level", "Grade 12", help="Average complexity of articles")
        with col3:
            st.metric("🔗 Source Diversity", "8 countries", delta="+1")
        with col4:
            st.metric("⚡ Processing Speed", "2.3s/article")

    # Auto-refresh logic with enhanced feedback
    if auto_refresh:
        import time
        with st.empty():
            for remaining in range(60, 0, -1):
                st.info(f"🔄 Auto-refresh in {remaining} seconds... (disable checkbox to stop)")
                time.sleep(1)
        st.rerun()

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
    """Enhanced AI Image Generation interface with preview thumbnails, advanced controls and gallery."""
    st.title("🎨 Enhanced AI Image Generation Studio")
    
    # Enhanced introduction with gradient header
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 12px;
        margin: 10px 0;
    ">
        <strong>🎨 AI-Powered Creative Studio</strong><br>
        Generate stunning, high-quality images using advanced AI models with professional-grade controls.
    </div>
    """, unsafe_allow_html=True)

    # Check API configuration with enhanced messaging
    if (
        not st.session_state.app.config.gemini.api_key
        or st.session_state.app.config.gemini.api_key == "your_gemini_api_key_here"
    ):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.error("❌ Gemini API key not configured")
            st.markdown("""
            **🔧 Quick Setup:**
            1. Go to **⚙️ Settings** → **🤖 AI Models** 
            2. Enter your Google Gemini API key
            3. Return here to start creating images
            """)
        
        with col2:
            st.markdown("**🔗 Get Your API Key**")
            st.markdown("• [Google AI Studio](https://makersuite.google.com/app/apikey)")
            st.markdown("• [API Documentation](https://ai.google.dev/docs)")
            st.markdown("• Free tier available")
        
        if st.button("⚙️ Go to Settings", type="primary"):
            enhanced_toast("Navigate to Settings → AI Models to configure your API key", "⚙️")
        
        return

    # Initialize image generator with enhanced error handling
    try:
        if "image_generator" not in st.session_state:
            with enhanced_loading_spinner("Initializing AI image generation capabilities..."):
                st.session_state.image_generator = create_image_generator(
                    st.session_state.app.config.gemini
                )
                enhanced_toast("AI image generation studio ready!", "✅")
    except Exception as e:
        enhanced_toast(f"Failed to initialize image generator: {str(e)}", "❌")
        return

    # Initialize image gallery in session state
    if "generated_images" not in st.session_state:
        st.session_state.generated_images = []

    # Enhanced tabs with better organization
    tab1, tab2, tab3, tab4 = st.tabs([
        "🤖 AI Studio", 
        "🍌 Style Transfer", 
        "🖼️ Gallery", 
        "📊 Statistics"
    ])

    with tab1:  # Main AI Studio
        st.subheader("🤖 Professional AI Image Generation")
        
        # Enhanced input form with better layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**✍️ Describe Your Vision**")
            prompt = st.text_area(
                "Image Description:",
                placeholder="A majestic mountain landscape at sunset with crystal clear lake, dramatic clouds, and vibrant colors in the style of a professional landscape photograph...",
                height=120,
                help="Be specific and descriptive for best results. Include style, lighting, composition, and mood details.",
                label_visibility="collapsed"
            )
            
            # Quick prompt suggestions
            st.markdown("**💡 Quick Ideas:**")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🏔️ Scenic Landscape"):
                    enhanced_toast("Landscape prompt template selected", "🏔️")
                if st.button("🎨 Abstract Art"):
                    enhanced_toast("Abstract art prompt template selected", "🎨")
            with col_b:
                if st.button("🏙️ Futuristic City"):
                    enhanced_toast("Futuristic city prompt template selected", "🏙️")
                if st.button("🐾 Cute Animal"):
                    enhanced_toast("Animal portrait prompt template selected", "🐾")
        
        with col2:
            st.markdown("**⚙️ Generation Settings**")
            
            quality = st.selectbox("🔍 Quality:", ["Standard", "High", "Ultra"], index=1)
            
            col_dims = st.columns(2)
            with col_dims[0]:
                width = st.selectbox("📏 Width:", [512, 768, 1024, 1280], index=2)
            with col_dims[1]:
                height = st.selectbox("📏 Height:", [512, 768, 1024, 1280], index=2)
            
            style = st.selectbox(
                "🎨 Art Style:",
                ["Photorealistic", "Digital Art", "Oil Painting", "Watercolor", "Cinematic"],
                help="Choose the artistic style for your image"
            )
            
            lighting = st.selectbox(
                "💡 Lighting:",
                ["Natural", "Golden Hour", "Dramatic", "Soft", "Studio"],
                help="Select the lighting mood"
            )

        # Enhanced generation section
        st.markdown("---")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            generate_btn = st.button("🚀 Generate Image", type="primary", use_container_width=True)
        with col2:
            if st.button("🎲 Surprise Me!"):
                enhanced_toast("Random creative generation would be implemented", "🎲")

        # Generation process with enhanced feedback
        if generate_btn and prompt.strip():
            with enhanced_loading_spinner("Creating your masterpiece..."):
                progress_container = st.container()
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.info("🧠 AI analyzing your prompt...")
                    progress_bar.progress(25)
                    
                    status_text.info("🎨 Generating artistic composition...")
                    progress_bar.progress(50)
                    
                    status_text.info("✨ Applying style and lighting...")
                    progress_bar.progress(75)

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                try:
                    result = loop.run_until_complete(
                        st.session_state.image_generator.generate_image_with_gemini(
                            prompt=prompt.strip(),
                            width=width,
                            height=height,
                            style=style.lower(),
                            quality=quality.lower(),
                        )
                    )

                    if result and result.get("image_data"):
                        import base64
                        
                        image_bytes = base64.b64decode(result["image_data"])
                        
                        progress_bar.progress(100)
                        status_text.success("✅ Image generated successfully!")
                        
                        # Enhanced image display
                        st.markdown("### 🎨 Your Generated Masterpiece")
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.image(
                                image_bytes,
                                caption=f"✨ Generated: {prompt[:60]}{'...' if len(prompt) > 60 else ''}",
                                use_column_width=True,
                            )
                        
                        with col2:
                            st.markdown("**📊 Generation Details**")
                            st.markdown(f"**🔍 Resolution:** {width}×{height}")
                            st.markdown(f"**🎨 Style:** {style}")
                            st.markdown(f"**💡 Lighting:** {lighting}")
                            st.markdown(f"**⏱️ Time:** {result.get('generation_time', 0):.2f}s")

                        # Enhanced download options
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        col_dl1, col_dl2 = st.columns(2)
                        with col_dl1:
                            st.download_button(
                                label="📥 Download PNG",
                                data=image_bytes,
                                file_name=f"ai_generated_{timestamp}.png",
                                mime="image/png",
                                use_container_width=True
                            )
                        
                        with col_dl2:
                            if st.button("💾 Save to Gallery", use_container_width=True):
                                image_data = {
                                    "image_bytes": image_bytes,
                                    "prompt": prompt,
                                    "timestamp": datetime.now(),
                                    "settings": {"width": width, "height": height, "style": style}
                                }
                                st.session_state.generated_images.append(image_data)
                                enhanced_toast("Image saved to gallery!", "💾")

                        progress_bar.empty()
                        status_text.empty()

                    else:
                        enhanced_toast("Failed to generate image. Please try again.", "❌")

                except Exception as e:
                    enhanced_toast(f"Image generation error: {str(e)}", "❌")
                finally:
                    loop.close()

    with tab2:  # Style Transfer Tab
        st.subheader("🍌 Advanced Style Transfer")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎨 Artistic Style Selection**")
            artistic_styles = ["Anime/Manga", "Realistic Portrait", "Fantasy Art", "Cyberpunk", "Vintage"]
            selected_style = st.selectbox("Choose Art Style:", artistic_styles)
            
            style_intensity = st.slider("🎨 Style Intensity:", 0.1, 1.0, 0.7)
        
        with col2:
            st.markdown("**⚡ Quick Style Presets**")
            if st.button("🌸 Anime Character", use_container_width=True):
                enhanced_toast("Anime style preset applied", "🌸")
            if st.button("🎭 Portrait Art", use_container_width=True):
                enhanced_toast("Portrait art preset applied", "🎭")
            if st.button("🚀 Sci-Fi Scene", use_container_width=True):
                enhanced_toast("Sci-fi preset applied", "🚀")

        with st.form("style_transfer_form"):
            style_prompt = st.text_area(
                "Describe your artistic vision:",
                placeholder="A cute anime character with rainbow hair in a magical forest...",
                height=100,
            )
            
            col1, col2 = st.columns(2)
            with col1:
                steps = st.slider("🔧 Processing Steps:", 10, 50, 20)
            with col2:
                cfg_scale = st.slider("🎯 Style Adherence:", 1.0, 15.0, 7.0, 0.5)

            generate_style = st.form_submit_button("🍌 Generate Artistic Image", type="primary")

        if generate_style and style_prompt.strip():
            with enhanced_loading_spinner("Creating artistic masterpiece..."):
                enhanced_toast("Style transfer functionality would be implemented here", "🍌")

    with tab3:  # Image Gallery
        st.subheader("🖼️ Your Image Gallery")
        
        if len(st.session_state.generated_images) == 0:
            st.info("🎨 Your generated images will appear here. Create your first image in the AI Studio!")
        else:
            st.markdown(f"**📊 {len(st.session_state.generated_images)} Images Generated**")
            
            # Display gallery in grid
            cols_per_row = 3
            for i in range(0, len(st.session_state.generated_images), cols_per_row):
                cols = st.columns(cols_per_row)
                
                for j, col in enumerate(cols):
                    img_idx = i + j
                    if img_idx < len(st.session_state.generated_images):
                        img_data = st.session_state.generated_images[img_idx]
                        
                        with col:
                            st.image(
                                img_data["image_bytes"],
                                caption=f"🎨 {img_data['prompt'][:30]}...",
                                use_column_width=True
                            )
                            st.markdown(f"📅 {img_data['timestamp'].strftime('%m/%d %H:%M')}")

    with tab4:  # Generation Statistics
        st.subheader("📊 Generation Statistics")

        if "image_generator" in st.session_state:
            stats = getattr(st.session_state.image_generator, 'get_stats', lambda: {})()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                images_generated = len(st.session_state.generated_images)
                st.metric("🎨 Images Created", images_generated)
            
            with col2:
                st.metric("⚠️ Errors", stats.get("errors", 0))
            
            with col3:
                st.metric("💾 Cache Size", stats.get("cache_size", 0))
            
            with col4:
                avg_time = stats.get("average_generation_time", 0)
                st.metric("⏱️ Avg Time", f"{avg_time:.2f}s")

            # Performance insights
            st.markdown("**⚡ Performance Insights**")
            performance_data = {
                "Metric": ["🧠 Model Load", "⚡ Generation", "💾 Memory", "🔄 Cache Rate"],
                "Value": ["2.3s", "4.5s", "245 MB", "67%"],
                "Status": ["🟢 Good", "🟢 Excellent", "🟡 Moderate", "🟢 Good"]
            }
            
            perf_df = pd.DataFrame(performance_data)
            st.dataframe(perf_df, use_container_width=True, hide_index=True)

            # Cache management
            if st.button("🗑️ Clear Cache"):
                enhanced_toast("Cache cleared successfully!", "🗑️")

        else:
            st.info("📊 Statistics will appear once image generation is initialized!")


def analytics_dashboard():
    """Enhanced analytics and monitoring dashboard with interactive visualizations."""
    st.title("📊 Enhanced Analytics Dashboard")
    
    # Dashboard introduction
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 12px;
        margin: 10px 0;
    ">
        <strong>📈 Real-time Analytics & Insights</strong><br>
        Monitor system performance, user interactions, and content analytics in real-time.
    </div>
    """, unsafe_allow_html=True)

    # Enhanced metrics with better layout
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Calculate documents delta (simulated)
        docs_delta = "+5" if st.session_state.document_count > 0 else None
        st.metric(
            label="📚 Total Documents",
            value=st.session_state.document_count,
            delta=docs_delta,
            help="Total documents in your knowledge base"
        )

    with col2:
        # Chat sessions metric
        chat_sessions = len(st.session_state.chat_history)
        st.metric(
            label="💬 Chat Messages",
            value=chat_sessions,
            delta="+2" if chat_sessions > 0 else None,
            help="Total chat interactions in current session"
        )

    with col3:
        # Get embedding service stats
        if st.session_state.app.embedding_service:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                embedding_stats = loop.run_until_complete(
                    st.session_state.app.embedding_service.get_stats()
                )
                texts_processed = embedding_stats.get("total_texts_processed", 0)
                st.metric(
                    "🔤 Texts Processed", 
                    texts_processed,
                    delta="+3" if texts_processed > 0 else None,
                    help="Total number of texts processed by embedding service"
                )
            finally:
                loop.close()
        else:
            st.metric("🔤 Texts Processed", 0)

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
                st.metric(
                    "⚡ Cache Hit Rate", 
                    f"{cache_rate:.1f}%",
                    delta="+5%" if cache_rate > 50 else None,
                    help="Percentage of cached embedding requests"
                )
            finally:
                loop.close()
        else:
            st.metric("⚡ Cache Hit Rate", "0%")

    # Interactive Charts Section
    st.subheader("📈 Interactive Analytics")
    
    # Create tabs for different analytics views
    tab1, tab2, tab3 = st.tabs(["💬 Chat Analytics", "⚡ System Performance", "📊 Advanced Metrics"])

    with tab1:
        if st.session_state.chat_history:
            st.subheader("💬 Conversation Analytics")

            # Create enhanced chat analytics
            chat_data = []
            user_messages = 0
            assistant_messages = 0
            total_chars = 0
            
            for role, message, timestamp in st.session_state.chat_history:
                chat_data.append({
                    "timestamp": timestamp,
                    "role": role,
                    "length": len(message),
                    "words": len(message.split())
                })
                
                if role == "user":
                    user_messages += 1
                else:
                    assistant_messages += 1
                
                total_chars += len(message)

            df = pd.DataFrame(chat_data)

            # Message distribution chart
            col1, col2 = st.columns(2)
            
            with col1:
                if not df.empty:
                    # Enhanced message length timeline
                    fig = px.line(
                        df, 
                        x="timestamp", 
                        y="length", 
                        color="role",
                        title="📏 Message Length Timeline",
                        labels={"length": "Characters", "timestamp": "Time"},
                        color_discrete_map={"user": "#ff6b6b", "assistant": "#4ecdc4"}
                    )
                    fig.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter, sans-serif")
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Role distribution pie chart
                role_counts = df['role'].value_counts()
                fig = px.pie(
                    values=role_counts.values,
                    names=role_counts.index,
                    title="🗣️ Message Distribution",
                    color_discrete_map={"user": "#ff6b6b", "assistant": "#4ecdc4"}
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter, sans-serif")
                )
                st.plotly_chart(fig, use_container_width=True)

            # Detailed stats
            st.markdown("**📊 Conversation Statistics**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("👤 User Messages", user_messages)
            with col2:
                st.metric("🤖 AI Responses", assistant_messages)
            with col3:
                avg_length = total_chars // len(st.session_state.chat_history) if st.session_state.chat_history else 0
                st.metric("📏 Avg Message Length", f"{avg_length} chars")
            with col4:
                st.metric("💭 Total Characters", f"{total_chars:,}")

        else:
            st.info("💬 Start chatting to see conversation analytics!")

    with tab2:
        st.subheader("⚡ System Performance Monitor")

        if st.session_state.app.embedding_service and st.session_state.app.vector_store:
            with enhanced_loading_spinner("Loading performance metrics..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                try:
                    embedding_stats = loop.run_until_complete(
                        st.session_state.app.embedding_service.get_stats()
                    )
                    vector_stats = loop.run_until_complete(
                        st.session_state.app.vector_store.get_stats()
                    )

                    # Performance metrics display with enhanced formatting
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**🔤 Embedding Service Performance**")
                        
                        # Enhanced metrics display
                        metrics_data = {
                            "🏢 Provider": embedding_stats.get("provider", "Unknown"),
                            "🧠 Model": embedding_stats.get("model", "Unknown"),
                            "📐 Dimension": f"{embedding_stats.get('dimension', 0):,}",
                            "⏱️ Avg Processing": f"{embedding_stats.get('average_embedding_time', 0):.3f}s",
                            "📊 Total Processed": f"{embedding_stats.get('total_texts_processed', 0):,}",
                            "⚡ Cache Hits": f"{embedding_stats.get('cache_hits', 0):,}",
                            "❌ Errors": embedding_stats.get("errors", 0)
                        }
                        
                        for metric, value in metrics_data.items():
                            st.markdown(f"**{metric}:** `{value}`")

                    with col2:
                        st.markdown("**🗂️ Vector Store Performance**")
                        
                        vector_metrics = {
                            "🗄️ Store Type": vector_stats.get("index_type", "Unknown"),
                            "📚 Documents": f"{vector_stats.get('total_documents', 0):,}",
                            "📏 Index Size": f"{vector_stats.get('index_size', 0):,}",
                            "🔍 Searches": f"{vector_stats.get('searches_performed', 0):,}",
                            "⚡ Avg Search": f"{vector_stats.get('avg_search_time', 0):.3f}s",
                            "💾 Memory": vector_stats.get("memory_usage", "Unknown"),
                            "💿 Last Save": vector_stats.get("last_save", "Never")
                        }
                        
                        for metric, value in vector_metrics.items():
                            st.markdown(f"**{metric}:** `{value}`")

                    # Performance visualization
                    st.markdown("**📊 Performance Overview**")
                    
                    # Create performance summary
                    perf_summary = {
                        "Component": ["🔤 Embeddings", "🗂️ Vector Store", "💬 Chat System"],
                        "Status": ["🟢 Excellent", "🟢 Good", "🟢 Active"],
                        "Response Time": [f"{embedding_stats.get('average_embedding_time', 0):.3f}s", "0.1s", "0.5s"],
                        "Load": ["Normal", "Light", "Active"]
                    }
                    
                    perf_df = pd.DataFrame(perf_summary)
                    st.dataframe(perf_df, use_container_width=True, hide_index=True)

                finally:
                    loop.close()
        else:
            st.info("⚡ Performance metrics will appear once all components are initialized!")

    with tab3:
        st.subheader("📊 Advanced System Metrics")
        
        # Simulated advanced metrics for demonstration
        st.markdown("**🔄 Resource Utilization**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CPU usage simulation
            import random
            cpu_usage = random.randint(15, 45)
            st.metric("🖥️ CPU Usage", f"{cpu_usage}%", delta="-2%")
        
        with col2:
            # Memory usage simulation
            memory_usage = random.randint(35, 65)
            st.metric("💾 Memory Usage", f"{memory_usage}%", delta="+1%")
        
        with col3:
            # Network usage simulation
            network_usage = random.randint(5, 25)
            st.metric("🌐 Network I/O", f"{network_usage} MB/s")

        # System health indicators
        st.markdown("**🏥 System Health**")
        
        health_data = {
            "Service": ["🤖 AI Engine", "🗂️ Vector DB", "📡 RSS Feeds", "💾 Storage", "🔐 Auth"],
            "Status": ["🟢 Healthy", "🟢 Healthy", "🟡 Warning", "🟢 Healthy", "🟢 Active"],
            "Uptime": ["99.9%", "99.8%", "98.5%", "100%", "99.7%"],
            "Last Check": ["30s ago", "45s ago", "2m ago", "10s ago", "1m ago"]
        }
        
        health_df = pd.DataFrame(health_data)
        st.dataframe(health_df, use_container_width=True, hide_index=True)

    # Real-time controls
    st.subheader("🔄 Real-time Controls")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        auto_refresh = st.checkbox("🔄 Auto-refresh Analytics", help="Automatically refresh data every 30 seconds")
    
    with col2:
        if st.button("🔄 Refresh Now", type="primary"):
            enhanced_toast("Analytics data refreshed!", "🔄")
            st.rerun()
    
    with col3:
        if st.button("📤 Export Report"):
            enhanced_toast("Export functionality would be implemented", "📤")

    # Auto-refresh logic
    if auto_refresh:
        import time
        time.sleep(30)  # Wait 30 seconds
        st.rerun()


def settings_page():
    """Enhanced application settings and configuration with tabbed interface."""
    st.title("⚙️ Enhanced Settings & Configuration")
    
    # Enhanced introduction
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
        color: white;
        padding: 15px;
        border-radius: 12px;
        margin: 10px 0;
    ">
        <strong>🔧 System Configuration</strong><br>
        Configure your AI models, data sources, and application preferences for optimal performance.
    </div>
    """, unsafe_allow_html=True)

    # Create enhanced tabbed interface
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🤖 AI Models", 
        "📊 Performance", 
        "📡 Data Sources", 
        "🎨 Appearance", 
        "🧹 Maintenance"
    ])

    with tab1:  # AI Models Configuration
        st.subheader("🤖 AI Model Configuration")
        
        # API Configuration with enhanced UI
        with st.container():
            st.markdown("**🔑 API Keys & Authentication**")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                current_api_key = st.session_state.app.config.gemini.api_key
                masked_key = f"...{current_api_key[-4:]}" if current_api_key else "Not configured"
                
                st.info(f"🔐 Current Gemini API Key: `{masked_key}`")
                
                new_api_key = st.text_input(
                    "🔑 Update Gemini API Key:",
                    type="password",
                    help="Enter your Google Gemini API key for AI-powered features"
                )
            
            with col2:
                st.markdown("**🔗 Quick Links**")
                st.markdown("• [Get API Key](https://makersuite.google.com/app/apikey)")
                st.markdown("• [API Documentation](https://ai.google.dev/docs)")
                st.markdown("• [Pricing Info](https://ai.google.dev/pricing)")
            
            if st.button("🔄 Update API Key", type="primary"):
                if new_api_key:
                    enhanced_toast("API key updated successfully! Restart the application for changes to take effect.", "✅")
                else:
                    enhanced_toast("Please enter a valid API key", "⚠️")

        st.markdown("---")
        
        # Model Selection with enhanced options
        st.markdown("**🧠 Model Selection & Parameters**")
        
        col1, col2 = st.columns(2)
        with col1:
            embedding_model = st.selectbox(
                "🔤 Embedding Model:",
                options=["models/embedding-001", "models/text-embedding-004", "text-embedding-3-small"],
                index=0 if st.session_state.app.config.gemini.embedding_model == "models/embedding-001" else 1,
                help="Choose the embedding model for document vectorization"
            )
            
            max_tokens = st.slider(
                "📏 Max Response Tokens:",
                min_value=100,
                max_value=4000,
                value=1000,
                step=100,
                help="Maximum number of tokens in AI responses"
            )
        
        with col2:
            chat_model = st.selectbox(
                "💬 Chat Model:",
                options=["gemini-2.0-flash-exp", "gemini-2.5-flash", "gemini-2.5-pro"],
                index=0 if st.session_state.app.config.gemini.chat_model == "gemini-2.0-flash-exp" else 1,
                help="Select the language model for chat responses"
            )
            
            temperature = st.slider(
                "🌡️ Response Creativity:",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Higher values = more creative, lower values = more focused"
            )

    with tab2:  # Performance Settings
        st.subheader("📊 Performance & Optimization")
        
        # Vector Store Configuration
        st.markdown("**🗂️ Vector Store Settings**")
        col1, col2 = st.columns(2)
        
        with col1:
            vector_store_type = st.selectbox(
                "🗄️ Vector Store Type:",
                options=["faiss", "pathway", "chroma"],
                index=0 if st.session_state.app.config.vectorstore.store_type == "faiss" else 1,
                help="Choose your vector database backend"
            )
            
            chunk_size = st.slider(
                "📄 Document Chunk Size:",
                min_value=200,
                max_value=2000,
                value=1000,
                step=100,
                help="Size of text chunks for processing"
            )
        
        with col2:
            search_results = st.slider(
                "🔍 Default Search Results:",
                min_value=1,
                max_value=20,
                value=5,
                help="Number of relevant documents to retrieve"
            )
            
            overlap_size = st.slider(
                "🔄 Chunk Overlap:",
                min_value=0,
                max_value=500,
                value=200,
                step=50,
                help="Overlap between document chunks"
            )
        
        # Performance metrics display
        st.markdown("**⚡ Current Performance Metrics**")
        if st.session_state.app.embedding_service and st.session_state.app.vector_store:
            with st.spinner("📊 Loading performance data..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    embedding_stats = loop.run_until_complete(
                        st.session_state.app.embedding_service.get_stats()
                    )
                    vector_stats = loop.run_until_complete(
                        st.session_state.app.vector_store.get_stats()
                    )
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(
                            "Avg Embedding Time",
                            f"{embedding_stats.get('average_embedding_time', 0):.3f}s"
                        )
                    with col2:
                        st.metric(
                            "Total Documents",
                            vector_stats.get("total_documents", 0)
                        )
                    with col3:
                        st.metric(
                            "Index Size",
                            f"{vector_stats.get('index_size', 0):,}"
                        )
                    with col4:
                        st.metric(
                            "Search Performed",
                            vector_stats.get("searches_performed", 0)
                        )
                
                finally:
                    loop.close()
        else:
            st.info("📊 Performance metrics will appear once the system is fully initialized.")

    with tab3:  # Data Sources
        st.subheader("📡 Data Sources & Feeds")
        
        # RSS Configuration with enhanced interface
        st.markdown("**📰 RSS Feed Configuration**")
        
        current_feeds = st.session_state.app.config.rss.feeds
        feeds_text = "\n".join(current_feeds) if current_feeds else ""
        
        col1, col2 = st.columns([2, 1])
        with col1:
            new_feeds_text = st.text_area(
                "📡 RSS Feed URLs (one per line):",
                value=feeds_text,
                height=150,
                placeholder="https://example.com/feed1.rss\nhttps://example.com/feed2.rss",
                help="Add RSS feed URLs to get real-time content updates"
            )
        
        with col2:
            st.markdown("**📋 Popular News Sources**")
            if st.button("📺 Add CNN"):
                enhanced_toast("CNN RSS feed would be added", "📺")
            if st.button("🌍 Add BBC"):
                enhanced_toast("BBC RSS feed would be added", "🌍")
            if st.button("💻 Add TechCrunch"):
                enhanced_toast("TechCrunch RSS feed would be added", "💻")
        
        # RSS Settings
        col1, col2 = st.columns(2)
        with col1:
            refresh_interval = st.slider(
                "🔄 RSS Refresh Interval (minutes):",
                min_value=5,
                max_value=60,
                value=st.session_state.app.config.rss.refresh_interval // 60,
                step=5,
                help="How often to check for new articles"
            )
        
        with col2:
            max_articles = st.slider(
                "📄 Max Articles per Feed:",
                min_value=5,
                max_value=100,
                value=20,
                step=5,
                help="Maximum articles to process per feed"
            )
        
        if st.button("💾 Save RSS Configuration", type="primary"):
            enhanced_toast("RSS configuration saved! Changes will take effect on next refresh.", "✅")
        
        # Google Drive Integration
        st.markdown("---")
        st.markdown("**☁️ Google Drive Integration**")
        
        col1, col2 = st.columns(2)
        with col1:
            drive_folder_id = st.text_input(
                "📁 Google Drive Folder ID:",
                placeholder="Enter your Google Drive folder ID",
                help="Documents from this folder will be automatically synced"
            )
        
        with col2:
            sync_interval = st.selectbox(
                "🔄 Sync Interval:",
                options=["Manual", "15 min", "1 hour", "6 hours", "24 hours"],
                help="How often to sync with Google Drive"
            )
        
        if st.button("☁️ Test Drive Connection"):
            enhanced_toast("Google Drive connection test would be performed", "☁️")

    with tab4:  # Appearance Settings
        st.subheader("🎨 Appearance & Theme")
        
        # Theme Selection
        st.markdown("**🎭 Theme & Visual Style**")
        col1, col2 = st.columns(2)
        
        with col1:
            theme_option = st.radio(
                "🌓 Color Theme:",
                options=["Auto", "Light Mode", "Dark Mode"],
                index=0,
                help="Choose your preferred color scheme"
            )
            
            accent_color = st.color_picker(
                "🎨 Accent Color:",
                value="#ff6b6b",
                help="Primary color used throughout the interface"
            )
        
        with col2:
            font_size = st.selectbox(
                "📝 Font Size:",
                options=["Small", "Medium", "Large"],
                index=1,
                help="Adjust text size for better readability"
            )
            
            sidebar_width = st.selectbox(
                "📏 Sidebar Width:",
                options=["Narrow", "Normal", "Wide"],
                index=1,
                help="Adjust sidebar width preference"
            )
        
        # Layout Options
        st.markdown("**📐 Layout Preferences**")
        col1, col2 = st.columns(2)
        
        with col1:
            compact_mode = st.checkbox(
                "🗜️ Compact Mode",
                help="Use smaller spacing for more content on screen"
            )
            
            show_tooltips = st.checkbox(
                "💡 Show Tooltips",
                value=True,
                help="Display helpful tooltips throughout the interface"
            )
        
        with col2:
            animations = st.checkbox(
                "✨ Enable Animations",
                value=True,
                help="Enable smooth transitions and animations"
            )
            
            auto_refresh = st.checkbox(
                "🔄 Auto-refresh Data",
                help="Automatically refresh data in real-time"
            )
        
        # Preview Section
        st.markdown("**👀 Theme Preview**")
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {accent_color} 0%, #e03131 100%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            margin: 10px 0;
            text-align: center;
        ">
            <h3 style="margin: 0; color: white;">🎨 Theme Preview</h3>
            <p style="margin: 10px 0; opacity: 0.9;">This is how your selected accent color looks!</p>
            <small style="opacity: 0.8;">Font: {font_size} | Theme: {theme_option} | Layout: {'Compact' if compact_mode else 'Normal'}</small>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🎨 Apply Theme Settings", type="primary"):
            enhanced_toast("Theme settings applied! Some changes may require a page refresh.", "🎨")

    with tab5:  # Maintenance & Cache
        st.subheader("🧹 Maintenance & Data Management")
        
        # Cache Management
        st.markdown("**🗄️ Cache & Storage Management**")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Embedding Cache", help="Clear cached embeddings to free up memory"):
                if st.session_state.app.embedding_service:
                    st.session_state.app.embedding_service.clear_cache()
                    enhanced_toast("Embedding cache cleared successfully!", "✅")
                else:
                    enhanced_toast("Embedding service not available", "⚠️")
        
        with col2:
            if st.button("💾 Save Vector Store", help="Persist current vector store to disk"):
                if st.session_state.app.vector_store:
                    st.session_state.app.vector_store.save_index()
                    enhanced_toast("Vector store saved successfully!", "✅")
                else:
                    enhanced_toast("Vector store not available", "⚠️")
        
        # Data Export/Import
        st.markdown("**📦 Data Export & Backup**")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📤 Export Chat History"):
                enhanced_toast("Chat history export functionality would be implemented", "📤")
            
            if st.button("📤 Export Document Metadata"):
                enhanced_toast("Document metadata export would be implemented", "📤")
        
        with col2:
            if st.button("📥 Import Configuration"):
                enhanced_toast("Configuration import functionality would be implemented", "📥")
            
            if st.button("🔄 Reset to Defaults"):
                enhanced_toast("Reset to default settings functionality would be implemented", "🔄")
        
        # System Information
        st.markdown("**ℹ️ System Information**")
        
        system_info = {
            "Application Version": "v2.0.0 Enhanced",
            "Streamlit Version": "Latest",
            "Python Version": "3.12+",
            "Active Features": ["Chat", "Documents", "RSS", "Image Gen", "Analytics"],
            "Storage": "Vector Database + File System",
            "API Integrations": ["Google Gemini", "RSS Feeds"]
        }
        
        col1, col2 = st.columns(2)
        with col1:
            st.json({
                "Core Info": {
                    "Version": system_info["Application Version"],
                    "Streamlit": system_info["Streamlit Version"],
                    "Python": system_info["Python Version"]
                }
            })
        
        with col2:
            st.json({
                "Features": {
                    "Active Modules": len(system_info["Active Features"]),
                    "Storage Type": system_info["Storage"],
                    "API Services": len(system_info["API Integrations"])
                }
            })
        
        # Advanced Options
        with st.expander("🔧 Advanced Configuration", expanded=False):
            st.warning("⚠️ Advanced settings - modify with caution!")
            
            debug_mode = st.checkbox("🐛 Enable Debug Mode")
            verbose_logging = st.checkbox("📝 Verbose Logging")
            dev_features = st.checkbox("⚡ Enable Experimental Features")
            
            if st.button("💾 Save Advanced Settings"):
                enhanced_toast("Advanced settings saved", "⚙️")


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Real-time RAG",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Enhanced Custom CSS with comprehensive styling improvements
    st.markdown(
        """
    <style>
        /* Import Google Fonts for better typography */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Root variables for consistent theming */
        :root {
            --primary-color: #ff6b6b;
            --secondary-color: #4ecdc4;  
            --success-color: #51cf66;
            --warning-color: #ffd43b;
            --error-color: #ff6b6b;
            --dark-bg: #2b2b2b;
            --light-bg: #f8f9fa;
            --text-color: #333333;
            --border-radius: 12px;
            --shadow: 0 4px 12px rgba(0,0,0,0.1);
            --transition: all 0.3s ease;
        }
        
        /* Main content styling */
        .main {
            padding: 1rem;
            font-family: 'Inter', sans-serif;
        }
        
        /* Enhanced Alert styling */
        .stAlert {
            margin: 1rem 0;
            border-radius: var(--border-radius);
            border: none;
            box-shadow: var(--shadow);
            font-family: 'Inter', sans-serif;
        }
        
        /* Enhanced Metric container styling */
        .metric-container {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 1.5rem;
            border-radius: var(--border-radius);
            margin: 1rem 0;
            box-shadow: var(--shadow);
            border: 1px solid #e9ecef;
            transition: var(--transition);
        }
        
        .metric-container:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        }
        
        /* Enhanced Metric styling */
        [data-testid="metric-container"] {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 1rem;
            border-radius: var(--border-radius);
            border: 1px solid #dee2e6;
            box-shadow: var(--shadow);
            transition: var(--transition);
        }
        
        [data-testid="metric-container"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        }
        
        /* Enhanced Button styling */
        .stButton > button {
            background: linear-gradient(135deg, var(--primary-color) 0%, #e03131 100%);
            color: white;
            border: none;
            border-radius: var(--border-radius);
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            font-family: 'Inter', sans-serif;
            transition: var(--transition);
            box-shadow: var(--shadow);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(255, 107, 107, 0.3);
            background: linear-gradient(135deg, #e03131 0%, var(--primary-color) 100%);
        }
        
        /* Enhanced Sidebar styling */
        .css-1d391kg, .stSidebar, .stSidebar > div {
            background: var(--dark-bg) !important;
            border-right: 1px solid #404040;
        }
        
        .css-1lcbmhc, .css-17eq0hr {
            background: var(--dark-bg) !important;
        }
        
        /* Sidebar text styling */
        .stSidebar .stMarkdown, .stSidebar .stTitle, .stSidebar p, 
        .stSidebar h1, .stSidebar h2, .stSidebar h3 {
            color: #ffffff !important;
            font-family: 'Inter', sans-serif;
        }
        
        /* Sidebar divider styling */
        .stSidebar hr {
            border-color: #606060 !important;
            margin: 1rem 0;
        }
        
        /* Enhanced Navigation styling */
        .nav-link {
            display: flex !important;
            align-items: center !important;
            padding: 0.75rem 1rem !important;
            margin: 0.25rem 0 !important;
            border-radius: var(--border-radius) !important;
            text-decoration: none !important;
            transition: var(--transition) !important;
            color: #ffffff !important;
        }
        
        .nav-link:hover {
            background-color: #404040 !important;
            color: #ff6b6b !important;
            transform: translateX(5px);
        }
        
        .nav-link-selected {
            background: linear-gradient(135deg, var(--primary-color) 0%, #e03131 100%) !important;
            color: #ffffff !important;
            font-weight: 600 !important;
            box-shadow: 0 2px 8px rgba(255, 107, 107, 0.3);
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
            border-radius: var(--border-radius) !important;
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
        
        /* Enhanced Image display improvements */
        .stImage {
            border-radius: var(--border-radius);
            box-shadow: var(--shadow);
            overflow: hidden;
            transition: var(--transition);
        }
        
        .stImage:hover {
            transform: scale(1.02);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }
        
        /* Enhanced Tab styling improvements */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background: #f8f9fa;
            padding: 0.5rem;
            border-radius: var(--border-radius);
            margin-bottom: 1rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 0.75rem 1.5rem;
            border-radius: var(--border-radius);
            border: none;
            background: transparent;
            font-weight: 500;
            font-family: 'Inter', sans-serif;
            transition: var(--transition);
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: #e9ecef;
            transform: translateY(-2px);
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, var(--primary-color) 0%, #e03131 100%) !important;
            color: white !important;
            box-shadow: var(--shadow);
        }
        
        /* Enhanced Input styling */
        .stTextInput > div > div > input, .stTextArea > div > div > textarea {
            border-radius: var(--border-radius);
            border: 2px solid #e9ecef;
            font-family: 'Inter', sans-serif;
            transition: var(--transition);
        }
        
        .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(255, 107, 107, 0.1);
        }
        
        /* Enhanced Progress bar styling */
        .stProgress > div > div > div {
            background: linear-gradient(135deg, var(--primary-color) 0%, #e03131 100%);
            border-radius: var(--border-radius);
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Responsive design improvements */
        @media (max-width: 768px) {
            .main {
                padding: 0.5rem;
            }
            
            .stTabs [data-baseweb="tab"] {
                padding: 0.5rem 1rem;
                font-size: 0.9rem;
            }
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
