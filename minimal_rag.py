"""
Minimal RAG Streamlit Application for testing
"""

import streamlit as st
import sys
import os

# Add current directory to path
sys.path.insert(0, os.getcwd())

st.set_page_config(
    page_title="Real-time RAG",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Real-time RAG - Testing Version")

# Test basic imports
with st.expander("🧪 Import Tests", expanded=True):
    try:
        from src.config import get_config

        st.success("✅ Config imported successfully")

        # Try to get config
        config = get_config()
        st.write(f"Config loaded: {type(config)}")
    except Exception as e:
        st.error(f"❌ Config import failed: {e}")

    try:
        from src.embeddings.embedder import EmbeddingService

        st.success("✅ EmbeddingService imported successfully")
    except Exception as e:
        st.error(f"❌ EmbeddingService import failed: {e}")

    try:
        from src.vectorstores.faiss_store import FAISSVectorStore

        st.success("✅ FAISSVectorStore imported successfully")
    except Exception as e:
        st.error(f"❌ FAISSVectorStore import failed: {e}")

# Simple chat interface
st.subheader("💬 Simple Chat Test")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is your question?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = f"You said: {prompt}. This is a test response!"
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

st.sidebar.success("🎯 Minimal RAG app is running!")
