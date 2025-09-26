# ⚡ Real-time RAG Playground ⚡

*Ingest • Embed • Retrieve • Chat with your data in real-time*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37+-red.svg)](https://streamlit.io/)

---

## 📚 Table of Contents
- [✨ Features](#-features)
- [🏗️ Architecture](#-architecture)
- [📁 Project Structure](#-project-structure)
- [🚀 Installation](#-installation)
- [⚙️ Configuration](#-configuration)
- [💬 Usage](#-usage)
- [🧪 Testing](#-testing)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [🙏 Acknowledgments](#-acknowledgments)

---

## ✨ Features

| 🚀 Ingestion (Multi-Source) | 🧹 Preprocessing | 💾 Vector Storage | 🤖 AI Features |
|-----------------------------|-----------------|-----------------|----------------|
| RSS feeds, Google Drive, File System watch | Cleaning, normalization, metadata extraction, intelligent chunking | FAISS (local), Pathway (streaming), easily extensible | Chat UI, Semantic Search, Context-aware answers with citations |

![Demo](docs/demo.gif)

---

## 🏗️ Architecture

<details>
<summary>View Diagram</summary>

```mermaid
graph LR
    A[📡 Data Sources\n(RSS, Drive, Files)] --> B[🔄 Ingestion Layer]
    B --> C[🧹 Preprocessing & Chunking]
    C --> D[🪶 Embedding Generation]
    D --> E[💾 Vector Store\n(FAISS / Pathway)]
    E --> F[🤖 Query Layer & LLM]
    F --> G[💬 Streamlit UI / FastAPI API]
````

</details>

---

## 📁 Project Structure

```text
anjali-project/
├── .env                    # Environment configuration
├── Dockerfile              # Docker container setup
├── docker-compose.yml      # Multi-service orchestration
├── README.md               # This documentation
├── requirements.txt        # Python dependencies
├── streamlit_app.py        # Main Streamlit app
├── minimal_rag.py          # Minimal RAG example
├── src/
│   ├── config.py           # Configuration management
│   ├── main.py             # FastAPI entry point
│   ├── utils/logger.py     # Structured logging
│   ├── preprocessing/cleaner.py
│   ├── ingestion/rss_ingest.py
│   ├── ingestion/drive_ingest.py
│   ├── ingestion/filewatch_ingest.py
│   ├── embeddings/embedder.py
│   ├── vectorstores/faiss_store.py
│   ├── vectorstores/pathway_client.py
│   └── langchain_wrapper/pathway_vectorstore.py
└── data/                   # Data directory for file monitoring
```

---

## 🚀 Installation

### Prerequisites

* Python 3.11+
* Docker & Docker Compose
* Git

### Quick Start

```bash
git clone https://github.com/your-username/real-time-rag-playground.git
cd real-time-rag-playground
cp .env.example .env
# Edit .env with your API keys and settings
docker-compose up --build
```

### Manual Installation

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Development Setup (Auto-Reload)

```bash
pip install -r requirements-dev.txt
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

---

## ⚙️ Configuration

Create `.env` based on `.env.example`:

```bash
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

VECTOR_STORE_TYPE=faiss  # Options: faiss, pathway
FAISS_INDEX_PATH=./data/faiss_index
PATHWAY_HOST=localhost
PATHWAY_PORT=8080

RSS_FEEDS=https://example.com/feed.xml,https://another.com/rss
GOOGLE_DRIVE_FOLDER_ID=your_folder_id
WATCH_DIRECTORIES=./data/documents

LOG_LEVEL=INFO
MAX_CHUNK_SIZE=1000
OVERLAP_SIZE=200
EMBEDDING_MODEL=text-embedding-ada-002
```

---

## 💬 Usage

### RSS Feed Ingestion

```bash
RSS_FEEDS=https://example.com/feed1.rss,https://example.com/feed2.rss
RSS_REFRESH_INTERVAL=300
```

### Google Drive Integration

```bash
GOOGLE_DRIVE_FOLDER_ID=your_folder_id
GOOGLE_CREDENTIALS_PATH=path/to/credentials.json
```

### File System Monitoring

```bash
WATCH_DIRECTORY=./data
WATCH_EXTENSIONS=.txt,.md,.pdf
```

### API Usage

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "your search query", "top_k": 5}'

curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{"documents": ["document content here"]}'
```

---

## 🧪 Testing

```bash
python -m pytest tests/
python -m pytest tests/test_rss.py -v

# Coverage
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add feature'`)
4. Push (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Guidelines:** PEP 8, add tests, update documentation, ensure all tests pass.

---

## 📄 License

MIT License — see the [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgments

* Embeddings by [OpenAI](https://openai.com/)
* Vector search by [FAISS](https://github.com/facebookresearch/faiss)
* Real-time processing with [Pathway](https://pathway.com/)
* UI built with [Streamlit](https://streamlit.io/)

```

