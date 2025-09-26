# Real-time RAG Playground

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37+-red.svg)](https://streamlit.io/)

A comprehensive real-time Retrieval-Augmented Generation (RAG) system that ingests data from multiple sources including RSS feeds, Google Drive, and local file monitoring, processes it through embeddings, and stores it in vector databases for efficient retrieval.

## ✨ Features

### 🔄 Multi-source Data Ingestion

- **RSS Feed Monitoring**: Real-time RSS feed processing with configurable refresh intervals
- **Google Drive Integration**: Automatic document syncing from Google Drive folders
- **File System Monitoring**: Watch directories for new or modified files

### 🗄️ Flexible Vector Storage

- **FAISS Vector Store**: High-performance local vector storage for development
- **Pathway Integration**: Real-time vector processing and streaming capabilities
- **Extensible Architecture**: Easy to add new vector store backends

### 🧹 Advanced Preprocessing Pipeline

- **Text Cleaning & Normalization**: Remove HTML, URLs, emails, and special characters
- **Intelligent Chunking**: Overlapping text chunks with configurable sizes
- **Metadata Extraction**: Preserve document metadata and timestamps

### 🤖 AI-Powered Features

- **Chat Interface**: Interactive conversations with your documents
- **Semantic Search**: Find relevant information using natural language queries
- **Context-Aware Responses**: LLM-powered answers with source citations

### 🎨 Modern Web Interface

- **Streamlit-based UI**: Clean, responsive web interface
- **Real-time Updates**: Live data ingestion and processing status
- **Interactive Exploration**: Browse documents and search results visually

## 📁 Project Structure

```text
anjali-project/
├── .env                    # Environment configuration (create from .env.example)
├── .env.example           # Environment variables template
├── Dockerfile            # Docker container configuration
├── docker-compose.yml    # Multi-service orchestration
├── LICENSE              # MIT License
├── README.md           # This documentation
├── pyproject.toml      # Python project configuration (in real_time_rag/)
├── requirements.txt    # Python dependencies
├── streamlit_app.py    # Main Streamlit application
├── minimal_rag.py      # Minimal RAG example
├── test_*.py          # Test files
├── src/               # Source code
│   ├── __init__.py
│   ├── config.py     # Configuration management
│   ├── main.py      # FastAPI application entry point
│   ├── utils/       # Utility modules
│   │   ├── __init__.py
│   │   └── logger.py # Structured logging configuration
│   ├── preprocessing/ # Data preprocessing
│   │   ├── __init__.py
│   │   └── cleaner.py # Text cleaning utilities
│   ├── ingestion/   # Data ingestion modules
│   │   ├── __init__.py
│   │   ├── rss_ingest.py     # RSS feed processing
│   │   ├── drive_ingest.py   # Google Drive integration
│   │   └── filewatch_ingest.py # File system monitoring
│   ├── embeddings/  # Embedding generation
│   │   ├── __init__.py
│   │   ├── embedder.py   # Embedding interface
│   │   └── typing.py     # Type definitions
│   ├── vectorstores/ # Vector storage backends
│   │   ├── __init__.py
│   │   ├── faiss_store.py     # FAISS implementation
│   │   └── pathway_client.py  # Pathway integration
│   ├── image_generation/ # Image generation services
│   │   ├── __init__.py
│   │   └── generator.py # Gemini image generation
│   └── langchain_wrapper/ # LangChain integration
│       ├── __init__.py
│       └── pathway_vectorstore.py # Custom vector store
├── real_time_rag/    # Alternative project structure
├── tests/           # Test suite
└── data/           # Data directory for file monitoring
```

## 🚀 Installation

### Prerequisites

- Python 3.11+
- Docker and Docker Compose (for containerized deployment)
- Git

### Quick Start

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/real-time-rag-playground.git
   cd real-time-rag-playground
   ```

2. Copy and configure environment variables:

   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

3. Build and run with Docker Compose:

   ```bash
   docker-compose up --build
   ```

### Manual Installation

1. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:

   ```bash
   streamlit run streamlit_app.py
   ```

### Development Setup

For development with hot reloading and additional tools:

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run with auto-reload
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Vector Store Configuration
VECTOR_STORE_TYPE=faiss  # Options: faiss, pathway
FAISS_INDEX_PATH=./data/faiss_index
PATHWAY_HOST=localhost
PATHWAY_PORT=8080

# Data Sources
RSS_FEEDS=https://example.com/feed.xml,https://another.com/rss
GOOGLE_DRIVE_FOLDER_ID=your_folder_id
WATCH_DIRECTORIES=./data/documents

# Application Settings
LOG_LEVEL=INFO
MAX_CHUNK_SIZE=1000
OVERLAP_SIZE=200
EMBEDDING_MODEL=text-embedding-ada-002
```

### Advanced Configuration

For production deployments, adjust these settings:

```bash
# Performance Tuning
BATCH_SIZE=32
MAX_WORKERS=4
CACHE_TTL=3600

# Security
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=localhost,127.0.0.1

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
```

## 📖 Usage

### RSS Feed Ingestion

Configure RSS feeds in your `.env` file:

```bash
RSS_FEEDS=https://example.com/feed1.rss,https://example.com/feed2.rss
RSS_REFRESH_INTERVAL=300
```

### Google Drive Integration

1. Set up Google API credentials
2. Configure the folder ID in `.env`:

   ```bash
   GOOGLE_DRIVE_FOLDER_ID=your_folder_id
   GOOGLE_CREDENTIALS_PATH=path/to/credentials.json
   ```

### File System Monitoring

Point the system to monitor a directory:

```bash
WATCH_DIRECTORY=./data
WATCH_EXTENSIONS=.txt,.md,.pdf
```

### API Usage

The system provides REST API endpoints:

```bash
# Query documents
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "your search query", "top_k": 5}'

# Ingest new documents
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{"documents": ["document content here"]}'
```

## 🧪 Testing

Run the test suite:

```bash
python -m pytest tests/
```

For specific module testing:

```bash
python -m pytest tests/test_rss.py -v
```

### Test Coverage

Generate coverage report:

```bash
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

## 🏗️ Architecture

### System Components

- **Ingestion Layer**: RSS feeds, Google Drive, file system monitoring
- **Processing Layer**: Text cleaning, chunking, embedding generation
- **Storage Layer**: FAISS or Pathway vector stores
- **Query Layer**: Semantic search with LLM-powered responses
- **API Layer**: FastAPI REST endpoints and Streamlit UI

### Data Flow

1. Data sources feed into ingestion modules
2. Documents are preprocessed and chunked
3. Embeddings are generated using OpenAI API
4. Vectors are stored in the configured vector store
5. Queries are processed through semantic search
6. Results are enhanced with LLM context

## 🔒 Security

- API keys stored securely in environment variables
- Input validation on all endpoints
- Rate limiting and authentication (planned)
- Secure credential management

## 🚀 Deployment

### Docker Deployment

```bash
docker-compose up --build -d
```

### Kubernetes Deployment

```bash
kubectl apply -f k8s/
```

### Cloud Deployment

- **AWS**: ECS/Fargate with RDS and S3
- **GCP**: Cloud Run with Firestore and Cloud Storage
- **Azure**: Container Apps with CosmosDB and Blob Storage

## 📊 Monitoring

### Metrics

- Query latency and throughput
- Ingestion success rates
- Vector store performance
- API endpoint health

### Logging

Structured logging with configurable levels:

```bash
LOG_LEVEL=DEBUG  # Options: DEBUG, INFO, WARNING, ERROR
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation
- Ensure all tests pass

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Embeddings provided by [OpenAI](https://openai.com/)
- Vector search powered by [FAISS](https://github.com/facebookresearch/faiss)
- Real-time processing with [Pathway](https://pathway.com/)
- UI built with [Streamlit](https://streamlit.io/)
