# Real-time RAG Playground

A comprehensive real-time Retrieval-Augmented Generation (RAG) system that ingests data from multiple sources including RSS feeds, Google Drive, and local file monitoring, processes it through embeddings, and stores it in vector databases for efficient retrieval.

## Features

- **Multi-source Data Ingestion**
  - RSS feed monitoring with configurable refresh intervals
  - Google Drive integration for document processing
  - Local file system monitoring with real-time updates
  
- **Flexible Vector Storage**
  - FAISS vector store for local development
  - Pathway integration for streaming data processing
  - Configurable embedding models (OpenAI, etc.)

- **Preprocessing Pipeline**
  - Text cleaning and normalization
  - Document chunking and metadata extraction
  - Support for multiple file formats (.txt, .md, .pdf)

- **LangChain Integration**
  - Custom vector store wrappers
  - Seamless integration with LangChain ecosystem
  - Support for various retrieval strategies

## Project Structure

```
real-time-rag-playground/
├── .env.example              # Environment variables template
├── Dockerfile               # Docker container configuration
├── docker-compose.yml       # Multi-service orchestration
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── tests/                  # Test suite
│   └── test_rss.py        # RSS ingestion tests
└── src/                   # Source code
    ├── __init__.py
    ├── config.py          # Configuration management
    ├── main.py           # Application entry point
    ├── utils/            # Utility modules
    │   ├── __init__.py
    │   └── logger.py     # Logging configuration
    ├── preprocessing/    # Data preprocessing
    │   ├── __init__.py
    │   └── cleaner.py    # Text cleaning utilities
    ├── ingestion/        # Data ingestion modules
    │   ├── __init__.py
    │   ├── rss_ingest.py     # RSS feed processing
    │   ├── drive_ingest.py   # Google Drive integration
    │   └── filewatch_ingest.py # File system monitoring
    ├── embeddings/       # Embedding generation
    │   ├── __init__.py
    │   ├── embedder.py   # Embedding interface
    │   └── typing.py     # Type definitions
    ├── vectorstores/     # Vector storage backends
    │   ├── __init__.py
    │   ├── pathway_client.py  # Pathway integration
    │   └── faiss_store.py     # FAISS implementation
    └── langchain_wrapper/ # LangChain integration
        ├── __init__.py
        └── pathway_vectorstore.py # Custom vector store
```

## Installation

### Using Docker (Recommended)

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd real-time-rag-playground
   ```

2. Copy and configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. Build and run with Docker Compose:
   ```bash
   docker-compose up --build
   ```

### Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables:
   ```bash
   cp .env.example .env
   # Configure your .env file
   ```

3. Run the application:
   ```bash
   python src/main.py
   ```

## Configuration

Key environment variables (see `.env.example` for complete list):

- `OPENAI_API_KEY`: Your OpenAI API key for embeddings
- `RSS_FEEDS`: Comma-separated list of RSS feed URLs
- `RSS_REFRESH_INTERVAL`: How often to check feeds (seconds)
- `VECTOR_STORE_TYPE`: Choose between 'faiss' or 'pathway'
- `WATCH_DIRECTORY`: Directory to monitor for file changes

## Usage

### RSS Feed Ingestion

Configure RSS feeds in your `.env` file:
```
RSS_FEEDS=https://example.com/feed1.rss,https://example.com/feed2.rss
RSS_REFRESH_INTERVAL=300
```

### Google Drive Integration

1. Set up Google API credentials
2. Configure the folder ID in `.env`:
   ```
   GOOGLE_DRIVE_FOLDER_ID=your_folder_id
   GOOGLE_CREDENTIALS_PATH=path/to/credentials.json
   ```

### File System Monitoring

Point the system to monitor a directory:
```
WATCH_DIRECTORY=./data
WATCH_EXTENSIONS=.txt,.md,.pdf
```

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

For specific module testing:
```bash
python -m pytest tests/test_rss.py -v
```

## Development

### Adding New Data Sources

1. Create a new ingestion module in `src/ingestion/`
2. Implement the ingestion interface
3. Register the source in the main configuration

### Custom Vector Stores

1. Implement the vector store interface in `src/vectorstores/`
2. Add configuration options
3. Update the factory pattern in the main module

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [LangChain](https://github.com/langchain-ai/langchain)
- Vector processing powered by [FAISS](https://github.com/facebookresearch/faiss) and [Pathway](https://pathway.com/)
- Embeddings provided by [OpenAI](https://openai.com/)