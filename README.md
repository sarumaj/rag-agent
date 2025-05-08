[![release](https://github.com/sarumaj/rag-agent/actions/workflows/release.yml/badge.svg)](https://github.com/sarumaj/rag-agent/actions/workflows/release.yml)
[![GitHub Release](https://img.shields.io/github/v/release/sarumaj/rag-agent?logo=github)](https://github.com/sarumaj/rag-agent/releases/latest)
[![Libraries.io dependency status for GitHub repo](https://img.shields.io/librariesio/github/sarumaj/rag-agent)](https://github.com/sarumaj/rag-agent/blob/main/pyproject.toml)

---

# RAG Pipeline

A powerful Retrieval-Augmented Generation (RAG) pipeline implementation that supports multiple document types, embedding models, and LLM providers.

## Features

- **Multiple Document Support**
  - PDF documents
  - Text files
  - HTML content

- **Flexible LLM Integration**
  - Ollama support (local models)
  - Hugging Face integration
  - Configurable model parameters

- **Advanced Document Processing**
  - Metadata extraction from file paths
  - Configurable text chunking
  - Duplicate detection
  - Progress tracking

- **Vector Store Features**
  - ChromaDB integration
  - Configurable embedding models
  - Multiple search strategies
  - Persistent storage

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rag-pipeline.git
cd rag-pipeline
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
```

4. Install Ollama (if using local models):
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

## Configuration

Create a `.env` file in the project root with your configuration:

```env
# Document source settings
PIPELINE_SOURCES='{"/path/to/your/documents": ["pdf", "mhtml"]}'

# Text splitting settings
PIPELINE_CHUNK_SIZE=1000
PIPELINE_CHUNK_OVERLAP=200

# Vector store settings
PIPELINE_PERSIST_DIRECTORY=chroma_db
PIPELINE_COLLECTION_NAME=default_collection

# Embedding model settings
PIPELINE_EMBEDDING_MODEL=all-MiniLM-L6-v2
PIPELINE_EMBEDDING_MODEL_KWARGS={"device": "cuda"}

# LLM settings
PIPELINE_LLM_PROVIDER=ollama  # or 'huggingface'
PIPELINE_LLM_MODEL=mistral
PIPELINE_LLM_MODEL_KWARGS={"temperature": 0.3}
PIPELINE_LLM_API_KEY=your_api_key  # Required for Hugging Face

# Retrieval settings
PIPELINE_SEARCH_TYPE=similarity  # or 'mmr', 'similarity_score_threshold'
PIPELINE_K=5
PIPELINE_SCORE_THRESHOLD=0.5
PIPELINE_FETCH_K=20
PIPELINE_LAMBDA_MULT=0.5
```

## Usage

### Basic Usage

```python
from pipeline import RAGPipeline, Settings

async def main():
    # Initialize with default settings
    async with RAGPipeline() as pipeline:
        # Load and process documents
        documents = await pipeline.load_documents()
        processed_docs = await pipeline.process_documents(documents)
        await pipeline.update_vectorstore(processed_docs)
        
        # Setup and run query
        await pipeline.setup_retrieval_chain()
        answer = await pipeline.run("Your question here")
        print(answer)

# Run the pipeline
import asyncio
asyncio.run(main())
```

### Custom Configuration

```python
config = Settings(
    pipeline_source="/path/to/documents",
    pipeline_source_type="pdf",
    pipeline_llm_provider="ollama",
    pipeline_llm_model="mixtral",
    pipeline_llm_model_kwargs={"temperature": 0.3}
)

async with RAGPipeline(config) as pipeline:
    # ... rest of the code
```

## Supported Models

### Ollama Models
- `mistral` (7B parameters)
- `mixtral` (8x7B parameters)
- `llama2` (7B parameters)
- `codellama` (Code specialized)
- `neural-chat` (Chat optimized)
- `dolphin-mixtral` (Chat optimized)
- And many more...

### Hugging Face Models
- `mistralai/Mistral-7B-Instruct-v0.2`
- `meta-llama/Llama-2-7b-chat-hf`
- `malteos/gpt2-wechsel-german`
- And many more...

## Document Processing

The pipeline supports various document types and processing options:

### PDF Processing
- Uses PyMuPDF for efficient PDF processing
- Extracts text and metadata
- Configurable processing modes

### Text Processing
- Configurable chunk sizes
- Overlap control
- Metadata extraction

### HTML Processing
- Clean HTML extraction
- Structured content handling
- Metadata preservation

## IX Archive Scraper

The project includes a specialized scraper for the IX archive with the following features:

### Features
- **Automated Article Download**
  - Downloads articles in multiple formats (PDF, MHTML)
  - Preserves article metadata and structure
  - Handles authentication and session management

- **Parallel Processing**
  - Concurrent article processing
  - Configurable thread pool
  - Progress tracking with tqdm

- **Robust Error Handling**
  - Automatic retries for failed downloads
  - Graceful cleanup on interruption
  - Detailed logging

### Configuration

Add the following to your `.env` file for IX scraper configuration:

```env
# IX Scraper settings
IX_SCRAPER_BASE_URL=https://www.heise.de
IX_SCRAPER_SIGN_IN_URL=https://www.heise.de/sso/login/
IX_SCRAPER_ARCHIVE_URL=https://www.heise.de/select/ix/archiv/
IX_SCRAPER_MAX_THREADS=10
IX_SCRAPER_MAX_CONCURRENT=10
IX_SCRAPER_TIMEOUT=30
IX_SCRAPER_RETRY_ATTEMPTS=5
IX_SCRAPER_OUTPUT_DIR=~/Downloads/ix
IX_SCRAPER_USERNAME=your_username
IX_SCRAPER_PASSWORD=your_password
IX_SCRAPER_OVERWRITE=false
IX_SCRAPER_EXPORT_FORMATS=["pdf"]
```

### Usage

```python
# pip install -e .[scraper]
from scrappers.ix import IXScraper, Settings

async def main():
    # Initialize with default settings
    async with IXScraper() as scraper:
        # Run the scraper
        await scraper.run()

# Run the scraper
import asyncio
asyncio.run(main())
```

### Export Formats

The scraper supports multiple export formats:

1. **PDF Export**
   - High-quality PDF output
   - Configurable page settings
   - Base64 encoded transfer

2. **MHTML Export**
   - Preserves web page structure
   - Includes all resources
   - Suitable for archival

### WebDriver Configuration

The scraper uses Selenium WebDriver with configurable options:
- Headless mode support
- Custom user agent
- Resource optimization
- Security settings

## Vector Store

The pipeline uses ChromaDB for vector storage with features like:
- Configurable embedding models
- Multiple search strategies
- Persistent storage
- Duplicate detection

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- LangChain for the RAG framework
- ChromaDB for vector storage
- Ollama for local LLM support
- Hugging Face for model hosting
