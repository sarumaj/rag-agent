import pytest
from unittest.mock import Mock, patch, AsyncMock, mock_open
import re

from pipeline.pipeline import Document
from pipeline.pipeline import Chroma
from pipeline.pipeline import HuggingFaceEmbeddings
from pipeline.pipeline import OllamaLLM
from pipeline.pipeline import HuggingFaceEndpoint
from pipeline.pipeline import StateGraph
from pipeline.pipeline import RAGPipeline
from pipeline.config import Settings, TXT_SOURCE


@pytest.fixture(params=[
    "ollama",
    "huggingface"
])
def llm_provider(request):
    return request.param


@pytest.fixture
def mock_settings(llm_provider):
    settings = {
        "pipeline_max_threads": 1,
        "pipeline_embedding_model": "sentence-transformers/all-mpnet-base-v2",
        "pipeline_embedding_model_kwargs": {},
        "pipeline_chunk_size": 100,
        "pipeline_chunk_overlap": 20,
        "pipeline_persist_directory": "/tmp/test_vectorstore",
        "pipeline_collection_name": "test_collection",
        "pipeline_sources": {
            "/tmp/test_source": [
                TXT_SOURCE,
            ]
        },
        "pipeline_prompt_template": "Test template: {context} {question}",
        "pipeline_search_type": "similarity",
        "pipeline_k": 3,
        "pipeline_llm_provider": llm_provider,
        "pipeline_llm_model": "llama2" if llm_provider == "ollama" else "test-model",
        "pipeline_llm_model_kwargs": {},
    }

    if llm_provider == "huggingface":
        settings["pipeline_llm_api_key"] = "test-key"

    with (
        patch("os.environ", {}),
        patch("builtins.open", mock_open(read_data=""))
    ):
        return Settings(**settings)


@pytest.fixture
def mock_documents():
    document_meta_pattern = re.compile(r'^.*/(?P<journal>[^/]+)/(?P<year>\d+)/(?P<month>\d+)/(?P<page>\d+)(?:\.\w+)?$')
    return [
        Document(
            page_content="Test content 1",
            metadata={"source": "2024-01-01", "title": "Test 1", "meta_pattern": document_meta_pattern}
        ),
        Document(
            page_content="Test content 2",
            metadata={"source": "2024-01-02", "title": "Test 2", "meta_pattern": document_meta_pattern}
        )
    ]


@pytest.mark.asyncio
async def test_pipeline_initialization(mock_settings):
    """Test RAGPipeline initialization."""
    with (
        patch("pipeline.pipeline.HuggingFaceEmbeddings") as mock_embeddings,
        patch("pipeline.pipeline.Chroma") as mock_chroma
    ):

        mock_embeddings.return_value = Mock(spec=HuggingFaceEmbeddings)
        mock_chroma.return_value = Mock(spec=Chroma)

        pipeline = RAGPipeline(config=mock_settings)

        assert pipeline._config == mock_settings
        assert pipeline._thread_pool._max_workers == mock_settings.pipeline_max_threads
        mock_embeddings.assert_called_once_with(
            model_name=mock_settings.pipeline_embedding_model,
            model_kwargs=mock_settings.pipeline_embedding_model_kwargs,
            encode_kwargs={"normalize_embeddings": True}
        )
        mock_chroma.assert_called_once()


@pytest.mark.asyncio
async def test_load_documents(mock_settings, mock_documents):
    """Test document loading."""
    with patch("pipeline.pipeline.DirectoryLoader") as mock_loader:
        mock_loader_instance = AsyncMock()
        mock_loader_instance.aload.return_value = mock_documents
        mock_loader.return_value = mock_loader_instance

        pipeline = RAGPipeline(config=mock_settings)
        documents = await pipeline.load_documents()

        assert documents == mock_documents
        mock_loader.assert_called_once()
        mock_loader_instance.aload.assert_called_once()


@pytest.mark.asyncio
async def test_process_documents(mock_settings, mock_documents):
    """Test document processing."""
    with patch("pipeline.pipeline.RecursiveCharacterTextSplitter") as mock_splitter:
        mock_splitter.return_value.split_documents.return_value = [
            Document(page_content="Chunk 1", metadata={"source": "2024-01-01"}),
            Document(page_content="Chunk 2", metadata={"source": "2024-01-01"})
        ]

        pipeline = RAGPipeline(config=mock_settings)
        processed_docs = await pipeline.process_documents(mock_documents)

        assert len(processed_docs) >= 2
        mock_splitter.assert_called_once()


@pytest.mark.asyncio
async def test_update_vectorstore(mock_settings, mock_documents):
    """Test vector store update."""
    with patch("pipeline.pipeline.Chroma") as mock_chroma:
        mock_chroma.return_value.aadd_documents = AsyncMock(return_value=["id1", "id2"])

        pipeline = RAGPipeline(config=mock_settings)
        ids = await pipeline.update_vectorstore(mock_documents)

        assert ids == ["id1", "id2"]
        mock_chroma.return_value.aadd_documents.assert_called_once()


@pytest.mark.asyncio
async def test_list_vectorstore(mock_settings, llm_provider):
    """Test vector store listing."""
    with patch("pipeline.pipeline.Chroma") as mock_chroma:
        mock_chroma.return_value.get.return_value = {
            "documents": ["doc1", "doc2"],
            "metadatas": [{"source": "2024-01-01"}, {"source": "2024-01-02"}]
        }

        pipeline = RAGPipeline(config=mock_settings)
        result = await pipeline.list_vectorstore()

        assert result == {
            "documents": ["doc1", "doc2"],
            "metadatas": [{"source": "2024-01-01"}, {"source": "2024-01-02"}]
        }
        mock_chroma.return_value.get.assert_called_once_with(include=["documents", "metadatas"])


@pytest.mark.asyncio
async def test_setup_retrieval_chain_ollama(mock_settings):
    """Test retrieval chain setup with Ollama."""
    if mock_settings.pipeline_llm_provider != "ollama":
        pytest.skip("Skipping test for non-Ollama provider")

    with (
        patch("pipeline.pipeline.OllamaLLM") as mock_llm,
        patch("pipeline.pipeline.StateGraph") as mock_graph
    ):

        mock_llm.return_value = Mock(spec=OllamaLLM)
        mock_graph.return_value = Mock(spec=StateGraph)

        pipeline = RAGPipeline(config=mock_settings)
        await pipeline.setup_retrieval_chain()

        assert pipeline._llm is not None
        assert pipeline._rag_chain is not None
        mock_llm.assert_called_once()


@pytest.mark.asyncio
async def test_setup_retrieval_chain_huggingface(mock_settings):
    """Test retrieval chain setup with HuggingFace."""
    if mock_settings.pipeline_llm_provider != "huggingface":
        pytest.skip("Skipping test for non-HuggingFace provider")

    with patch("pipeline.pipeline.HuggingFaceEndpoint") as mock_llm, \
         patch("pipeline.pipeline.StateGraph") as mock_graph:

        mock_llm.return_value = Mock(spec=HuggingFaceEndpoint)
        mock_graph.return_value = Mock(spec=StateGraph)

        pipeline = RAGPipeline(config=mock_settings)
        await pipeline.setup_retrieval_chain()

        assert pipeline._llm is not None
        assert pipeline._rag_chain is not None
        mock_llm.assert_called_once()


@pytest.mark.asyncio
async def test_run_pipeline(mock_settings):
    """Test pipeline execution."""
    with patch("pipeline.pipeline.StateGraph") as mock_graph:
        mock_chain = AsyncMock()
        mock_chain.ainvoke.return_value = {"answer": "Test answer"}
        mock_graph.return_value = mock_chain

        pipeline = RAGPipeline(config=mock_settings)
        pipeline._rag_chain = mock_chain

        answer = await pipeline.run("Test question")
        assert answer == "Test answer"
        mock_chain.ainvoke.assert_called_once_with({"question": "Test question"})
