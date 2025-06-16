import pytest
from unittest.mock import Mock, patch, AsyncMock, mock_open
import re

from pipeline.chroma import ChromaRAGPipeline, ChromaRAGPipelineConfig
from pipeline.config import TXT_SOURCE
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEndpoint
from langgraph.graph import StateGraph


@pytest.fixture(params=[
    "ollama",
    "huggingface"
])
def llm_provider(request: pytest.FixtureRequest):
    match request.param:
        case "ollama":
            return request.param
        case "huggingface":
            return request.param
        case _:
            raise NotImplementedError(f"Invalid LLM provider: {request.param}")


@pytest.fixture(params=[
    "chroma",
])
def vector_store(request: pytest.FixtureRequest):
    match request.param:
        case "chroma":
            return request.param
        case _:
            raise NotImplementedError(f"Invalid vector store: {request.param}")


@pytest.fixture
def mock_settings(llm_provider: str, vector_store: str):
    settings = {
        "pipeline_max_threads": 1,
        "pipeline_embedding_model": "sentence-transformers/all-mpnet-base-v2",
        "pipeline_embedding_model_kwargs": {},
        "pipeline_chunk_size": 100,
        "pipeline_chunk_overlap": 20,
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

    match llm_provider:
        case "huggingface":
            settings["pipeline_llm_api_key"] = "test-key"

    match vector_store:
        case "chroma":
            settings.update({
                "pipeline_persist_directory": "/tmp/test_vectorstore",
                "pipeline_collection_name": "test_collection",
            })
            with (
                patch("os.environ", {}),
                patch("builtins.open", mock_open(read_data=""))
            ):
                return ChromaRAGPipelineConfig(**settings)
        case _:
            raise NotImplementedError(f"Invalid vector store: {vector_store}")


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
async def test_pipeline_initialization(mock_settings, vector_store):
    """Test RAGPipeline initialization."""
    match vector_store:
        case "chroma":
            with (
                patch("pipeline.chroma.vectorstore.HuggingFaceEmbeddings") as mock_embeddings,
                patch("pipeline.chroma.vectorstore.Chroma") as mock_chroma,
            ):
                mock_embeddings.return_value = Mock(spec=HuggingFaceEmbeddings)
                mock_chroma.return_value = Mock(spec=Chroma)

                pipeline = ChromaRAGPipeline(config=mock_settings)
                mock_embeddings.assert_called_once_with(
                    model_name=mock_settings.pipeline_embedding_model,
                    model_kwargs=mock_settings.pipeline_embedding_model_kwargs,
                    encode_kwargs={"normalize_embeddings": True}
                )
                mock_chroma.assert_called_once()
                assert pipeline._config == mock_settings
        case _:
            raise NotImplementedError(f"Invalid vector store: {vector_store}")


@pytest.mark.asyncio
async def test_load_documents(mock_settings, mock_documents, vector_store):
    """Test document loading."""
    match vector_store:
        case "chroma":
            with patch("pipeline.chroma.vectorstore.DirectoryLoader") as mock_loader:
                mock_loader_instance = AsyncMock()
                mock_loader_instance.aload.return_value = mock_documents
                mock_loader.return_value = mock_loader_instance

                pipeline = ChromaRAGPipeline(config=mock_settings)
                documents = await pipeline.load_documents()

                assert documents == mock_documents
                mock_loader.assert_called_once()
                mock_loader_instance.aload.assert_called_once()
        case _:
            raise NotImplementedError(f"Invalid vector store: {vector_store}")


@pytest.mark.asyncio
async def test_process_documents(mock_settings, mock_documents, vector_store):
    """Test document processing."""
    match vector_store:
        case "chroma":
            with patch("pipeline.chroma.vectorstore.RecursiveCharacterTextSplitter") as mock_splitter:
                mock_splitter.return_value.split_documents.return_value = [
                    Document(page_content="Chunk 1", metadata={"source": "2024-01-01"}),
                    Document(page_content="Chunk 2", metadata={"source": "2024-01-01"})
                ]

                pipeline = ChromaRAGPipeline(config=mock_settings)
                processed_docs = await pipeline.process_documents(mock_documents)

                assert len(processed_docs) >= 2
                mock_splitter.assert_called_once()
        case _:
            raise NotImplementedError(f"Invalid vector store: {vector_store}")


@pytest.mark.asyncio
async def test_update_vectorstore(mock_settings, mock_documents, vector_store):
    """Test vector store update."""
    match vector_store:
        case "chroma":
            with patch("pipeline.chroma.vectorstore.Chroma") as mock_chroma:
                mock_chroma.return_value.aadd_documents = AsyncMock(return_value=["id1", "id2"])
                pipeline = ChromaRAGPipeline(config=mock_settings)
                ids = await pipeline.update_vectorstore(mock_documents)
                assert ids == ["id1", "id2"]
                mock_chroma.return_value.aadd_documents.assert_called_once()
        case _:
            raise NotImplementedError(f"Invalid vector store: {vector_store}")


@pytest.mark.asyncio
async def test_list_vectorstore(mock_settings, vector_store):
    """Test vector store listing."""
    match vector_store:
        case "chroma":
            with patch("pipeline.chroma.vectorstore.Chroma") as mock_chroma:
                mock_chroma.return_value.get.return_value = {
                    "documents": ["doc1", "doc2"],
                    "metadatas": [{"source": "2024-01-01"}, {"source": "2024-01-02"}]
                }
                pipeline = ChromaRAGPipeline(config=mock_settings)
                result = await pipeline.list_vectorstore()
                assert result == {
                    "documents": ["doc1", "doc2"],
                    "metadatas": [{"source": "2024-01-01"}, {"source": "2024-01-02"}]
                }
                mock_chroma.return_value.get.assert_called_once_with(include=["documents", "metadatas"])
        case _:
            raise NotImplementedError(f"Invalid vector store: {vector_store}")


@pytest.mark.asyncio
async def test_setup_retrieval_chain_ollama(mock_settings, vector_store):
    """Test retrieval chain setup with Ollama."""
    if mock_settings.pipeline_llm_provider != "ollama":
        pytest.skip("Skipping test for non-Ollama provider")

    match vector_store:
        case "chroma":
            with (
                patch("pipeline.chroma.vectorstore.OllamaLLM") as mock_llm,
                patch("pipeline.chroma.vectorstore.StateGraph") as mock_graph
            ):
                mock_llm.return_value = Mock(spec=OllamaLLM)
                mock_graph.return_value = Mock(spec=StateGraph)

                pipeline = ChromaRAGPipeline(config=mock_settings)
                await pipeline.setup_retrieval_chain()

                assert pipeline._llm is not None
                assert pipeline._rag_chain is not None
                mock_llm.assert_called_once()
        case _:
            raise NotImplementedError(f"Invalid vector store: {vector_store}")


@pytest.mark.asyncio
async def test_setup_retrieval_chain_huggingface(mock_settings, vector_store):
    """Test retrieval chain setup with HuggingFace."""
    if mock_settings.pipeline_llm_provider != "huggingface":
        pytest.skip("Skipping test for non-HuggingFace provider")

    match vector_store:
        case "chroma":
            with (
                patch("pipeline.chroma.vectorstore.HuggingFaceEndpoint") as mock_llm,
                patch("pipeline.chroma.vectorstore.StateGraph") as mock_graph
            ):
                mock_llm.return_value = Mock(spec=HuggingFaceEndpoint)
                mock_graph.return_value = Mock(spec=StateGraph)

                pipeline = ChromaRAGPipeline(config=mock_settings)
                await pipeline.setup_retrieval_chain()

                assert pipeline._llm is not None
                assert pipeline._rag_chain is not None
                mock_llm.assert_called_once()
        case _:
            raise NotImplementedError(f"Invalid vector store: {vector_store}")


@pytest.mark.asyncio
async def test_run_pipeline(mock_settings, vector_store):
    """Test pipeline execution."""
    match vector_store:
        case "chroma":
            with patch("pipeline.chroma.vectorstore.StateGraph") as mock_graph:
                mock_chain = AsyncMock()

                class MockAsyncIterator:
                    def __init__(self):
                        def _generator():
                            yield {"context": "Important context"}
                            yield {"answer": "Test answer"}

                        self.generator = _generator()

                    def __aiter__(self):
                        return self

                    async def __anext__(self):
                        try:
                            return next(self.generator)
                        except StopIteration:
                            raise StopAsyncIteration

                mock_chain.astream = Mock(return_value=MockAsyncIterator())
                mock_graph.return_value = mock_chain

                pipeline = ChromaRAGPipeline(config=mock_settings)
                pipeline._rag_chain = mock_chain
                response = await pipeline.run("Test question")
                assert response == {"context": "Important context", "answer": "Test answer"}

                mock_chain.astream.assert_called_once_with({"question": "Test question"}, stream_mode="updates")
        case _:
            raise NotImplementedError(f"Invalid vector store: {vector_store}")
