from typing import List, TypedDict, Dict, Any, Optional, Literal
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    PyMuPDFLoader,
    UnstructuredHTMLLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from chromadb.config import Settings as ChromaSettings
import re
from tqdm import tqdm
import json

from .config import Settings

try:
    from importlib import import_module
    LOADERS_AVAILABLE = all([
        import_module("pymupdf") is not None,
        import_module("unstructured") is not None,
    ])
except (ImportError, ModuleNotFoundError):
    LOADERS_AVAILABLE = False

    class NotImported:
        def __getattr__(self, item):
            raise ModuleNotFoundError(
                "Loader dependencies are not installed. "
                "Please install them using: pip install 'rag-agent[loaders]'"
            )

        def __call__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "Loader dependencies are not installed. "
                "Please install them using: pip install 'rag-agent[loaders]'"
            )

    globals().update(dict.fromkeys(
        [
            "PyMuPDFLoader",
            "UnstructuredHTMLLoader",
        ],
        NotImported()
    ))


logging.basicConfig(
    level=logging.INFO,
    style='{',
    format='{asctime} - {levelname} - {message}',
)

logger = logging.getLogger("rag_agent.pipeline")


class RAGPipeline:
    """RAG pipeline implementation using LangChain."""

    def __init__(self, config: Optional[Settings] = None):
        """Initialize the RAG pipeline.

        Args:
            config: Optional configuration object. If not provided, defaults will be used.
        """
        self._config = config or Settings()
        logger.info(f"Initializing RAG pipeline with config: {self._config.model_dump_json(indent=2)}")
        self._thread_pool = ThreadPoolExecutor(max_workers=self._config.pipeline_max_threads)

        try:
            self._loop = asyncio.get_event_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

        self._embedding_model = HuggingFaceEmbeddings(
            model_name=self._config.pipeline_embedding_model,
            model_kwargs=self._config.pipeline_embedding_model_kwargs,
            encode_kwargs={"normalize_embeddings": True}
        )
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._config.pipeline_chunk_size,
            chunk_overlap=self._config.pipeline_chunk_overlap
        )
        self._vectorstore = Chroma(
            embedding_function=self._embedding_model,
            persist_directory=self._config.pipeline_persist_directory,
            collection_name=self._config.pipeline_collection_name,
            client_settings=ChromaSettings(
                anonymized_telemetry=False
            )
        )

        match self._config.pipeline_llm_provider:
            case "ollama":
                self._llm = OllamaLLM(
                    model=self._config.pipeline_llm_model,
                    **self._config.pipeline_llm_model_kwargs
                )
            case "huggingface":
                if not self._config.pipeline_llm_api_key:
                    raise ValueError("Hugging Face API key is required")
                self._llm = HuggingFaceEndpoint(
                    repo_id=self._config.pipeline_llm_model,
                    task="text-generation",
                    huggingface_api_token=self._config.pipeline_llm_api_key,
                    **self._config.pipeline_llm_model_kwargs
                )
            case _:
                raise ValueError(f"Unsupported LLM provider: {self._config.pipeline_llm_provider}")

        self._rag_chain = None

    async def load_documents(self) -> List[Document]:
        """Load documents from various sources asynchronously.

        Returns:
            List of loaded documents
        """
        try:
            loader_cls = None
            loader_kwargs = {}

            documents = []
            for path, sources in self._config.pipeline_sources.items():
                for source in sources:
                    logger.info(f"Loading documents from {path} using {source.source_type} loader")
                    match source.source_type:
                        case "txt":
                            loader_cls = TextLoader
                            loader_kwargs = {"autodetect_encoding": True}
                        case "pdf":
                            loader_cls = PyMuPDFLoader
                            loader_kwargs = {
                                "mode": "single"
                            }
                        case "html":
                            loader_cls = UnstructuredHTMLLoader
                            loader_kwargs = {"mode": "single", "strategy": "fast"}
                        case _:
                            raise ValueError(f"Unsupported source type: {source.source_type}")

                    loader = DirectoryLoader(
                        path,
                        glob=source.glob_pattern,
                        loader_cls=loader_cls,
                        loader_kwargs=loader_kwargs,
                        use_multithreading=True,
                        max_concurrency=self._config.pipeline_max_threads,
                        show_progress=True,
                    ) if source.glob_pattern else loader_cls(
                        path,
                        loader_kwargs=loader_kwargs
                    )

                    for doc in await loader.aload():
                        meta_pattern = (
                            source.meta_pattern
                            if isinstance(source.meta_pattern, re.Pattern)
                            else re.compile(source.meta_pattern)
                        )
                        doc.metadata.update({
                            "meta_pattern": meta_pattern.pattern,
                            **((
                                match.groupdict() or
                                dict(((str(i), g) for i, g in enumerate(match.groups())))
                            ) if (match := meta_pattern.match(doc.metadata.get('source', ''))) else {})
                        })
                        documents.append(doc)

                    logger.info(f"Loaded {len(documents)} documents from {path} using {source.source_type} loader")

            return documents

        except Exception as e:
            logging.error(f"Error loading documents: {str(e)}")
            raise

    async def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process and split documents into chunks asynchronously.

        Args:
            documents: List of documents to process

        Returns:
            List of processed document chunks
        """
        logger.info(f"Processing {len(documents)} documents")
        processed_docs = []

        with tqdm(total=len(documents), desc="Processing documents") as pbar:
            for doc in documents:
                chunks = await self._loop.run_in_executor(
                    self._thread_pool,
                    self._text_splitter.split_documents,
                    [doc]
                )
                processed_docs.extend(chunks)
                pbar.update(1)
                pbar.set_postfix({"chunks": len(processed_docs)})

        logger.info(f"Total processed chunks: {len(processed_docs)}")
        return processed_docs

    async def update_vectorstore(self, documents: List[Document]) -> List[str]:
        """Update the vector store with new documents asynchronously.

        Args:
            documents: List of processed document chunks

        Returns:
            List of document ids
        """

        documents = list(filter_complex_metadata(documents))
        logger.info(f"Updating vector store with {len(documents)} documents")

        with tqdm(total=len(documents), desc="Updating vector store") as pbar:
            ids = []
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_ids = await self._vectorstore.aadd_documents(batch)
                ids.extend(batch_ids)
                pbar.update(len(batch))
                pbar.set_postfix({"processed": len(ids)})

        logger.info(f"Added {len(ids)} documents to vector store")
        return ids

    async def list_vectorstore(self) -> Dict[str, Any]:
        """List all documents in the vector store asynchronously.

        Returns:
            List of document ids
        """
        return await self._loop.run_in_executor(
            self._thread_pool,
            lambda: self._vectorstore.get(
                include=["documents", "metadatas"]
            )
        )

    async def setup_retrieval_chain(self, context_format: Literal["json", "markdown"] = "json"):
        """Set up the retrieval-augmented generation chain asynchronously.

        Args:
            context_format: Format of the context to be used in the prompt
        """
        prompt = ChatPromptTemplate.from_template(self._config.pipeline_prompt_template)

        retrieval_settings = {
            "search_type": self._config.pipeline_search_type,
            "k": self._config.pipeline_k,
        }

        match self._config.pipeline_search_type:
            case "similarity":
                pass
            case "similarity_score_threshold":
                retrieval_settings["score_threshold"] = self._config.pipeline_score_threshold
            case "mmr":
                retrieval_settings.update({
                    "fetch_k": self._config.pipeline_fetch_k,
                    "lambda_mult": self._config.pipeline_lambda_mult
                })
            case _:
                raise ValueError(f"Unsupported search type: {self._config.pipeline_search_type}")

        retriever = self._vectorstore.as_retriever(**retrieval_settings)

        class State(TypedDict):
            context: str
            question: str
            answer: str

        def retrieve(state: State) -> State:
            results = retriever.invoke(state["question"])
            return {"context": results}

        def generate(state: State) -> State:
            match context_format:
                case "json":
                    formatted_context = json.dumps([
                        {
                            **doc.metadata,
                            "Content": doc.page_content
                        }
                        for doc in state["context"]
                    ])
                case "markdown":
                    keys = {key for doc in state["context"] for key in doc.metadata.keys()}
                    formatted_context = "\n".join([
                        "| " + " | ".join(keys) + " |",
                        "| " + " | ".join(["---"] * len(keys)) + " |",
                        *[
                            "| " + " | ".join([str(doc.get(key, "")) for key in keys]) + " |"
                            for doc in state["context"]
                        ],
                    ])
                case _:
                    raise ValueError(f"Unsupported context format: {context_format}")

            prompt_template = prompt.invoke({
                "context": f"```{context_format}\n\n{formatted_context}\n\n```",
                "question": state["question"]
            })
            response = self._llm.invoke(prompt_template)
            return {"answer": response, "context": formatted_context}

        graph = StateGraph(State)
        graph.add_node("retrieve", retrieve)
        graph.add_edge(START, "retrieve")
        graph.add_node("generate", generate)
        graph.add_edge("retrieve", "generate")

        self._rag_chain = graph.compile()

    async def run(self, question: str) -> str:
        """Run the RAG pipeline asynchronously.

        Args:
            question: Question to answer

        Returns:
            Answer to the question
        """
        if not self._rag_chain:
            raise ValueError("Retrieval chain not initialized. Call setup_retrieval_chain() first.")

        response = {}
        async for chunk in self._rag_chain.astream({"question": question}, stream_mode="updates"):
            response.update(chunk)
        return response

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self._thread_pool.shutdown(wait=True)
