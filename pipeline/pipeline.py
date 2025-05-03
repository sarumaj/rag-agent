from typing import List
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredPDFLoader,
    DirectoryLoader,
    UnstructuredHTMLLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores.utils import filter_complex_metadata
from os import environ

from .config import Settings

logging.basicConfig(
    level=logging.INFO,
    style='{',
    format='{asctime} - {levelname} - {message}',
)

logger = logging.getLogger("rag_pipeline")


class RAGPipeline:
    """RAG pipeline implementation using LangChain."""

    def __init__(
        self,
        config: Settings = Settings()
    ):
        """Initialize the RAG pipeline.

        Args:
            config: Optional configuration object. If not provided, defaults will be used.
        """
        self._config = config
        logger.info(f"Initializing RAG pipeline with config: {self._config.model_dump_json(indent=2)}")
        environ["HUGGINGFACEHUB_API_TOKEN"] = self._config.llm_api_key
        self._thread_pool = ThreadPoolExecutor(max_workers=self._config.max_threads)

        try:
            self._loop = asyncio.get_event_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

        self._embedding_model = HuggingFaceEmbeddings(
            model_name=self._config.embedding_model,
            model_kwargs=self._config.embedding_model_kwargs,
        )
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._config.chunk_size,
            chunk_overlap=self._config.chunk_overlap
        )
        self._vectorstore = None
        self._llm = None
        self._rag_chain = None

    async def load_documents(self) -> List[dict]:
        """Load documents from various sources asynchronously.

        Returns:
            List of loaded documents
        """
        try:
            loader_cls = None
            loader_kwargs = {}

            match self._config.source_type:
                case "file":
                    loader_cls = TextLoader
                    loader_kwargs = {"autodetect_encoding": True}
                case "pdf":
                    loader_cls = UnstructuredPDFLoader
                    loader_kwargs = {"mode": "elements", "strategy": "fast"}
                case "html":
                    loader_cls = UnstructuredHTMLLoader
                    loader_kwargs = {"mode": "elements", "strategy": "fast"}
                case _:
                    raise ValueError(f"Unsupported source type: {self._config.source_type}")

            loader = DirectoryLoader(
                self._config.source,
                glob=self._config.glob_pattern,
                loader_cls=loader_cls,
                loader_kwargs=loader_kwargs,
                use_multithreading=True,
                max_concurrency=self._config.max_threads,
                show_progress=True,
            ) if self._config.glob_pattern else loader_cls(
                self._config.source,
                loader_kwargs=loader_kwargs
            )

            documents = await self._loop.run_in_executor(self._thread_pool, loader.load)
            print("Documents loaded:", len(documents))
            return documents

        except Exception as e:
            logging.error(f"Error loading documents: {str(e)}")
            raise

    async def process_documents(self, documents: List[dict]) -> List[dict]:
        """Process and split documents into chunks asynchronously.

        Args:
            documents: List of documents to process

        Returns:
            List of processed document chunks
        """
        return await self._loop.run_in_executor(
            self._thread_pool,
            self._text_splitter.split_documents,
            documents
        )

    async def create_vectorstore(self, documents: List[dict]):
        """Create and persist vector store from documents asynchronously.

        Args:
            documents: List of processed document chunks
        """
        self._vectorstore = await self._loop.run_in_executor(
            self._thread_pool,
            lambda: Chroma.from_documents(
                documents=filter_complex_metadata(documents),
                embedding=self._embedding_model,
                persist_directory=self._config.persist_directory,
                collection_name=self._config.collection_name,
            )
        )

    async def load_vectorstore(self):
        """Load existing vector store from disk asynchronously."""
        self._vectorstore = await self._loop.run_in_executor(
            self._thread_pool,
            lambda: Chroma(
                persist_directory=self._config.persist_directory,
                embedding_function=self._embedding_model,
                collection_name=self._config.collection_name,
            )
        )

    async def setup_retrieval_chain(self):
        """Set up the retrieval-augmented generation chain asynchronously."""
        if not self._vectorstore:
            raise ValueError("Vector store not initialized. Call create_vectorstore() or load_vectorstore() first.")

        prompt = ChatPromptTemplate.from_template(self._config.prompt_template)
        retrieval_settings = {
            "search_type": self._config.search_type,
            "k": self._config.k,
        }

        match self._config.search_type:
            case "similarity":
                pass
            case "similarity_score_threshold":
                retrieval_settings["score_threshold"] = self._config.score_threshold
            case "mmr":
                retrieval_settings.update({
                    "fetch_k": self._config.fetch_k,
                    "lambda_mult": self._config.lambda_mult
                })
            case _:
                raise ValueError(f"Unsupported search type: {self._config.search_type}")

        retriever = self._vectorstore.as_retriever(**retrieval_settings)
        match self._config.llm_provider:
            case "huggingface":
                self._llm = HuggingFaceEndpoint(
                    repo_id=self._config.llm_model,
                    task="text-generation",
                    huggingface_api_token=self._config.llm_api_key,
                    **self._config.llm_model_kwargs,
                )
            case _:
                raise ValueError(f"Unsupported LLM provider: {self._config.llm_provider}")

        self._rag_chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough()
            } | prompt | self._llm | StrOutputParser()
        )

    async def run(self, question: str) -> str:
        """Run the RAG pipeline asynchronously.

        Args:
            question: Question to answer

        Returns:
            Answer to the question
        """
        if not self._rag_chain:
            raise ValueError("Retrieval chain not initialized. Call setup_retrieval_chain() first.")

        return await self._loop.run_in_executor(
            self._thread_pool,
            self._rag_chain.invoke,
            question
        )

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self._thread_pool.shutdown(wait=True)
