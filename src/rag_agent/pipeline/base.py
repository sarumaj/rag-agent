from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Literal
from langchain_core.documents import Document

from .config import PipelineSettings


class BaseRAGPipeline(ABC):
    """Abstract base class for RAG pipeline implementations."""

    def __init__(self, config: Optional[PipelineSettings] = None):
        """Initialize the RAG pipeline.

        Args:
            config: Optional configuration object. If not provided, defaults will be used.
        """
        self._config = config or PipelineSettings()

    @abstractmethod
    async def update_vectorstore(self, documents: List[Document]) -> List[str]:
        """Update the vector store with new documents asynchronously.

        Args:
            documents: List of processed document chunks

        Returns:
            List of document ids
        """
        pass

    @abstractmethod
    async def list_vectorstore(self) -> Dict[str, Any]:
        """List all documents in the vector store asynchronously.

        Returns:
            Dictionary containing documents and their metadata
        """
        pass

    @abstractmethod
    async def setup_retrieval_chain(self, context_format: Literal["json", "markdown"] = "json"):
        """Set up the retrieval-augmented generation chain asynchronously.

        Args:
            context_format: Format of the context to be used in the prompt
        """
        pass

    @abstractmethod
    async def run(self, question: str) -> Dict[str, Any]:
        """Run the RAG pipeline asynchronously.

        Args:
            question: Question to answer

        Returns:
            Answer to the question
        """
        pass

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass
