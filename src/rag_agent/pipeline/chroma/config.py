from pydantic import Field

from ..config import PipelineSettings


class Settings(PipelineSettings):
    """Configuration settings for the RAG pipeline."""
    pipeline_persist_directory: str = Field(
        default="chroma_db",
        description="Directory to store the ChromaDB database"
    )
    pipeline_collection_name: str = Field(
        default="default_collection",
        description="Name of the ChromaDB collection to use",
        min_length=3,
        max_length=512
    )
