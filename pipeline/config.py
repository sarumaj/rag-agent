from pydantic import Field
from typing import Optional, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration for RAGPipeline."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="_",
        frozen=True,
        extra="ignore"
    )

    max_threads: int = Field(default=10, description="Maximum number of threads")

    # Embedding model settings
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Name of the HuggingFace embedding model to use"
    )
    embedding_model_kwargs: dict = Field(
        default={"device": "cuda"},
        description="Keyword arguments for the embedding model"
    )

    # Document source settings
    source: str = Field(description="Path to the document or URL")
    source_type: Literal["file", "pdf", "html", "directory"] = Field(
        default="file",
        description="Type of source"
    )
    glob_pattern: str = Field(
        default="**/*",
        description="Glob pattern for matching files in directory"
    )

    # Text splitting settings
    chunk_size: int = Field(
        default=1000,
        ge=100,
        description="Size of text chunks for splitting documents"
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        description="Overlap between chunks"
    )

    # Vector store settings
    persist_directory: str = Field(
        default="chroma_db",
        description="Directory to store the vector database"
    )
    collection_name: str = Field(
        default="default_collection",
        description="Name of the ChromaDB collection to use",
        min_length=3,
        max_length=512
    )

    # Retrieval settings
    search_type: Literal["similarity", "mmr", "similarity_score_threshold"] = Field(
        default="similarity",
        description="Type of search to perform"
    )
    k: int = Field(
        default=5,
        ge=1,
        description="Number of documents to retrieve"
    )
    score_threshold: Optional[float] = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score (for similarity_score_threshold)"
    )
    fetch_k: Optional[int] = Field(
        default=20,
        ge=1,
        description="Number of documents to fetch before filtering (for MMR)"
    )
    lambda_mult: Optional[float] = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Diversity factor (for MMR, 0-1)"
    )

    # Prompt template
    prompt_template: Optional[str] = Field(
        default="""Answer the question based on the following context:

        Context: {context}

        Question: {question}

        Answer:""",
        description="Optional custom prompt template for the RAG chain"
    )

    # LLM settings
    llm_api_key: str = Field(
        default="",
        description="API key for the LLM provider"
    )
    llm_model: str = Field(
        default="mistralai/Mistral-7B-Instruct-v0.2",
        description="Name of the HuggingFace LLM model to use"
    )
    llm_model_kwargs: dict = Field(
        default={"temperature": 0.3, "max_new_tokens": 250},
        description="Keyword arguments for the LLM model"
    )
    llm_provider: Literal["huggingface"] = Field(
        default="huggingface",
        description="Provider of the LLM model"
    )
