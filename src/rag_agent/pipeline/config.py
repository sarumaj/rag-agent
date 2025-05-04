from pydantic import Field, field_validator, ValidationError, BaseModel
from typing import Optional, Literal, Union
from pydantic_settings import BaseSettings, SettingsConfigDict
import re
import os
import json


class DocumentSource(BaseModel):
    source_type: Literal["txt", "pdf", "html"] = Field(default="txt", description="Type of source")
    meta_pattern: re.Pattern = Field(default_factory=lambda: re.compile(
            r"^.*/(?P<journal>[^/]+)/"
            r"(?P<year>\d+)/"
            r"(?P<month>\d+)/"
            r"(?P<page>\d+)(?:\.\w+)?$"
        ),
        description="Regex pattern for extracting metadata from the source path"
    )
    glob_pattern: str = Field(
        default="**/*",
        description="Glob pattern for matching files in directory"
    )

    @field_validator("meta_pattern", mode="before")
    @classmethod
    def parse_meta_pattern(cls, v: str) -> re.Pattern:
        try:
            return re.compile(v) if not isinstance(v, re.Pattern) else v
        except re.error:
            raise ValidationError(f"Invalid regex pattern: {v}")


PDF_SOURCE = DocumentSource(
    source_type="pdf",
    glob_pattern="**/*.pdf",
)

TXT_SOURCE = DocumentSource(
    source_type="txt",
    glob_pattern="**/*.txt",
)

MHTML_SOURCE = DocumentSource(
    source_type="html",
    glob_pattern="**/*.mhtml",
)

HTML_SOURCE = DocumentSource(
    source_type="html",
    glob_pattern="**/*.html",
)


class Settings(BaseSettings):
    """Configuration for RAGPipeline."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="_",
        frozen=True,
        extra="ignore",
        case_sensitive=False,
    )

    pipeline_max_threads: int = Field(default=10, description="Maximum number of threads")

    # Embedding model settings
    pipeline_embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Name of the sentence-transformers model to use"
    )
    pipeline_embedding_model_kwargs: dict = Field(
        default={"device": "cuda"},
        description="Keyword arguments for the embedding model"
    )

    # Document source settings
    pipeline_sources: dict[str, list[Union[str, DocumentSource]]] = Field(
        default_factory=lambda: {
            os.getcwd(): [PDF_SOURCE, MHTML_SOURCE]
        },
        description="Sources to be used for the pipeline"
    )

    # Text splitting settings
    pipeline_chunk_size: int = Field(
        default=1000,
        ge=100,
        description="Size of text chunks for splitting documents"
    )
    pipeline_chunk_overlap: int = Field(
        default=200,
        ge=0,
        description="Overlap between chunks"
    )

    # Vector store settings
    pipeline_persist_directory: str = Field(
        default="chroma_db",
        description="Directory to store the vector database"
    )
    pipeline_collection_name: str = Field(
        default="default_collection",
        description="Name of the ChromaDB collection to use",
        min_length=3,
        max_length=512
    )

    # Retrieval settings
    pipeline_search_type: Literal["similarity", "mmr", "similarity_score_threshold"] = Field(
        default="similarity",
        description="Type of search to perform"
    )
    pipeline_k: int = Field(
        default=5,
        ge=1,
        description="Number of documents to retrieve"
    )
    pipeline_score_threshold: Optional[float] = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score (for similarity_score_threshold)"
    )
    pipeline_fetch_k: Optional[int] = Field(
        default=20,
        ge=1,
        description="Number of documents to fetch before filtering (for MMR)"
    )
    pipeline_lambda_mult: Optional[float] = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Diversity factor (for MMR, 0-1)"
    )

    # Prompt template
    pipeline_prompt_template: Optional[str] = Field(
        default="""You are a helpful assistant that can answer questions about the given context.
        You have access to the iX-Magazine for professional IT experts.
        If you don't have enough information to answer the question, say so.

        For each piece of information you use, please provide the source (Title, Journal, Year, Month and Page).
        Answer in the language of the question.

        Answer the question based on the following context:

        Context: {context}

        Question: {question}

        Answer:""",
        description="Optional custom prompt template for the RAG chain"
    )

    # LLM settings
    pipeline_llm_provider: Literal["ollama", "huggingface"] = Field(
        default="ollama",
        description="Provider of the LLM model"
    )
    pipeline_llm_model: str = Field(
        default="mistral",
        description="Name of the LLM model to use"
    )
    pipeline_llm_model_kwargs: dict = Field(
        default={"temperature": 0.3},
        description="Keyword arguments for the LLM model"
    )
    pipeline_llm_api_key: Optional[str] = Field(
        default=None,
        description="API key for the LLM provider (required for Hugging Face)",
        repr=False
    )

    @field_validator("pipeline_sources", mode="before")
    @classmethod
    def parse_sources(
        cls,
        v: Union[str, dict[str, list[Union[str, DocumentSource]]]]
    ) -> dict[str, list[DocumentSource]]:
        if not isinstance(v, (str, dict)):
            return v

        sources = {}
        for path, source_list in (json.loads(v) if isinstance(v, str) else v).items():
            sources[path] = []
            for source in source_list:
                if isinstance(source, str):
                    match source.lower():
                        case "pdf":
                            sources[path].append(PDF_SOURCE)
                        case "mhtml":
                            sources[path].append(MHTML_SOURCE)
                        case "html":
                            sources[path].append(HTML_SOURCE)
                        case "txt":
                            sources[path].append(TXT_SOURCE)
                        case _:
                            raise ValidationError(f"Invalid source type: {source}")
                elif isinstance(source, DocumentSource):
                    sources[path].append(source)
                elif isinstance(source, dict):
                    try:
                        sources[path].append(DocumentSource(**source))
                    except ValidationError:
                        raise ValidationError(f"Invalid source: {source}")
                else:
                    raise ValidationError(f"Invalid source: {source}")
        return sources
