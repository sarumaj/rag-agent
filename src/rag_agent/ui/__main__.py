from typing import Dict, Any, TypedDict, NotRequired
import traceback
from itertools import cycle
import argparse
import json
import asyncio
from datetime import datetime
from pathlib import Path

from ..pipeline.pipeline import RAGPipeline
from ..pipeline.config import (
    Settings,
    DocumentSource,
    PDF_SOURCE,
    MHTML_SOURCE,
    HTML_SOURCE,
    TXT_SOURCE,
)

try:
    import panel as pn
    UI_FRAMEWORK_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    UI_FRAMEWORK_AVAILABLE = False

    class NotImported:
        exc = ModuleNotFoundError(
            "Panel is not installed. "
            "Please install it using: pip install 'rag-agent[ui]'"
        )

        def __getattr__(self, item):
            raise self.exc

        def __call__(self, *args, **kwargs):
            raise self.exc

    globals().update(dict.fromkeys(
        [
            "panel",
            "pn",
        ],
        NotImported()
    ))

pn.extension('jsoneditor', notifications=True)


class ChatWithConfigurableMessages(pn.chat.ChatInterface):
    """A chat interface with context toggle functionality."""
    def __init__(self, message_kwargs: dict[str, Any] = {}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._message_kwargs = message_kwargs

    def send(
        self,
        value: Any,
        user: str | None = None,
        avatar: str | None = None,
        respond: bool = True,
        trigger_post_hook: bool = True,
        **kwargs: dict[str, Any]
    ):
        """Send a message with context toggle functionality."""
        kwargs.update(self._message_kwargs)
        super(ChatWithConfigurableMessages, self).send(
            value=value,
            user=user,
            avatar=avatar,
            respond=respond,
            trigger_post_hook=trigger_post_hook,
            **kwargs
        )


class MessageWithContext(pn.chat.ChatMessage):
    """A message with context toggle functionality."""

    class MessageParams(TypedDict):
        object: str
        user: str
        avatar: str
        tooltip: str

    def __init__(self, object: Dict[str, Any], **kwargs):
        self._ctx_params = cycle([
            MessageWithContext.MessageParams(
                object=object["generate"]["answer"],
                user="Assistant",
                avatar="🤖",
                tooltip="Show context",
            ),
            MessageWithContext.MessageParams(
                object="**Context**:\n\n```json\n{context}\n```".format(context=json.dumps(
                    json.loads(object['generate']['context']),
                    indent=2
                )),
                user="System",
                avatar="⚙️",
                tooltip="Hide context",
            ),
        ])

        params = next(self._ctx_params)
        kwargs.update(**{k: v for k, v in params.items() if k not in ["tooltip"]})
        super().__init__(**kwargs)

        toggle_context_icon = pn.widgets.ToggleIcon(
            description=params["tooltip"],
            icon="zoom-in-area",
            active_icon="zoom-out-area",
        )
        self._icons_row.insert(0, toggle_context_icon)
        toggle_context_icon.param.watch(self._toggle_context, "value")

    def _toggle_context(self, event: Any) -> None:
        """Toggle between answer and context views."""
        params = next(self._ctx_params)
        event.obj.description = params["tooltip"]
        self.object = params["object"]
        self.param.update(
            user=params["user"],
            avatar=params["avatar"],
        )


class RAGChatInterface:
    """Main chat interface for RAG system."""

    class InitialEvent:
        def __init__(self, obj: pn.widgets.Widget):
            self.new = obj.value
            self.old = None
            self.type = "change"
            self.obj = obj
            self.cls = obj

    class SourceConfig(TypedDict):
        """Source configuration."""
        sources: pn.widgets.JSONEditor

    class EmbeddingConfig(TypedDict):
        """Embedding model configuration."""
        model: pn.widgets.TextInput
        device: pn.widgets.Select

    class TextSplittingConfig(TypedDict):
        """Text splitting configuration."""
        chunk_size: pn.widgets.IntInput
        chunk_overlap: pn.widgets.IntInput

    class VectorStoreConfig(TypedDict):
        """Vector store configuration."""
        persist_directory: pn.widgets.TextInput
        collection_name: pn.widgets.TextInput

    class RetrievalConfig(TypedDict):
        """Retrieval configuration.

        Optional fields are dependent on search_type:
        - score_threshold: Required for similarity_score_threshold
        - fetch_k, lambda_mult: Required for mmr
        """
        search_type: pn.widgets.Select
        k: pn.widgets.IntInput
        score_threshold: NotRequired[pn.widgets.FloatInput]
        fetch_k: NotRequired[pn.widgets.IntInput]
        lambda_mult: NotRequired[pn.widgets.FloatInput]

    class LLMConfig(TypedDict):
        """LLM configuration.

        Optional fields:
        - api_key: Required for huggingface provider
        """
        provider: pn.widgets.Select
        model: pn.widgets.TextInput
        temperature: pn.widgets.FloatInput
        api_key: NotRequired[pn.widgets.PasswordInput]

    class ConfigWidgets(TypedDict):
        """All configuration widgets."""
        sources: list['RAGChatInterface.SourceConfig']
        embedding: 'RAGChatInterface.EmbeddingConfig'
        text_splitting: 'RAGChatInterface.TextSplittingConfig'
        vector_store: 'RAGChatInterface.VectorStoreConfig'
        retrieval: 'RAGChatInterface.RetrievalConfig'
        llm: 'RAGChatInterface.LLMConfig'

    class ConfigSections(TypedDict):
        """UI sections for configuration."""
        sources: pn.Column
        embedding: pn.Column
        text_splitting: pn.Column
        vector_store: pn.Column
        retrieval: pn.Column
        llm: pn.Column

    def __init__(self):
        self._pipeline: RAGPipeline | None = None
        self._reload_documents = False
        self._pipeline_config = Settings()
        self._config_widgets = self._create_config_widgets()
        self._chat_interface = self._create_chat_interface()
        self._layout = self._create_layout()

    def _create_config_widgets(self) -> list[pn.widgets.Widget]:
        """Create the configuration section with widgets for all pipeline settings."""
        widgets: RAGChatInterface.ConfigWidgets = {
            "sources": {
                "sources": pn.widgets.JSONEditor(
                    name="Sources",
                    value={
                        k: [
                            v.as_dict() if isinstance(v, DocumentSource) else v for v in vv
                        ] for k, vv in self._pipeline_config.pipeline_sources.items()
                    },
                    mode="text",
                    search=False,
                    menu=False,
                    sizing_mode="stretch_width",
                    schema={
                        "type": "object",
                        "description": "Sources to be used for the pipeline",
                        "patternProperties": {
                            "^.+$": {
                                "type": "array",
                                "items": {
                                    "oneOf": [
                                        {
                                            "type": "string",
                                            "enum": ["pdf", "mhtml", "html", "txt"]
                                        },
                                        {
                                            "type": "object",
                                            "properties": {
                                                "source_type": {
                                                    "type": "string",
                                                    "enum": ["txt", "pdf", "html"]
                                                },
                                                "meta_pattern": {
                                                    "type": "string",
                                                    "description": "Regex pattern for extracting metadata from the source path"
                                                },
                                                "glob_pattern": {
                                                    "type": "string",
                                                    "description": "Glob pattern for matching files in directory"
                                                }
                                            },
                                            "required": ["source_type", "glob_pattern"],
                                            "additionalProperties": False
                                        }
                                    ]
                                }
                            }
                        },
                        "additionalProperties": False
                    },
                ),
            },
            "embedding": {
                "model": pn.widgets.TextInput(
                    name="Embedding Model",
                    value=self._pipeline_config.pipeline_embedding_model,
                    placeholder="e.g., all-MiniLM-L6-v2"
                ),
                "device": pn.widgets.AutocompleteInput(
                    name="Embedding Device",
                    value=self._pipeline_config.pipeline_embedding_model_kwargs.get("device", "cuda"),
                    options=["cuda", "cpu", "mps"],
                    restrict=False,
                ),
            },
            "text_splitting": {
                "chunk_size": pn.widgets.IntInput(
                    name="Chunk Size",
                    value=self._pipeline_config.pipeline_chunk_size,
                    start=100,
                    step=100
                ),
                "chunk_overlap": pn.widgets.IntInput(
                    name="Chunk Overlap",
                    value=self._pipeline_config.pipeline_chunk_overlap,
                    start=0,
                    step=50
                ),
            },
            "vector_store": {
                "persist_directory": pn.widgets.TextInput(
                    name="Persist Directory",
                    value=self._pipeline_config.pipeline_persist_directory,
                    placeholder="e.g., chroma_db"
                ),
                "collection_name": pn.widgets.TextInput(
                    name="Collection Name",
                    value=self._pipeline_config.pipeline_collection_name,
                    placeholder="e.g., default_collection"
                ),
            },
            "retrieval": {
                "search_type": pn.widgets.Select(
                    name="Search Type",
                    value=self._pipeline_config.pipeline_search_type,
                    options=["similarity", "mmr", "similarity_score_threshold"]
                ),
                "k": pn.widgets.IntInput(
                    name="Number of Documents (k)",
                    value=self._pipeline_config.pipeline_k,
                    start=1,
                    step=1
                ),
                "score_threshold": pn.widgets.FloatInput(
                    name="Score Threshold",
                    value=self._pipeline_config.pipeline_score_threshold or 0.5,
                    start=0.0,
                    end=1.0,
                    step=0.1
                ),
                "fetch_k": pn.widgets.IntInput(
                    name="Fetch k",
                    value=self._pipeline_config.pipeline_fetch_k or 20,
                    start=1,
                    step=1
                ),
                "lambda_mult": pn.widgets.FloatInput(
                    name="Lambda Multiplier",
                    value=self._pipeline_config.pipeline_lambda_mult or 0.5,
                    start=0.0,
                    end=1.0,
                    step=0.1
                ),
            },
            "llm": {
                "provider": pn.widgets.Select(
                    name="LLM Provider",
                    value=self._pipeline_config.pipeline_llm_provider,
                    options=["ollama", "huggingface"]
                ),
                "model": pn.widgets.TextInput(
                    name="LLM Model",
                    value=self._pipeline_config.pipeline_llm_model,
                    placeholder="e.g., mistral"
                ),
                "temperature": pn.widgets.FloatInput(
                    name="Temperature",
                    value=self._pipeline_config.pipeline_llm_model_kwargs.get("temperature", 0.3),
                    start=0.0,
                    end=1.0,
                    step=0.1
                ),
                "api_key": pn.widgets.PasswordInput(
                    name="API Key",
                    value=self._pipeline_config.pipeline_llm_api_key or "",
                    placeholder="Enter API key for Hugging Face"
                ),
            },
        }

        sections: RAGChatInterface.ConfigSections = {
            "sources": pn.Column(
                widgets["sources"]["sources"],
                pn.layout.Divider(),
                name="Source Settings"
            ),
            "embedding": pn.Column(
                widgets["embedding"]["model"],
                widgets["embedding"]["device"],
                pn.layout.Divider(),
                name="Embedding Model Settings"
            ),
            "text_splitting": pn.Column(
                widgets["text_splitting"]["chunk_size"],
                widgets["text_splitting"]["chunk_overlap"],
                pn.layout.Divider(),
                name="Text Splitting Settings"
            ),
            "vector_store": pn.Column(
                widgets["vector_store"]["persist_directory"],
                widgets["vector_store"]["collection_name"],
                pn.layout.Divider(),
                name="Vector Store Settings"
            ),
            "retrieval": pn.Column(
                widgets["retrieval"]["search_type"],
                widgets["retrieval"]["k"],
                pn.layout.Divider(),
                name="Retrieval Settings"
            ),
            "llm": pn.Column(
                widgets["llm"]["provider"],
                widgets["llm"]["model"],
                widgets["llm"]["temperature"],
                pn.layout.Divider(),
                name="LLM Settings"
            ),
        }

        def update_llm_section(event: Any):
            match event.new:
                case "huggingface":
                    if widgets["llm"]["api_key"] not in sections["llm"]:
                        sections["llm"].insert(-1, widgets["llm"]["api_key"])
                case _:
                    if widgets["llm"]["api_key"] in sections["llm"]:
                        sections["llm"].remove(widgets["llm"]["api_key"])

        def update_retrieval_section(event: Any):
            for widget in [
                widgets["retrieval"]["score_threshold"],
                widgets["retrieval"]["fetch_k"],
                widgets["retrieval"]["lambda_mult"]
            ]:
                if widget in sections["retrieval"]:
                    sections["retrieval"].remove(widget)

            match event.new:
                case "mmr":
                    sections["retrieval"].insert(-1, widgets["retrieval"]["fetch_k"])
                    sections["retrieval"].insert(-1, widgets["retrieval"]["lambda_mult"])
                case "similarity_score_threshold":
                    sections["retrieval"].insert(-1, widgets["retrieval"]["score_threshold"])

        update_llm_section(RAGChatInterface.InitialEvent(widgets["llm"]["provider"]))
        update_retrieval_section(RAGChatInterface.InitialEvent(widgets["retrieval"]["search_type"]))

        widgets["llm"]["provider"].param.watch(update_llm_section, "value")
        widgets["retrieval"]["search_type"].param.watch(update_retrieval_section, "value")

        reload_documents_checkbox = pn.widgets.Checkbox(
            name="Reload Documents",
            value=False,
        )
        reload_documents_checkbox.param.watch(self._toggle_reload_documents, "value")
        apply_button = pn.widgets.Button(
            name="Apply Changes",
            button_type="primary"
        )
        apply_button.on_click(self._apply_config_changes)

        return [
            *sections.values(),
            reload_documents_checkbox,
            apply_button,
        ]

    def _toggle_reload_documents(self, event: Any) -> None:
        """Toggle the reload documents checkbox."""
        self._reload_documents = event.new

    def _apply_config_changes(self, event: Any) -> None:
        """Apply configuration changes and reinitialize the pipeline."""
        event.obj.disabled = True
        try:
            sources_section = self._config_widgets[0]
            embedding_section = self._config_widgets[1]
            text_splitting_section = self._config_widgets[2]
            vector_store_section = self._config_widgets[3]
            retrieval_section = self._config_widgets[4]
            llm_section = self._config_widgets[5]

            config_kwargs = {
                "pipeline_sources": {
                    k: [
                        DocumentSource.from_dict(v) if isinstance(v, dict) else
                        {
                            "pdf": PDF_SOURCE,
                            "mhtml": MHTML_SOURCE,
                            "html": HTML_SOURCE,
                            "txt": TXT_SOURCE,
                        }[v] if isinstance(v, str) and v in ["pdf", "mhtml", "html", "txt"] else v
                        for v in vv
                    ] for k, vv in sources_section[0].value.items()
                },
                "pipeline_embedding_model": embedding_section[0].value,
                "pipeline_embedding_model_kwargs": {"device": embedding_section[1].value},
                "pipeline_chunk_size": text_splitting_section[0].value,
                "pipeline_chunk_overlap": text_splitting_section[1].value,
                "pipeline_persist_directory": vector_store_section[0].value,
                "pipeline_collection_name": vector_store_section[1].value,
                "pipeline_search_type": retrieval_section[0].value,
                "pipeline_k": retrieval_section[1].value,
                "pipeline_llm_provider": llm_section[0].value,
                "pipeline_llm_model": llm_section[1].value,
                "pipeline_llm_model_kwargs": {"temperature": llm_section[2].value},
            }

            match retrieval_section[0].value:
                case "similarity_score_threshold":
                    config_kwargs["pipeline_score_threshold"] = float(retrieval_section[2].value)
                case "mmr":
                    config_kwargs["pipeline_fetch_k"] = int(retrieval_section[2].value)
                    config_kwargs["pipeline_lambda_mult"] = float(retrieval_section[3].value)

            match llm_section[0].value:
                case "huggingface":
                    config_kwargs["pipeline_llm_api_key"] = llm_section[3].value

            self._pipeline_config = Settings(**config_kwargs)
            self._pipeline = None
            asyncio.run(self._initialize_pipeline())

        except Exception as e:
            pn.state.notifications.error(
                f"Error applying configuration: {str(e)}\n\n{traceback.format_exc()}",
                duration=10000
            )
        else:
            pn.state.notifications.success("Configuration applied successfully!", duration=3000)
        finally:
            event.obj.disabled = False

    async def _initialize_pipeline(self) -> None:
        """Initialize the RAG pipeline."""
        if self._pipeline is None:
            self._pipeline = RAGPipeline(config=self._pipeline_config)

            if self._reload_documents:
                documents = await self._pipeline.load_documents()
                processed_documents = await self._pipeline.process_documents(documents)
                await self._pipeline.update_vectorstore(processed_documents)

            await self._pipeline.setup_retrieval_chain(context_format="json")

    async def _process_question(self, question: str) -> Dict[str, Any]:
        """Process a question through the RAG pipeline."""
        if self._pipeline is None:
            await self._initialize_pipeline()
        return await self._pipeline.run(question)

    async def _chat_callback(
        self,
        contents: str,
        user: str,
        instance: pn.chat.ChatInterface
    ):
        """Callback function for chat interface."""
        try:
            match user:
                case "Assistant" | "System":
                    pass

                case "User":
                    message = MessageWithContext(
                        object=await self._process_question(contents),
                        show_copy_icon=True,
                        show_edit_icon=False,
                        show_timestamp=True,
                        show_reaction_icons=False,
                    )
                    instance.send(value=message, user=None, avatar=None)
                    return

                case _:
                    raise ValueError(f"Invalid user: {user}")

        except Exception as e:
            yield pn.chat.ChatMessage(
                object=f"**Error**: {str(e)}\n\n```python\n{traceback.format_exc()}\n```",
                user="System",
                avatar="⚠️"
            )

    def _create_chat_interface(self) -> pn.chat.ChatInterface:
        """Create the chat interface component."""
        def save_chat_history(instance: pn.chat.ChatInterface, event: Any):
            """Save the chat history to a file."""
            filename = Path(f"chat_history_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.html")
            try:
                instance.save(filename)
            except Exception as e:
                pn.state.notifications.error(f"Error saving chat history: {str(e)}", duration=10000)
            else:
                pn.state.notifications.success(
                    f"Chat history saved successfully to: {filename.resolve()}!",
                    duration=3000
                )

        return ChatWithConfigurableMessages(
            message_kwargs={"show_reaction_icons": False},
            callback=self._chat_callback,
            widgets=pn.chat.ChatAreaInput(
                placeholder="Ask your question!",
                sizing_mode="stretch_width"
            ),
            sizing_mode="stretch_width",
            show_send=True,
            show_stop=True,
            show_rerun=True,
            show_undo=False,
            show_clear=True,
            show_button_name=False,
            avatar="👤",
            height=600,
            button_properties={"save": {"icon": "device-floppy", "callback": save_chat_history}}
        )

    def _create_layout(self) -> pn.template.BootstrapTemplate:
        """Create the main layout."""
        return pn.template.BootstrapTemplate(
            title="RAG Agent Chat Interface",
            main=[self._chat_interface],
            sidebar=[
                pn.layout.Column(
                    *self._config_widgets[:-2],
                    self._config_widgets[-2],
                    pn.layout.Row(
                        pn.layout.HSpacer(),
                        self._config_widgets[-1],
                    ),
                )
            ],
            sidebar_width=350,
            theme=pn.template.DefaultTheme,
            collapsed_sidebar=True,
        )

    def serve(self, port: int = 8501) -> None:
        """Serve the application."""
        pn.state.onload(self._initialize_pipeline)
        pn.serve(
            self._layout,
            port=port,
            show=False,
            title="RAG Chat Interface",
            allow_websocket_origin=["*"],
        )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="RAG Chat Interface")
    parser.add_argument("--port", type=int, default=8501, help="Port to serve the application on")
    args = parser.parse_args()

    chat = RAGChatInterface()
    chat.serve(port=args.port)


if __name__ == "__main__":
    main()
