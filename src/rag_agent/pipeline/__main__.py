import asyncio
import argparse

from .chroma import ChromaRAGPipeline


async def amain(args: argparse.Namespace):
    match args.database_engine:
        case 'chroma':
            _cls = ChromaRAGPipeline
        case _:
            raise ValueError(f"Invalid pipeline: {args.pipeline}")

    async with _cls() as pipeline:
        if args.setup:
            documents = await pipeline.load_documents()
            processed_docs = await pipeline.process_documents(documents)
            await pipeline.update_vectorstore(processed_docs)

        await pipeline.setup_retrieval_chain()

        answer = await pipeline.run(args.question)
        print("Answer:", answer, end="\n\n")


def main():
    parser = argparse.ArgumentParser(
        prog="rag_agent.pipeline",
        description="RAG Pipeline CLI",
        add_help=True
    )
    parser.add_argument('--question', '-q', required=True, type=str, help='Question to ask the RAG system')
    parser.add_argument('--setup', '-s', action='store_true', help='Setup knowledge base (load and process documents)')
    parser.add_argument(
        '--database-engine',
        default="chroma",
        type=str,
        help='Database engine to use',
        choices=['chroma']
    )
    args = parser.parse_args()

    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
