import asyncio
import argparse
from .pipeline import RAGPipeline


async def amain(args: argparse.Namespace):
    async with RAGPipeline() as pipeline:
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
    args = parser.parse_args()

    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
