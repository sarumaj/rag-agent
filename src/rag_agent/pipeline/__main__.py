import asyncio
import sys
from .pipeline import RAGPipeline


async def main():
    async def ainput(string: str) -> str:
        await asyncio.to_thread(sys.stdout.write, f'{string} ')
        return (await asyncio.to_thread(sys.stdin.readline)).rstrip('\n')

    async with RAGPipeline() as pipeline:
        setup = await ainput("Setup pipeline? (y/n)")
        if setup == "y":
            documents = await pipeline.load_documents()
            processed_docs = await pipeline.process_documents(documents)
            await pipeline.update_vectorstore(processed_docs)
            await pipeline.setup_retrieval_chain()
        question = await ainput("Enter a question: ")
        answer = await pipeline.run(question)
        print("Answer:", answer, end="\n\n")

if __name__ == "__main__":
    asyncio.run(main())
