import asyncio

from .pipeline import RAGPipeline


async def main():
    async with RAGPipeline() as pipeline:
        await pipeline.load_vectorstore()
        print("Vector store loaded")
        # documents = await pipeline.load_documents()
        # print("Documents loaded:", len(documents))
        # await asyncio.sleep(10)
        # processed_docs = await pipeline.process_documents(documents)
        # print("Documents processed:", len(processed_docs))
        # await asyncio.sleep(10)
        # await pipeline.create_vectorstore(processed_docs)
        # print("Vector store created")

        await asyncio.sleep(10)
        await pipeline.setup_retrieval_chain()
        print("Retrieval chain setup")
        await asyncio.sleep(10)
        answer = await pipeline.run("Wie kann man eine LLM lokal betreiben?")
        print(answer)

if __name__ == "__main__":
    asyncio.run(main())
