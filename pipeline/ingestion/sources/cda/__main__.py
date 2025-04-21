import asyncio

from .downloader import CDActionDownloader


async def main() -> None:
    downloader = CDActionDownloader()
    await downloader.run()

if __name__ == "__main__":
    asyncio.run(main())
