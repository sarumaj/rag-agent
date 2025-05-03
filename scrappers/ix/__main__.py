import asyncio

from .scrapper import IXScraper


async def main():
    await IXScraper().run()


if __name__ == "__main__":
    asyncio.run(main())
