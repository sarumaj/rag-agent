import asyncio

from .scraper import IXScraper


async def main():
    async with IXScraper() as scraper:
        await scraper.run()


if __name__ == "__main__":
    asyncio.run(main())
