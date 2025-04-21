import asyncio

from .scrapper import ArchiveScraper


async def main():
    scraper = ArchiveScraper()
    await scraper.run()


if __name__ == "__main__":
    asyncio.run(main())
