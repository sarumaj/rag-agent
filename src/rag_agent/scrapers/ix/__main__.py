import asyncio
import argparse

from .scraper import IXScraper


async def amain(args: argparse.Namespace):
    async with IXScraper() as scraper:
        await scraper.run()


def main():
    parser = argparse.ArgumentParser(
        prog="rag_agent.scrapers.ix",
        description="IX Scraper CLI",
        add_help=True
    )
    args = parser.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
