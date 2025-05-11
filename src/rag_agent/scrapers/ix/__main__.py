import asyncio
import argparse

from .scraper import IXScraper
from .downloader import IXDownloader


async def amain(args: argparse.Namespace):
    class_ = IXDownloader if args.download else IXScraper
    async with class_() as scraper:
        await scraper.run()


def main():
    parser = argparse.ArgumentParser(
        prog="rag_agent.scrapers.ix",
        description="IX Scraper CLI",
        add_help=True
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the archive",
    )
    args = parser.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
