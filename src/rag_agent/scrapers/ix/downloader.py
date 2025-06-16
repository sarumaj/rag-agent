from aiohttp import ClientSession, ClientResponse
from tqdm import tqdm
from pathlib import Path
from asyncio import create_task, gather
import asyncio
import re
import logging
import signal

from .scraper import IXScraper

logging.basicConfig(
    level=logging.INFO,
    style='{',
    format='{asctime} - {levelname} - {message}',
)

logger = logging.getLogger("rag_agent.scrapers.ix")


class IXDownloader(IXScraper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._shutdown_event = asyncio.Event()
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        loop = asyncio.get_event_loop()

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig,
                lambda s=sig: asyncio.create_task(self._handle_shutdown(s))
            )

    async def _handle_shutdown(self, sig: signal.Signals):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {sig.name}, initiating graceful shutdown...")
        self._shutdown_event.set()

        for task in self._running_tasks:
            if not task.done():
                task.cancel()

        if self._running_tasks:
            await asyncio.gather(*self._running_tasks, return_exceptions=True)

        logger.info("Shutdown complete")
        await self._cleanup()

    async def _download_archive(self):
        """Download the entire archive hierarchy."""
        await self._initialize_drivers()
        self._setup_signal_handlers()

        try:
            async with self._DriverContext(self) as driver:
                cookies = {cookie['name']: cookie['value'] for cookie in driver.get_cookies()}

            async with ClientSession(
                connector=self._connector,
                timeout=self._timeout,
                cookies=cookies
            ) as session:
                pbar = tqdm(desc="Fetching archive structure")
                issues_links = await self._fetch_links(
                    session,
                    url=self._config.ix_scraper_archive_url,
                    class_name="archive__years__link"
                )
                pbar.close()

                with tqdm(total=len(issues_links), desc="Downloading issues") as pbar:
                    tasks = []
                    for issues_link in issues_links:
                        if self._shutdown_event.is_set():
                            break

                        task = create_task(
                            coro=self._download_articles(session, issues_link),
                            name=issues_link,
                        )
                        self._running_tasks.add(task)

                        def callback(task: asyncio.Task):
                            self._running_tasks.discard(task)
                            pbar.update(1)

                        task.add_done_callback(callback)
                        tasks.append(task)

                    await gather(*tasks)
        finally:
            await self._cleanup()

    async def _download_articles(
        self,
        session: ClientSession,
        issues_link: str,
    ):
        """Download all articles for a single year concurrently."""
        await asyncio.gather(*[
            asyncio.create_task(
                coro=self._download_article(session, issue_link),
                name=issue_link,
            )
            for issue_link in await self._fetch_links(
                session,
                url=f"{self._config.ix_scraper_base_url}{issues_link}",
                class_name="archive__year__link"
            )
        ])

    async def _download_article(self, session: ClientSession, issue_link: str):
        """Download an article."""
        for link in await self._fetch_links(
            session,
            url=f"{self._config.ix_scraper_base_url}{issue_link}",
            class_name="issue__button issue__button--pdf issue-download-link"
        ):

            async def save(file_path: Path, response: ClientResponse):
                if not file_path.parent.exists():
                    file_path.parent.mkdir(parents=True, exist_ok=True)

                with open(file_path, "wb") as f:
                    logger.debug(f"Saving article to {file_path}")
                    async for chunk in response.content.iter_chunked(8192):
                        f.write(chunk)

            logger.debug(f"Downloading article {link}")
            parts = tuple(filter(None, re.split(r"\D+/?", link)))
            if len(parts) != 2:
                logger.error(f"Invalid article URL: {link}")
                continue

            issue_year, issue_number = parts
            file_name = f"ix.{issue_year[-2:]}.{int(issue_number):02d}.pdf"
            async with session.get(
                f"{self._config.ix_scraper_base_url}{link}",
                headers={"Accept": (
                    "application/pdf,"
                    "binary/octet-stream,"
                    "*/*"
                )}
            ) as response:
                try:
                    response.raise_for_status()
                except Exception as e:
                    logger.error(f"Error downloading article {link}: {e}")
                    return

                match response.content_type:
                    case "binary/octet-stream":
                        await save(self._config.ix_scraper_output_dir / issue_year / file_name, response)

                    case _:
                        content = await response.text()
                        logger.error(
                            "Unexpected content type: {content_type}, link: {link}, "
                            "headers: {headers}, content: {content}"
                            .format(
                                content_type=response.content_type,
                                link=f"{self._config.ix_scraper_base_url}{link}",
                                headers=response.headers,
                                content=content[:100] + "..." if len(content) > 100 else content
                            )
                        )

    async def run(self):
        """Run the downloader."""
        try:
            await self._download_archive()
        finally:
            await self._cleanup()
