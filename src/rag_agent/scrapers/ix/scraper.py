import asyncio
from asyncio import Semaphore, Queue, create_task, gather, Event
import json
import signal
import re
import base64
from typing import List, Tuple, Set, Optional
import logging
import traceback
from aiohttp import ClientSession, ClientTimeout, TCPConnector
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from .article import Article, ArticleEncoder, ArticleDecoder
from .config import Settings

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.chrome.webdriver import WebDriver
    SCRAPER_AVAILABLE = True
except ImportError:
    SCRAPER_AVAILABLE = False

    class NotImported:
        def __getattr__(self, item):
            raise ModuleNotFoundError(
                "Selenium dependencies are not installed. "
                "Please install them using: pip install 'rag-agent[scraper]'"
            )

        def __call__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "Selenium dependencies are not installed. "
                "Please install them using: pip install 'rag-agent[scraper]'"
            )

    globals().update(dict.fromkeys(
        [
            "webdriver",
            "Options",
            "Service",
            "By",
            "WebDriverWait",
            "EC",
            "TimeoutException",
            "ChromeDriverManager",
            "WebDriver",
        ],
        NotImported()
    ))


logging.basicConfig(
    level=logging.INFO,
    style='{',
    format='{asctime} - {levelname} - {message}',
)

logger = logging.getLogger("rag_agent.scrapers.ix")


class IXScraper:
    """Scraper for the ix archive."""
    class _DriverContext:
        """Context manager for webdriver management."""

        def __init__(self, scraper: 'IXScraper'):
            self.scraper = scraper
            self.driver = None

        async def __aenter__(self) -> webdriver.Chrome:
            """Get a driver from the pool."""
            self.driver = await self.scraper._driver_queue.get()
            return self.driver

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            """Return the driver to the pool."""
            if self.driver:
                await self.scraper._driver_queue.put(self.driver)
                self.driver = None

    def __init__(self, config: Optional[Settings] = None):
        self._config = config or Settings()
        self._semaphore = Semaphore(self._config.ix_scraper_max_concurrent)
        self._connector = TCPConnector(limit=self._config.ix_scraper_max_concurrent)
        self._timeout = ClientTimeout(total=self._config.ix_scraper_timeout)
        self._config.ix_scraper_output_dir.mkdir(exist_ok=True, parents=True)
        self._driver_queue = Queue()
        self._thread_pool = ThreadPoolExecutor(max_workers=self._config.ix_scraper_max_threads)
        self._service = Service(ChromeDriverManager().install())
        self._shutdown_event = Event()
        self._running_tasks: Set[asyncio.Task] = set()
        self._file_lock = asyncio.Lock()

        logger.info(f"Using config: {self._config.model_dump_json(indent=2)}")

        self._lookup_registry: tuple[Article, ...] = ()
        if (path := (self._config.ix_scraper_output_dir / 'articles.json')).exists():
            try:
                with open(path, 'r') as f:
                    self._lookup_registry = tuple(el["article"] for el in json.load(f, cls=ArticleDecoder)["articles"])
                logger.info(f"Loaded lookup registry with {len(self._lookup_registry)} articles")
                logger.info(f"Lookup registry head: {self._lookup_registry[0] if self._lookup_registry else None}")
            except Exception as e:
                logger.error(f"Failed to load lookup registry: {e}")
        else:
            logger.info("Lookup registry not found")

        try:
            self._loop = asyncio.get_event_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

    def _setup_driver(self) -> WebDriver:
        """Setup Chrome driver with options.

        Returns:
            WebDriver: A configured Chrome webdriver instance

        Raises:
            WebDriverException: If there's an issue creating the driver
            ImportError: If selenium dependencies are not installed
        """
        try:
            options = Options()
            for option in self._config.ix_scraper_webdriver_options:
                options.add_argument(option)

            return webdriver.Chrome(service=self._service, options=options)
        except Exception as e:
            logger.error(f"Failed to setup driver: {e}")
            raise

    def _login_driver_sync(self, driver: webdriver.Chrome) -> bool:
        """Synchronous login for a single driver. Returns True if successful."""
        try:
            driver.get(self._config.ix_scraper_sign_in_url)

            wait = WebDriverWait(driver, self._config.ix_scraper_timeout)
            username_field = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input#login-user"))
            )
            username_field.send_keys(self._config.ix_scraper_username)

            password_field = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input#login-password"))
            )
            password_field.send_keys(self._config.ix_scraper_password)

            submit_button = wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button[type='submit']"))
            )
            submit_button.click()

            try:
                error_element = wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "a-notify[type='error']"))
                )
                if error_element.is_displayed():
                    logger.error(f"Login error: {error_element.text}")
                    return False

            except TimeoutException:
                wait.until(lambda driver: driver.current_url != self._config.ix_scraper_sign_in_url)
                return True

        except Exception as e:
            logger.error(f"Login failed: {str(e)}")
            return False

    async def _initialize_drivers(self):
        """Initialize a pool of drivers."""
        drivers = []
        try:
            drivers = [self._setup_driver() for _ in range(self._config.ix_scraper_max_threads)]
            successful_logins = 0
            with tqdm(total=len(drivers), desc="Initializing drivers") as pbar:
                async def login_with_progress(driver):
                    nonlocal successful_logins
                    for attempt in range(self._config.ix_scraper_retry_attempts):
                        success = await self._loop.run_in_executor(
                            self._thread_pool,
                            self._login_driver_sync,
                            driver
                        )
                        if success:
                            successful_logins += 1
                            pbar.update(1)
                            return
                        else:
                            logger.warning(f"Login attempt {attempt + 1} failed, retrying...")
                            await asyncio.sleep(1 * (attempt + 1))

                tasks = []
                for idx, driver in enumerate(drivers):
                    task = create_task(
                        coro=login_with_progress(driver),
                        name=f"Login {idx}",
                    )
                    self._running_tasks.add(task)

                    def callback(task: asyncio.Task):
                        self._running_tasks.discard(task)

                    task.add_done_callback(callback)
                    tasks.append(task)

                await asyncio.gather(*tasks)

            if successful_logins != len(drivers):
                raise Exception(f"Failed to login to {len(drivers) - successful_logins} drivers")

            for driver in drivers:
                await self._driver_queue.put(driver)

        except Exception as e:
            logger.error(f"Failed to initialize drivers: {e}")
            for driver in drivers:
                try:
                    driver.quit()
                except Exception:
                    pass
            raise

    async def _cleanup_drivers(self):
        """Clean up all drivers in the pool."""
        while not self._driver_queue.empty():
            driver = await self._driver_queue.get()
            driver.quit()

    def _lookup_article(self, issue_year: str, issue_number: str, article_id: str) -> Article:
        """Lookup an article by issue year, issue number, and article id."""
        for article in self._lookup_registry:
            if all([
                article.issue_year == issue_year,
                article.issue_number == issue_number,
                article.id == article_id,
            ]):
                return article
        return None

    async def _process_article(self, article: Article, pbar: tqdm) -> Tuple[Article, bool]:
        """Process a single article using a driver from the pool."""
        async with self._DriverContext(self) as driver:
            try:
                pbar.set_description(f"Processing {article.issue_year}.{article.issue_number}.{article.id}")
                return await self._loop.run_in_executor(
                    self._thread_pool,
                    self._capture_article,
                    driver,
                    article,
                )
            except Exception as e:
                logger.error(f"Failed to process {article.id}: {str(e)}")
                return article, False

    def _prepare_document(self, driver: webdriver.Chrome) -> None:
        """Prepare the document for PDF generation by cleaning up the HTML."""
        driver.execute_script("""
            // Get the main content
            const main = document.querySelector('main');
            if (!main) return;

            // Create a new document
            const newDoc = document.implementation.createHTMLDocument();

            // Copy all styles from the original document
            Array.from(document.styleSheets).forEach(sheet => {
                try {
                    const style = document.createElement('style');
                    style.textContent = Array.from(sheet.cssRules)
                        .map(rule => rule.cssText)
                        .join('\\n');
                    newDoc.head.appendChild(style);
                } catch (e) {
                    // Skip stylesheets that can't be accessed
                }
            });

            // Add override styles
            const overrideStyle = document.createElement('style');
            overrideStyle.textContent = `
                main {
                    margin: 0 !important;
                    width: 100% !important;
                    max-width: 100% !important;
                }
                body {
                    padding: 0 !important;
                    background-color: #fff !important;
                    margin: 0 !important;
                }
                html {
                    background-color: #fff !important;
                    --akwa-body-bg-color: #fff !important;
                    --akwa-bg-accent-color: #fff !important;
                    --akwa-bg-color: #fff !important;
                    --akwa-border-alternate-color: #fff !important;
                    --akwa-ads-bg-color: #fff !important;
                }
                @media print {
                    body {
                        break-before: avoid;
                        break-after: avoid;
                    }
                    main {
                        break-before: avoid;
                        break-after: avoid;
                    }
                    img {
                        break-inside: avoid;
                        max-width: 100% !important;
                        height: auto !important;
                    }
                    p {
                        text-align: justify !important;
                        text-justify: auto !important;
                        hyphens: auto !important;
                        orphans: 3 !important;
                        widows: 3 !important;
                    }
                }
            `;
            newDoc.head.appendChild(overrideStyle);

            // Remove unwanted elements
            const elementsToRemove = [
                '.purchase--single',
                '.pwsp',
                '.bottom-links',
                '.comment',
                'script[type="text/javascript"]'
            ];

            elementsToRemove.forEach(selector => {
                main.querySelectorAll(selector).forEach(el => el.remove());
            });

            // Fix lazy-loaded images and wrap them in containers
            main.querySelectorAll('a-img').forEach(aImg => {
                const img = document.createElement('img');
                const src = aImg.getAttribute('src');
                const style = aImg.getAttribute('style') || '';
                const alt = aImg.getAttribute('alt') || '';

                const container = document.createElement('div');
                container.style.cssText = style;

                img.src = src;
                img.alt = alt;
                img.style.cssText = `
                    max-width: 100%;
                    height: auto;
                    object-fit: contain;
                    display: block;
                `;

                container.appendChild(img);
                aImg.replaceWith(container);
            });

            // Wait for all images to load
            return new Promise((resolve) => {
                const images = main.querySelectorAll('img');
                let loaded = 0;
                const total = images.length;

                if (total === 0) {
                    resolve();
                    return;
                }

                const loadImage = (img) => {
                    return new Promise((imgResolve) => {
                        if (img.complete) {
                            imgResolve();
                            return;
                        }

                        img.onload = () => imgResolve();
                        img.onerror = () => imgResolve();
                    });
                };

                const checkProgress = () => {
                    loaded++;
                    if (loaded === total) {
                        resolve();
                    }
                };

                Promise.all(Array.from(images).map(img =>
                    loadImage(img).then(checkProgress)
                )).catch(() => resolve());
            }).then(() => {
                // Clone the main content
                const mainClone = main.cloneNode(true);
                newDoc.body.appendChild(mainClone);

                // Replace the current document
                document.replaceChild(newDoc.documentElement, document.documentElement);
            });
        """)

    def _capture_article(self, driver: webdriver.Chrome, article: Article) -> Tuple[Article, bool]:
        """Capture article using Chrome DevTools Protocol.

        Args:
            driver: The Chrome webdriver instance
            article: The article to capture

        Returns:
            Tuple[Article, bool]: The article and a boolean indicating success

        Raises:
            ValueError: If no export formats are specified
            TimeoutException: If page load times out
            WebDriverException: If there's an issue with the webdriver
            OSError: If there's an issue writing files
        """
        try:
            reference_article = None
            if not self._config.ix_scraper_overwrite:
                reference_article = self._lookup_article(article.issue_year, article.issue_number, article.id)
                if (
                    reference_article is not None and
                    len(reference_article.export_formats) * len(reference_article.files) == 0
                ):
                    logger.warning("No export formats specified for reference article")
                    reference_article = None

                logger.debug(
                    f"Found reference article: {reference_article} for {article}" if reference_article
                    else f"No reference article found for {article}"
                )

                if reference_article is not None and all([
                    (self._config.ix_scraper_output_dir / path).exists()
                    for path in getattr(reference_article, "files", [])
                ]) and all([
                    export_format in reference_article.export_formats
                    for export_format in self._config.ix_scraper_export_formats
                ]):
                    return reference_article, False

            driver.get(article.url)

            WebDriverWait(driver, self._config.ix_scraper_timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "main"))
            )

            # Extract article data from script
            article_data = driver.execute_script("""
                return window.articleData || {};
            """)

            # Validate extracted data
            if not article_data:
                logger.warning(f"No article data found for {article}")
            else:
                try:
                    article.issue_name = str(article_data["issueName"])
                    article.title = str(article_data["title"])
                    article.page = str(article_data["page"])
                except KeyError as e:
                    logger.warning(f"Missing key {e} in article data for {article}, article data: {article_data}")

            output_path_base = (
                self._config.ix_scraper_output_dir /
                article.issue_year /
                article.issue_number /
                f"{article.page or article.id or '0'}.ext"
            )

            self._prepare_document(driver)

            try:
                output_path_base.parent.mkdir(exist_ok=True, parents=True)
            except OSError as e:
                logger.error(f"Failed to create directory {output_path_base.parent}: {e}")
                raise

            for export_format in self._config.ix_scraper_export_formats:
                target_path = output_path_base.with_suffix(export_format.extension)
                if (
                    not self._config.ix_scraper_overwrite and
                    target_path.exists()
                ):
                    article.files.append(target_path)
                    article.export_formats.append(export_format)
                    logger.debug(
                        f"Skipping {export_format.extension} for article {article} "
                        "because it already exists"
                    )
                    continue

                try:
                    result = driver.execute_cdp_cmd(cmd=export_format.command, cmd_args=export_format.options)
                    data = base64.b64decode(result['data']) if export_format.base64encoded else result['data']
                    kwargs = {'mode': 'wb'} if isinstance(data, bytes) else {'mode': 'w', 'encoding': 'utf-8'}

                    try:
                        with open(file=target_path, **kwargs) as target:
                            target.write(data)
                    except OSError as e:
                        logger.error(f"Failed to write file {target_path}: {e}")
                        raise

                except Exception as e:
                    logger.warning(f"Failed to export {export_format.extension} for article {article.id}: {e}")
                    continue

                else:
                    article.files.append(target_path)
                    article.export_formats.append(export_format)

            return article, len(article.files) * len(article.export_formats) > 0

        except TimeoutException as e:
            logger.error(f"Timeout while processing article {article.id}: {e}")
            raise
        except Exception as e:
            trace = traceback.format_exc()
            logger.error(f"Failed to capture article {article.id}: {str(e)}")
            logger.error(trace)
            raise

    async def _fetch_links(self, session: ClientSession, url: str, class_name: str) -> List[str]:
        """Fetch links from a page with the given class name."""
        async with self._semaphore:
            async with session.get(url) as response:
                return [
                    link['href'] for link in BeautifulSoup(await response.text(), 'html.parser').
                    find_all('a', class_=class_name)
                ] if response.status == 200 else []

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        for sig in (signal.SIGTERM, signal.SIGINT):
            self._loop.add_signal_handler(sig, self._shutdown_event.set)

    async def _cleanup(self):
        """Cleanup resources during shutdown.

        This method ensures all resources are properly cleaned up, including:
        - Cancelling all running tasks
        - Closing all webdriver instances
        - Shutting down the thread pool
        """
        logger.info("Starting cleanup...")
        for task in self._running_tasks:
            if not task.done():
                logger.debug(f"Cancelling task {task.get_name()}")
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Error while cancelling task {task.get_name()}: {e}")

        try:
            await self._cleanup_drivers()
        except Exception as e:
            logger.error(f"Error while cleaning up drivers: {e}")

        try:
            self._thread_pool.shutdown(wait=True)
        except Exception as e:
            logger.error(f"Error while shutting down thread pool: {e}")

        logger.info("Cleanup completed")

    async def _process_archive(self):
        """Process the entire archive hierarchy."""
        await self._initialize_drivers()
        self._setup_signal_handlers()

        try:
            async with self._DriverContext(self) as driver:
                cookies = {cookie['name']: cookie['value'] for cookie in driver.get_cookies()}

            articles_to_process = []
            processed_articles = []
            path = self._config.ix_scraper_output_dir / 'articles.json'
            if path.exists():
                suffix = 0
                while True:
                    if not (new_path := path.with_suffix(f".json.bak{suffix if suffix > 0 else ''}")).exists():
                        path.rename(new_path)
                        break
                    suffix += 1

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

                with tqdm(total=len(issues_links), desc="Scanning issues") as pbar:
                    tasks = []
                    for issues_link in issues_links:
                        if self._shutdown_event.is_set():
                            break
                        task = create_task(
                            coro=self._process_archive_issues(session, issues_link),
                            name=issues_link,
                        )
                        self._running_tasks.add(task)

                        def callback(task: asyncio.Task):
                            self._running_tasks.discard(task)
                            pbar.update(1)

                        task.add_done_callback(callback)
                        tasks.append(task)

                    for articles in await gather(*tasks):
                        if self._shutdown_event.is_set():
                            break
                        articles_to_process.extend(articles)

            if len(articles_to_process) > 0 and not self._shutdown_event.is_set():
                with tqdm(total=len(articles_to_process), desc="Processing articles") as pbar:
                    tasks = []
                    for article in articles_to_process:
                        if self._shutdown_event.is_set():
                            break

                        task = create_task(
                            coro=self._process_article(article, pbar),
                            name=f"{article.issue_year}.{article.issue_number}.{article.id}",
                        )
                        self._running_tasks.add(task)

                        async def callback(task: asyncio.Task):
                            self._running_tasks.discard(task)
                            pbar.update(1)
                            try:
                                article, processed = task.result()
                                async with self._file_lock:
                                    processed_articles.append({
                                        "article": article,
                                        "processed": processed
                                    })
                                    with open(path, 'w') as f:
                                        json.dump({"articles": processed_articles}, f, indent=2, cls=ArticleEncoder)
                            except Exception as e:
                                logger.error(f"Failed to update articles.json: {e}")

                        task.add_done_callback(lambda t: asyncio.create_task(callback(t)))
                        tasks.append(task)

                    if tasks:
                        await gather(*tasks)

                if not self._shutdown_event.is_set():
                    self._lookup_registry = tuple(el["article"] for el in processed_articles)

        finally:
            await self._cleanup()

    async def _process_archive_issues(
        self,
        session: ClientSession,
        issues_link: str,
    ) -> List[Article]:
        """Process a single year's issues concurrently."""
        return [
            article for articles in await asyncio.gather(*[
                asyncio.create_task(
                    coro=self._process_archive_issue(session, issue_link),
                    name=issue_link,
                )
                for issue_link in await self._fetch_links(
                    session,
                    url=f"{self._config.ix_scraper_base_url}{issues_link}",
                    class_name="archive__year__link"
                )
            ]) for article in articles
        ]

    async def _process_archive_issue(
        self,
        session: ClientSession,
        issue_link: str,
    ) -> List[Article]:
        """Process a single month's articles."""
        articles = []
        for link in await self._fetch_links(
            session,
            url=f"{self._config.ix_scraper_base_url}{issue_link}",
            class_name="share-link share-link--read"
        ):
            parts = re.split(r"\D+/", link)[-1].rsplit("/", 2)
            if len(parts) not in (2, 3):
                logger.error(f"Invalid article URL: {link}")
                continue

            issue_year, issue_number, article_id = parts if len(parts) == 3 else (*parts, "")
            articles.append(Article(
                id=article_id,
                issue_year=issue_year,
                issue_number=issue_number,
                url=f"{self._config.ix_scraper_base_url}{link}",
            ))

        return articles

    async def run(self):
        """Run the scraper."""
        try:
            await self._process_archive()
        finally:
            await self._cleanup()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self._thread_pool.shutdown(wait=True)
