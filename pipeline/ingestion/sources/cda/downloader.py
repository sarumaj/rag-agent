from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
from urllib.parse import unquote
import re
import asyncio
from aiohttp import ClientSession, ClientTimeout
from typing import Dict, List, Optional
from pathlib import Path
from .config import Config
from ....setup.logging import getLogger


logger = getLogger("cda.downloader")


class CDActionDownloader:
    """A class to handle downloading files from CDAction website."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize the downloader with configuration."""
        self._config = config or Config()
        self._service: Optional[Service] = None
        self._driver: Optional[webdriver.Chrome] = None
        self._wait: Optional[WebDriverWait] = None
        self._session: Optional[ClientSession] = None
        self._semaphore: Optional[asyncio.Semaphore] = None

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to be safe for filesystem."""
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        filename = filename.strip('. ')
        if not filename:
            filename = 'unnamed_file'
        if len(filename) > 200:
            filename = filename[:200]
        return filename

    async def _download_file(self, download_url: str, link_text: str) -> bool:
        """Download a single file with retry logic."""
        async with self._semaphore:
            logger.info(f"Starting download: {link_text}")

            fn = Path(self._sanitize_filename(link_text)).with_suffix('.pdf')
            fp = self._config.output_dir / fn

            if fp.exists():
                logger.info(f"Skipping existing file: {fn}")
                return True

            for attempt in range(self._config.max_retries):
                try:
                    async with self._session.get(
                        download_url,
                        timeout=ClientTimeout(total=self._config.timeout),
                    ) as response:
                        if response.status == 200:
                            if cd := response.headers.get('Content-Disposition'):
                                if match := re.search(r'filename="([^"]+)"', cd):
                                    fn = Path(self._sanitize_filename(unquote(match.group(1)))).with_suffix('.pdf')
                                    fp = self._config.output_dir / fn

                                    if fp.exists():
                                        logger.info(f"File already exists: {fn}")
                                        return True

                            with open(fp, 'wb') as f:
                                chunks = response.content.iter_chunked(8192)
                                async for chunk in chunks:
                                    f.write(chunk)

                            logger.info(f"Successfully downloaded: {fn}")
                            return True
                        else:
                            logger.error(
                                "Failed to download {}: HTTP {}".format(
                                    link_text,
                                    response.status,
                                )
                            )
                            if attempt < self._config.max_retries - 1:
                                await asyncio.sleep(2 ** attempt)
                                continue

                            return False

                except asyncio.TimeoutError:
                    logger.error("Timeout downloading {}".format(link_text))
                    if attempt < self._config.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue

                    return False

                except Exception as e:
                    logger.error(
                        "Error downloading {}: {}".format(
                            link_text,
                            e,
                        )
                    )
                    if attempt < self._config.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue

                    return False

            return False

    async def _download_all_files(
        self,
        download_links: List[webdriver.remote.webelement.WebElement],
        cookies: Dict[str, str],
    ) -> None:
        """Download all files concurrently."""
        self._semaphore = asyncio.Semaphore(self._config.max_concurrent)
        timeout = ClientTimeout(total=self._config.timeout)
        async with ClientSession(
            cookies=cookies,
            timeout=timeout,
            headers={'User-Agent': 'Mozilla/5.0'}
        ) as self._session:
            tasks = [
                self._download_file(
                    link.get_attribute("href"),
                    link.text.strip(),
                )
                for link in download_links
            ]

            results = await asyncio.gather(*tasks)
            successful_downloads = sum(1 for result in results if result)
            logger.info(
                "Download complete! Successfully downloaded {} out of {} files.".format(
                    successful_downloads,
                    len(download_links),
                )
            )

    def _setup_driver(self) -> None:
        """Set up the Selenium WebDriver."""
        options = Options()
        for option in self._config.webdriver_options:
            options.add_argument(option)

        self._service = Service(ChromeDriverManager().install())
        self._driver = webdriver.Chrome(service=self._service, options=options)
        self._wait = WebDriverWait(self._driver, self._config.timeout)

    def _login(self) -> None:
        """Perform login to CDAction website."""
        logger.info("Opening login page")
        self._driver.get(self._config.sign_in_url)

        logger.info("Locating login link")
        login_link = self._wait.until(
            EC.element_to_be_clickable(
                (By.CSS_SELECTOR, "a[onclick*='moOAuthLoginNew']")
            )
        )
        logger.info("Clicking login link")
        login_link.click()

        logger.info("Locating username field")
        username_field = self._wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "input#auth"))
        )
        logger.info("Filling in username")
        username_field.send_keys(self._config.username)

        logger.info("Locating password field")
        password_field = self._driver.find_element(By.CSS_SELECTOR, "input#password")
        logger.info("Filling in password")
        password_field.send_keys(self._config.password)

        logger.info("Locating submit button")
        submit_button = self._driver.find_element(
            By.CSS_SELECTOR,
            "button[type='submit']"
        )
        logger.info("Clicking submit button")
        submit_button.click()

        logger.info("Waiting for page to load")
        time.sleep(3)

        logger.info("Locating second submit button")
        second_submit = self._wait.until(
            EC.element_to_be_clickable(
                (By.CSS_SELECTOR, "button[type='submit']")
            )
        )
        logger.info("Clicking second submit button")
        second_submit.click()

        logger.info("Waiting for login to complete")
        time.sleep(3)

    def _get_download_links(self) -> List[webdriver.remote.webelement.WebElement]:
        """Navigate to downloads page and get all download links."""

        logger.info("Navigating to downloads page")
        self._driver.get(self._config.download_url)

        logger.info("Waiting for downloads page to load")
        self._wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "body")))

        self._config.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Extracting download links")
        download_links = self._driver.find_elements(
            By.CSS_SELECTOR,
            "a.woocommerce-MyAccount-downloads-file"
        )
        logger.info(f"Found {len(download_links)} download links")
        return download_links

    async def run(self) -> None:
        """Main method to perform the download process."""
        try:
            self._setup_driver()
            self._login()
            download_links = self._get_download_links()

            cookies = {
                cookie['name']: cookie['value']
                for cookie in self._driver.get_cookies()
            }
            await self._download_all_files(download_links, cookies)

        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            raise

        finally:
            if self._driver:
                self._driver.quit()
