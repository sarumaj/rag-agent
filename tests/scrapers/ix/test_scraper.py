import asyncio
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

try:
    from scrapers.ix.scraper import WebDriver
    from scrapers.ix.scraper import Service
    from scrapers.ix.scraper import TimeoutException
    from scrapers.ix.scraper import IXScraper
    from scrapers.ix.config import Settings, ExportFormat
    from scrapers.ix.article import Article
    from scrapers.ix.scraper import SCRAPER_AVAILABLE
except (ModuleNotFoundError, ImportError) as e:
    pytest.skip(reason=str(e), allow_module_level=True)
else:
    if SCRAPER_AVAILABLE is not True:
        pytest.skip(reason="scraper dependencies are not installed", allow_module_level=True)


@pytest.fixture
def mock_settings():
    return Settings(
        ix_scraper_max_threads=1,
        ix_scraper_max_concurrent=1,
        ix_scraper_timeout=1,
        ix_scraper_retry_attempts=1,
        ix_scraper_output_dir=Path("/tmp/test_output"),
        ix_scraper_webdriver_options=["--headless"],
        ix_scraper_export_formats=[ExportFormat(extension=".pdf", command="print")],
        ix_scraper_overwrite=False,
        ix_scraper_username="test_username",
        ix_scraper_password="test_password",
    )


@pytest.fixture
def mock_driver():
    driver = Mock(spec=WebDriver)
    driver.current_url = "https://example.com"
    return driver


@pytest.fixture
def mock_webdriver():
    with patch("scrapers.ix.scraper.webdriver") as mock:
        mock.Chrome.return_value = Mock(spec=WebDriver)
        yield mock


@pytest.fixture
def mock_service():
    with patch("scrapers.ix.scraper.Service") as mock:
        mock.return_value = Mock(spec=Service)
        yield mock


@pytest.fixture
def mock_chrome_driver_manager():
    with patch("scrapers.ix.scraper.ChromeDriverManager") as mock:
        mock.return_value.install.return_value = "/path/to/chromedriver"
        yield mock


@pytest.mark.asyncio
async def test_scraper_initialization(mock_settings, mock_webdriver, mock_service, mock_chrome_driver_manager):
    """Test IXScraper initialization."""
    scraper = IXScraper(config=mock_settings)

    assert scraper._config == mock_settings
    assert scraper._semaphore._value == mock_settings.ix_scraper_max_concurrent
    assert scraper._connector.limit == mock_settings.ix_scraper_max_concurrent
    assert scraper._timeout.total == mock_settings.ix_scraper_timeout
    assert scraper._thread_pool._max_workers == mock_settings.ix_scraper_max_threads
    assert scraper._shutdown_event.is_set() is False


@pytest.mark.asyncio
async def test_driver_setup(mock_settings, mock_webdriver, mock_service, mock_chrome_driver_manager):
    """Test driver setup."""
    scraper = IXScraper(config=mock_settings)
    driver = scraper._setup_driver()

    mock_webdriver.Chrome.assert_called_once()
    assert isinstance(driver, Mock)


@pytest.mark.asyncio
async def test_driver_login_success(mock_settings, mock_driver):
    """Test successful driver login."""
    scraper = IXScraper(config=mock_settings)

    # Mock WebDriverWait and its methods
    with patch("scrapers.ix.scraper.WebDriverWait") as mock_wait, \
         patch("scrapers.ix.scraper.EC") as mock_ec:

        mock_ec.presence_of_element_located.return_value = Mock()
        mock_ec.element_to_be_clickable.return_value = Mock()

        error_check_mock = Mock()
        error_check_mock.until.side_effect = TimeoutException()

        url_check_mock = Mock()
        url_check_mock.until.return_value = True

        mock_wait.return_value.until.side_effect = [
            Mock(send_keys=Mock()),  # username field
            Mock(send_keys=Mock()),  # password field
            Mock(click=Mock()),      # submit button
            error_check_mock.until,  # error element check (should timeout)
            url_check_mock.until     # URL check (should succeed)
        ]

        scraper._login_driver_sync(mock_driver)
        mock_wait.assert_called_once()
        calls = mock_wait.return_value.until.call_args_list
        assert len(calls) == 4


@pytest.mark.asyncio
async def test_driver_login_failure(mock_settings, mock_driver):
    """Test failed driver login."""
    scraper = IXScraper(config=mock_settings)

    # Mock WebDriverWait and its methods
    with patch("scrapers.ix.scraper.WebDriverWait") as mock_wait:
        mock_wait.return_value.until.side_effect = [
            Mock(send_keys=Mock()),  # username field
            Mock(send_keys=Mock()),  # password field
            Mock(click=Mock()),      # submit button
            Mock(is_displayed=Mock(return_value=True), text="Login failed")  # error element
        ]

        success = scraper._login_driver_sync(mock_driver)
        assert success is False


@pytest.mark.asyncio
async def test_driver_initialization(mock_settings, mock_webdriver, mock_service, mock_chrome_driver_manager):
    """Test driver initialization."""
    scraper = IXScraper(config=mock_settings)

    with patch.object(scraper, "_login_driver_sync", return_value=True):
        await scraper._initialize_drivers()

        assert not scraper._driver_queue.empty()
        driver = await scraper._driver_queue.get()
        assert isinstance(driver, Mock)


@pytest.mark.asyncio
async def test_driver_cleanup(mock_settings, mock_webdriver, mock_service, mock_chrome_driver_manager):
    """Test driver cleanup."""
    scraper = IXScraper(config=mock_settings)

    # Initialize a driver
    with patch.object(scraper, "_login_driver_sync", return_value=True):
        await scraper._initialize_drivers()

    # Cleanup drivers
    await scraper._cleanup_drivers()
    assert scraper._driver_queue.empty()


@pytest.mark.asyncio
async def test_article_lookup(mock_settings):
    """Test article lookup functionality."""
    scraper = IXScraper(config=mock_settings)

    # Create test articles
    article1 = Article(
        id="123",
        issue_year="2024",
        issue_number="1",
        url="https://example.com/article1"
    )
    article2 = Article(
        id="456",
        issue_year="2024",
        issue_number="2",
        url="https://example.com/article2"
    )

    # Set lookup registry
    scraper._lookup_registry = (article1, article2)

    # Test lookup success
    found = scraper._lookup_article("2024", "1", "123")
    assert found == article1

    # Test lookup failure
    not_found = scraper._lookup_article("2024", "3", "789")
    assert not_found is None


@pytest.mark.asyncio
async def test_article_processing(mock_settings, mock_driver):
    """Test article processing."""
    scraper = IXScraper(config=mock_settings)

    # Create a mock driver queue
    scraper._driver_queue = asyncio.Queue()
    await scraper._driver_queue.put(mock_driver)

    article = Article(
        id="123",
        issue_year="2024",
        issue_number="1",
        url="https://example.com/article"
    )

    # Mock the progress bar
    mock_pbar = Mock()

    # Mock the capture_article method
    with patch.object(scraper, "_capture_article", return_value=(article, True)):
        result = await scraper._process_article(article, mock_pbar)
        assert result == (article, True)
        scraper._capture_article.assert_called_once_with(mock_driver, article)


@pytest.mark.asyncio
async def test_scraper_context_manager(mock_settings, mock_webdriver, mock_service, mock_chrome_driver_manager):
    """Test scraper context manager functionality."""
    async with IXScraper(config=mock_settings) as scraper:
        assert isinstance(scraper, IXScraper)
        with (
            patch.object(scraper, "_setup_driver", return_value=Mock()) as mock_setup_driver,
            patch.object(scraper, "_login_driver_sync") as mock_login_driver,
        ):
            await scraper._initialize_drivers()
            mock_setup_driver.assert_called_once()
            mock_login_driver.assert_called_once()

        # Test that drivers are initialized
        assert not scraper._driver_queue.empty()
