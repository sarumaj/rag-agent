import dataclasses as dc
from typing import Any
from pathlib import Path
from getpass import getpass


@dc.dataclass
class ExportFormat:
    extension: str
    command: str
    options: dict[str, Any] = dc.field(default_factory=dict)
    base64encoded: bool = False


MHTML_EXPORT_FORMAT = ExportFormat(
    extension=".mhtml",
    command="Page.captureSnapshot",
    options={
        "format": "mhtml",
    },
)

PDF_EXPORT_FORMAT = ExportFormat(
    extension=".pdf",
    command="Page.printToPDF",
    options={
        "landscape": False,
        "displayHeaderFooter": False,
        "printBackground": True,
        "scale": 1,
        "paperWidth": 8.268,      # in inches (A4)
        "paperHeight": 11.693,    # in inches (A4)
        "marginTop": 0.3,         # in inches
        "marginRight": 0.3,       # in inches
        "marginBottom": 0.3,      # in inches
        "marginLeft": 0.3,        # in inches
        "pageRanges": "1-",
        "preferCSSPageSize": True,
        "transferMode": "ReturnAsBase64",
    },
    base64encoded=True,
)


@dc.dataclass
class Config:
    """Configuration for the archive scraper.

    Attributes:
        base_url: Base URL for the website
        sign_in_url: URL for signing in
        archive_url: URL for the archive
        max_threads: Maximum number of webdrivers to use
        max_concurrent: Maximum number of concurrent requests
        timeout: Timeout for webdriver operations
        retry_attempts: Number of times to retry failed logins
        output_dir: Directory to save downloaded articles
        username: Username for login
        password: Password for login
        webdriver_options: Options for Chrome webdriver
        export_formats: List of export formats to use
        overwrite: Whether to overwrite existing files
    """
    base_url: str = "https://www.heise.de"
    sign_in_url: str = "https://www.heise.de/sso/login/?forward=https%3A%2F%2Fwww.heise.de%2Fix"
    archive_url: str = "https://www.heise.de/select/ix/archiv/"
    max_threads: int = 10
    max_concurrent: int = 10
    timeout: int = 30
    retry_attempts: int = 5
    output_dir: Path = Path("~").expanduser().resolve() / "Downloads" / "ix"
    username: str = input("Enter your username: ")
    password: str = getpass("Enter your password: ")
    webdriver_options: list[str] = dc.field(default_factory=lambda: [
        '--headless',
        '--no-sandbox',
        '--disable-dev-shm-usage',
        '--kiosk-printing',
        '--force-color-profile=srgb',
        '--disable-software-rasterizer',
        '--disable-gpu',
        '--disable-dev-tools',
        '--disable-extensions',
        '--disable-logging',
        '--log-level=3',
        '--silent',
        '--disable-blink-features=AutomationControlled',
        '--disable-features=IsolateOrigins,site-per-process',
        '--disable-site-isolation-trials',
    ])
    export_formats: list[ExportFormat] = dc.field(default_factory=lambda: [
        PDF_EXPORT_FORMAT,
    ])
    overwrite: bool = False
