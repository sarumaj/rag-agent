import dataclasses as dc
from pathlib import Path
from getpass import getpass


@dc.dataclass
class Config:
    """Configuration for the downloader."""
    sign_in_url: str = "https://sklep.cdaction.pl/moje-konto/"
    download_url: str = "https://sklep.cdaction.pl/moje-konto/pliki-do-pobrania/"
    max_retries: int = 3
    max_concurrent: int = 10
    timeout: int = 30
    output_dir: Path = Path("~").expanduser().resolve() / "Downloads" / "cdaction"
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
