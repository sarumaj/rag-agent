from typing import Any, Union
from pathlib import Path
from getpass import getpass
from pydantic import BaseModel, Field, field_validator, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
import re
import json


class ExportFormat(BaseModel):
    extension: str = Field(description="The extension of the file to be exported")
    command: str = Field(description="The command to be used to export the file")
    options: dict[str, Any] = Field(default_factory=dict, description="The options to be used to export the file")
    base64encoded: bool = Field(default=False, description="Whether the file should be base64 encoded")


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


class Settings(BaseSettings):
    """Settings for the archive scraper that can be loaded from environment variables.

    Environment variables should be prefixed with 'IX_' to avoid conflicts.
    For example: IX_BASE_URL, IX_USERNAME, etc.
    """
    model_config = SettingsConfigDict(
        env_prefix="IX_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="_",
        case_sensitive=False,
        frozen=True,
        extra="ignore",
    )

    ix_scraper_base_url: str = Field(
        default="https://www.heise.de",
        description="Base URL of the archive"
    )
    ix_scraper_sign_in_url: str = Field(
        default="https://www.heise.de/sso/login/?forward=https%3A%2F%2Fwww.heise.de%2Fix",
        description="Sign in URL of the archive"
    )
    ix_scraper_archive_url: str = Field(
        default="https://www.heise.de/select/ix/archiv/",
        description="Archive URL of the archive"
    )
    ix_scraper_max_threads: int = Field(default=10, description="Maximum number of threads")
    ix_scraper_max_concurrent: int = Field(default=10, description="Maximum number of concurrent requests")
    ix_scraper_timeout: int = Field(default=30, description="Timeout for the requests")
    ix_scraper_retry_attempts: int = Field(default=5, description="Number of retry attempts")
    ix_scraper_output_dir: Path = Field(
        default_factory=lambda: Path("~").expanduser().resolve() / "Downloads" / "ix",
        description="Output directory for the downloaded files"
    )
    ix_scraper_username: str = Field(
        default_factory=lambda: getpass("[IX] Enter your username: "),
        description="Username for the archive"
    )
    ix_scraper_password: str = Field(
        default_factory=lambda: getpass("[IX] Enter your password: "),
        description="Password for the archive",
        repr=False
    )
    ix_scraper_webdriver_options: list[str] = Field(default_factory=lambda: [
        '--headless',
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
    ], description="The options to be used to configure the webdriver")
    ix_scraper_export_formats: list[ExportFormat] = Field(
        default_factory=lambda: [PDF_EXPORT_FORMAT],
        description="The formats to be used to export the files"
    )
    ix_scraper_overwrite: bool = Field(
        default=False,
        description="Whether to overwrite existing files"
    )

    @classmethod
    def for_testing(cls) -> 'Settings':
        """Create a Settings instance with test values."""
        return cls(
            ix_scraper_max_threads=1,
            ix_scraper_max_concurrent=1,
            ix_scraper_timeout=1,
            ix_scraper_retry_attempts=1,
            ix_scraper_output_dir=Path("/tmp/test_output"),
            ix_scraper_webdriver_options=["--headless"],
            ix_scraper_export_formats=[ExportFormat(extension=".pdf", command="print")],
            ix_scraper_overwrite=False,
        )

    @field_validator("ix_scraper_webdriver_options", mode="before")
    @classmethod
    def parse_cli_options(cls, v: Union[str, list[str]]) -> list[str]:
        return v.split(" ") if isinstance(v, str) else v

    @field_validator("ix_scraper_export_formats", mode="before")
    @classmethod
    def parse_export_formats(cls, v: Union[str, list[ExportFormat]]) -> list[ExportFormat]:
        if not isinstance(v, str):
            return v

        export_formats = []
        for export_format in re.split(r",\s*", v):
            match export_format:
                case "pdf":
                    export_formats.append(PDF_EXPORT_FORMAT)
                case "mhtml":
                    export_formats.append(MHTML_EXPORT_FORMAT)
                case _:
                    try:
                        export_formats.append(ExportFormat(**json.loads(export_format)))
                    except (json.JSONDecodeError, ValidationError):
                        raise ValidationError(f"Invalid export format: {export_format}")

        return export_formats
