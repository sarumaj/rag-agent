from typing import Any
from pathlib import Path
from getpass import getpass
from pydantic import BaseModel, Field, field_validator, ValidationInfo
from pydantic_settings import BaseSettings, SettingsConfigDict


class ExportFormat(BaseModel):
    extension: str
    command: str
    options: dict[str, Any] = Field(default_factory=dict)
    base64encoded: bool = Field(default=False)


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
        frozen=True,
        extra="ignore"
    )

    base_url: str = Field(
        default="https://www.heise.de",
        description="Base URL of the archive"
    )
    sign_in_url: str = Field(
        default="https://www.heise.de/sso/login/?forward=https%3A%2F%2Fwww.heise.de%2Fix",
        description="Sign in URL of the archive"
    )
    archive_url: str = Field(
        default="https://www.heise.de/select/ix/archiv/",
        description="Archive URL of the archive"
    )
    max_threads: int = Field(default=10, description="Maximum number of threads")
    max_concurrent: int = Field(default=10, description="Maximum number of concurrent requests")
    timeout: int = Field(default=30, description="Timeout for the requests")
    retry_attempts: int = Field(default=5, description="Number of retry attempts")
    output_dir: Path = Field(
        default_factory=lambda: Path("~").expanduser().resolve() / "Downloads" / "ix",
        description="Output directory for the downloaded files"
    )
    username: str = Field(default="", description="Username for the archive")
    password: str = Field(default="", description="Password for the archive")
    webdriver_options: list[str] = Field(default_factory=lambda: [
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
    export_formats: list[ExportFormat] = Field(default_factory=lambda: [PDF_EXPORT_FORMAT])
    overwrite: bool = Field(default=False)

    @field_validator("username", "password", mode="before")
    @classmethod
    def prompt_if_empty(cls, v: str, info: ValidationInfo) -> str:
        if not v:
            match info.field_name:
                case "username":
                    return input("Enter your username: ")
                case "password":
                    return getpass("Enter your password: ")
                case _:
                    raise ValueError(f"Invalid field name: {info.field_name}")
        return v

    @field_validator("webdriver_options", mode="before")
    @classmethod
    def parse_cli_options(cls, v: Any, info: ValidationInfo) -> list[str]:
        match info.field_name:
            case "webdriver_options":
                return v.split(" ") if isinstance(v, str) else v
            case _:
                raise ValueError(f"Invalid field name: {info.field_name}")
