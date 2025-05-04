from abc import ABC, abstractmethod


class Scraper(ABC):
    """Abstract base class for scrapers."""

    @abstractmethod
    async def run(self):
        """Run the scraper."""
        pass

    @abstractmethod
    async def __aenter__(self):
        """Async context manager entry."""
        return self

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass


__all__ = [k for k, v in globals().items() if v in (Scraper,)]
