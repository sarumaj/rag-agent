from abc import ABC, abstractmethod


class Scrapper(ABC):
    """Abstract base class for scrapers."""

    @abstractmethod
    async def run(self):
        """Run the scraper."""
        pass


__all__ = [Scrapper.__name__]
