import logging

logging.basicConfig(
    level=logging.INFO,
    style='{',
    format='{asctime} - {levelname} - {message}',
)

getLogger = logging.getLogger
