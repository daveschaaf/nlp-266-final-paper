import sys
import logging

from .constants import PROJECT_ROOT

LOG_DIR = PROJECT_ROOT / 'logs'

def setup_logging() -> None:

    logger = logging.getLogger("dearborn")
    if logger.handlers:
        return

    logger.setLevel(logging.DEBUG)

    format = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    LOG_DIR.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(LOG_DIR / 'app.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(format)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(format)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
