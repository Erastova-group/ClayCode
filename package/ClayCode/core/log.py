import logging

__all__ = ["logger"]

logging.basicConfig(format="%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
