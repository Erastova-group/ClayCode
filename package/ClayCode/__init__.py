from __future__ import annotations

from .core.log import logger

__all__ = [
    "analysis",
    "builder",
    "siminp",
    "logger",
]
import importlib.metadata

__version__ = importlib.metadata.version("ClayCode")

# TODO: remove siminp?
# TODO: take out setup/edit/molecule insertion from analysis
# TODO: Now analysis has analysis but also checks
# TODO: make config (for setup) become a base for all
