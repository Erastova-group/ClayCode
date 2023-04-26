from __future__ import annotations
from .core.log import logger

# from .core.consts import (
#     KWD_DICT,
#     exec_time,
#     exec_date,
#     AA,
#     FF,
#     MDP,
#     CLAYS,
#     IONS,
#     SOL,
#     SOL_DENSITY,
#     UCS,
#     FILE_SEARCHSTR_LIST,
#     DATA,
# )


__all__ = [
    "analysis",
    "builder",
    "siminp",
    "logger",
    # "KWD_DICT",
    # "exec_date",
    # "exec_time",
    # "AA",
    # "FF",
    # "MDP",
    # "CLAYS",
    # "IONS",
    # "SOL",
    # "SOL_DENSITY",
    # "UCS",
    # "FILE_SEARCHSTR_LIST",
    # "DATA",
]
import importlib.metadata
__version__ = importlib.metadata.version("ClayCode")
# TODO: remove siminp?
# TODO: take out setup/edit/molecule insertion from analysis
# TODO: Now analysis has analysis but also checks
# TODO: make config (for setup) become a base for all
