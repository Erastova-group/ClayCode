from __future__ import annotations
import logging

import importlib_resources
import pkg_resources

from ClayCode.core.consts import (
    KWD_DICT,
    exec_time,
    exec_date,
    AA,
    FF,
    MDP,
    CLAYS,
    IONS,
    SOL,
    SOL_DENSITY,
    UCS,
    FILE_SEARCHSTR_LIST,
    DATA,
)

logging.basicConfig(format="%(message)s", level=logging.INFO)
logger = logging.getLogger("ClayCode")

__all__ = ["analysis", "builder", "siminp", "config", "logger"]  # , 'check', 'edit']

# TODO: remove siminp?
# TODO: take out setup/edit/molecule insertion from analysis
# TODO: Now analysis has analysis but also checks
# TODO: make config (for setup) become a base for all
