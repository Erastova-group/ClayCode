from __future__ import annotations
import logging

from ClayCode.config.consts import (
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
    DATA
)

__all__ = ["analysis", "builder", "siminp", "config", 'logger']  # , 'check', 'edit']


logging.basicConfig(format="%(message)s", level=logging.INFO)
logger = logging.getLogger('ClayCode')

# TODO: remove siminp?
# TODO: take out setup/edit/molecule insertion from analysis
# TODO: Now analysis has analysis but also checks
# TODO: make config (for setup) become a base for all
