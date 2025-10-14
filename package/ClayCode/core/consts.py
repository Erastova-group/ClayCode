#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r""":mod:`ClayCode.core.consts` --- Constants
============================================
"""
import logging

# from typing import Any, Dict
import shutil
from datetime import datetime, timezone

# from caseless_dictionary import CaselessDict

__all__ = [
    "exec_time",
    "exec_date",
    "IONS",
    "SOL",
    "SOL_DENSITY",
    "LINE_LENGTH",
    "TABSIZE",
    "ANGSTROM",
]

# from ClayCode.data.consts import MDP, MDP_DEFAULTS, TOP_KWDS

IONS: list = ["Cl", "Na", "Ca", "K", "Mg", "Cs"]
SOL_DENSITY: int = 1000  # g L-1
SOL: str = "SOL"

shandler = logging.StreamHandler()
shandler.setLevel(logging.INFO)

exec_datetime: str = datetime.now(timezone.utc).strftime("%y%m%d%H%M")
exec_date: str = datetime.now(timezone.utc).strftime("%y/%m/%d")
exec_time: str = datetime.now(timezone.utc).strftime("%H:%M")


LINE_LENGTH: int = shutil.get_terminal_size().columns

TABSIZE: int = 4

ANGSTROM: str = "\u212B"
GREATER_EQUAL: str = "\u2265"
LESS_EQUAL: str = "\u2264"
