#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r""":mod:`ClayCode.core.consts` --- Constants
============================================
"""
import logging
from datetime import datetime, timezone
from typing import Any, Dict

from caseless_dictionary import CaselessDict

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

from ClayCode.data.consts import MDP, MDP_DEFAULTS, TOP_KWDS

IONS = ["Cl", "Na", "Ca", "K", "Mg", "Cs"]
SOL_DENSITY = 1000  # g L-1
SOL = "SOL"

shandler = logging.StreamHandler()
shandler.setLevel(logging.INFO)

exec_datetime = datetime.now(timezone.utc).strftime("%y%m%d%H%M")
exec_date = datetime.now(timezone.utc).strftime("%y/%m/%d")
exec_time = datetime.now(timezone.utc).strftime("%H:%M")


LINE_LENGTH: int = 100

TABSIZE = 4

ANGSTROM = "\u212B"
GREATER_EQUAL = "\u2265"
LESS_EQUAL = "\u2264"
