#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
r""":mod:`ClayCode.core` --- Core functions and classes
===========================================
"""
from __future__ import annotations

import MDAnalysis

MDAnalysis.stop_logging()
import logging

from ClayCode.core.log import ClayCodeLogger

logging.setLoggerClass(ClayCodeLogger)


__all__ = ["ClayCodeLogger"]
