#!/usr/bin/env python3
from __future__ import annotations
import logging
import os
import pathlib as pl
import re
import shutil
import sys
import tempfile
from functools import partial, update_wrapper
from pathlib import Path, PosixPath
import pickle as pkl
from typing import (
    NoReturn,
    Union,
    List,
    Optional,
    Literal,
    TypeVar,
    overload,
    Tuple,
    Callable,
    Dict,
    Any,
    cast,
    Sequence,
)

import MDAnalysis
import MDAnalysis as mda
import MDAnalysis.coordinates
import numpy as np
import pandas as pd
from MDAnalysis import Universe
from MDAnalysis.lib.distances import minimize_vectors
from MDAnalysis.lib.mdamath import triclinic_vectors
from numpy.typing import NDArray

from ClayCode.config._consts import SOL, SOL_DENSITY, IONS, MDP, FF, DATA, AA, UCS
from ClayCode.core import gmx

def init_ff(ff: Dict[])
