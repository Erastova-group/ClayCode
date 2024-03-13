#!/usr/bin/env python3
from __future__ import annotations

import logging
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from MDAnalysis.coordinates.TRJ import logger as amber_logger

amber_logger.setLevel(logging.ERROR)

from ClayCode.core.cctypes import (
    AnyDict,
    AnyDir,
    AnyFile,
    FileNameMatchSelector,
    NumOrListDictOf,
    NumOrListOf,
    PathOrStr,
    PathType,
    StrNum,
    StrNumOrListDictOf,
    StrOrListDictOf,
    StrOrListOf,
)
from ClayCode.core.classes import (
    BasicPath,
    BasicPathList,
    Definition,
    Dir,
    DirFactory,
    FFDir,
    FFList,
    File,
    FileFactory,
    FileList,
    ForceField,
    GROFile,
    GROList,
    ITPFile,
    ITPList,
    MDPFile,
    MDPList,
    MoleculeParameter,
    MoleculeParameters,
    Parameter,
    ParameterBase,
    ParameterFactory,
    Parameters,
    ParametersBase,
    ParametersFactory,
    PathFactory,
    PathList,
    PathListFactory,
    SimDir,
    SystemParameter,
    SystemParameters,
    TOPFile,
    TOPList,
    YAMLFile,
)
from ClayCode.core.log import ClayCodeLogger
from ClayCode.core.parsing import ArgsFactory, BuildArgs, SiminpArgs, parser

__all__ = [
    "ClayCodeLogger",
    "StrOrListOf",
    "StrOrListDictOf",
    "StrNum",
    "StrNumOrListDictOf",
    "FileNameMatchSelector",
    "AnyFile",
    "AnyDir",
    "PathType",
    "PathOrStr",
    "AnyDict",
    "Definition",
    "File",
    "FileList",
    "Dir",
    "MDPFile",
    "GROList",
    "ITPList",
    "ITPFile",
    "MDPList",
    "YAMLFile",
    "FFDir",
    "ForceField",
    "TOPFile",
    "GROFile",
    "FFList",
    "PathList",
    "PathFactory",
    "ParameterFactory",
    "Parameter",
    "Parameters",
    "ParametersFactory",
    "MoleculeParameter",
    "BasicPathList",
    "BasicPath",
    "SimDir",
    "TOPList",
    "ParameterBase",
    "ParametersBase",
    "PathListFactory",
    "SystemParameters",
    "MoleculeParameters",
    "SystemParameter",
    "SiminpArgs",
    "parser",
    "BuildArgs",
    "ArgsFactory",
    "DirFactory",
    "FileFactory",
    "PathListFactory",
    "NumOrListOf",
    "NumOrListDictOf",
]
logging.setLoggerClass(
    ClayCodeLogger
)  # from Bio import BiopythonDeprecationWarning
