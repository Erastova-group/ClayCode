#!/usr/bin/env python3
from __future__ import annotations

import logging
import warnings

# from Bio import BiopythonDeprecationWarning


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

from ClayCode.core import (
    ArgsFactory,
    BasicPath,
    BasicPathList,
    BuildArgs,
    ClayCodeLogger,
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
    SiminpArgs,
    SystemParameter,
    SystemParameters,
    TOPFile,
    TOPList,
    YAMLFile,
    parser,
)

logging.setLoggerClass(ClayCodeLogger)

from ClayCode.core.cctypes import (
    AnyDict,
    AnyDir,
    AnyFile,
    FileNameMatchSelector,
    PathOrStr,
    PathType,
    StrNum,
    StrNumOrListDictOf,
    StrOrListDictOf,
    StrOrListOf,
)

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
]

logging.setLoggerClass(ClayCodeLogger)
