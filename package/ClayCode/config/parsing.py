import logging
from abc import ABC, abstractmethod
from argparse import ArgumentParser
import sys
from collections import UserDict

import pandas as pd
import yaml
from typing import Dict

from ClayCode.config.classes import File, Dir, ForceField, UnitCell

logger = logging.getLogger(File(__file__).stem)

logger.setLevel(logging.DEBUG)

__all__ = ["ArgsFactory", "_Args", "parser"]

parser: ArgumentParser = ArgumentParser(
    "ClayCode",
    description="Automatically generate atomistic clay models.",
    add_help=True,
    allow_abbrev=False,
)

subparsers = parser.add_subparsers(help="Select option.", dest="option")

# Model setup parser
buildparser = subparsers.add_parser("build", help="Setup clay models.")

# Model build specifications
buildparser.add_argument(
    "-f",
    type=File,
    help="YAML file with build parameters",
    metavar="yaml_file",
    dest="yaml_file",
)

# Clay model composition
buildparser.add_argument(
    "-comp",
    type=File,
    help="CSV file with clay composition",
    metavar="csv_file",
    dest="csv_file",
)

# Clay model modification parser
editparser = subparsers.add_parser("edit", help="Edit clay models.")

# Clay simulation analysis parser
analysisparser = subparsers.add_parser("analysis", help="Analyse clay simulations.")

# Clay simulation check parser
checkparser = subparsers.add_parser("check", help="Check clay simulation _data.")

# TODO: add plotting?
#
# parser.add_argument('build',
#                     required=False,
#                     default=False,
#                     nargs=0,
#                     action='store_true',
#                     dest='BUILDER')
#
# parser.add_argument('siminp',
#                     required=False,
#                     default=False,
#                     action='store_true',
#                     dest='SIMINP')
#


class _Args(ABC, UserDict):
    option = None

    def __init__(self, data: dict):
        self.data = data

    def __repr__(self):
        return f"{self.__class__.__name__}({self.data.__repr__()})"

    def __str__(self):
        return f"{self.__class__.__name__}({self.data.__str__()})"

    @abstractmethod
    def process(self):
        pass


class BuildArgs(_Args):
    option = "build"

    _arg_names = [
        "SYSNAME",
        "BUILD",
        "CLAY_TYPE",
        "X_CELLS",
        "Y_CELLS",
        "N_SHEETS",
        "IL_SOLV",
        "UC_NUMLIST",
        "UC_RATIOS",
        "ION_WATERS",
        "UC_WATERS",
        "BOX_HEIGHT",
        "BULK_SOLV",
        "BULK_IONS",
        "CLAY_COMP",
    ]

    def __init__(self):
        self.x_cells: int = None
        self.y_cells: int = None
        self.boxheight: float = None
        self.n_sheets: int = None
        self._uc_stem: str = None
        self.name: str = None
        self.outpath: Dir = None
        self._raw_comp: pd.DataFrame
        self._corr_comp: pd.DataFrame
        self._uc_comp: pd.DataFrame
        self.ff = Dict[str, ForceField]
        self.ucs = Dict[str, UnitCell]

    def read_yaml(self):
        with open(self.data["yaml_file"], "r") as file:
            self.__yaml_data = yaml.safe_load(file)
        logger.info(f"Reading {file.name!r}")
        for k, v in self.__yaml_data.items():
            if k in self._arg_names:
                self.data[k] = v
                logger.info(f"{k!r} = {v!r}")
            else:
                raise KeyError(f"Unrecognised argument {k!r}!")
        try:
            csv_file = File(self.data["csv_file"])
        except KeyError:
            pass
        try:
            yaml_csv_file = File(self.data["CLAY_COMP"])
        except KeyError:
            pass
        if "csv_file" in self.data.keys() and "CLAY_COMP" in self.data.keys():
            if csv_file.absolute() == yaml_csv_file.absolute():
                logger.info(f"Clay composition {csv_file.absolute()} specified twice.")
                self.data["CLAY_COMP"] = csv_file
                self.data.pop("csv_file")
            else:
                raise ValueError(
                    f"Two non-identical clay composition files specified:"
                    f"\n\t1) {csv_file}\n\t2) {yaml_csv_file}"
                )
        elif "csv_file" in self.data.keys():
            self.data["CLAY_COMP"] = csv_file
            self.data.pop("csv_file")
        elif "CLAY_COMP" in self.data.keys():
            self.data["CLAY_COMP"] = yaml_csv_file
        else:
            raise ValueError(f"No csv file with clay composition specified!")

    def process(self):
        ...


class AnalysisArgs(_Args):
    option = "analysis"


class CheckArgs(_Args):
    option = "check"


class EditArgs(_Args):
    option = "edit"


class ArgsFactory:
    _options = {
        "build": BuildArgs,
        "edit": EditArgs,
        "check": CheckArgs,
        "analysis": AnalysisArgs,
    }

    @classmethod
    def get_subclass(cls, parse_args):
        if type(parse_args) != dict:
            data = parse_args.__dict__
        option = data.pop("option")
        try:
            _cls = cls._options[option]
        except KeyError:
            raise KeyError(f"{option!r} is not known!")
        return _cls(data)


p = parser.parse_args(["build", "-f", "build/tests/_data/input.yaml", "-comp", "a.csv"])
args: BuildArgs = ArgsFactory().get_subclass(p)
if type(args) == BuildArgs:
    args.read_yaml()
    ...
