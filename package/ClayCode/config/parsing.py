import logging
import os
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from collections import UserDict

import pandas as pd
import yaml
from typing import Dict

from ClayCode import UCS, FF
from ClayCode.builder.exp import ClayComposition
from ClayCode.config.classes import File, Dir
from ClayCode.core.utils import init_path

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
buildparser = subparsers.add_parser("builder", help="Setup clay models.")

# Model builder specifications
buildparser.add_argument(
    "-f",
    type=File,
    help="YAML file with builder parameters",
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
    required=False,
)

# Clay model modification parser
editparser = subparsers.add_parser("edit", help="Edit clay models.")

# Clay simulation analysis parser
analysisparser = subparsers.add_parser("analysis", help="Analyse clay simulations.")

# Clay simulation check parser
checkparser = subparsers.add_parser("check", help="Check clay simulation _data.")

# TODO: add plotting?
#
# parser.add_argument('builder',
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

    @abstractmethod
    def check(self):
        pass


class BuildArgs(_Args):
    """Parameters for clay model setup with :mod:`ClayCode.builder`"""

    option = "builder"
    from ClayCode.builder.consts import UC_CHARGE_OCC as _charge_occ_df
    from ClayCode.builder.consts import BUILD_DEFAULTS as _build_defaults

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
        "OUTPATH",
        "FF",
    ]

    def __init__(self, data) -> None:
        super().__init__(data)
        self.filestem: str = None
        self.uc_df: pd.DataFrame = None
        self.uc_charges: pd.DataFrame = None
        self.x_cells: int = None
        self.y_cells: int = None
        self.boxheight: float = None
        self.n_sheets: int = None
        self._uc_name: str = None
        self._uc_stem: str = None
        self.name: str = None
        self.outpath: Dir = None
        self._raw_comp: pd.DataFrame = None
        self._corr_comp: pd.DataFrame = None
        self._uc_comp: pd.DataFrame = None
        self.ff = None
        self.ucs = None
        self.il_solv = None
        self.read_yaml()
        self.check()
        self.process()
        self.get_uc_data()
        self.get_exp_data()

    def read_yaml(self) -> None:
        """Read clay model builder specifications from yaml file."""
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
            csv_file = File(self.data["csv_file"], check=True)
        except TypeError:
            self.data.pop("csv_file")
        try:
            yaml_csv_file = File(self.data["CLAY_COMP"], check=True)
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

    def check(self) -> None:
        try:
            self.name = self.data["SYSNAME"]
            logger.info(f"Setting name = {self.name!r}")
        except KeyError:
            raise KeyError(f"Clay system name must be given")
        try:
            uc_type = self.data["CLAY_TYPE"]
            if (
                uc_type in self._charge_occ_df.index.get_level_values("value").unique()
            ) and (UCS / uc_type).is_dir():
                self._uc_name = uc_type
                self._uc_stem = self._uc_name[:2]
                logger.info(f"Setting UC type = {self._uc_name!r}")
        except KeyError:
            raise KeyError(f"Unknown unit cell type {uc_type!r}")
        il_solv = self._charge_occ_df.loc[pd.IndexSlice["T", self._uc_name], ["solv"]]
        try:
            selected_solv = self.data["IL_SOLV"]
            if il_solv is False and selected_solv is True:
                raise ValueError(
                    f"Invalid interlayer solvation ({selected_solv}) for selected clay type {self._uc_name}!"
                )
        except KeyError:
            self.il_solv = il_solv
        for prm in [
            "BUILD",
            "X_CELLS",
            "Y_CELLS",
            "N_SHEETS",
            "UC_WATERS",
            "BOX_HEIGHT",
            "BULK_IONS",
            "FF",
        ]:
            try:
                prm_value = self.data[prm]
            except KeyError:
                prm_value = self._build_defaults[prm]
            setattr(self, prm.lower(), prm_value)
        try:
            outpath = self.data["OUTPATH"]
            self.outpath = Dir(outpath, check=False)

        except KeyError:
            raise KeyError(f"No output directory specified")

    def process(self):
        self.filestem = f"{self.name}_{self.x_cells}_{self.y_cells}"
        self.outpath = self.outpath / self.name
        init_path(self.outpath)
        self.get_ff_data()
        self.get_uc_data()

    def get_ff_data(self):
        from ClayCode.config.classes import ForceField
        water_sel_dict = {"SPC": ["ClayFF_Fe", ["spc", "interlayer_spc"]]}
        ff_dict = {}
        for ff_key, ff_sel in self.ff.items():
            if ff_key == "WATER":
                ff_sel = water_sel_dict[ff_sel][0]
            ff_dict[ff_key.lower()] = ForceField(FF / ff_sel)
        self.ff = ff_dict

    def get_uc_data(self):
        from ClayCode.builder.exp import UCData
        self.uc_data = UCData(UCS / self._uc_name, uc_stem=self._uc_stem, ff=self.ff["clay"])
        occ = self.uc_data.occupancies
        ch = self.uc_data.oxidation_numbers
        atc = self.uc_data.atomic_charges
        ...

    def get_exp_data(self):
        from ClayCode.builder.exp import ClayComposition
        csv_fname = self.data["CLAY_COMP"]
        clay_atoms = self.uc_data.df.index
        clay_atoms.append(pd.MultiIndex.from_tuples([("O", "fe_tot")]))

        exp_comp = ClayComposition(self.name, self.data['CLAY_COMP'], self.uc_data
        )
        # exp_comp = pd.read_csv()
        ...

class AnalysisArgs(_Args):
    option = "analysis"

    def __init__(self, data):
        super().__init__(data)


class CheckArgs(_Args):
    option = "check"

    def __init__(self, data):
        super().__init__(data)


class EditArgs(_Args):
    option = "edit"

    def __init__(self, data):
        super().__init__(data)


class ArgsFactory:
    _options = {
        "builder": BuildArgs,
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
        print(_cls)
        return _cls(data)


p = parser.parse_args(
    ["builder", "-f", "builder/tests/data/input.yaml"]
)  # , "-comp", "a.csv"])
args = ArgsFactory()
b = args.get_subclass(p)
...
