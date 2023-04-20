#!/usr/bin/env python3
import argparse
import logging
import re
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from collections import UserDict
from functools import cached_property

import numpy as np
import pandas as pd
import yaml

from ClayCode import UCS, FF, logger
from ClayCode.builder.claycomp import MatchClayComposition, InterlayerIons, BulkIons
from ClayCode.core.classes import File, Dir, init_path
from ClayCode.core.lib import get_ion_charges
from ClayCode.core.utils import get_header, get_subheader


__all__ = {
    "ArgsFactory",
    "parser",
    "BuildArgs",
    "EditArgs",
    "CheckArgs",
    "EquilibrateArgs",
    "PlotArgs",
    "AnalysisArgs",
}


parser: ArgumentParser = ArgumentParser(
    "ClayCode",
    description="Automatically generate atomistic clay models.",
    add_help=True,
    allow_abbrev=False,
)

parser.add_argument(
    "--debug",
    help="Debug run",
    action="store_const",
    const=logging.DEBUG,
    default=logging.INFO,
    dest="DEBUG",
)

subparsers = parser.add_subparsers(help="Select option.", dest="option")
subparsers.required = True

# Model setup parser
buildparser = subparsers.add_parser("builder", help="Setup clay models.")

# Model builder specifications
buildparser.add_argument(
    "-f",
    type=File,
    help="YAML file with builder parameters",
    metavar="yaml_file",
    dest="yaml_file",
    required=True,
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

editparser.add_argument(
    "-c",
    help="system coordinates",
    required=True,
    dest="ingro",
    metavar="input_grofile",
)
editparser.add_argument(
    "-p",
    help="system topology",
    required=False,
    dest="intop",
    default=None,
    metavar="input_topfile",
)
editparser.add_argument(
    "--solvate-bulk",
    help="Solvate box",
    action="store_true",
    default=False,
    dest="bulk_solv",
)
editparser.add_argument(
    "--add-resnum",
    help="Add residue numbers to coordinate file",
    action="store_true",
    default=False,
)
editparser.add_argument(
    "-neutralise",
    help="Neutralise system with selected ion types",
    nargs="?",
    type=str,
    metavar="ion_types",
    dest="neutral_ions",
)

editparser.add_argument(
    "-odir",
    help="Output directory",
    required=False,
    default=None,
    type=Dir,
    dest="odir",
    metavar="output_directory",
)

editparser.add_argument(
    "-o",
    help="Output naming pattern",
    type=str,
    required=True,
    dest="new_name",
    metavar="output_filestem",
)

edit_subparsers = editparser.add_subparsers(
    help="Molecule addition subparsers", dest="add_mol"
)

aa_parser = edit_subparsers.add_parser("add_aa", help="Amino acid addition")
aa_parser.add_argument(
    "-aa",
    help="Amino acid types",
    required=True,
    nargs="+",
    type=str,
    dest="aa",
    metavar="aa_types",
)
aa_parser.add_argument(
    "-pH", help="pH value", dest="pH", type=float, default=7, metavar="pH"
)

aa_parser.add_argument(
    "-replace",
    help="Molecule type to replace",
    required=False,
    dest="replace_type",
    metavar="replace_moltype",
)

aa_add_group = aa_parser.add_mutually_exclusive_group(required=True)
aa_add_group.add_argument(
    "-n_mols",
    help="Insert number of molecules",
    type=int,
    dest="n_mols",
    metavar="n_mols",
)

aa_add_group.add_argument(
    "-conc",
    help="Insert concentration",
    type=float,
    dest="conc",
    required=False,
    metavar="concentration",
)

ion_parser = edit_subparsers.add_parser("add_ions", help="Bulk ion addition")

ion_parser.add_argument(
    "-pion",
    help="Cation type(s)",
    required=False,
    nargs="+",
    type=str,
    dest="pion",
    metavar="cation_type",
)
ion_parser.add_argument(
    "-nion",
    help="Anion type(s)",
    required=False,
    nargs="+",
    type=str,
    dest="nion",
    metavar="anion_type",
)


ion_add_group = ion_parser.add_mutually_exclusive_group(required=True)
ion_add_group.add_argument(
    "-n_atoms",
    help="Insert number of atoms",
    type=int,
    dest="n_mols",
    metavar="n_atoms",
)

ion_add_group.add_argument(
    "-conc",
    help="Insert concentration",
    type=float,
    dest="conc",
    required=False,
    metavar="concentration",
)

# Clay simulation analysis parser
analysisparser = subparsers.add_parser("analyse", help="Analyse clay simulations.")

# plot analysis results
plotparser = subparsers.add_parser("plot", help="Plot simulation analysis results")

# Clay simulation check parser
checkparser = subparsers.add_parser("check", help="Check clay simulation data.")

equilparser = subparsers.add_parser(
    "equilibrate", help="Generate clay model equilibration run input files."
)
equilparser.add_argument(
    "-d_space", help="d-spacing in A", metavar="d_spacing", dest="d_space", type=float
)
equilparser.add_argument(
    "-n_wat",
    help="number of water molecules to remove per cycle per unit cell",
    metavar="n_waters",
    dest="n_wat",
    type=float,
)

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


def read_yaml_decorator(f):
    def wrapper(self: _Args):
        assert isinstance(self, _Args), f"Wrong class for decorator"
        with open(self.data["yaml_file"], "r") as file:
            self.__yaml_data = yaml.safe_load(file)
        logger.info(f"Reading {file.name!r}:\n")
        for k, v in self.__yaml_data.items():
            if k in self._arg_names:
                self.data[k] = v
                logger.info(f"\t{k} = {v!r}")
            else:
                raise KeyError(f"Unrecognised argument {k}!")
        return f(self)

    return wrapper


class _Args(ABC, UserDict):
    option = None

    _arg_names = []

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
        "UC_INDEX_LIST",
        "UC_RATIOS_LIST",
        "ION_WATERS",
        "UC_WATERS",
        "BOX_HEIGHT",
        "BULK_SOLV",
        "BULK_IONS",
        "CLAY_COMP",
        "OUTPATH",
        "FF",
        "GMX",
    ]

    def __init__(self, data) -> None:
        super().__init__(data)
        self._target_comp = None
        self._uc_data = None
        self.filestem: str = None
        # self.uc_df: pd.DataFrame = None
        self.uc_charges: pd.DataFrame = None
        self.x_cells: int = None
        self.y_cells: int = None
        self.boxheight: float = None
        self.n_sheets: int = None
        self._uc_name: str = None
        self.uc_stem: str = None
        self.name: str = None
        self.outpath: Dir = None
        self._raw_comp: pd.DataFrame = None
        self._corr_comp: pd.DataFrame = None
        self._uc_comp: pd.DataFrame = None
        self.ff = None
        self.ucs = None
        self.il_solv = None
        # self.read_yaml()
        # self.check()
        self.process()

    @read_yaml_decorator
    def read_yaml(self) -> None:
        """Read clay model builder specifications from yaml file."""
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
            logger.info(f"\nSetting name: {self.name!r}")
        except KeyError:
            raise KeyError(f"Clay system name must be given")
        try:
            uc_type = self.data["CLAY_TYPE"]
            if (
                uc_type in self._charge_occ_df.index.get_level_values("value").unique()
            ) and (UCS / uc_type).is_dir():
                self._uc_name = uc_type
                self.uc_stem = self._uc_name[:2]
                logger.debug(f"Setting unit cell type: {self._uc_name!r}")
        except KeyError:
            raise KeyError(f"Unknown unit cell type {uc_type!r}")
        il_solv = self._charge_occ_df.loc[
            pd.IndexSlice["T", self._uc_name], ["solv"]
        ].values[0]
        try:
            selected_solv = self.data["IL_SOLV"]
            if il_solv == False and selected_solv is True:
                raise ValueError(
                    f"Invalid interlayer solvation ({selected_solv}) for selected clay type {self._uc_name}!"
                )
            self.il_solv = il_solv
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
            "BULK_SOLV",
            "FF",
            "UC_INDEX_LIST",
            "UC_RATIOS_LIST",
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
        try:
            GMX = self.data["GMX"]
        except KeyError:
            setattr(self, GMX, self._build_defaults["GMX"])
            logger.info(f"Using default GROMACS alias: {GMX}")

    def process(self):
        logger.info(get_header("Getting build parameters"))
        self.read_yaml()
        self.check()
        self.filestem = f"{self.name}_{self.x_cells}_{self.y_cells}"
        self.outpath = self.outpath / self.name
        init_path(self.outpath)
        logger.info(f"Setting output directory: {self.outpath}")
        self.get_ff_data()
        self.get_uc_data()
        self.get_exp_data()
        self.match_uc_combination()
        self.get_il_ions()
        self.get_il_solvation_data()
        self.get_bulk_ions()

    def get_ff_data(self):
        from ClayCode.core.classes import ForceField

        water_sel_dict = {"SPC": ["ClayFF_Fe", ["spc", "interlayer_spc"]]}
        ff_dict = {}
        logger.info(get_subheader(f"Getting force field data"))
        for ff_key, ff_sel in self.ff.items():
            if ff_key == "WATER":
                ff_sel = water_sel_dict[ff_sel][0]
            ff = ForceField(FF / ff_sel)
            ff_dict[ff_key.lower()] = ff
            logger.info(f"\t{ff_key}: {ff.name}")
            itp_str = "\n\t\t".join(ff.itp_filelist.stems)
            logger.info(f"\t\t{itp_str}\n")
        self.ff = ff_dict

    def get_uc_data(self):
        from ClayCode.builder.claycomp import UCData

        self._uc_data = UCData(
            UCS / self._uc_name, uc_stem=self.uc_stem, ff=self.ff["clay"]
        )
        occ = self._uc_data.occupancies
        ch = self._uc_data.oxidation_numbers
        atc = self._uc_data.atomic_charges

    def get_exp_data(self):
        from ClayCode.builder.claycomp import TargetClayComposition

        # csv_fname = self.data["CLAY_COMP"]
        clay_atoms = self._uc_data.df.index
        clay_atoms.append(pd.MultiIndex.from_tuples([("O", "fe_tot")]))

        self._target_comp = TargetClayComposition(
            self.name, self.data["CLAY_COMP"], self._uc_data
        )
        self._target_comp.write_csv(self.outpath)

    def get_il_solvation_data(self):
        if self.il_solv == True:
            n_ions = np.sum([*self.n_il_ions.values()])
            if hasattr(self, "ion_waters") and not (
                hasattr(self, "uc_waters") or hasattr(self, "spacing_waters")
            ):
                waters = self.ion_waters
                if isinstance(waters, dict):
                    assert waters.keys() in self.ion_df.index
                    for ion_type in waters.keys():
                        waters[ion_type] *= self.n_il_ions[ion_type]
                    waters = np.sum([*waters.values()]) + n_ions
                else:
                    waters *= n_ions
                self.n_waters = waters + n_ions
                self.il_solv_height = None
            elif hasattr(self, "uc_waters") and not (
                hasattr(self, "ion_waters") or hasattr(self, "spacing_waters")
            ):
                self.n_waters = self.uc_waters * self.sheet_n_cells + n_ions
                self.il_solv_height = None
            elif hasattr(self, "spacing_waters") and not (
                hasattr(self, "uc_waters") or hasattr(self, "ion_waters")
            ):
                self.n_waters = None
                self.il_solv_height = self.spacing_waters
            else:
                raise AttributeError(
                    "Number of water molecules or interlayer solvent height to add must be "
                    'specified through either "ION_WATERS", '
                    '"UC_WATERS" or "SPACING_WATERS".'
                )

    @property
    def uc_df(self):
        return self._uc_data.df

    @property
    def target_df(self):
        return self._target_comp.df

    @property
    def match_df(self):
        return self._match_comp.df

    @property
    def ion_df(self):
        return self._target_comp.ion_df

    @cached_property
    def sheet_n_cells(self):
        return self.x_cells * self.y_cells

    def match_uc_combination(self):
        self.match_comp = MatchClayComposition(self._target_comp, self.sheet_n_cells)
        self.match_comp.write_csv(self.outpath)

    @property
    def match_df(self) -> pd.DataFrame:
        return self.match_comp.match_composition

    @property
    def sheet_uc_weights(self) -> pd.Series:
        return self.match_comp.uc_weights

    @property
    def sheet_uc_ids(self):
        return self.match_comp.uc_ids

    @property
    def match_charge(self):
        return self.match_comp.match_charge

    def get_il_ions(self):
        tot_charge = self.match_charge["tot"]
        if tot_charge != 0:
            self.il_ions = InterlayerIons(
                tot_charge=tot_charge, ion_ratios=self.ion_df, n_ucs=self.sheet_n_cells
            )

    def get_bulk_ions(self):
        self._bulk_ions = BulkIons(self.bulk_ions, self._build_defaults["BULK_IONS"])

    @property
    def bulk_ion_conc(self):
        return self._bulk_ions.tot_conc

    @property
    def bulk_ion_df(self):
        return self._bulk_ions.conc

    @cached_property
    def default_bulk_pion(self):
        neutralise_ions = self._bulk_ions.neutralise_ions
        return tuple(
            *neutralise_ions[neutralise_ions["charge"] > 0]["conc"].reset_index().values
        )

    @cached_property
    def default_bulk_nion(self):
        neutralise_ions = self._bulk_ions.neutralise_ions
        return tuple(
            *neutralise_ions[neutralise_ions["charge"] < 0]["conc"].reset_index().values
        )

    @property
    def n_il_ions(self):
        return self.il_ions.numbers


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

    _arg_names = [
        "ingro",
        "intop",
        "bulk_solv",
        "neutral_ions",
        "odir",
        "new_name",
        "add_mol",
    ]
    _mol_args = {
        "aa": ["pH", "conc", "n_mols", "replace_type"],
        "ions": ["pion", "n_ion", "conc", "n_mols", "replace_type"],
    }

    def __init__(self, data):
        super().__init__(data)


class PlotArgs(_Args):
    option = "plot"

    def __init__(self, data):
        super().__init__(data)


class EquilibrateArgs(_Args):
    option = "equilibrate"
    # TODO: Add csv or yaml for eq run prms

    def __init__(self, data):
        super().__init__(data)

    def process(self):
        # convert d-spacing from A to nm
        self.d_space /= 10


class ArgsFactory:
    _options = {
        "builder": BuildArgs,
        "edit": EditArgs,
        "check": CheckArgs,
        "analysis": AnalysisArgs,
        "equilibrate": EquilibrateArgs,
    }

    @classmethod
    def init_subclass(cls, parse_args):
        if type(parse_args) != dict:
            data = parse_args.__dict__
        option = data.pop("option")
        try:
            _cls = cls._options[option]
        except KeyError:
            raise KeyError(f"{option!r} is not known!")
        print(_cls)
        return _cls(data)
