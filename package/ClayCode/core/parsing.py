#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r""":mod:`ClayCode.core.parsing` --- Argument parsing module
===========================================================
This module provides classes for parsing command line arguments.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import sys
from abc import ABC, abstractmethod
from argparse import ArgumentParser, FileType
from collections import UserDict
from functools import cached_property
from typing import Any, Dict, List, Tuple, Type, Union

import MDAnalysis
import numpy as np
import pandas as pd
import yaml
from caseless_dictionary import CaselessDict
from ClayCode import PathType
from ClayCode.addmols.consts import ADDMOLS_DEFAULTS as _addmols_defaults
from ClayCode.addmols.consts import ADDTYPES
from ClayCode.builder.consts import BUILDER_DATA
from ClayCode.core.classes import (
    BasicPath,
    Dir,
    File,
    ForceField,
    PathFactory,
    init_path,
)
from ClayCode.core.consts import ANGSTROM
from ClayCode.core.lib import select_clay
from ClayCode.core.utils import (
    get_debugheader,
    get_header,
    get_subheader,
    parse_yaml,
    select_file,
)
from ClayCode.data.consts import FF, MDP_DEFAULTS, UCS, USER_UCS
from ClayCode.siminp.writer import MDPRunGenerator

__all__ = {
    "ArgsFactory",
    "parser",
    "BuildArgs",
    "EditArgs",
    "CheckArgs",
    "SiminpArgs",
    "PlotArgs",
    "AnalysisArgs",
}

logger = logging.getLogger(__name__)

parser: ArgumentParser = ArgumentParser(
    "ClayCode",
    description="Automatically generate atomistic clay models.",
    add_help=True,
    allow_abbrev=False,
)

parser.add_argument(
    "--debug",
    help="Debug run",
    action="store_true",
    default=False,
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

buildparser.add_argument(
    "-mdp",
    type=File,
    help="YAML file with mdp parameter options for energy minimisation run.",
    metavar="yaml_file",
    dest="mdp_prms",
    required=False,
)

buildparser.add_argument(
    "--manual_setup",
    help="Ask for confirmation at each model setup stage.",
    dest="manual_setup",
    action="store_true",
    default=False,
    required=False,
)

buildparser.add_argument(
    "--backup",
    help="Backup old files instead of overwriting them.",
    dest="backup",
    action="store_true",
    default=False,
    required=False,
)

buildparser.add_argument(
    "-max_ucs",
    help="Set maximum number of unit cells used in model.",
    dest="MAX_UCS",
    default=None,
    required=False,
)
buildparser.add_argument(
    "--save-progress",
    help="Save progress",
    dest="SAVE_PROGRESS",
    action="store_true",
    default=False,
    required=False,
)
buildparser.add_argument(
    "--load-progress",
    help="Load progress",
    dest="LOAD_PROGRESS",
    action="store_true",
    default=False,
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
analysisparser = subparsers.add_parser(
    "analysis", help="Analyse clay simulations."
)

analysisparser.add_argument(
    "-f",
    type=File,
    help="YAML file with analysis parameters",
    metavar="yaml_file",
    dest="yaml_file",
    required=True,
)

# analysisparser.add_argument(
#     "-name", type=str, help="System name", dest="sysname", required=True
# )
# analysisparser.add_argument(
#     "-inp",
#     type=str,
#     help="Input file names",
#     nargs=2,
#     metavar=("coordinates", "trajectory"),
#     dest="infiles",
#     required=False,
# )
# analysisparser.add_argument(
#     "-inpname",
#     type=str,
#     help="Input file names",
#     metavar="name_stem",
#     dest="inpname",
#     required=False,
# )
# analysisparser.add_argument(
#     "-uc",
#     type=str,
#     help="Clay unit cell type",
#     dest="clay_type",
#     required=True,
# )
# analysisparser.add_argument(
#     "-sel",
#     type=str,
#     nargs="+",
#     help="Atom type selection",
#     dest="sel",
#     required=True,
# )
# analysisparser.add_argument(
#     "-n_bins",
#     default=None,
#     type=int,
#     help="Number of bins in histogram",
#     dest="n_bins",
# )
# analysisparser.add_argument(
#     "-bin_step",
#     type=float,
#     default=None,
#     help="bin size in histogram",
#     dest="bin_step",
# )
# analysisparser.add_argument(
#     "-xyrad",
#     type=float,
#     default=3,
#     help="xy-radius for calculating z-position clay surface",
#     dest="xyrad",
# )
# analysisparser.add_argument(
#     "-cutoff",
#     type=float,
#     default=20,
#     help="cutoff in z-direction",
#     dest="cutoff",
# )
#
# analysisparser.add_argument(
#     "-start",
#     type=int,
#     default=None,
#     help="First frame for analysis.",
#     dest="start",
# )
# analysisparser.add_argument(
#     "-step",
#     type=int,
#     default=None,
#     help="Frame steps for analysis.",
#     dest="step",
# )
# analysisparser.add_argument(
#     "-stop",
#     type=int,
#     default=None,
#     help="Last frame for analysis.",
#     dest="stop",
# )
# analysisparser.add_argument(
#     "-out",
#     type=str,
#     help="Filename for results pickle.",
#     dest="save",
#     default=True,
# )
# analysisparser.add_argument(
#     "-check_traj",
#     type=int,
#     default=False,
#     help="Expected trajectory length.",
#     dest="check_traj_len",
# )
# analysisparser.add_argument(
#     "--write_z",
#     type=str,
#     default=True,
#     help="Binary array output of selection z-distances.",
#     dest="write",
# )
# analysisparser.add_argument(
#     "--overwrite",
#     action="store_true",
#     default=False,
#     help="Overwrite existing z-distance array data.",
#     dest="overwrite",
# )
# analysisparser.add_argument(
#     "--update",
#     action="store_true",
#     default=False,
#     help="Overwrite existing trajectory and coordinate array data.",
#     dest="new",
# )

# plot analysis results
plotparser = subparsers.add_parser(
    "plot", help="Plot simulation analysis results"
)
plotparser.add_argument(
    "-f",
    required=True,
    type=File,
    help="File with plotting specifications",
    metavar="yaml_file",
    dest="yaml_file",
)
plotgroup = plotparser.add_mutually_exclusive_group()
plotgroup.add_argument("-lines", action="store_true", default=False)
plotgroup.add_argument("-bars", action="store_true", default=False)
plotgroup.add_argument("-gbars", action="store_true", default=False)
atype_split_parser = plotparser.add_subparsers(dest="split_atypes")
atype_data_parser = atype_split_parser.add_parser("get_data")
atype_data_parser.add_argument(
    "-new_bins", type=float, default=None, required=False, dest="new_bins"
)
atype_data_parser.add_argument(
    "--overwrite", default=False, action="store_true", dest="overwrite"
)
atype_data_parser.add_argument(
    "-datadir", type=Dir, required=True, dest="datadir"
)
atype_data_parser.add_argument(
    "--load", type=bool, required=False, dest="load"
)
atype_data_parser.add_argument(
    "-save", type=BasicPath, default=False, dest="save"
)

# Clay simulation check parser
checkparser = subparsers.add_parser(
    "check", help="Check clay simulation data."
)

siminpparser = subparsers.add_parser(
    "siminp", help="Generate clay model equilibration run input files."
)

siminpparser.add_argument(
    "-f",
    help="Equilibration parameter yaml file.",
    type=File,
    required=False,
    dest="yaml_file",
)

dataparser = subparsers.add_parser(
    "data", help="Add new unit cell types to database."
)

dataparser.add_argument(
    "-f",
    help="YAML file with unit cell data",
    type=File,
    required=True,
    dest="yaml_file",
)


# siminp_subparsers = siminpparser.add_subparsers()

# dspace_arg_group = siminpparser.add_argument_group("il_spacing")
#
# remove_wat_group = siminpparser.add_mutually_exclusive_group(required=False)
#
# dspace_arg_group.add_argument(
#     "-dspace",
#     help="d-spacing in {ANGSTROM}",
#     metavar="d_spacing",
#     dest="D_SPACE",
#     type=float,
#     required=True,
# )

# remove_wat_group.add_argument(
#     "-uc_wat",
#     help="number of water molecules to remove per cycle per unit cell",
#     metavar="n_waters",
#     dest="UC_WAT",
#     type=float,
#     default=0.1,
#     required=False,
# )
#
# remove_wat_group.add_argument(
#     "-sheet_wat",
#     help="number of water molecules to remove per cycle per sheet",
#     metavar="n_waters",
#     dest="SHEET_WAT",
#     type=float,
#     default=None,
#     required=False,
# )
#
# remove_wat_group.add_argument(
#     "-percent_wat",
#     help="percentage of inital water molecules to remove per cycle per sheet",
#     metavar="n_waters",
#     dest="PERCENT_WAT",
#     type=float,
#     default=None,
#     required=False,
# )
#
# dspace_arg_group.add_argument(
#     "-remove_steps",
#     help="water removal interval",
#     metavar="remove_steps",
#     dest="REMOVE_STEPS",
#     type=int,
#     default=1000000,
#     required=False,
# )


# def valid_run_type(run_name):
#     try:
#         run_type, run_id = run_name.split("_")
#         assert re.match(
#             r"[0-9]*", run_id
#         ), f"Invalid run id option: {run_id!r}, must be numeric"
#     except ValueError as e:
#         run_type = run_name
#     assert run_type in [
#         "EQ",
#         "P",
#     ], f'Invalid run type option: {run_type!r}, select either "EQ" or "P"'
#     return run_name
#
#
# siminpparser.add_argument(
#     "-runs",
#     help="Run type specifications",
#     # type=str,
#     nargs="+",
#     type=valid_run_type,
# )
#
# siminpparser.add_argument("-run_config")

# TODO: add plotting?


def read_yaml_path_decorator(*path_args):
    def read_yaml_decorator(f):
        def wrapper(self: _Args, enumerate_duplicates=False):
            assert isinstance(self, _Args), "Wrong class for decorator"
            try:
                with open(self.data["yaml_file"], "r") as file:
                    self.__yaml_data = parse_yaml(
                        file, enumerate_duplicates=enumerate_duplicates
                    )
            except FileNotFoundError:
                logger.error(
                    f'{File(self.data["yaml_file"]).resolve()} not found!\nAborting...'
                )
                sys.exit(2)
            logger.finfo(f"Reading {file.name!r}:\n")
            for k, v in self.__yaml_data.items():
                if k in self._arg_names:
                    if k in path_args:
                        path = BasicPath(v).resolve()
                        if path.suffix != "":
                            path_list = [path]
                            for key in ["yaml_file", "INPATH"]:
                                try:
                                    self.data[key] = BasicPath(
                                        self.data[key], check=False
                                    )
                                except KeyError:
                                    pass
                                else:
                                    if self.data[key].suffix != "":
                                        self.data[key] = self.data[key].parent
                                    path_list.extend(
                                        [
                                            self.data[key]
                                            / path.relative_to(path.cwd()),
                                            self.data[key] / v,
                                        ]
                                    )
                            for path_option in path_list:
                                try:
                                    path_option = File(path_option, check=True)
                                except FileNotFoundError:
                                    found = False
                                    pass
                                else:
                                    path = path_option
                                    found = True
                                    break
                            if not found:
                                raise FileNotFoundError(
                                    f"File {v!r} not found!"
                                )
                                #  # try:  #     path = File(path, check=True)  # except FileNotFoundError:  #     try:  #         path = File(  #             self.data["yaml_file"].parent / v,  #             check=True,  #         )  #     except FileNotFoundError:  #         try:  #             path = File(  #                 self.data["INPATH"]  #                 / path.relative_to(path.cwd()),  #                 check=True,  #             )  #             found = True  #         except KeyError:  #             found = False  #         except FileNotFoundError:  #             found = False  #         finally:  #             if not found:  #                 raise FileNotFoundError(  #                     f"File {v!r} not found!"  #                 )
                        else:
                            path = Dir(path)
                        v = str(path)
                    self.data[k] = v
                    if type(v) != dict:
                        logger.finfo(kwd_str=f"\t{k} = ", message=f"{v!r}")
                else:
                    raise KeyError(f"Unrecognised argument {k}!")
            return f(self)

        return wrapper

    return read_yaml_decorator


class _Args(ABC, UserDict):
    option = None
    _arg_defaults = {}
    _arg_names = []
    data = {}

    def __init__(self, data: dict):
        self.data = data
        self.ff = None
        self.__yaml_data = None

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

    def _set_attributes(
        self, data_keys: List[str], optional: List[str] = None
    ):
        if optional is None:
            optional = []
        for prm in data_keys:
            try:
                prm_value = self.data[prm]
            except KeyError:
                try:
                    prm_value = self._arg_defaults[prm]
                except KeyError:
                    if prm in optional:
                        prm_value = None
                    else:
                        raise KeyError(f"Missing required parameter {prm!r}")
            setattr(self, prm.lower(), prm_value)

    @property
    def mdp_parameters(self) -> CaselessDict[str, str]:
        if not hasattr(self, "_mdp_defaults"):
            self._mdp_defaults = MDP_DEFAULTS
        return self._mdp_defaults

    def get_ff_data(self) -> Dict[str, Type["ForceField"]]:
        water_sel_dict = {"SPC": ["ClayFF_Fe", ["spc", "interlayer_spc"]]}
        ff_dict = {}
        logger.info(get_subheader("Getting force field data"))
        for ff_key, ff_sel in self.ff.items():
            if ff_key == "WATER":
                ff_sel = water_sel_dict[ff_sel][0]
            ff = ForceField(FF / ff_sel)
            ff_dict[ff_key.lower()] = ff
            logger.finfo(kwd_str=f"{ff_key}: ", message=f"{ff.name}")
            itp_str = f"\n\t".join(ff.itp_filelist.stems)
            logger.info(f"\t{itp_str}\n")
        self.ff = ff_dict

    @property
    def ff_data(self) -> Dict[str, Type["ForceField"]]:
        if not hasattr(self, "ff"):
            self.get_ff_data()
        return self.ff

    def _select_clay(
        self, universe: MDAnalysis.Universe
    ) -> MDAnalysis.Universe:
        return select_clay(universe, self.ff)

    @property
    def update_mdp_parameters(
        self, key: str, prm_dict: Union[CaselessDict[str, Any], Dict[str, Any]]
    ):
        mdp_prms = self.mdp_defaults
        try:
            mdp_prms[key].update(CaselessDict(prm_dict))
        except KeyError:
            mdp_prms[key] = CaselessDict(prm_dict)
        self._mdp_defaults = mdp_prms

    def _get_gmx_prms(self):
        try:
            GMX = self.data["GMX"]
        except KeyError:
            GMX = self._arg_defaults["GMX"]
            logger.finfo(
                kwd_str="Using default GROMACS alias: ", message=f"{GMX}"
            )
        setattr(self, "gmx_alias", GMX)

    def _get_outpath(self):
        try:
            outpath = self.data["OUTPATH"]
            self.outpath = Dir(outpath, check=False)
        except KeyError:
            raise KeyError("No output directory specified")


class BuildArgs(_Args):
    """Parameters for clay model setup with :mod:`ClayCode.builder`
    :param data: dictionary of command line arguments
    :type data: dict
    """

    option = "builder"
    from ClayCode.builder.consts import BUILD_DEFAULTS as _build_defaults
    from ClayCode.data.consts import UC_CHARGE_OCC as _charge_occ_df

    _arg_names = [
        "SYSNAME",
        "BUILD",
        "CLAY_TYPE",
        "X_CELLS",
        "Y_CELLS",
        "N_SHEETS",
        "IL_SOLV",
        "UC_INDEX_RATIOS",
        "IL_ION_RATIOS",
        "ION_WATERS",
        "UC_WATERS",
        "SPACING_WATERS",
        "DEFAULT_D_SPACE",
        "BOX_HEIGHT",
        "BULK_SOLV",
        "BULK_IONS",
        "CLAY_COMP",
        "OUTPATH",
        "FF",
        "GMX",
        "OCC_TOL",
        "SEL_PRIORITY",
        "CHARGE_PRIORITY",
        "MDP_PRMS",
        "ZERO_THRESHOLD",
        "Z_PADDING",
        "MIN_IL_HEIGHT",
        "MAX_UCS",
        "MATCH_TOLERANCE",
        "EM_FREEZE_CLAY",
        "EM_CONSTRAIN_CLAY_DIST",
        "SAVE_PROGRESS",
        "LOAD_PROGRESS",
    ]
    _arg_defaults = _build_defaults

    def __init__(self, data, debug_run=False) -> None:
        super().__init__(data)
        self.debug_run = debug_run
        self._target_comp = None
        self._uc_data = None
        self.filestem = None
        self.uc_charges = None
        self.x_cells = None
        self.y_cells = None
        self.boxheight = None
        self.n_sheets = None
        self._uc_name = None
        self.uc_stem = None
        self.name = None
        self._outpath = None
        self.zero_threshold = None
        self._raw_comp = None
        self._corr_comp = None
        self._uc_comp = None
        self.ff = None
        self.ucs = None
        self.il_solv = None
        self.process()

    @read_yaml_path_decorator("CLAY_COMP", "OUTPATH")
    def read_yaml(self) -> None:
        """Read clay model builder specifications
        and mdp_parameter defaults from yaml file."""
        files_dict = {"csv_file": "CLAY_COMP", "mdp_prms": "MDP_PRMS"}
        description_dict = {
            "csv_file": "clay composition",
            "mdp_prms": "mdp options",
        }
        for cmdline_dest, yaml_kwd in files_dict.items():
            try:
                csv_file = File(self.data[cmdline_dest], check=True)
            except TypeError:
                self.data.pop(cmdline_dest)
            try:
                yaml_csv_file = File(self.data[yaml_kwd], check=True)
            # except FileNotFoundError:
            #     yaml_csv_file = BasicPath(self.data[yaml_kwd]).resolve().relative_to(BasicPath('.').resolve())
            #     yaml_csv_file = File(self.data['yaml_file'].parent / yaml_csv_file, check=True)
            except KeyError:
                pass
            if (
                cmdline_dest in self.data.keys()
                and yaml_kwd in self.data.keys()
            ):
                if csv_file.absolute() == yaml_csv_file.absolute():
                    logger.finfo(
                        f"{description_dict.get(cmdline_dest)} {csv_file.absolute()} specified twice."
                    )
                    self.data[yaml_kwd] = csv_file
                    self.data.pop(cmdline_dest)
                else:
                    raise ValueError(
                        f"Two non-identical {description_dict.get(cmdline_dest)} files specified:"
                        f"\n\t1) {csv_file}\n\t2) {yaml_csv_file}"
                    )
            elif cmdline_dest in self.data.keys():
                self.data[yaml_kwd] = csv_file
                self.data.pop(cmdline_dest)
            elif yaml_kwd in self.data.keys():
                self.data[yaml_kwd] = yaml_csv_file
            elif (
                cmdline_dest == "csv_file"
                and "UC_INDEX_RATIOS" not in self.data.keys()
            ):
                raise ValueError(
                    "No CSV file with clay composition ('CLAY_COMP') or unit cell ratios ('UC_INDEX_RATIOS') specified!"
                )
            else:
                logger.debug("No mdp parameters specified")

    def check(self) -> None:
        try:
            self.name = self.data["SYSNAME"]
            logger.finfo(
                kwd_str="Setting name: ",
                message=f"{self.name!r}",
                initial_linebreak="\n",
            )
        except KeyError:
            raise KeyError("Clay system name must be given")
        try:
            uc_type = self.data["CLAY_TYPE"]
            if (
                uc_type
                in self._charge_occ_df.index.get_level_values("value").unique()
            ):
                if (UCS / uc_type).is_dir():
                    self._uc_path = UCS / uc_type
                elif (USER_UCS / uc_type).is_dir():
                    self._uc_path = USER_UCS / uc_type
                else:
                    raise ValueError(f"Unknown unit cell type {uc_type!r}!")
                self._uc_path = Dir(self._uc_path)
                self._uc_name = uc_type
                self._tbc = None
                tbc_match = re.search(
                    "[A-Z]+[0-9]+", uc_type, flags=re.IGNORECASE
                )
                if tbc_match:
                    pass
                self.uc_stem = self._uc_path.itp_filelist.filter(
                    "[A-Z][A-Z0-9]\d\d\d"
                )[0].stem[:-3]
                logger.debug(f"Setting unit cell type: {self._uc_name!r}")
            else:
                raise ValueError(f"Unknown unit cell type {uc_type!r}!")
        except ValueError as e:
            logger.error(f"{e}\nAborting...")
            sys.exit(2)
        except KeyError:
            logger.error("No unit cell type specified!")
            sys.exit(2)
        il_solv = self._charge_occ_df.loc[
            pd.IndexSlice["T", self._uc_name], ["solv"]
        ].values[0]
        try:
            selected_solv = self.data["IL_SOLV"]
            if not il_solv and selected_solv:
                raise ValueError(
                    f"Invalid interlayer solvation ({selected_solv}) for selected clay type {self._uc_name}!"
                )
            self.il_solv = selected_solv
        except KeyError:
            self.il_solv = il_solv
        if self.il_solv:
            il_solv_prms = [
                prm
                for prm in self.data.keys()
                if prm in ["UC_WATERS", "ION_WATERS", "SPACING_WATERS"]
            ]
            assert len(il_solv_prms) == 1, (
                f"Only one interlayer solvation specification allowed!\n Found {len(il_solv_prms)}: "
                + ", ".join(il_solv_prms)
            )
        data_keys = [
            "BUILD",
            "X_CELLS",
            "Y_CELLS",
            "N_SHEETS",
            "UC_WATERS",
            "DEFAULT_D_SPACE",
            "BOX_HEIGHT",
            "BULK_IONS",
            "BULK_SOLV",
            "FF",
            "UC_INDEX_RATIOS",
            "IL_ION_RATIOS",
            "OCC_TOL",
            "SEL_PRIORITY",
            "CHARGE_PRIORITY",
            "ZERO_THRESHOLD",
            "Z_PADDING",
            "MIN_IL_HEIGHT",
            "MATCH_TOLERANCE",
        ]
        self._set_attributes(data_keys=data_keys)
        if self.z_padding <= 0:
            raise ValueError(f"Interlayer padding must be > 0 {ANGSTROM}")
        # setattr(self, "mdp_parameters", MDP_DEFAULTS)
        setattr(self, "max_ucs", self.data["MAX_UCS"])
        setattr(self, "save_progress", self.data["SAVE_PROGRESS"])
        setattr(self, "load_progress", self.data["LOAD_PROGRESS"])
        try:
            mdp_prm_file = self.data["MDP_PRMS"]
            with open(mdp_prm_file, "r") as mdp_file:
                mdp_prms = yaml.safe_load(mdp_file)
                self.mdp_parameters.update_mdp_parameters("EM", mdp_prms)
        except KeyError:
            pass
        self.em_freeze_clay = self._charge_occ_df.loc[
            pd.IndexSlice["T", self._uc_name], ["em_freeze"]
        ].values[0]
        self.em_constrain_clay_dist = self._charge_occ_df.loc[
            pd.IndexSlice["T", self._uc_name], ["em_constrain_clay_dist"]
        ].values[0]
        self.em_n_runs = self._charge_occ_df.loc[
            pd.IndexSlice["T", self._uc_name], ["em_n_runs"]
        ].values[0]
        self._get_outpath()
        self._get_gmx_prms()

    def _get_zarr_storage(self, name, **kwargs):
        import zarr

        return zarr.storage.DirectoryStore(BUILDER_DATA / f"{name}", **kwargs)

    def process(self):
        logger.info(get_header("Getting build parameters"))
        self.read_yaml()
        self.check()
        self.manual_setup = self.data["manual_setup"]
        self.backup = self.data["backup"]
        self.filestem = f"{self.name}_{self.x_cells}_{self.y_cells}"
        # self.outpath = self.outpath / self.name
        os.makedirs(self.outpath, exist_ok=True)
        logger.set_file_name(
            new_filepath=self.outpath, new_filename=self.filestem
        )
        # init_path(self.outpath)
        logger.finfo(
            kwd_str="Setting output directory: ", message=f"{self.outpath}"
        )
        self.get_ff_data()
        self.get_uc_data()
        self.get_exp_data()
        self.match_uc_combination()
        self.get_il_ions()
        self.get_il_solvation_data()
        self.get_bulk_ions()

    @property
    def outpath(self):
        return self._outpath

    @outpath.setter
    def outpath(self, path: Union[str, PathType]):
        try:
            outpath = BasicPath(path, check=False)
        except TypeError:
            logger.error(f"Invalid type {type(path)} for output path!")
        else:
            if (
                self._outpath is None
                or outpath.resolve() != self._outpath.resolve()
            ):
                outpath = PathFactory(outpath)
                if type(outpath) == FileType or outpath.name == self.name:
                    outpath = Dir(outpath.parent / self.name, check=False)
                elif outpath.name != self.name:
                    outpath = outpath / self.name
                init_path(outpath)
                if self._outpath is not None:
                    for file in self._outpath.iterdir():
                        if file == outpath:
                            continue
                        elif file.is_dir() and file.name not in ["EM", "RUNS"]:
                            continue
                        shutil.move(file, outpath / file.name)
                self._outpath = outpath
                logger.set_file_name(
                    new_filepath=self._outpath, new_filename=self.filestem
                )

    def get_uc_data(self, reset=False):
        from ClayCode.builder.claycomp import UCData

        self._uc_data = UCData(
            self._uc_path,
            uc_stem=self.uc_stem,
            ff=self.ff["clay"],
            reset=reset,
        )
        occ = self._uc_data.occupancies
        ch = self._uc_data.oxidation_numbers
        atc = self._uc_data.atomic_charges

    def get_exp_data(self):
        if not self.uc_index_ratios:
            from ClayCode.builder.claycomp import TargetClayComposition

            clay_atoms = self._uc_data.df.index
            clay_atoms.append(pd.MultiIndex.from_tuples([("O", "fe_tot")]))
            self._target_comp = TargetClayComposition(
                name=self.name,
                csv_file=self.data["CLAY_COMP"],
                uc_data=self._uc_data,
                occ_tol=self.occ_tol,
                sel_priority=self.sel_priority,
                charge_priority=self.charge_priority,
                manual_setup=self.manual_setup,
                occ_correction_threshold=self.zero_threshold,
            )
            # if self.name not in self.data["CLAY_COMP"].keys():
            #     logger.finfo(f'Invalid system name {self.name!r}!\nAvailable options are:'+', '.join(self.data["CLAY_COMP"].keys()))
            #     sys.exit(1)
            self._target_comp.write_csv(self.outpath, backup=self.backup)
            self._ion_df = self._target_comp.ion_df

    def _was_specified(self, parameter: str) -> bool:
        return parameter.upper() in self.data.keys()

    def get_il_solvation_data(self):
        n_ions = np.sum([*self.n_il_ions.values()])
        if self.il_solv == True:
            if self._was_specified("ion_waters") and not (
                self._was_specified("uc_waters")
                or self._was_specified("spacing_waters")
            ):
                waters = self.data["ION_WATERS"]
                if isinstance(waters, dict):
                    assert waters.keys() in self.ion_df.index
                    for ion_type in waters.keys():
                        waters[ion_type] *= self.n_il_ions[ion_type]
                    waters = np.sum([*waters.values()]) + n_ions
                else:
                    waters *= n_ions
                self.n_waters = waters + n_ions
                self.il_solv_height = None
            elif self._was_specified("uc_waters") and not (
                self._was_specified("ion_waters")
                or self._was_specified("spacing_waters")
            ):
                self.n_waters = self.uc_waters * self.sheet_n_cells + n_ions
                self.il_solv_height = None
            elif self._was_specified("spacing_waters") and not (
                self._was_specified("uc_waters")
                or self._was_specified("ion_waters")
            ):
                self.n_waters = None
                self.il_solv_height = self.data["SPACING_WATERS"]
            else:
                raise AttributeError(
                    "Number of water molecules or interlayer solvent height to add must be "
                    'specified through either "ION_WATERS", '
                    '"UC_WATERS" or "SPACING_WATERS".'
                )
        else:
            self.n_waters = None
            if self._was_specified("spacing_waters"):
                self.il_solv_height = self.data["SPACING_WATERS"]
            else:
                self.n_waters = n_ions
                self.il_solv_height = None  # self.il_solv_height = (  #     n_ions * self.default_d_space  # ) / self.sheet_n_cells

    @property
    def uc_df(self):
        return self._uc_data.df

    @property
    def target_df(self):
        return self._target_comp.df

    # @property
    # def match_df(self):
    #     return self._match_comp.df

    @property
    def ion_df(self):
        return self._ion_df

    @cached_property
    def sheet_n_cells(self):
        return self.x_cells * self.y_cells

    def match_uc_combination(self):
        if not self.uc_index_ratios:
            from ClayCode.builder.claycomp import MatchClayComposition

            try:
                self.match_comp = MatchClayComposition(
                    target_composition=self._target_comp,
                    sheet_n_ucs=self.sheet_n_cells,
                    manual_setup=self.manual_setup,
                    ignore_threshold=self.zero_threshold,
                    debug_run=self.debug_run,
                    max_ucs=self.max_ucs,
                    max_dist=self.match_tolerance,
                )
            except Exception as e:
                logger.info(f"{e}")
                self.get_uc_data(reset=True)
                self.get_exp_data()
                self.match_comp = MatchClayComposition(
                    target_composition=self._target_comp,
                    sheet_n_ucs=self.sheet_n_cells,
                    manual_setup=self.manual_setup,
                    ignore_threshold=self.zero_threshold,
                    debug_run=self.debug_run,
                    max_ucs=self.max_ucs,
                    max_dist=self.match_tolerance,
                )
            finally:
                self.match_comp.write_csv(self.outpath, backup=self.backup)
        else:
            from ClayCode.builder.claycomp import (
                TargetClayComposition,
                UCClayComposition,
            )

            self.match_comp = UCClayComposition(
                sheet_n_ucs=self.sheet_n_cells,
                uc_data=self._uc_data,
                uc_index_ratios=self.uc_index_ratios,
                name=self.name,
            )
            ion_ratios = pd.Series(self.il_ion_ratios, name="at-type")
            ion_idx = pd.MultiIndex.from_product(
                [["I"], ion_ratios.index.values]
            )
            ion_ratios.reindex(ion_idx)
            self._ion_df = TargetClayComposition.get_ion_numbers(
                ion_ratios, self.match_comp.match_charge["tot"]
            )

    @property
    def match_df(self) -> pd.DataFrame:
        return self.match_comp.match_composition

    @property
    def sheet_uc_weights(self) -> pd.Series:
        return self.match_comp.uc_weights

    @property
    def sheet_uc_ids(self):
        """Unit cell IDs in clay sheet.
        :rtype: NDArray[str]"""
        return self.match_comp.uc_ids

    @property
    def match_charge(self):
        """Total charge of clay model."""
        return self.match_comp.match_charge

    def get_il_ions(self) -> None:
        """Get interlayer ion concentrations."""
        from ClayCode.builder.claycomp import InterlayerIons

        tot_charge = self.match_charge["tot"]
        if np.isclose(tot_charge, 0.0):
            self.il_ions = InterlayerIons(
                tot_charge=tot_charge,
                ion_ratios=self.ion_df,
                n_ucs=self.sheet_n_cells,
            )
        else:
            self.il_ions = InterlayerIons(
                tot_charge=tot_charge,
                ion_ratios=self.ion_df,
                n_ucs=self.sheet_n_cells,
            )

    def get_bulk_ions(self) -> None:
        """Get bulk ion concentrations."""
        from ClayCode.builder.claycomp import BulkIons

        self.bulk_ions = BulkIons(
            self.bulk_ions, self._build_defaults["BULK_IONS"]
        )

    @property
    def bulk_ion_conc(self) -> float:
        """Total bulk ion concentration.
        :rtype: float"""
        return self.bulk_ions.tot_conc

    @property
    def bulk_ion_df(self) -> pd.DataFrame:
        """DataFrame with bulk ion concentrations.
        :rtype: pd.DataFrame"""
        return self.bulk_ions.conc

    @cached_property
    def default_bulk_pion(self) -> Tuple[str, float]:
        """Default bulk cation species to neutralise excess charge.
        :rtype: Tuple[str, float]"""
        neutralise_ions = self.bulk_ions.neutralise_ions
        return tuple(
            *neutralise_ions[neutralise_ions["charge"] > 0]["conc"]
            .reset_index()
            .values
        )

    @cached_property
    def default_bulk_nion(self):
        """Default bulk anion species to neutralise excess charge.
        :rtype: Tuple[str, float]"""
        neutralise_ions = self.bulk_ions.neutralise_ions
        return tuple(
            *neutralise_ions[neutralise_ions["charge"] < 0]["conc"]
            .reset_index()
            .values
        )

    # @cached_property
    # def neutral_bulk_ions(self) -> Union[pd.DataFrame, None]:
    #     """Number of bulk ions to neutralise excess charge.
    #     :rtype: pd.DataFrame or None"""
    #     try:
    #         neutral_df = self._neutral_bulk_ions.df
    #     except AttributeError:
    #         neutral_df = None
    #     finally:
    #         return neutral_df

    @property
    def n_il_ions(self) -> Dict[str, int]:
        """Number of interlayer ions per clay sheet
        :rtype: Dict[str, int]"""
        return self.il_ions.numbers


class AnalysisArgs(_Args):
    """Parameters for analysing simulation run data with :mod:`ClayCode.analysis`"""

    option = "analysis"
    _arg_names = [
        "TYPE",
        "INPATH",
        "INFILES" "OUTPATH",
        "CLAY_TYPE",
        "SEL",
        "BIN_STEP",
        "CUTOFF",
        "START",
        "STEP",
        "STOP",
        "OUTNAME",
        "CHECK_TRAJ_LEN",
        "WRITE",
        "OVERWRITE",
        "XYRAD" "WRITE_Z",
        "UPDATE",
        "N_BINS" "NAME",
    ]

    def __init__(self, data: Dict[str, Any], debug_run: bool = False):
        super().__init__(data)
        self.debug_run = debug_run
        self.process()

    def check(self):
        pass

    @read_yaml_path_decorator("OUTPATH", "INPATH")
    def read_yaml(self):
        pass

    def process(self):
        self.read_yaml()
        self.check()


class CheckArgs(_Args):
    """Parameters for checking simulation run data with :mod:`ClayCode.check`
    :param data: dictionary of arguments
    :type data: Dict[str, Any]"""

    option = "check"

    def __init__(self, data: Dict[str, Any]):
        super().__init__(data)

    def check(self):
        pass


class EditArgs(_Args):
    """Parameters for editing clay model with :mod:`ClayCode.edit`

    :param data: dictionary of arguments
    :type data: Dict[str, Any]"""

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

    def __init__(self, data: Dict[str, Any]):
        super().__init__(data)


class PlotArgs(_Args):
    """Parameters for plotting analysis data with :mod:`ClayCode.plot`        :param data: dictionary of arguments
    :type data: Dict[str, Any]"""

    option = "plot"
    _arg_names = [
        "plot_type",
        "PLOT_SEL",
        "OUTPATH",
        "INPATH",
        "IGNORE_SUM",
        "YMIN",
        "YMAX",
        "XMIN",
        "XMAX",
        "YLABEL",
        "XLABEL",
        "CUTOFF",
        "BINS",
        "USE_ABS",
        "NAMESTEM",
        "ANALYSIS",
        "ATOMS",
        "AAS",
        "IONS",
        "CLAYS",
        "OTHER",
        "PH",
        "GROUPED",
        "TABLE",
        "EDGES",
        "ADD_BULK",
        "X",
        "Y",
        "PLSEL",
        "COLOURS",
        "NO_ATOMNAME",
        "FIGSIZE",
        "DATA2D",
        "ZDATA",
        "NEW_BINS",
        "OVERWRITE",
        "DATADIR",
        "LOAD",
        "SAVE",
    ]

    def __init__(self, data: Dict[str, Any], debug_run: bool = False):
        super().__init__(data)
        self.debug_run = debug_run
        self.process()

    def process(self):
        """Process simulation input arguments."""
        logger.info(get_header("Getting simulation input parameters"))
        self.read_yaml(enumerate_duplicates=True)
        self.check()

    @read_yaml_path_decorator("OUTPATH", "INPATH", "INGRO", "INTOP")
    def read_yaml(self) -> None:
        """Read plot specifications from yaml file."""
        pass

    def check(self):
        pass


class SiminpArgs(_Args):
    """Parameters for setting up GROMACS simulations with :mod:`ClayCode.siminp`
    :param data: dictionary of arguments
    :type data: Dict[str, Any]
    :debug_run: flag for running in debug mode
    :type debug_run: bool"""

    option = "siminp"
    from ClayCode.siminp.consts import SIMINP_DEFAULTS as _siminp_defaults

    # TODO: Add csv or yaml for eq run prms
    _run_order = [
        "EM",
        "EQ_NVT_F",
        "EQ_NpT_F",
        "EQ_NVT_R",
        "EQ_NpT_R",
        "EQ_NVT",
        "EQ_NpT",
        "D_SPACE",
        "[A-Z0-9_]",
    ]

    _arg_names = [
        "SIMINP",
        "OUTPATH",
        "SYSNAME",
        "INPATH",
        "INGRO",
        "INTOP",
        "RUN_SCRIPT",
        "INP_FILENAME",
        "GRO_NAME",
        "TOP_NAME",
        "RUN_PRMS",
        "MDP_PRMS",
        "GMX",
        "GMX_VERSION",
        "FF",
        "RUN_PATH",
        "MDRUN_PRMS",
        "SHELL",
        "HEADER",
    ]
    _arg_defaults = _siminp_defaults

    def __init__(self, data: Dict[str, Any], debug_run: bool = False):
        super().__init__(data)
        self.run_sequence = []
        self.debug_run = debug_run
        self.process()

    @read_yaml_path_decorator("OUTPATH", "INPATH", "INGRO", "INTOP")
    def read_yaml(self, enumerate_duplicates=True) -> None:
        """Read clay model builder specifications
        and mdp_parameter defaults from yaml file."""
        pass

    def _get_run_specs(self):
        """Get run specifications from YAML file data."""
        try:
            run_specs: dict = self.data["SIMINP"]
        except KeyError:
            logger.error(f"No run specifications found!")
            sys.exit(1)
        else:
            logger.finfo("Selected run types:")
            run_id = 0
            runs = {}
            matches = []
            for run_type in self._run_order:
                remove_keys = []
                for run_spec, run_options in sorted(run_specs.items()):
                    match = re.match(f"{run_type}[_0-9]*", run_spec)
                    if match:
                        # run_type = match.group(0)
                        matches.append((run_spec, match.group(0), run_options))
                        remove_keys.append(
                            run_spec
                        )  # else:  # run_type = run_spec  # non_matches.append((run_spec, run_spec, run_options))  # matches[run_spec] = (match.group(0), run_options)  # matches.append(run_spec, match.group(0), run_options)
                for match in remove_keys:
                    run_specs.pop(match)
            non_matches = [(k, k, v) for k, v in sorted(run_specs.items())]
            for run_name, run_match, run_options in [*matches, *non_matches]:
                run_id += 1
                if run_id == 1:
                    self.mdp_generator.add_run(
                        run_id,
                        run_name,
                        igro=self.ingro,
                        itop=self.intop,
                        odir=self.outpath,
                        deffnm=run_name,
                        run_dir=self.run_path,
                        **run_options,
                    )
                else:
                    self.mdp_generator.add_run(
                        run_id,
                        run_name,
                        odir=self.outpath,
                        deffnm=run_name,
                        run_dir=self.run_path,
                        **run_options,
                    )
                # run_options = run_specs.pop(run_name)
                # runs[run_id] = {match_run: run_options}
                # mdp_generator.add_run(
                #     run_id=run_id,
                #     name=match_run,
                # )
                # run_generator = MDPRunGenerator(mdp_prms=)
                # run = run_factory.init_subclass(
                #     match_run, run_id, run_options
                # )
                logger.finfo(f"\t{run_id}: {run_name}")  # return runs

    def check(self):
        """Check that all required arguments are present and set instance attributes."""

        data_keys = [
            "GMX",
            "GMX_VERSION",
            "INGRO",
            "OUTPATH",
            "INPATH",
            "INTOP",
            "FF",
            "SCRIPT_TEMPLATE",
            "RUN_PATH",
            "MDRUN_PRMS",
            "SHELL",
            "HEADER",
            "RUN_SCRIPT_NAME",
        ]
        paths = ["OUTPATH", "INPATH", "INGRO"]
        if [k in self.data.keys() for k in paths].count(True) == 0:
            raise ValueError(
                f"At leat one of {', '.join(paths)} must be specified for simulation input generation!"
            )
        elif [k in self.data.keys() for k in ["INPATH", "OUTPATH"]].count(
            True
        ) == 1:
            self.data["INPATH"] = self.data["OUTPATH"] = (
                self.data["INPATH"] or self.data["OUTPATH"]
            )
        elif [k in self.data.keys() for k in ["INPATH", "OUTPATH"]].count(
            True
        ) == 2:
            pass
        else:
            self.data["INPATH"] = self.data["OUTPATH"] = self.data[
                "INGRO"
            ].parent
        if "INGRO" not in self.data.keys():
            logger.finfo(
                f'Selecting latest GRO file from "{self.data["INPATH"]}:"'
            )
            self.data["INGRO"] = select_file(
                self.data["INPATH"], suffix=".gro", how="latest"
            )
        if "INTOP" not in self.data.keys():
            logger.finfo(
                f'Selecting latest TOP file from "{self.data["INPATH"]}:"',
                indent="\t",
            )
            self.data["INTOP"] = select_file(
                self.data["INPATH"], suffix=".top", how="latest"
            )
        if "RUN_PATH" not in self.data.keys():
            self.data["RUN_PATH"] = self.data["OUTPATH"]
        self._set_attributes(
            data_keys=data_keys,
            optional=[
                "INTOP",
                "SCRIPT_TEMPLATE",
                "MDRUN_PRMS",
                "HEADER",
                "RUN_SCRIPT_NAME",
            ],
        )

    def process(self):
        """Process simulation input arguments."""
        logger.info(get_header("Getting simulation input parameters"))
        self.read_yaml(enumerate_duplicates=True)
        self.check()
        self.get_ff_data()
        if self.data["SIMINP"]:
            prms_dict = {"mdp_prms": None, "run_prms": None}
            for k in prms_dict:
                try:
                    prms_dict[k] = self.data[k.upper()]
                except KeyError:
                    pass
            self.mdp_generator = MDPRunGenerator(
                gmx_alias=self.gmx,
                gmx_version=self.gmx_version,
                mdp_prms=prms_dict["mdp_prms"],
                run_options=prms_dict["run_prms"],
            )
        self._get_run_specs()  # if "dspace" in self.data:  #     logger.info(get_subheader("d-spacing equilibration parameters"))  #     self.d_spacing = self.data[  #         "D_SPACE"  #     ]  # convert d-spacing from A to nm  #     self.n_wat = self.data["n_wat"]  #     self.n_steps = self.data["n_steps"]  #     self.data["runs"].append("D_SPACE")  #     logger.finfo(f"Target spacing: {self.d_spacing:2.2f} {ANGSTROM}")  #     logger.finfo(  #         f"Removal interval: {self.n_wat:2.2f} water molecules per unit cell every {self.n_steps} steps"  #     )  # if len(self.mdp_generator._runs) != 0:  #     prms_dict = {"mdp_prms": None, "run_prms": None}  #     for k in prms_dict:  #         try:  #             prms_dict[k] = self.data[k.upper()]  #         except KeyError:  #             pass  #     self.run_sequence = self.data["runs"]  #     assigned_id = 1  #     logger.info(get_subheader("Selected the following run types:"))  #     for run_id, run_dict in sorted(self.run_sequence.items()):  #         for run_name, run_prms in run_dict.items():  #             try:  #                 run_type, assigned_id = run_name.split("_")  #             except ValueError:  #                 run_type = run_name  #                 assigned_id = run_id  #             finally:  #                 # if run_id != 1:  #         if prev == run_type:  #             if assigned_id == 1:  #                 self.run_sequence[run_id - 1] = {  #                     f"{prev}_{assigned_id}": run_prms  #                 }  #             assigned_id += 1  #             self.run_sequence[run_id] = {  #                 f"{run_type}_{assigned_id}": run_prms  #             }  #         else:  #             assigned_id = 1  # prev = run_type  #  # # for run_id, run_name in enumerate(self.run_sequence):  # logger.finfo(  #     f"\t{run_id}: {next(iter(self.run_sequence[run_id].keys()))}"  # )

    def write_runs(self, shell, header, script_name):
        self.mdp_generator.write_runs(
            self.outpath,
            ff=self.ff_data,
            run_dir=self.run_path,
            mdrun_prms=self.mdrun_prms,
            run_script_template=header,
            shell=shell,
            run_script_name=script_name,
        )


class DataArgs(_Args):
    """Class for handling arguments for adding or modifying interlayer data with :mod:`ClayCode.data`.
    :param data: command line arguments
    :param type: Dict[str, Any]
    :param debug_run: debug run flag
    :param type: bool"""

    option = "data"
    _arg_names = [
        "OUTPATH",
        "INGRO",
        "UC_TYPE",
        "UC_NAME",
        "SUBSTITUTIONS",
        "DEFAULT_SOLV",
    ]

    _arg_defaults = {"SUBSTITUTIONS": None, "DEFAULT_SOLV": None}

    def __init__(self, data, debug_run=False):
        super().__init__(data)
        self.process()

    def process(self):
        """Process data arguments."""
        self.read_yaml()
        self.check()

    @read_yaml_path_decorator("OUTPATH", "INGRO")
    def read_yaml(self):
        """Read specifications from yaml file."""
        pass

    def check(self) -> None:
        """Check that all required arguments are present."""
        self._set_attributes(data_keys=self._arg_names)


class AddMolsArgs(_Args):
    """Class for handling arguments for adding molecules with :mod:`ClayCode.addmols`."""

    option = "addmols"
    _arg_names = [
        "ODIR",
        "SYSNAME",
        "ADDTYPE",
        "MOLTYPES",
        "INGRO",
        "INTOP",
        "CONC",
        "PH",
        "FF",
        "NEUTRAL_IONS",
    ]
    _arg_defaults = _addmols_defaults

    def __init__(self, data, debug_run=False):
        super().__init__(data)
        self.process()

    def process(self):
        """Process data arguments."""
        self.read_yaml()
        self.check()

    @read_yaml_path_decorator("OUTPATH", "INGRO")
    def read_yaml(self):
        """Read specifications from yaml file."""
        pass

    def check(self) -> None:
        """Check that all required arguments are present."""
        if self.data["ADDTYPE"] not in ADDTYPES:
            raise ValueError(
                f"Invalid parameter {self.data['ADDTYPE']!r} for 'addmols'!\nAvailable options are: {', '.join(ADDTYPES)}"
            )
        self._set_attributes(data_keys=self._arg_names)


class ArgsFactory:
    """Factory class for initialising parser argument classes."""

    _options = {
        "builder": BuildArgs,
        "edit": EditArgs,
        "check": CheckArgs,
        "analysis": AnalysisArgs,
        "siminp": SiminpArgs,
        "data": DataArgs,
        "addmols": AddMolsArgs,
        "plot": PlotArgs,
    }

    @classmethod
    def init_subclass(cls, parse_args):
        """Initialise parser argument class based on command line arguments.
        :param parse_args: command line arguments
        :param type: Dict[str, Any]
        :return: parser argument class
        :return type: _Args subclass"""
        if type(parse_args) != dict:
            data = parse_args.__dict__
        option = data.pop("option")
        debug_run = data.pop("DEBUG")
        if debug_run:
            logger.info(get_debugheader("DEBUG RUN"))
        try:
            _cls = cls._options[option]
        except KeyError:
            raise KeyError(f"{option!r} is not known!")
        return _cls(data, debug_run=debug_run)
