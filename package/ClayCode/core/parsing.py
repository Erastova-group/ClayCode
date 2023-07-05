#!/usr/bin/env python3

import logging
import os
import re
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from collections import UserDict
from functools import cached_property
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml
from ClayCode.builder.claycomp import (
    BulkIons,
    InterlayerIons,
    MatchClayComposition,
)
from ClayCode.core.classes import BasicPath, Dir, File, init_path
from ClayCode.core.consts import FF, MDP_DEFAULTS, UCS
from ClayCode.core.utils import get_header, get_subheader

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
    "analyse", help="Analyse clay simulations."
)

# plot analysis results
plotparser = subparsers.add_parser(
    "plot", help="Plot simulation analysis results"
)

# Clay simulation check parser
checkparser = subparsers.add_parser(
    "check", help="Check clay simulation data."
)

siminpparser = subparsers.add_parser(
    "siminp", help="Generate clay model equilibration run input files."
)
siminp_subparsers = siminpparser.add_subparsers()

dspace_arg_group = siminpparser.add_argument_group(
    "il_spacing"
)  # , help='Equilibrate d-spacing')

dspace_arg_group.add_argument(
    "-dspace",
    help="d-spacing in \u00C5",
    metavar="d_spacing",
    dest="dspace",
    type=float,
    required=True,
)

dspace_arg_group.add_argument(
    "-n_wat",
    help="number of water molecules to remove per cycle per unit cell",
    metavar="n_waters",
    dest="n_wat",
    type=float,
    default=0.2,
    required=False,
)

dspace_arg_group.add_argument(
    "-n_steps",
    help="water removal interval",
    metavar="n_steps",
    dest="n_steps",
    type=int,
    default=1000000,
    required=False,
)


def valid_run_type(run_name):
    try:
        run_type, run_id = run_name.split("_")
        assert re.match(
            r"[0-9]*", run_id
        ), f"Invalid run id option: {run_id!r}, must be numeric"
    except ValueError as e:
        run_type = run_name
    assert run_type in [
        "EQ",
        "P",
    ], f'Invalid run type option: {run_type!r}, select either "EQ" or "P"'
    return run_name


siminpparser.add_argument(
    "-runs",
    help="Run type specifications",
    # type=str,
    nargs="+",
    type=valid_run_type,
)

siminpparser.add_argument("-run_config")

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


def read_yaml_path_decorator(*path_args):
    def read_yaml_decorator(f):
        def wrapper(self: _Args):
            assert isinstance(self, _Args), "Wrong class for decorator"
            with open(self.data["yaml_file"], "r") as file:
                self.__yaml_data = yaml.safe_load(file)
            logger.info(f"Reading {file.name!r}:\n")
            for k, v in self.__yaml_data.items():
                if k in self._arg_names:
                    if k in path_args:
                        path = BasicPath(v).resolve()
                        if path.suffix != "":
                            try:
                                path = File(path, check=True)
                            except FileNotFoundError:
                                path = File(
                                    self.data["yaml_file"].parent / v,
                                    check=True,
                                )
                        else:
                            path = Dir(path)
                        v = str(path)
                    self.data[k] = v
                    logger.info(f"\t{k} = {v!r}")
                else:
                    raise KeyError(f"Unrecognised argument {k}!")
            return f(self)

        return wrapper

    return read_yaml_decorator


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
    from ClayCode.builder.consts import BUILD_DEFAULTS as _build_defaults
    from ClayCode.builder.consts import UC_CHARGE_OCC as _charge_occ_df

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
        self.zero_threshold: float = None
        self._raw_comp: pd.DataFrame = None
        self._corr_comp: pd.DataFrame = None
        self._uc_comp: pd.DataFrame = None
        self.mdp_parameters: Dict[str, Any] = None
        self.ff = None
        self.ucs = None
        self.il_solv = None
        # self.read_yaml()
        # self.check()
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
                    logger.info(
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
            elif cmdline_dest == "csv_file":
                raise ValueError(
                    "No csv file with clay composition specified!"
                )
            else:
                logger.debug("No mdp parameters specified")

    def check(self) -> None:
        try:
            self.name = self.data["SYSNAME"]
            logger.info(f"\nSetting name: {self.name!r}")
        except KeyError:
            raise KeyError("Clay system name must be given")
        try:
            uc_type = self.data["CLAY_TYPE"]
            if (
                uc_type
                in self._charge_occ_df.index.get_level_values("value").unique()
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
            self.il_solv = selected_solv
        except KeyError:
            self.il_solv = il_solv
        il_solv_prms = [
            prm
            for prm in self.data.keys()
            if prm in ["UC_WATERS", "ION_WATERS", "SPACING_WATERS"]
        ]
        assert len(il_solv_prms) == 1, (
            f"Only one interlayer solvation specification allowed!\n Found {len(il_solv_prms)}: "
            + ", ".join(il_solv_prms)
        )
        for prm in [
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
            "UC_INDEX_LIST",
            "UC_RATIOS_LIST",
            "OCC_TOL",
            "SEL_PRIORITY",
            "CHARGE_PRIORITY",
            "ZERO_THRESHOLD",
        ]:
            try:
                prm_value = self.data[prm]
            except KeyError:
                prm_value = self._build_defaults[prm]
            setattr(self, prm.lower(), prm_value)
        setattr(self, "mdp_parameters", MDP_DEFAULTS)
        try:
            mdp_prm_file = self.data["MDP_PRMS"]
            with open(mdp_prm_file, "r") as mdp_file:
                mdp_prms = yaml.safe_load(mdp_file)
                self.mdp_parameters["EM"].update(mdp_prms)
        except KeyError:
            pass
        try:
            outpath = self.data["OUTPATH"]
            self.outpath = Dir(outpath, check=False)
        except KeyError:
            raise KeyError("No output directory specified")
        try:
            GMX = self.data["GMX"]
        except KeyError:
            GMX = self._build_defaults["GMX"]
            logger.info(f"Using default GROMACS alias: {GMX}")
        setattr(self, "gmx_alias", GMX)

    def process(self):
        logger.info(get_header("Getting build parameters"))
        self.read_yaml()
        self.check()
        self.manual_setup = self.data["manual_setup"]
        self.filestem = f"{self.name}_{self.x_cells}_{self.y_cells}"
        self.outpath = self.outpath / self.name
        os.makedirs(self.outpath, exist_ok=True)
        logger.set_file_name(
            new_filepath=self.outpath, new_filename=self.filestem
        )
        # logger.rename_log_file(
        #     new_filepath=self.outpath,
        #     new_filename=BasicPath(self.outpath.name),
        # )
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
        logger.info(get_subheader("Getting force field data"))
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
        self._target_comp.write_csv(self.outpath)

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
            if self._was_specified["SPACING_WATERS"]:
                self.il_solv_height = self.data["SPACING_WATERS"]
            else:
                self.il_solv_height = (
                    n_ions * self.default_d_space
                ) / self.sheet_n_cells

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
        return self._target_comp.ion_df

    @cached_property
    def sheet_n_cells(self):
        return self.x_cells * self.y_cells

    def match_uc_combination(self):
        self.match_comp = MatchClayComposition(
            target_composition=self._target_comp,
            sheet_n_ucs=self.sheet_n_cells,
            manual_setup=self.manual_setup,
            ignore_threshold=self.zero_threshold,
        )
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

    def get_bulk_ions(self):
        self._bulk_ions = BulkIons(
            self.bulk_ions, self._build_defaults["BULK_IONS"]
        )

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
            *neutralise_ions[neutralise_ions["charge"] > 0]["conc"]
            .reset_index()
            .values
        )

    @cached_property
    def default_bulk_nion(self):
        neutralise_ions = self._bulk_ions.neutralise_ions
        return tuple(
            *neutralise_ions[neutralise_ions["charge"] < 0]["conc"]
            .reset_index()
            .values
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


class SiminpArgs(_Args):
    option = "siminp"
    # TODO: Add csv or yaml for eq run prms
    _run_order = ["EQ", "P"]

    def __init__(self, data):
        super().__init__(data)
        self.process()

    def process(self):
        logger.info(get_header("Getting simulation input parameters"))
        if "dspace" in self.data:
            logger.info(get_subheader("d-spacing equilibration parameters"))
            self.d_spacing = (
                self.data["dspace"] / 10
            )  # convert d-spacing from A to nm
            self.n_wat = self.data["n_wat"]
            self.n_steps = self.data["n_steps"]
            self.data["runs"].append("D_SPACE")
            logger.info(f"Target spacing: {self.d_spacing:2.2f} \u00C5")
            logger.info(
                f"Removal interval: {self.n_wat:2.2f} water molecules per unit cell every {self.n_steps} steps"
            )
        if self.data["runs"] is not None:
            self.run_sequence = sorted(self.data["runs"])
            assigned_id = 1
            for run_id, run_name in enumerate(self.run_sequence):
                try:
                    run_type, assigned_id = run_name.split("_")
                except ValueError:
                    run_type = run_name
                    if run_id != 0:
                        if prev == run_type:
                            if assigned_id == 1:
                                self.run_sequence[
                                    run_id - 1
                                ] = f"{prev}_{assigned_id}"
                            assigned_id += 1
                            self.run_sequence[
                                run_id
                            ] = f"{run_type}_{assigned_id}"
                        else:
                            assigned_id = 1
                prev = run_type
            logger.info(get_subheader("Selected the following run types:"))
            for run_id, run_name in enumerate(self.run_sequence):
                logger.info(f"\t{run_id + 1}: {run_name}")

    def check(self):
        ...


class ArgsFactory:
    _options = {
        "builder": BuildArgs,
        "edit": EditArgs,
        "check": CheckArgs,
        "analysis": AnalysisArgs,
        "siminp": SiminpArgs,
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
        # print(_cls)
        return _cls(data)
