#!/usr/bin/env python3

import logging
import os
import pickle as pkl
import re
import shutil
from argparse import ArgumentParser
from functools import cached_property
from pathlib import Path, PosixPath
from typing import (
    Callable,
    Dict,
    List,
    Literal,
    NoReturn,
    Optional,
    Tuple,
    Union,
    cast,
)

import MDAnalysis
import numpy as np
import pandas as pd
from ClayAnalysis import CLAYS, FF
from ClayAnalysis.classes import SimDir
from ClayAnalysis.lib import (
    add_resnum,
    fix_gro_residues,
    get_ion_charges,
    get_system_charges,
    neutralise_system,
    remove_excess_ions,
    rename_il_solvent,
)
from ClayAnalysis.plots import RawData
from ClayAnalysis.setup import add_aa, run_em
from ClayAnalysis.utils import (
    change_suffix,
    convert_str_list_to_arr,
    copy_final_setup,
    execute_bash_command,
    grep_file,
    str_to_int,
)

tpr_logger = logging.getLogger("MDAnalysis.topology.TPRparser").setLevel(
    level=logging.WARNING
)


__all__ = ["DirCheck", "RunChecks"]

N_ION_DICT = {0.25: {"NAu-1-fe": 28, "NAu-2-fe": 21}}

ION_CHARGE_DICT = get_ion_charges()

CLAY_DICT = {"NAu-1-fe": "D2", "NAu-2-fe": "D2"}

logger = logging.getLogger(Path(__file__).stem)


class DirCheck:
    """Class for MD simulation data checks.
    Checks for single simulation runs
    that are saved within one directory:
    1. presence of (coordinate, topology, trajectory) files
    2. trajectory length
    3. charge neutrality
    4. bulk ion concentration
    5. correct bulk ion species
    6. correct amino acid addition
    7. Readability of files
    """

    error_dict = {
        "analysis": False,
        "files": False,
        "charge": False,
        "ion_conc": False,
        "setup": False,
        "ion_species": False,
        "aa_species": False,
        "sim_run": False,
        "traj_len": False,
    }

    def __init__(
        self,
        name: str,
        path: Union[Path, str],
        subfolder: str = "",
        trr_sel: Union[Literal["latest"], Literal["largest"]] = "largest",
        conc: float = 0.25,
    ) -> None:
        try:
            self._n_ion_dict = N_ION_DICT[conc]
        except KeyError:
            logger.error(
                f"No ion numbers found for concentration of {conc} mol L-1"
            )
            raise KeyError(
                f"No ion numbers found for concentration of {conc} mol L-1"
            )
        self.reset_errors()
        self.name = name
        self.path = SimDir(path, check=False)
        self.clay, self.ion, self.aa = self.idx = self.path.idx
        self.strinfo = f"{self.clay} {self.ion} {self.aa}"
        if (type(subfolder) == str and subfolder != "") or subfolder is None:
            self.check_subfolders(subfolder=subfolder)
        if self.path.is_dir():
            logger.info(f"Clay: {self.clay}, ion: {self.ion}, aa: {self.aa}")
            self.path.trr_sel = trr_sel
            logger.info(f"Path: {self.path.resolve()}")
            logger.debug(f"Looking for file types: {self.path.suffices}")
            self.trr_sel = trr_sel
            self.__fixes = []
            self.complete = False

        else:
            logger.debug(f"{self.path!r} is not a directory!")
            self.error_dict = None
            self.path = None
            self.__fixes = ["setup"]

    def check_subfolders(self, subfolder: Union[str, None]) -> None:
        """Check if current path is already a new setup and update path accordingly."""
        if subfolder != "":
            subfolder = self.path / subfolder
            self.path = SimDir(subfolder, check=False)
            self.path.subfolder = True
            logger.debug(f"Updating path: {self.path.resolve()}")

    def reset_errors(self) -> None:
        """Set all errors to False"""
        for k in self.error_dict.keys():
            self.error_dict[k] = False

    @cached_property
    def system_log_charge(self) -> Union[int, None]:
        """Get system charge from gmx mdrun .log file"""
        charge_str = grep_file(self.path.log, "System total charge")
        try:
            charge = re.match("[\sA-Za-z]+:\s+([-]*\d+).*", charge_str).group(
                1
            )
            charge = str_to_int(charge)
        except AttributeError:
            charge = None
        return charge

    @cached_property
    def system_gro_charge(self) -> Union[int, None]:
        """Get system total charge from _gro file and force field data."""
        try:
            charge = get_system_charges(crds=self.path._gro)
        except:
            charge = None
        return charge

    @property
    def _universe(self) -> Union[MDAnalysis.Universe, None]:
        """Constructs universe from '.tpr' (and '.trr' if present)
        Returns None if no topology found.
        :return: Universe
        :rtype: Union[MDAnalyis.Universe, None]
        """
        u = None
        try:
            tpr = str(self.path.tpr.resolve())
            format = "TPR"
            u = MDAnalysis.Universe(tpr, format=format)
        except AttributeError:
            try:
                logger.info("Could not find '.tpr', trying '._gro'")
                gro = str(self.path._gro.resolve())
                format = "GRO"
                u = MDAnalysis.Universe(gro, format=format)
            except AttributeError:
                logger.error("Could not find files to construct Universe.")
        if u is not None:
            try:
                traj = str(self.path.trr.resolve())
                format = "TRR"
                u.load_new(traj, format=format)
            except AttributeError:
                logger.error("No trajectory found")
                # self.error_dict['sim_run'] = True
            except OSError:
                logger.error("Empty trajectory")
                # self.error_dict['sim_run'] = True
        return u

    def _get_ion_numbers(self, anion: str = "Cl") -> Tuple[int, int]:
        """Get number of bulk ion additions from
        :param anion: anion type
        :type anion: str
        :return: number of bulk cations, number of bulk anions
        :rtype: int, int
        """
        with open(self.path.top, "r") as topfile:
            topstr = topfile.read()
        topstr = re.search(
            r"\[ system \](.*)", topstr, flags=re.MULTILINE | re.DOTALL
        ).group(1)
        cations = re.findall(
            rf"(?<=SOL).*{self.ion}\s+(\d+)\n",
            topstr,
            flags=re.DOTALL | re.MULTILINE,
        )
        anions = re.findall(
            rf"(?<=SOL).*{anion}\s+(\d+)\n",
            topstr,
            flags=re.DOTALL | re.MULTILINE,
        )
        return str_to_int(cations), str_to_int(anions)

    def run(self, traj_len: int, anion="Cl") -> None:
        """Run checks on simulation data in specified path"""
        # Skip if simulation directory does not exist
        if self.path is None:
            pass
        else:
            traj_len = int(traj_len)
            # 1. File existence checks
            if len(self.path.missing) != 0:
                logger.debug(
                    f"Missing output files in {self.path.name}!\n"
                    f"Found {self.path.suffices}, "
                    f"expected {list(self.path.pathdict.keys())}"
                )
                self.error_dict["files"] = self.path.missing
                logger.error(
                    f"1. Filenumber checks NOT passed: "
                    f"missing = {self.error_dict['files']}"
                )
            else:
                logger.info(f"1. Filenumber checks passed.")
            # 2. System charge check
            charge = None
            if charge is None and self.path.tpr is not None:
                charge = int(
                    np.sum(
                        MDAnalysis.Universe(
                            str(self.path.tpr), format="TPR"
                        ).atoms.charges
                    )
                )
                logger.info(f"Using tpr charge")
            if charge is None and self.path._gro is not None:
                logger.info(f"Using _gro charge")
                charge = self.system_gro_charge
            if charge is None:
                logger.error(f"2. Charge not checked: Files missing!")
            elif charge == 0:
                logger.info("2. Charge check passed.")
            else:
                logger.error(
                    f"2. Charge check NOT passed: " f"charge = {charge}"
                )
                self.error_dict["charge"] = charge

            # 3. Trajectory length check
            if "trr" in self.path.suffices and (
                "tpr" in self.path.suffices or "_gro" in self.path.suffices
            ):
                u = self._universe
                if u is None:
                    # self.error_dict["analysis"] = True
                    logger.error("3. Could not read trajectory.")
                else:
                    try:
                        # Get frame number of trajectory
                        found_frames = self._universe.trajectory.n_frames
                    except AttributeError:
                        # No trajectory found
                        found_frames = 0
                    if found_frames == traj_len:
                        logger.debug(
                            f"Correct trajectory length: " f"{traj_len} frames"
                        )
                        logger.info("3. Frame number checks passed.")
                    elif found_frames == 0:
                        logger.error(
                            "3. Frame number checks NOT passed: "
                            f"found {found_frames} frames"
                        )
                        self.error_dict["sim_run"] = True
                    else:
                        logger.error(
                            "3. Frame number checks NOT passed: "
                            f"found {found_frames} frames"
                        )
                        self.error_dict["traj_len"] = True
            elif "tpr" in self.path.suffices or "_gro" in self.path.suffices:
                logger.error("3. Could not read trajectory.")
                # u = self._universe
            else:
                logger.error(
                    "3. Could not read trajectory or coordinates/topology."
                )
            # 4. Ion concentration check
            if "top" in self.path.suffices:
                n_cations, n_anions = self._get_ion_numbers(anion=anion)
                if self._n_ion_dict[self.clay] in n_cations:
                    n_add = self.get_n_bulk_ion_additions(anion=anion)
                    add_species = self.get_n_bulk_ion_add_species(anion=anion)
                    if n_add in [0, 1] or add_species in [0, 1]:
                        numbers = True
                    elif n_add == 2 or add_species == 2:
                        add_dict = self.get_bulk_ion_add_dict()
                        cat_charge = ION_CHARGE_DICT[self.ion]
                        if cat_charge > 1:
                            an_charge_remainder = (
                                add_dict[anion] * ION_CHARGE_DICT[anion]
                            )
                            if cat_charge + an_charge_remainder > 0:
                                numbers = True
                            else:
                                numbers = False
                        else:
                            numbers = False
                    else:
                        numbers = False
                        # with open(self.path.top, 'r') as topfile:
                        #     topstr = topfile.read()
                        # topstr = re.search(r'(?<=\[ molecules \])(.*)', topstr, flags=re.DOTALL | re.MULTILINE)
                        logger.error(
                            f"4 a. Ion number checks NOT passed: "
                            f"found {n_add} bulk additions"
                            f"found {add_species} added species"
                        )
                        self.error_dict["ion_conc"] = (n_cations, n_anions)
                else:
                    numbers = False
                if numbers is True:
                    logger.info(f"4 a. Ion number checks passed.")
                else:
                    logger.error(f"4 a. Ion number checks NOT passed: ")
                    self.error_dict["ion_conc"] = True
                if self.aa != "ctl_7":
                    aa = self.aa.strip("_7")
                    with open(self.path.top) as topfile:
                        topstr = topfile.read()
                    topstr = re.search(
                        r"(?<=\[ system \])(.*)",
                        topstr,
                        flags=re.MULTILINE | re.DOTALL,
                    ).group(1)
                    aa_add = re.search(
                        rf"{aa.upper()}7",
                        topstr,
                        flags=re.MULTILINE | re.DOTALL,
                    )
                    if aa_add is not None:
                        aa_match = aa_add.group(0)
                        self.error_dict["aa_species"] = aa_match
                        logger.error(
                            f"4 b. AA species checks not passed, found {aa_match}"
                        )
                    else:
                        logger.info(f"4 b. AA species checks passed")
            else:
                logger.error(
                    "4 a. Ion concentration not checked: No topology found!"
                )
            # 5. Trajectory exists checks
            if "trr" not in self.path.suffices:
                logger.error("5. Trajectory exists check NOT passed!\n")
                self.error_dict["sim_run"] = True
            else:
                logger.info(f"5. Trajectory exists check passed.\n")

            error_list = self.errors
            logger.info(f"Getting fixes for errors: {error_list}")
            missing = self.path.missing
            if len(error_list) == 0:
                logger.info(f"COMPLETE: no fixes for {self.strinfo}\n")
                self.complete = True
            elif error_list == ["traj_len"]:
                logger.info(
                    f"INCOMPLETE RUN: extend simulation for {self.strinfo}\n"
                )
                self.fixes = "extend"
            elif (
                ("_gro" in missing or "top" in missing)
                or "ion_species" in error_list
                or "aa_species" in error_list
            ):
                logger.info(f"SETUP: repeat setup {self.strinfo}\n")
                self.fixes = "setup"
            elif "charge" in error_list or "ion_conc" in error_list:
                logger.info(
                    f"ION NUMBERS/CHARGE: correct ion numbers for {self.strinfo}\n"
                )
                self.fixes = "neutral"
            elif "sim_run" in error_list:
                logger.info(
                    f"NO RUN: start simulation run for {self.strinfo}\n"
                )
                self.fixes = "run"
            elif "analysis" in error_list:
                self.fixes = "analysis"
                logger.info(
                    f"UNEXPECTED ERROR: check failed for {self.strinfo}\n"
                )

    @property
    def fixes(self) -> Union[str, None]:
        """Check that number of assigned fixes <= 1.
        Returns None if no fixes necessary.
        Raises IndexError if more than 1 fix assigned.
        """
        if len(self.__fixes) == 0:
            fixes = None
        elif len(self.__fixes) == 1:
            fixes = self.__fixes[0]
        else:
            raise IndexError(f"Found {len(self.__fixes)} fixes, expected 1.")
        return fixes

    @fixes.setter
    def fixes(self, fix: str) -> None:
        """
        Add new fix to simulation run fixes list
        :param fix: selected fix
        :type fix: str
        """
        if str not in self.__fixes:
            self.__fixes.append(fix)

    @property
    def errors(self) -> List[str]:
        """Return list of keys in self.error_dict where dict values != False"""
        return [k for k, v in self.error_dict.items() if v is not False]

    # def _neutral_fpath(self, fname: Path, suffix=None, odir=None) -> Path:
    #     if isinstance(fname, str):
    #         fname = Path(fname)
    #     if suffix is None:
    #         suffix = fname.suffix
    #     else:
    #         suffix = f".{suffix.lstrip('.')}"
    #     if odir is None:
    #         odir = SimDir(fname.parent, check=False)
    #     else:
    #         odir = Path(odir)
    #     if not odir.is_dir():
    #         os.makedirs(odir)
    #     return odir / f"{fname.stem.rstrip('_neutral')}_neutral{suffix}"

    def bulk_ion_addition_decorator(f):
        def bulk_ion_additions_wrapper(self, anion: str = "Cl"):
            n_ions = self._n_ion_dict[self.clay]
            with open(self.path.top, "r") as topfile:
                topstr = topfile.read()
            extra_ion_matches = re.search(
                rf"system.*(SOL\s+\d+\n)(.*)({self.ion}\s+{n_ions}\n{anion}\s+[0-9]+\n)[A-Z1-9]+\s+[0-9]+",
                topstr,
                flags=re.MULTILINE | re.DOTALL,
            )
            try:
                extra_ion_matches = extra_ion_matches.group(2)
                extra_ion_matches = extra_ion_matches.splitlines()
            except AttributeError:
                extra_ion_matches = []
            if len(extra_ion_matches) >= 2:
                extra = []
                for match in extra_ion_matches:
                    extra.append(
                        re.match(r"[A-Za-z]+\s+[0-9]+", match).group(0)
                    )

            else:
                extra = extra_ion_matches
            return f(extra)

        return bulk_ion_additions_wrapper

    @bulk_ion_addition_decorator
    def get_n_bulk_ion_additions(extra_ion_matches: List[str]) -> int:
        return len(extra_ion_matches)

    @bulk_ion_addition_decorator
    def get_n_bulk_ion_add_species(extra_ion_matches: List[str]) -> int:
        ions = None
        matches = convert_str_list_to_arr(extra_ion_matches)
        try:
            ions = np.unique(matches[:, 0])
        except IndexError:
            if len(matches) == 0:
                ions = []
        if ions is not None:
            return len(ions)

    @bulk_ion_addition_decorator
    def get_bulk_ion_add_dict(extra_ion_matches: List[str]) -> Dict[str, int]:
        matches = convert_str_list_to_arr(extra_ion_matches)
        if len(matches) != 0:
            ion_df = pd.DataFrame.from_records(matches).groupby(0).sum()
            ion_df = ion_df.astype({1: int})
            ion_dict = ion_df.to_dict()[1]
        else:
            ion_dict = {}
        return ion_dict

    def get_ion_numbers(self, anion="Cl") -> Tuple[int, int]:
        if self.system_log_charge < 0:
            npatoms = abs(self.system_log_charge // ION_CHARGE_DICT[self.ion])
            remainder = abs(self.system_log_charge % ION_CHARGE_DICT[self.ion])
            if remainder != 0:
                npatoms += 1
                nnatoms = 1
            else:
                nnatoms = 0

        else:
            nnatoms = abs(self.system_log_charge * ION_CHARGE_DICT[anion])
            npatoms = 0
        return nnatoms, npatoms

    # def neutralize(
    #     self,
    #     odir: Path,
    #     anion: str = "Cl",
    #     overwrite=False,
    #     rm_tempfiles=True,
    #     outname="neutral",
    # ):
    #     if self.error_dict["charge"] is False:
    #         logger.info("Fix: Neutralising excess charge.")
    #     if self.error_dict["ion_conc"] is False:
    #         logger.info("Fix: Correcting ion concentrations.")
    #     _gro = Path(self.path._gro)
    #     top = Path(self.path.top)
    #     ndir = odir / "neutral"
    #     odir = odir / ".tmp"
    #     if ndir.exists():
    #         logger.info(f"{ndir} already exists.")
    #     if not ndir.is_dir() or overwrite is True:
    #         new_gro = self._neutral_fpath(_gro, odir=odir)
    #         new_top = self._neutral_fpath(top, odir=odir)
    #         end_top = self._neutral_fpath(outname, suffix="top", odir=odir)
    #         add_resnum(crdin=_gro, crdout=new_gro)
    #         rename_il_solvent(crdin=new_gro, crdout=new_gro)
    #         remove_excess_ions(
    #             crdin=new_gro,
    #             topin=top,
    #             topout=new_top,
    #             crdout=new_gro,
    #             n_ions=self._n_ion_dict[self.clay],
    #             ion_type=self.ion,
    #         )
    #         assert new_gro.exists()
    #         logger.debug(f"New files: {new_gro}, {new_top}")
    #
    #         logger.debug("Neutralising:")
    #         n_replaced = neutralise_system(
    #             odir=odir,
    #             crdin=new_gro,
    #             topin=new_top,
    #             topout=end_top,
    #             nion=anion,
    #             pion=self.ion,
    #         )
    #         if n_replaced != 0:
    #             logger.info(f"Successful replacement of {n_replaced} ions.")
    #             for file in odir.glob(r"#*.*#"):
    #                 os.remove(file)
    #                 logger.debug(file)
    #             _ = run_em(
    #                 mdp="em.mdp",
    #                 crdin=new_gro,
    #                 topin=end_top,
    #                 tpr=odir / "em.tpr",
    #                 odir=odir,
    #                 outname=f"{self.aa}_neutral",
    #             )
    #             logger.info(f"Writing final output to {ndir!r}.")
    #             # write final coordinates and topology to output directory
    #             copy_final_setup(outpath=ndir, tmpdir=odir, rm_tempfiles=rm_tempfiles)
    #         else:
    #             logger.info(f"Ion insertion not successful, found {n_replaced}!")
    #             shutil.rmtree(odir)


class RunChecks:
    def __init__(
        self,
        root_dir: Optional[Union[str, Path]] = None,
        alt_dir: Optional[Union[str, Path]] = None,
        clays: Optional[List[str]] = None,
        ions: Optional[List[str]] = None,
        aas: Optional[List[str]] = None,
        new_dirs: Optional[List[str]] = None,
        data_df_file: Optional[Union[str, Path]] = None,
        df: Optional[pd.DataFrame] = None,
        odir: Optional[Union[str, Path]] = None,
    ):
        self.fixed_df = None
        idx_names = [*RawData.idx_names, "paths"]
        self.errors = list(DirCheck.error_dict.keys())
        err_col_names = [*self.errors, "exists", "complete", "fixes"]
        self.suffices = SimDir._suffices
        col_names = [*err_col_names, *self.suffices]

        self.root_dir = str(Path(root_dir).resolve())
        if df is not None or data_df_file is not None:
            self.new = True
            if df is not None:
                if type(df) != pd.DataFrame:
                    logger.error(
                        f"df: Expected pd.DataFrame, found {type(df)}"
                    )
                else:
                    self.df.update(df)
                    self.new = False
                    logger.info(f"Using data from df")
            elif data_df_file is not None and Path(data_df_file).is_file():
                logger.info(f"Loading df from {Path(data_df_file).suffix!r}")
                if Path(data_df_file).suffix == ".p":  # print('reading df')
                    with open(data_df_file, "rb") as file:
                        df = pkl.load(file)
                    self.df = df
                    self.new = False
                elif Path(data_df_file).suffix == ".csv":
                    logger.info(Path(data_df_file))
                    self.df = pd.read_csv(str(Path(data_df_file).resolve()))
                    self.df.set_index(idx_names, inplace=True)
                    self.new = False
                else:
                    logger.info(f"Invalid format")
            else:
                logger.info(f"Invalid df specification")
                logger.info(f"Using data from {data_df_file!r}")
            root_idx = (
                self.df.index.get_level_values("root").astype(str).unique()
            )
            if str(self.root_dir) not in root_idx:
                raise ValueError("Wrong root directory specified")
            if len(root_idx) == 2:
                self.alt_dir = [
                    root for root in root_idx if root != self.root_dir
                ][0]
            elif len(root_idx) > 2:
                raise IndexError(
                    f"Expected a maximum of one extra run directory, "
                    f"found {len(root_idx)}"
                )
            if len(self.df.columns) == len(col_names):
                self.df.columns = col_names
                if (self.df.columns == col_names).all():
                    logger.info(f"Successfully loaded DataFrame.")
                else:
                    logger.error(f"Error creating the dataframe.")
            elif len(self.df.columns) == len(err_col_names):
                self.df.columns = err_col_names
                if (self.df.columns == err_col_names).all():
                    logger.info("Adding file columns")
                    for c_name in self.suffices:
                        self.df[c_name] = np.NaN
                else:
                    logger.error(f"Error creating the dataframe.")
            else:
                logger.error(f"Error creating the dataframe.")
                raise RuntimeError(f"Error loading the dataframe.")

        else:
            self.new = True
            logger.info(f"getting data: {root_dir}, {alt_dir}")
            # search root_dir (and alt_dir) for sim data directories and files
            self.__data = RawData(
                root_dir=root_dir,
                alt_root=alt_dir,
                clays=clays,
                ions=ions,
                aas=aas,
                new_dirs=new_dirs,
                odir=odir,
            )
            self.new_dirs = self.__data.new_dirs
            self.idx_iter = self.__data.idx_iter
            check_df = self.__data.df.copy()
            check_df = check_df.stack(dropna=False)
            err_dict = dict(
                map(
                    lambda k: (k, np.full_like(check_df.values, np.NaN)),
                    DirCheck.error_dict.keys(),
                )
            )
            err_df = pd.DataFrame(err_dict, index=check_df.index)
            err_df["exists"] = check_df
            err_df["complete"] = False
            err_df["fixes"] = np.NaN
            for c_name in self.suffices:
                err_df[c_name] = np.NaN
            logger.info(f"Creating new DataFrame")
            self.alt_dir = self.__data.alt
            try:
                self.alt_dir = str(self.__data.alt.resolve())
            except:
                logger.info("No alternative root directory.")
            self.df = err_df
            if (
                len(self.df.columns) == len(err_col_names)
                and (self.df.columns == err_col_names).all()
            ):
                logger.info("Adding file columns")
                for c_name in self.suffices:
                    self.df[c_name] = np.NaN

        self.results = pd.DataFrame(
            index=self.df.index.droplevel(["root", "paths"]).unique(),
            columns=[
                "complete",
                "extend",
                "run",
                "neutral",
                "setup",
                "missing",
                "error",
                *self.suffices,
            ],
            dtype="object",
        )
        self.results["path"] = np.NaN
        if self.new is False:
            self.get_results(save=False)

    @staticmethod
    def col_sel(col_sel: Union[str, List[str]]):
        return pd.IndexSlice[:, col_sel]

    def run(self, traj_len, savedest=None):
        if self.new is True:
            for path_parts in self.idx_iter:
                for dir in ["orig", *self.new_dirs]:
                    clay, ion, aa = path_parts[1:]
                    inpath = path_parts[0]
                    if dir == "orig":
                        dirstr = ""
                    else:
                        dirstr = dir
                    idsl = pd.IndexSlice
                    checks = DirCheck(
                        name="b",
                        path=inpath / f"{clay}/{ion}/{aa}",
                        subfolder=dirstr,
                    )
                    if checks.path is not None:
                        checks.run(traj_len=traj_len)
                        for err_key, err_val in checks.error_dict.items():
                            if err_val is not False:
                                err_val = True
                            self.df.loc[
                                tuple([*path_parts, dir]), err_key
                            ] = err_val
                        if checks.path is not None:
                            for f_type in self.suffices:
                                file = getattr(checks.path, f_type.strip("."))
                                if file is not None:
                                    self.df.loc[
                                        tuple([*path_parts, dir]), f_type
                                    ] = str(file.resolve())
                        if checks.fixes is not None:
                            self.df.loc[
                                tuple([*path_parts, dir]), "fixes"
                            ] = checks.fixes
            self.get_results(save=savedest)

        else:
            self.get_results(save=savedest)
            logger.info(f"Checks already run.")

    def fix(
        self,
        odir: Optional[Union[Path, str]] = None,
        conc: float = 0.25,
        anion: str = "Cl",
        rm_tempfiles: bool = False,
        overwrite: bool = False,
        pH: int = 7,
        ff: Union[str, Path] = FF,
        datpath: Optional[Union[Path, str]] = None,
    ):
        # keep track of output directory
        _odir = odir
        # get fixes from check results
        fix_df = self.results.copy()
        fix_df["fixes"] = np.nan
        fix_df.pop("complete")
        fixes = ["extend", "run", "neutral", "setup", "missing", "error"]
        # summarise fix columns in new column and remove from df
        for fix in fixes:
            fix_df["fixes"].loc[fix_df[fix] == True] = fix
            fix_df.pop(fix)
        fix_df = fix_df.dropna(how="any")
        # get df to track fixing process
        fixed_df = pd.DataFrame(index=fix_df.index, columns=fix_df.columns)
        fixed_df["fixes"] = fix_df["fixes"]
        # get path for writing output / reading input
        if datpath is None:
            datpath = Path.cwd()
        else:
            datpath = Path(datpath)
        # load old fixing results
        if datpath.suffix == ".p":
            logger.info(f"Loading fixes: {datpath.resolve()!r}")
            with open(datpath.resolve(), "rb") as df_pkl:
                load_fix_df = pkl.load(df_pkl)
                fixed_df.update(load_fix_df)
                datpath = datpath.parent
        # apply fixes
        else:
            # print(fixed_df)
            if not datpath.is_dir():
                os.makedirs(datpath)
            for run in fix_df.iterrows():
                index, cols = run[0], run[1]
                slice = pd.IndexSlice[index]
                # set output directory
                if _odir is not None:
                    logger.info(f"Setting path to: {_odir}")
                    path = Path(_odir)
                # get output directory from df
                else:
                    logger.info(f'Choosing path from cols: {cols["path"]}')
                    path = Path(cols["path"])
                fix = cols["fixes"]
                # others = [f for f in ['setup', 'neutral'] if f != fix]
                # print(f"fix: {fix}")
                if fix == "extend":
                    gro = Path(cols["._gro"])
                    top = Path(cols[".top"])
                    fixed_df.loc[slice, ["._gro", ".top", "path", "fixes"]] = (
                        gro.resolve(),
                        top.resolve(),
                        path,
                        fix,
                    )
                    # print(fix_df)

                elif fix == "run":
                    gro = Path(cols["._gro"])
                    top = Path(cols[".top"])
                    fixed_df.loc[slice, ["._gro", ".top", "path", "fixes"]] = (
                        gro.resolve(),
                        top.resolve(),
                        path,
                        fix,
                    )
                    # print(fix_df)
                # create new setup
                elif fix in ["setup", "neutral"]:
                    # print(fix)
                    while "setup" in str(path.resolve()).split(
                        "/"
                    ) or "neutral" in str(path.resolve()).split("/"):
                        path = path.parent
                    if not path.is_dir():
                        logger.info(f"{path} does not exist.")
                        pass
                    else:
                        aa = index[2].strip("_7")
                        ion = index[1]
                        clay = index[0]
                        clay_type = CLAY_DICT[clay]
                        logger.info(f"{aa}, {clay}, {ion}")
                        relpath = Path(f"{clay}/{ion}/{aa}_7")
                        try:
                            setup_root = path.relative_to(
                                Path(f"{clay}/{ion}/{aa}_7")
                            )
                        except ValueError:
                            setup_root = path
                        odir = setup_root / relpath / "setup"
                        # print(odir, odir.exists())

                        # transform ion concentration 0.25 mol/L -> '250' mmol/L
                        i_conc = re.sub(
                            r"([0-9]*)[.]([0-9]*)", r"\2", f"{conc}"
                        )
                        if len(i_conc) > 3:
                            raise ValueError(
                                f"Expected max 3 digits for conc, found {len(i_conc)}"
                            )
                        while len(i_conc) < 3:
                            i_conc += "0"
                        # get setup coordinate file
                        crdin = CLAYS / f"{clay}/{i_conc}/{ion}/inp._gro"
                        # check and fix interlayer solvent/residue numbers
                        fix_gro_residues(crdin=crdin, crdout=crdin)
                        assert Path(
                            crdin
                        ).exists(), f"{Path(crdin)} does not exist"
                        # add aa molecules and neutralise charges

                        gro, top = add_aa(
                            aa=aa,
                            clay_type=clay_type,
                            conc=int(i_conc),
                            pion=ion,
                            nion=anion,
                            pH=pH,
                            ff_path=ff,
                            crdin=crdin,
                            odir=odir,
                            posfile="pos.dat",
                            rm_tempfiles=rm_tempfiles,
                            new_outpath=False,
                            overwrite=overwrite,
                            outname_suffix="setup",
                        )
                        # write to dataframe
                        fixed_df.loc[
                            slice, ["._gro", ".top", "path", "fixes"]
                        ] = (gro.resolve(), top.resolve(), path, fix)
        self.fixed_df = fixed_df.dropna(subset=["._gro", ".top"]).loc[
            pd.IndexSlice[:], ["._gro", ".top", "path", "fixes"]
        ]
        # save applied fixes
        for fix in self.fixed_df["fixes"].unique():
            if len(self.fixed_df["fixes"] == fix) != 0:
                self.write_fixed(
                    sel="paths", which=fix, outname=None, odir=datpath
                )
                self.write_fixed(
                    sel="files", which=fix, outname=None, odir=datpath
                )
        with open(datpath / "fixed_df.p", "wb") as outfile:
            pkl.dump(self.fixed_df, outfile)
        logger.info(f"Writing fixed_results to {outfile.name!r}")

    def save(self, savedest: Union[str, Path]):
        savename = self.__get_savename(
            savename=savedest, suffix="p", default_name="checks_df"
        )
        with open(savename, "wb") as outfile:
            pkl.dump(self.df, outfile)
            logger.info(f"Writing data to {str(savename)}")
        savename = change_suffix(savename, "csv")
        self.df.to_csv(savename)
        logger.info(f"Writing DataFrame to {str(savename)}")
        for res_column in [
            "complete",
            "extend",
            "run",
            "neutral",
            "setup",
            "missing",
            "error",
        ]:
            notna_results = self.results.dropna(how="all")
            if len(notna_results[res_column]) != 0:
                for sel in ["paths", "files"]:
                    self.write_results(
                        sel=sel, which=res_column, odir=savedest
                    )

    def add_path(self, df, drop_index=False):
        """Add new 'path' column to df from index ['root', 'paths'].
        (df['path'] = df['root'] / df['paths'] where df['root'] is Path)
        If drop_index is True, remove ['root', 'paths'] from index levels.
        This is done by choosing by giving priority to fixed directories:
        'setup' > 'neutral' > original directory
        and alternative root directory over original root directory.
        Returns simulation data file names and root directory."""
        idx_enum = {
            "orig": 1,
            "neutral": 2,
            "setup": 3,
            self.root_dir: 2,
            self.alt_dir: 1,
        }
        df = df.copy()
        if len(df) != 0:
            path = df.index.to_frame(index=False)
            path["paths"].where(path["paths"] != "orig", "", inplace=True)
            relpath = path.iloc[:, 1:-1]
            vec_path_join = np.vectorize(
                lambda path, root, new: str(
                    (Path(path) / f"{root}/{new}").resolve()
                )
            )
            relpath = np.apply_along_axis(lambda x: "/".join(x), 1, relpath)
            path = vec_path_join(path["root"].values, relpath, path["paths"])
            df["path"] = path
            if drop_index is True:
                df.reset_index(["root", "paths"], inplace=True)
                if df.index.has_duplicates:
                    duplicates_dict = {}
                    duplicates = df.index.duplicated(keep=False)
                    duplicates_df = df[duplicates]
                    duplicates_group = duplicates_df.groupby(
                        duplicates_df.index
                    )
                    for duplicate in duplicates_group:
                        for id_val in ["root", "paths"]:
                            idx, grouped_df = duplicate
                            nunique = grouped_df[id_val].nunique()
                            if nunique != 1:
                                key = grouped_df[
                                    grouped_df[id_val].apply(
                                        lambda x: idx_enum[str(x)]
                                    )
                                    == np.max(
                                        grouped_df[id_val].apply(
                                            lambda x: idx_enum[str(x)]
                                        )
                                    )
                                ]
                                duplicates_dict[idx] = key
                    for k, v in duplicates_dict.items():
                        df.loc[k, :] = v
                    df.drop_duplicates(inplace=True)
            df = df.loc[self.col_sel([*self.suffices, "path"])]
        else:
            df = pd.DataFrame(columns=[*self.suffices, "path"])
        return df

    def get_results(self, save: Union[Literal[False], str, Path] = False):
        complete_df = self.add_path(self.completed, drop_index=True)
        self.results.update(complete_df, overwrite=False)
        self.results.loc[complete_df.index, "complete"] = True
        self.results.loc[complete_df.index, :] = self.results.loc[
            complete_df.index, :
        ].fillna(value=False)

        fixes = self.results.columns.tolist()
        for fix in fixes[1:-2]:
            fix_df = self.notna[self.notna["fixes"] == fix]
            fix_df = self.add_path(fix_df, drop_index=True)
            self.results.update(fix_df, overwrite=False)
            self.results.loc[fix_df.index, fix] = self.results.loc[
                fix_df.index, fix
            ].fillna(value=True)

            self.results.loc[fix_df.index, :] = self.results.loc[
                fix_df.index, :
            ].fillna(value=False)
        err_idx = pd.IndexSlice[self.results.isnull().all(axis=1)]
        err_df = self.notna.copy()
        err_df = (
            err_df.reset_index(["root", "paths"])
            .loc[err_idx]
            .set_index(["root", "paths"], append=True)
        )
        err_df.index = err_df.index.reorder_levels(self.notna.index.names)
        err_df = self.add_path(err_df, drop_index=True)
        self.results.update(err_df, overwrite=False)
        self.results.loc[err_idx, "error"] = self.results.loc[
            err_idx, "error"
        ].fillna(value=True)
        self.results.loc[err_idx, :] = self.results.loc[err_idx, :].fillna(
            value=False
        )
        if save is not False:
            savename = self.__get_savename(save, suffix=".csv")
            self.results.to_csv(str(savename.resolve()))
            logger.info(
                f"Writing results DataFrame to {str(savename.resolve())}"
            )
            with open(str(change_suffix(savename, "p")), "wb") as pklfile:
                pkl.dump(self.results, pklfile)

    @staticmethod
    def __get_savename(
        savename: Optional[Union[str, Path]] = None,
        suffix: str = "dat",
        default_name: str = "results",
    ):
        try:
            savename = Path(savename)
            if savename.suffix == "":
                if not savename.is_dir():
                    os.makedirs(savename)
                savename = savename / f'{default_name}.{suffix.strip(".")}'
        except NotImplementedError:
            savename = Path(f'{default_name}.{suffix.strip(".")}')
        if savename.is_file():
            backup = savename.parent / f"{savename.name}.bk"
            shutil.move(savename, backup)
        if not savename.parent.is_dir():
            os.makedirs(savename.parent)
        return savename.resolve()

    def write_fixed(
        self,
        sel: Union[Literal["paths"], Literal["files"]],
        which: Union[Literal["run"], Literal["neutral"], Literal["setup"]],
        outname=None,
        odir=None,
    ):
        if self.fixed_df is None:
            raise ValueError(f"No fixes to write!")
        else:
            suffix_dict = {"paths": "csv", "files": "csv"}
            col_dict = {"paths": "path", "files": ["._gro", ".top", "path"]}
            write_dict = {
                "paths": cast(
                    Callable[[pd.Series], None],
                    lambda x, y: x.to_csv(y, index=False, header=False),
                ),
                "files": cast(
                    Callable[[pd.Series], None],
                    lambda x, y: x.to_csv(y, index=False, header=False),
                ),
            }
            if odir is None:
                odir = Path("results")
            else:
                odir = Path(odir)
            if outname is None:
                outname = f"fixed_{which}_{sel}"
            else:
                outname = Path(outname).stem
            outname = self.__get_savename(
                savename=odir, suffix=suffix_dict[sel], default_name=outname
            )
            write_slice = self.fixed_df.dropna()[
                self.fixed_df.dropna()["fixes"] == which
            ].copy()
            if len(write_slice) != 0:
                write_slice = write_slice.loc[
                    pd.IndexSlice[:], col_dict[sel]
                ].copy()
                write_dict[sel](write_slice, outname)
                logger.info(
                    f"Wrote {sel} to '{str(outname.parent)}/{outname.name}'"
                )

    def write_results(
        self,
        sel: Union[Literal["paths"], Literal["files"]],
        which: Union[
            Literal["complete"],
            Literal["run"],
            Literal["neutral"],
            Literal["setup"],
            Literal["missing"],
            Literal["error"],
        ],
        outname=None,
        odir=None,
    ):
        suffix_dict = {"paths": "csv", "files": "json"}
        col_dict = {"paths": "path", "files": [*self.suffices, "path"]}
        write_dict = {
            "paths": cast(
                Callable[[pd.Series], None],
                lambda x, y: x.to_csv(y, index=False, header=False),
            ),
            "files": cast(
                Callable[[pd.Series], None],
                lambda x, y: x.to_json(y, orient="index", indent=4),
            ),
        }
        if odir is None:
            odir = Path("results")
        else:
            odir = Path(odir)
        if outname is None:
            outname = f"{which}_{sel}"
        else:
            outname = Path(outname).stem
        outname = self.__get_savename(
            savename=odir, suffix=suffix_dict[sel], default_name=outname
        )
        if self.results[which].hasnans:
            self.get_results()
        write_slice = self.results.dropna()[self.results.dropna()[which]]
        if sel == "paths":
            logger.info(f"{which}: {len(write_slice)} entries")
        write_dict[sel](write_slice.loc[self.col_sel(col_dict[sel])], outname)
        logger.info(f"Wrote {sel} to '{str(outname.parent)}/{outname.name}'")

    # def write(self, odir=None):
    #     if odir is None:
    #         odir = Path.cwd() / "checks"
    #     elif type(odir) != Path:
    #         odir = Path(odir)
    #     if not odir.is_dir():
    #         os.makedirs(odir)
    #     complete_fname = odir / "completed.dat"
    #     completed = self.completed
    #     paths = completed["paths"]
    #     paths[paths == "orig"] = ""
    #     completed = completed.values
    #     with open(complete_fname, "w") as c_file:
    #         for row in completed:
    #             path = row[0] / "/".join(row[1:])
    #             c_file.write(str(path))
    #     return completed

    @property
    def notna(self):
        return self.df.dropna(how="all", subset="exists")

    @property
    def full_df(self):
        return self.df

    @cached_property
    def missing_paths(self):
        return self.df["exists"].isna().index.to_frame(index=False).values

    @cached_property
    def exists_idx(self):
        exists = self.df.loc[self.df["exists"].notna()].copy()
        return exists.index

    @cached_property
    def completed(self) -> pd.DataFrame:
        other = self.exists_idx
        exists_sel = self.df.loc[
            self.df.index.intersection(other), self.errors
        ]
        self.df.loc[exists_sel.index, "complete"] = np.all(
            ~(exists_sel.values.astype(bool)), keepdims=True, axis=1
        )
        return self.notna.loc[self.notna["complete"]]

    @staticmethod
    def write_complete_rsync_script(
        dest: Union[str, Path],
        complete_df_json: Union[str, Path],
        outname: Union[str, Path],
        odir: Union[str, Path] = "complete_runs",
        n_jobs: int = 8,
        header_file="../data/scripts/rsync_complete_header.sh",
    ):
        outname = Path(outname).with_suffix(".sh")
        if not outname.parent.is_dir():
            os.mkdir(outname.parent)
        odir = (Path(dest) / Path(odir).name).resolve()
        df = pd.read_json(complete_df_json, orient="index")
        n_jobs = int(n_jobs)
        length = len(df.index)
        split = int(length // n_jobs)
        split += np.ceil((length % n_jobs) / n_jobs).astype(int)
        with open(header_file, "r") as rsync_file:
            rsync_head = rsync_file.read()
        array = np.arange(1, n_jobs + 1).astype(str)
        array = ",".join(array.tolist())
        rsync_head = RunChecks.sub_parts(
            r"(--array)=ARRAY", rf"\1={array}", rsync_head
        )
        outscript = open(outname, "w+")
        outscript.write(rsync_head)
        outscript.write("\nif [[ ${SLURM_ARRAY_TASK_ID} -eq 1 ]]; then\n")
        ci = 2
        for c, v in df.iterrows():
            if ci % split == 0:
                n_task = int(ci // split) + 1
                logger.info(f"{ci}, {n_task}")
                outscript.write(
                    "\nelif [[ ${SLURM_ARRAY_TASK_ID} -eq"
                    + f" {n_task} ]]; then\n"
                )
            v.pop("path")
            new_path = re.findall("[^\s^'^,^(^)]+", c)
            new_path = [part.strip('"') for part in new_path]
            outname = "_".join(new_path)
            new_path = "/".join(new_path)
            new_path = Path(f"{new_path}")
            outdir = odir / new_path
            # print(f"mkdir -p {outdir}")
            outscript.write(f"\tmkdir -p {outdir}\n")
            for file in v:
                if file is not False:
                    file = Path(file).resolve()
                    # print(f"rsync -auvz {file} {outdir}/{outname}{file.suffix}")
                    outscript.write(
                        f"\trsync -auvz {file} {outdir}/{outname}{file.suffix}\n"
                    )
            ci += 1
        outscript.write("\nfi\n")
        outscript.close()
        logger.info(f"Wrote rsync script to {outscript.name!r}")

    def get_submit_script_str(
        self,
        template: Union[str, Path],
        dest: Union[str, Path],
        gro: Union[Path, str],
        top: Union[Path, str],
        clay: str,
        ion: str,
        aa: str,
        mdp: Union[Path, str],
    ):
        aa_codes = {
            "ala": "A",
            "arg": "R",
            "asn": "N",
            "asp": "D",
            "cys": "C",
            "gln": "Q",
            "glu": "E",
            "gly": "G",
            "his": "H",
            "ile": "I",
            "leu": "L",
            "lys": "K",
            "met": "M",
            "phe": "F",
            "pro": "P",
            "ser": "S",
            "thr": "T",
            "trp": "W",
            "tyr": "Y",
            "val": "V",
            "ctl": "O",
        }
        template = Path(template)
        with open(template, "r") as aa_script:
            script_str = aa_script.read()
        aa = aa.lower()
        claynum = re.search("\d", clay).group(0)
        dest = Path(dest)
        mdp = Path(mdp)
        script_str = self.sub_parts(
            r"(-J\s*) RUNNAME",
            rf"\1 {claynum}-{ion}-{aa_codes[aa]}",
            script_str,
        )
        script_str = self.sub_parts(r"(aa=)AA", rf"\1{aa}", script_str)
        script_str = self.sub_parts(r"(dir=)DIR", rf"\1{dest}", script_str)
        script_str = self.sub_parts(r"(mdp=)MDP", rf"\1{mdp.name}", script_str)
        script_str = self.sub_parts(
            r"(ingro=)INGRO", rf"\1{dest / gro.name}", script_str
        )
        script_str = self.sub_parts(
            r"(intop=)INTOP", rf"\1{dest / top.name}", script_str
        )
        return script_str

    @staticmethod
    def sub_parts(match: str, substr: str, input_str: str) -> str:
        """Use `re.sub` with MULTILINE and DOTALL flags.
        Substitute match by substr in input_str"""
        new_str = re.sub(
            match, substr, input_str, flags=re.MULTILINE | re.DOTALL
        )
        return new_str

    def write_fixed_rsync_script(
        self,
        remote: str,
        dest: Union[str, Path],
        df: pd.DataFrame,
        outname: Union[str, Path],
        root_dir: Union[Path, str, None] = None,
        fix: Union[str, None] = None,
    ):
        # print(df)
        if root_dir is None:
            root_dir = Path(self.root_dir).resolve()
        else:
            root_dir = Path(root_dir)
        root_dir = root_dir.resolve()
        if len(df) != 0:
            dest = Path(dest).resolve()
            outname = Path(outname)
            if outname.suffix == "":
                if not outname.is_dir():
                    os.mkdir(outname)
                outname = outname / outname.name
            outname = Path(outname).with_suffix(".sh")
            rsync_name = outname.with_stem(
                outname.stem + "_rsync"
            ).with_suffix(".sh")
            outscript = open(outname, "w+")
            rsync_script = open(rsync_name, "w+")
            for idx, cols in df.iterrows():
                clay, ion, aa = idx
                top, gro, path = (
                    Path(cols[".top"]),
                    Path(cols["._gro"]),
                    Path(cols["path"]),
                )
                if type(path) not in [Path, PosixPath]:
                    path = gro.parent.resolve()
                try:
                    searchpath = (
                        f"{dest.parent}/{gro.relative_to(root_dir).parent}"
                    )
                    rel_gro = gro.relative_to(root_dir)
                    rel_top = top.relative_to(root_dir)
                except ValueError:
                    searchpath = f"{dest.parent}/{gro.relative_to(self.root_dir).parent}"
                    rel_gro = gro.relative_to(self.root_dir)
                    rel_top = top.relative_to(self.root_dir)
                script_dir = Path(searchpath).resolve()
                while aa in str(script_dir.resolve()).split("/"):
                    script_dir = script_dir.parent.resolve()
                rel_dir = script_dir.relative_to(self.root_dir)
                if fix == "extend":
                    tpr = self.results[
                        self.results["._gro"].astype(str) == str(gro.resolve())
                    ][".tpr"]
                    tpr = Path(tpr.values[0]).resolve()
                    cpt = Path(tpr).with_suffix(".cpt")
                    out = execute_bash_command(
                        f"if ssh {remote} [[ -f {cpt.resolve()} ]]; then echo {cpt.resolve()}; fi;",
                        text=True,
                        capture_output=True,
                    )
                    # print('exists', out.stdout)
                    if out.stdout != "":
                        script_str = self.get_cont_script_str(
                            template="../data/scripts/aa_template_cont.sh",
                            dest=tpr.relative_to(script_dir).parent,
                            tpr=tpr,
                            clay=clay,
                            ion=ion,
                            aa=aa.strip("_7"),
                        )
                        script_ext = "cont"
                    else:
                        # print(f'does not exist')
                        fix = "run"
                if fix != "extend":
                    script_str = self.get_submit_script_str(
                        template="../data/scripts/aa_template.sh",
                        dest=searchpath,
                        gro=gro,
                        top=top,
                        clay=clay,
                        ion=ion,
                        aa=aa.strip("_7"),
                        mdp="../data/MDP/ads.mdp",
                    )
                    script_ext = fix
                if script_ext is None:
                    script_ext = "setup"
                script_name = f'{aa.strip("_7")}_{script_ext}.sh'
                out = execute_bash_command(
                    f"if ssh {remote} [[ ! -d {searchpath} ]]; then echo {searchpath}; fi;",
                    text=True,
                    capture_output=True,
                )
                # print("out", out.stdout)
                if out.stdout != "":
                    rsync_script.write(
                        f"rsync -rauvz --relative {root_dir}/./{rel_gro} {remote}:{dest.parent}/\n"
                    )
                    rsync_script.write(
                        f"rsync -rauvz --relative {root_dir}/./{rel_top} {remote}:{dest.parent}/\n"
                    )
                    outscript.write(f"cd {dest.parent}/{rel_dir}\n")
                    outscript.write(
                        f"sbatch {dest.parent}/{rel_dir}/{script_name}\n"
                    )
                    rsync_script.write(
                        f"rsync -auvz --relative {root_dir}/./{rel_dir}/{script_name} {remote}:{dest.parent}/\n"
                    )
                    with open(
                        f"{root_dir}/{rel_dir}/{script_name}", "w"
                    ) as submit_script:
                        submit_script.write(script_str)
                elif fix in ["run", "extend"]:
                    outscript.write(f"cd {dest.parent}/{rel_dir}\n")
                    outscript.write(
                        f"sbatch {dest.parent}/{rel_dir}/{script_name}\n"
                    )
                    rsync_script.write(
                        f"rsync -auvz --relative {root_dir}/./{rel_dir}/{script_name} {remote}:{dest.parent}/\n"
                    )
                    with open(
                        f"{root_dir}/{rel_dir}/{script_name}", "w"
                    ) as submit_script:
                        submit_script.write(script_str)
            rsync_script.write(f"rsync -auvz {outname} {remote}:{dest}/\n")
            outscript.close()
            rsync_script.close()
            logger.info(
                f"Wrote fixed rsync script to {rsync_name.resolve()!r}"
            )
            logger.info(
                f"Wrote fixed run submit_script to {outname.resolve()!r}"
            )
            logger.info(
                f"Submit script destination: '{remote}:{dest.resolve()}'"
            )

    def get_cont_script_str(
        self,
        template: Union[str, Path],
        dest: Union[str, Path],
        tpr: Union[Path, str],
        clay: str,
        ion: str,
        aa: str,
    ):
        aa_codes = {
            "ala": "A",
            "arg": "R",
            "asn": "N",
            "asp": "D",
            "cys": "C",
            "gln": "Q",
            "glu": "E",
            "gly": "G",
            "his": "H",
            "ile": "I",
            "leu": "L",
            "lys": "K",
            "met": "M",
            "phe": "F",
            "pro": "P",
            "ser": "S",
            "thr": "T",
            "trp": "W",
            "tyr": "Y",
            "val": "V",
            "ctl": "O",
        }
        template = Path(template)
        with open(template, "r") as aa_script:
            script_str = aa_script.read()
        aa = aa.lower()
        claynum = re.search("\d", clay).group(0)
        dest = Path(dest)
        tpr = Path(tpr)
        script_str = self.sub_parts(
            r"(-J\s*) RUNNAME",
            rf"\1 {claynum}-{ion}-{aa_codes[aa]}",
            script_str,
        )
        script_str = self.sub_parts(r"(aa=)AA", rf"\1{aa}", script_str)
        script_str = self.sub_parts(r"(dir=)DIR", rf"\1{dest}", script_str)
        script_str = self.sub_parts(
            r"(deffnm=[${]+dir[}]/)TPR", rf"\1{tpr.stem}", script_str
        )
        print(script_str)
        return script_str


if __name__ == "__main__":
    logger.info(f"Using MDAnalysis {MDAnalysis.__version__}")
    check_parser = ArgumentParser("checks", allow_abbrev=False)

    check_parser.add_argument(
        "-p",
        type=str,
        help="data directory path",
        metavar="root_dir",
        dest="root_dir",
        required=True,
    )
    check_parser.add_argument(
        "-a",
        type=str,
        help="alternative data directory path",
        metavar="alt_root",
        dest="alt_root",
        default=None,
    )
    check_parser.add_argument(
        "-t",
        type=int,
        help="expected trajectory length",
        metavar="traj_len",
        dest="traj_len",
    )
    check_parser.add_argument(
        "-w",
        type=str,
        help="write results to text files to directory",
        metavar="write_dir",
        dest="write_dir",
        default=None,
    )
    check_parser.add_argument(
        "-s",
        type=str,
        help="Subfolder name",
        metavar="subdir",
        dest="subdir",
        default=None,
    )
    check_parser.add_argument(
        "-largest_traj",
        action="store_const",
        const="largest",
        default="latest",
        help="Select largest trajectory file instead of most recent",
        dest="trr_sel",
    )
    check_parser.add_argument(
        "-fix",
        action="store_true",
        default=False,
        required=False,
        help="Apply fixes for failed run checks.",
        dest="fix",
    )

    check_parser.add_argument(
        "-update",
        action="store_true",
        default=False,
        required=False,
        help="Update even if a dataframe pickle exists for [-in].",
        dest="update",
    )

    check_parser.add_argument(
        "-in",
        default=None,
        required=False,
        help="Load dataframe from pickle.",
        dest="load",
    )
    check_parser.add_argument(
        "-out",
        default=None,
        required=False,
        help="Save dataframe to pickle.",
        dest="savedata",
    )
    #
    # parser.add_argument(
    #     "-p",
    #     type=str,
    #     help="data directory path",
    #     metavar="root_dir",
    #     dest="root_dir",
    #     required=True
    # )
    # parser.add_argument(
    #     "-a",
    #     type=str,
    #     help="alternative data directory path",
    #     metavar="alt_root",
    #     dest="alt_root",
    #     default=None
    # )
    # parser.add_argument(
    #     "-t",
    #     type=int,
    #     help="expected trajectory length",
    #     metavar="traj_len",
    #     dest="traj_len",
    # )
    # parser.add_argument(
    #     "-w",
    #     type=str,
    #     help="write results to text files to directory",
    #     metavar="write_dir",
    #     dest="write_dir",
    #     default=None,
    # )
    # parser.add_argument("-n", type=str, help="check name", metavar="name", dest="name")
    # parser.add_argument(
    #     "-s",
    #     type=str,
    #     help="Subfolder name",
    #     metavar="subdir",
    #     dest="subdir",
    #     default=None,
    # )
    # parser.add_argument(
    #     "-largest_traj",
    #     action="store_const",
    #     const="largest",
    #     default="latest",
    #     help="Select largest trajectory file instead of most recent",
    #     dest="trr_sel",
    # )
    # parser.add_argument(
    #     "-fix",
    #     action="store_true",
    #     default=False,
    #     required=False,
    #     help="Apply fixes for failed run checks.",
    #     dest="fix",
    # )

    # args = check_parser.parse_args(sys.argv[1:])
    # r = RunChecks(
    #     "/nobackup/projects/bddur15/1_NON_aa_ads/",
    #     data_df_file="/storage/runcheck_data/checks_df.csv",
    # )
    # if args.rsync is not False:
    #     rsync_dest = Path(args.rsync).resolve()
    rsync_dest = "/nobackup/projects/bddur15/1_NON_aa_ads/scripts"
    fix = False
    r = RunChecks(
        "/nobackup/projects/bddur15/1_NON_aa_ads/",
        "new_runs",
        data_df_file="/storage/ClayAnalysis/results/runcheck_results/checks_df.csv",
    )
    r.run(
        traj_len=35001,
        savedest="/storage/ClayAnalysis/results/runcheck_results_local/",
    )
    # r.save(savedest="/storage/runcheck_aa_local_up")
    if fix is True:
        # logger.info(f"Getting fixes:")
        # r.fix(
        #     odir="/storage/run_data_finished/",
        #     overwrite=False,
        #     rm_tempfiles=True,
        #     datpath="/storage/ClayAnalysis/results/runcheck_results_local/",
        # )
        if rsync_dest is not None:
            # neutral_idx = r.fixed_df[r.fixed_df["fixes"] == "neutral"].index
            # setup_idx = r.fixed_df[r.fixed_df["fixes"] == "setup"].index
            # rsync_idx = neutral_idx.union(setup_idx)
            # rsync_df = r.fixed_df.loc[rsync_idx]
            # logger.info(f"Writing new setup rsync.")
            # r.write_fixed_rsync_script(
            #     df=rsync_df,
            #     remote="bede",
            #     outname=f"/storage/fixed_setup_remaining_fixed_up",
            #     dest=rsync_dest,
            #     root_dir="/storage/run_data_finished/",
            #     fix="setup",
            # )
            # logger.info(f"New setup rsync written.")
            # rsync_df = r.fixed_df[r.fixed_df["fixes"] == "run"]
            # logger.info(f"Writing run rsync.")
            # r.write_fixed_rsync_script(
            #     df=rsync_df,
            #     remote="bede",
            #     outname=f"/storage/fixed_setup_remaining_runs_up",
            #     dest=rsync_dest,
            #     root_dir="/storage/run_data_finished/",
            #     fix="run",
            # )
            # logger.info(f"Run rsync written.")
            # rsync_df = r.fixed_df[r.fixed_df["fixes"] == "extend"]
            # r.write_fixed_rsync_script(
            #     df=rsync_df,
            #     remote="bede",
            #     outname=f"/storage/ClayAnalysis/results/runcheck_results_local/rsync",
            #     dest=rsync_dest,
            #     root_dir="/storage/run_data_finished/",
            #     fix="extend",
            # )
            ...
    logger.info(f"Writing rsync script")
    r.write_complete_rsync_script(
        odir="complete",
        outname="/storage/all_complete_rsync/all_complete_rsync.sh",
        dest=Path(rsync_dest).parent,
        complete_df_json="/storage/ClayAnalysis/results/runcheck_results/complete_files.json",
    )
