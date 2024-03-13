#!/usr/bin/env python3
""":mod:`claycomp` --- ClayCode builder module for processing clay compositions"""
from __future__ import annotations

import itertools
import logging
import math
import os
import pickle as pkl
import re
import shutil
import sys
import tempfile
from abc import ABC, abstractmethod
from functools import (
    cache,
    cached_property,
    partialmethod,
    singledispatchmethod,
    wraps,
)
from pathlib import PosixPath
from typing import (
    Callable,
    Dict,
    List,
    Literal,
    NoReturn,
    Optional,
    Tuple,
    Union,
)

import dask.array
import dask.bag
import dask.dataframe as dd
import numpy as np
import pandas as pd
import yaml
import zarr
from ClayCode.builder.consts import BUILDER_DATA
from ClayCode.builder.utils import get_checked_input, select_input_option
from ClayCode.core.classes import Dir, File, ITPFile, PathFactory, YAMLFile
from ClayCode.core.consts import LINE_LENGTH
from ClayCode.core.lib import get_ion_charges
from ClayCode.core.utils import (
    backup_files,
    get_arr_bytes,
    get_debugheader,
    get_subheader,
)
from ClayCode.data.consts import CLAYFF_AT_TYPES, UCS
from MDAnalysis.lib.mdamath import triclinic_vectors
from numba import njit, prange
from numpy._typing import NDArray
from tqdm.auto import tqdm
from zarr.errors import PathNotFoundError

__all__ = [
    "TargetClayComposition",
    "UCData",
    "UnitCell",
    "UCClayComposition",
    "MatchClayComposition",
    "Ions",
    "InterlayerIons",
    "BulkIons",
]

logger = logging.getLogger(__name__)

MAX_STORE_ARR_SIZE = 536870912


class UnitCell(ITPFile):
    """Class for handling single unit cell data"""

    @property
    def idx(self) -> str:
        """Unit cell index
        :return: unit cell index
        :rtype: str"""
        return self.stem[-3:]

    @property
    def clay_type(self) -> str:
        """Clay type
        :return: clay type
        :rtype: str"""
        return self.parent.name

    @property
    def uc_stem(self) -> str:
        """Unit cell stem
        :return: unit cell stem
        :rtype: str"""
        return self.stem[:-3]

    @cached_property
    def atom_df(self) -> pd.DataFrame:
        """Atom type dataframe
        :return: atom type dataframe
        :rtype: pd.DataFrame"""
        atoms = self.get_parameter("atoms")
        return atoms.df

    @cached_property
    def charge(self) -> float:
        """Unit cell charge
        :return: unit cell charge
        :rtype: float"""
        return self.atom_df["charge"].sum().round(6)


class UCData(Dir):
    """Class for handling unit cell data collection of a single clay type"""

    _suffices = [".gro", ".itp"]
    _sheet_grouper = pd.Grouper(level="sheet", sort=False)

    def __init__(
        self, path: Dir, uc_stem=None, ff=None, write: bool = True, reset=False
    ):
        """Initialize UCData object
        :param path: path to unit cell data directory
        :type path: Dir
        :param uc_stem: unit cell stem, defaults to None
        :type uc_stem: str, optional
        :param ff: force field, defaults to None
        :type ff: str, optional"""
        from ClayCode.core.classes import ForceField

        if uc_stem is None:
            self.uc_stem = self.uc_itp_filelist[0].stem[:-3]
        else:
            self.uc_stem: str = uc_stem
        logger.info(get_subheader("Getting unit cell data"))
        self.ff: ForceField = ForceField(ff)
        self.__uc_idxs: list = list(map(lambda x: str(x[-3:]), self.available))
        self.uc_idxs = self.__uc_idxs.copy()
        self.atomtypes: pd.DataFrame = self.ff["atomtypes"].df
        self._full_df = None
        self._df = None
        if self.__get_full_df(write=write, reset=reset) is False:
            self.__get_full_df(write, reset=True)
        self.__get_df()
        self.__atomic_charges = None
        self.group_id = None
        self.__gro_groups = None
        self.__itp_groups = None
        self.__dimensions = None
        self.__uc_groups = None
        self.get_uc_groups(write=write, reset=reset)
        logger.finfo(f"Found {self.n_groups} {self.name!r} unit cell groups:")
        self.get_uc_group_base_compositions()
        self.idx_sel = (
            self.df.index.get_level_values("sheet").unique().to_list()
        )

    @property
    def uc_gro_filelist(self):
        return self.gro_filelist.filter(f"{self.uc_stem}[0-9][0-9][0-9]")

    def check_ucs(self):
        uc_error_charges = {}
        for uc in sorted(self.uc_list):
            if uc.idx in self.uc_idxs:
                if not (np.isclose(uc.charge, np.rint(uc.charge))):
                    uc_error_charges[uc.idx] = uc.charge
        if uc_error_charges:
            error_str = "\n\t".join(
                [
                    f"{idx:5}: {uc_error_charges[idx]:2.5f}"
                    for idx in sorted(uc_error_charges.keys())
                ]
            )
            logger.error(
                f"Found unit cells with non-integer charge:\n\n\tuc-id: charge\n\t{error_str}"
            )
            self.__abort()

    @staticmethod
    def __abort():
        logger.finfo("Aborting model construction.", initial_linebreak=True)
        sys.exit(0)

    def get_uc_group_base_compositions(self):
        for n_group, group_ids in self.group_iter():
            group_id_str = f", {self.uc_stem}".join(group_ids)
            logger.finfo(
                kwd_str=f"\tGroup {n_group:2d}: ",
                message=f"{self.uc_stem}{group_id_str}\n",
            )
            self.__base_ucs[n_group] = self.__get_base_ucs(group_ids)

    def get_uc_groups(self, write=True, reset=False):
        if reset and (self.path / "uc_groups.pkl").is_file():
            os.remove(self.path / "uc_groups.pkl")
        box_dims = {}
        uc_groups = {}
        gro_groups = {}
        itp_groups = {}
        dimensions = {}
        bbox_height = {}
        dim_str = None
        get_group = lambda: box_dims[dim_str]
        n_group = 0
        self.__dimensions = {}
        self.__bbox_height = {}
        extract_id = lambda file: file.stem[-3:]
        if not (self.path / "uc_groups.pkl").is_file() or not write:
            for uc in sorted(self.uc_gro_filelist):
                uc_dimensions = uc.universe.dimensions
                bbox_height_new = np.ediff1d(uc.universe.atoms.bbox()[:, 2])[0]
                dim_str = "_".join(uc_dimensions.round(3).astype(str))
                if dim_str not in box_dims.keys():
                    bbox_height[n_group] = bbox_height_new
                    box_dims[dim_str] = n_group
                    uc_groups[n_group] = [extract_id(uc)]
                    gro_groups[n_group] = [uc]
                    itp_groups[n_group] = [uc.with_suffix(".itp")]
                    dimensions[n_group] = uc_dimensions
                    n_group += 1
                else:
                    if np.less(
                        bbox_height_new, bbox_height[box_dims[dim_str]]
                    ):
                        bbox_height[get_group()] = bbox_height_new
                    uc_groups[get_group()].append(extract_id(uc))
                    gro_groups[get_group()].append(uc)
                    itp_groups[get_group()].append(uc.with_suffix(".itp"))
                    dimensions[get_group()] = uc_dimensions
            self.__bbox_height = bbox_height
            self.__dimensions = dimensions
            self.__uc_groups = uc_groups
            self.__gro_groups = gro_groups
            self.__itp_groups = itp_groups
            if write:
                with open(self.path / "uc_groups.pkl", "wb") as f:
                    pkl.dump(
                        {
                            "bbox_height": bbox_height,
                            "dimensions": dimensions,
                            "uc_groups": uc_groups,
                            "gro_groups": gro_groups,
                            "itp_groups": itp_groups,
                        },
                        f,
                    )
        else:
            with open(self.path / "uc_groups.pkl", "rb") as f:
                uc_groups = pkl.load(f)
            self.__bbox_height = uc_groups["bbox_height"]
            self.__dimensions = uc_groups["dimensions"]
            self.__uc_groups = uc_groups["uc_groups"]
            self.__gro_groups = uc_groups["gro_groups"]
            self.__itp_groups = uc_groups["itp_groups"]
        self.__base_ucs = {}

    @property
    def uc_itp_filelist(self):
        return self.itp_filelist.filter(f"{self.uc_stem}[0-9][0-9][0-9]")

    @property
    def bbox_height(self):
        if self.group_id is not None:
            return self.__bbox_height[self.group_id]
        else:
            return self.__bbox_height

    @property
    def bbox_z_shift(self):
        if self.group_id is not None:
            uc_u = self.__gro_groups[self.group_id][0].universe
            not_oh = uc_u.select_atoms("not name [oOhH]*")
            uc_cog_z = not_oh.center_of_geometry()[2]
            shift = self.bbox_height / 2 - uc_cog_z
            return shift
        else:
            raise ValueError("No group selected!")

    def __get_z_shift(self, uc_id):
        base_uc = self.uc_base[uc_id]

    def __get_base_ucs(self, uc_ids: List[str]) -> Union[List[str], str]:
        uc_df = self._df.loc[:, uc_ids]
        base_sel = uc_df.loc[
            :, uc_df[uc_df != 0].count() == min(uc_df[uc_df != 0].count())
        ]
        base_sel = base_sel.columns[0]
        return base_sel

    @property
    def uc_base(self):
        if self.group_id is not None:
            return self.__base_ucs[self.group_id]
        else:
            logger.error("No unit cell group selected.")

    @property
    def base_ucs(self):
        return self.__base_ucs

    @property
    def group_gro_filelist(self):
        if self.group_id is not None:
            return self.__gro_groups[self.group_id]
        else:
            return self.__gro_groups

    @cached_property
    def _gro_df(self):
        gro_df = pd.DataFrame(
            index=np.arange(1, np.max(self.n_atoms) + 1, dtype=np.int_),
            columns=[*self.df.columns],
        ).stack(dropna=False)
        gro_df.index.names = ["atom-id", "uc-id"]
        gro_cols = pd.DataFrame(index=gro_df.index, columns=["x", "y", "z"])
        gro_df = gro_df.to_frame(name="at-type").join(gro_cols, how="outer")
        gro_df.reset_index(level="uc-id", inplace=True)
        gro_df["uc-id"] = gro_df["uc-id"].apply(
            lambda uc_id: f"1{self.uc_stem}{uc_id}"
        )
        gro_df.set_index("uc-id", inplace=True, append=True)
        gro_df.index = gro_df.index.reorder_levels(["uc-id", "atom-id"])
        gro_df.sort_index(level="uc-id", inplace=True, sort_remaining=True)
        for gro in self.uc_gro_filelist:
            if gro.df.index.unique().size > 1:
                raise ValueError(
                    "Multiple unit cells in single GRO file not supported!"
                )
            if f"{gro.df.index[0]}" != f"1{gro.stem}":
                raise ValueError(
                    f"Residue name must match file name:\nExpected 1{gro.stem}, found {gro.df.index[0]}"
                )
            #     raise ValueError(f"Residue name must match file name:\nExpected 1{gro.stem}, found {gro.df.index[0]}")
            n_atoms = self.n_atoms.filter(regex=gro.stem[-3:]).values[0]
            gro_df.update(gro.df.set_index("atom-id", append=True))
        if gro_df.isna().any().any():
            raise ValueError("Encountered invalid GRO file!")
        return gro_df

    @property
    def gro_df(self):
        regex = "|".join([f"{self.uc_stem}{uc_id}" for uc_id in self.uc_idxs])
        return (
            self._gro_df.reset_index("atom-id")
            .filter(regex=regex, axis=0)
            .set_index("atom-id", append=True)
        )

    @property
    def group_itp_filelist(self):
        if self.group_id is not None:
            return self.__itp_groups[self.group_id]
        else:
            return self.__itp_groups

    def group_iter(self) -> Tuple[int, List[str]]:
        for group_id in sorted(self.__uc_groups.keys()):
            yield group_id, self.__uc_groups[group_id]

    @property
    def uc_groups(self):
        return self.__uc_groups

    def select_group(self, group_id: int):
        self.group_id = group_id
        try:
            self.uc_idxs = self.uc_groups[group_id]
            uc_idx_str = f", {self.uc_stem}".join(self.uc_idxs)
            logger.finfo(
                kwd_str="Selected unit cells: ",
                message=f"{self.uc_stem}{uc_idx_str}\n",
                initial_linebreak=True,
            )
        except KeyError:
            raise KeyError(f"{group_id} is an invalid group id!")

    def reset_selection(self):
        self.group_id = None
        self.uc_idxs = self.__uc_idxs.copy()

    def select_ucs(self, uc_ids: List[str]):
        assert np.isin(uc_ids, self.uc_idxs).all(), "Invalid selection"
        self.uc_idxs = uc_ids

    @property
    def n_groups(self):
        return len(self.__uc_groups.keys())

    @property
    def full_df(self) -> pd.DataFrame:
        return self._full_df.sort_index(
            ascending=False, level="sheet", sort_remaining=True
        )

    @property
    def dimensions(self):
        if self.group_id is not None:
            return self.__dimensions[self.group_id]
        else:
            return self.__dimensions

    @property
    def df(self) -> pd.DataFrame:
        return (
            self._df.loc[:, self.uc_idxs]
            .sort_index(ascending=False, level="sheet", sort_remaining=True)
            .sort_index(axis=1)
        )

    def __get_full_df(self, write=True, reset=False):
        finished = False
        while finished is False:
            finished = True
            if reset is True and (self.path / "full_df.csv").is_file():
                os.remove(self.path / "full_df.csv")
                if (self.path / "uc_groups.pkl").is_file():
                    os.remove(self.path / "uc_groups.pkl")
            idx = self.atomtypes.iloc[:, 0].copy()
            idx = idx.reindex([*idx.index.values, len(idx)])
            idx.iloc[-1] = "itp_charge"
            cols = [*self.uc_idxs, "charge", "sheet"]
            self._full_df: pd.DataFrame = pd.DataFrame(
                index=idx, columns=cols, dtype=np.float64
            )
            self._full_df["charge"].update(
                self.atomtypes.set_index("at-type")["charge"]
            )
            self.__get_df_sheet_annotations()
            self._full_df["sheet"].fillna("X", inplace=True)
            self._full_df.fillna(0, inplace=True)
            # self._full_df = dd.from_pandas(self._full_df, chunksize=1000)
            if not (self.path / "full_df.csv").is_file() or not write:
                for uc in self.uc_list:
                    try:
                        atoms: UnitCell = uc.atom_df
                        self._full_df[f"{uc.idx}"].update(
                            atoms.value_counts("at-type")
                        )
                        self._full_df.loc[
                            "itp_charge", f"{uc.idx}"
                        ] = uc.charge
                        self._full_df[f"{uc.idx}"].fillna(0, inplace=True)
                    except AttributeError:
                        logger.finfo(f"Invalid unit cell {uc.name!r}")
                        for suffix in [".gro", ".itp"]:
                            try:
                                remove = select_input_option(
                                    instance_or_manual_setup=True,
                                    query=f"Remove invalid unit cell {uc.idx}? [y]es/[n]o (default y)\n",
                                    options=["y", "n", ""],
                                    result=None,
                                    result_map={
                                        "y": True,
                                        "n": False,
                                        "": True,
                                    },
                                )
                                if remove:
                                    os.remove(uc.with_suffix(suffix))
                            except FileNotFoundError:
                                pass
                self._full_df.set_index("sheet", append=True, inplace=True)
                self._full_df.sort_index(
                    inplace=True, level=1, sort_remaining=True
                )
                self._full_df.index = self._full_df.index.reorder_levels(
                    ["sheet", "at-type"]
                )
                self.check_ucs()
                if write:
                    self._full_df.to_csv(self.path / "full_df.csv")
            else:
                self._full_df = pd.read_csv(
                    self.path / "full_df.csv", index_col=[0, 1]
                )
                try:
                    if not np.equal(
                        self._full_df.loc["itp_charge", "itp_charge"],
                        np.rint(self._full_df.loc["itp_charge", "itp_charge"]),
                    ).all():
                        finished = False
                except KeyError:
                    finished = False
            if (
                sorted(self._full_df.columns)[:-1] != sorted(self.uc_idxs)
            ) or finished is False:
                os.remove(self.path / "full_df.csv")
                if (self.path / "uc_groups.pkl").is_file():
                    os.remove(self.path / "uc_groups.pkl")
                finished = False

    def __get_df_sheet_annotations(self):
        old_index = self._full_df.index
        regex_dict = {
            "T": r"[a-z]+t",
            "O": r"[a-z]*[a-gi-z][o2]",
            "C": "charge",
        }
        index_extension_list = []
        for key in regex_dict.keys():
            for element in old_index:
                match = re.fullmatch(regex_dict[key], element)
                if match is not None:
                    index_extension_list.append((key, match.group(0)))
        new_index = pd.MultiIndex.from_tuples(index_extension_list)
        new_index = new_index.to_frame().set_index(1)
        self._full_df["sheet"].update(new_index[0])
        self._full_df.loc["itp_charge", "sheet"] = "itp_charge"

    def __get_df(self):
        # self._df = dd.from_pandas(self._full_df, chunksize=1000)
        # self._df.reset_index('at-type', inplace=True)
        # self._df.filter(regex=r"^(?![X].*)", axis=0, inplace=True)
        self._df = (
            self.full_df.copy()
            .reset_index("at-type")
            .filter(regex=r"^(?![X].*)", axis=0)
        )
        self._df = (
            self._df.reset_index()
            .set_index(["sheet", "at-type"])
            .sort_index(axis=1)
        )
        self._df.drop("itp_charge", inplace=True)

    @cached_property
    def uc_list(self) -> List[UnitCell]:
        uc_list = [UnitCell(itp) for itp in self.uc_itp_filelist]
        return uc_list

    @cached_property
    def occupancies(self) -> Dict[str, int]:
        return self._get_occupancies(self.df)

    @cached_property
    def ff_charge(self) -> pd.Series:
        charge = self.full_df.apply(
            lambda x: x * self.full_df["charge"], raw=True
        )
        total_charge = (
            charge.loc[:, self.uc_idxs].sum().astype(np.float32).round(4)
        )
        total_charge.name = "charge"
        if total_charge.any() > 1.001:
            self._full_df = self._full_df[
                total_charge[
                    np.less(
                        1.002,
                        total_charge,
                    )
                ]
            ]
        return total_charge

    @cached_property
    def tot_charge(self) -> pd.Series:
        total_charge = self.full_df.loc[("itp_charge", "itp_charge")].filter(
            regex="[0-9]+"
        )
        return total_charge.sort_index()

    @cached_property
    def n_atoms(self):
        return (
            self.full_df.drop("itp_charge")
            .filter(regex="[0-9]+")
            .astype(int)
            .sum(axis=0)
        )

    @cached_property
    def uc_composition(self) -> pd.DataFrame:
        return self.full_df.reindex(self.atomtypes, fill_value=0).filter(
            regex=r"^(?![oOhH].*)", axis=0
        )

    @cached_property
    def oxidation_numbers(self) -> Dict[str, int]:
        """Get dictionary of T and O element oxidation numbers for 0 charge sheet"""
        ox_dict = self._get_oxidation_numbers(
            self.occupancies, self.df, self.tot_charge
        )[1]
        return dict(
            zip(ox_dict.keys(), list(map(lambda x: int(x), ox_dict.values())))
        )

    @cached_property
    def idxs(self) -> np.array:
        return self.full_df.columns

    def check(self) -> None:
        if not self.is_dir():
            raise FileNotFoundError(f"{self.name} is not a directory.")

    @cached_property
    def available(self) -> List[ITPFile]:
        return self.uc_itp_filelist._extract_parts(
            part="stem", pre_reset=False
        )

    # def __str__(self):
    #     return f"{self.__class__.__name__}({self.name!r})"
    #
    # def __repr__(self):
    #     return f"{self.__class__.__name__}({self.name!r})"

    @property
    def path(self) -> Dir:
        return self

    @staticmethod
    def _get_occupancies(df) -> Dict[str, int]:
        try:
            occ = (
                df.sort_index(level="sheet", sort_remaining=True)
                .groupby("sheet")
                .sum()
                .aggregate("drop_duplicates", axis="columns")
            )
        except ValueError:
            occ = (
                df.sort_index(level="sheet", sort_remaining=True)
                .groupby("sheet")
                .sum()
            )
        occ = occ[occ != 0].dropna()
        idx = (
            df.sort_index(level="sheet", sort_remaining=True)
            .index.get_level_values("sheet")
            .drop_duplicates()
        )
        occ_dict = dict(zip(idx.values, np.ravel(occ.values)))
        return occ_dict

    @property
    def atomic_charges(self) -> pd.Series:
        """Get dictionary of unit cell atomic charges
        :return: dictionary of unit cell atomic charges
        :rtype: Dict[str , int]"""
        return self._get_oxidation_numbers(
            self.occupancies, self.df, self.tot_charge
        )[0]

    @singledispatchmethod
    @staticmethod
    def _get_oxidation_numbers(
        occupancies: Dict[Union[Literal["T"], Literal["O"]], int],
        df: Union[pd.DataFrame, pd.Series],
        tot_charge: Optional = None,
        sum_dict: bool = True,
    ) -> Union[Dict[Union[Literal["T"], Literal["O"]], int], pd.DataFrame]:
        """Get oxidation numbers from unit cell composition and occupancies.
        :param occupancies: dictionary of unit cell sheet occupancies
        :type occupancies: Dict[Union[Literal['T'], Literal['O']], int]
        :param df: unit cell composition dataframe
        :type df: Union[pd.DataFrame, pd.Series]
        :param tot_charge: total charge of unit cell composition, defaults to None
        :type tot_charge: Optional[int]"""
        raise NotImplementedError

    @_get_oxidation_numbers.register(dict)
    @staticmethod
    def _(
        occupancies: Dict[str, int],
        df: Union[pd.DataFrame, pd.Series],
        tot_charge: Optional = None,
        sum_dict: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, int], pd.DataFrame]:
        """Get oxidation numbers from unit cell composition and occupancies"""
        ox_dict = UCData._get_ox_dict()
        # df = df.loc[['T','O']]
        ox_df: pd.DataFrame = df.copy()
        # ox_df = ox_df.drop('itp_charge')
        try:
            ox_df = ox_df.loc[~(ox_df == 0).all(1), :]
        except ValueError:
            ox_df = ox_df.loc[~(ox_df == 0)]

        at_types: pd.DataFrame = ox_df.index.get_level_values(
            "at-type"
        ).to_frame()
        at_types.index = ox_df.index
        # for idx_entry in ("O", "fe_tot"), ('itp_charge', 'itp_charge')]:
        try:
            at_types.drop(("O", "fe_tot"), inplace=True)

        except KeyError:
            pass
        at_types = at_types.applymap(lambda x: ox_dict[x])
        if tot_charge is not None:
            _ox_df = ox_df.loc[:, tot_charge.abs() == tot_charge.abs().min()]
            # ox_df.drop_duplicates(inplace=True)
            if not _ox_df.empty:
                # _ox_df = ox_df.max(axis=1)#.groupby('sheet', group_keys=False).apply(lambda x: x.sort_values(ascending=False).head(1))
                ox_df = _ox_df
        # ox_df = ox_df.max(axis=1)
        # ox_df.name = 'charge'
        if sum_dict is True:  # ox_df.groupby("sheet").count().max() > 1 and
            # if tot_charge is not None:
            #     _ox_df = ox_df.loc[:, tot_charge.abs() == tot_charge.abs().min()]
            #     # ox_df.drop_duplicates(inplace=True)
            #     if not _ox_df.empty:
            #         ox_df = _ox_df
            col = (
                ox_df.groupby("sheet")
                .apply(lambda x: x.max())
                .sum()
                .sort_values(ascending=False)
                .head(1)
                .index[0]
            )
            ox_df = ox_df.loc[:, col]
            ox_df.sort_index(level="sheet", sort_remaining=True, inplace=True)
            ox_df.name = "charge"
            ox_df = ox_df.groupby("sheet", group_keys=False, sort=True).apply(
                lambda x: x[x == x.max()]
            )
            # _ox_df = ox_df.groupby('sheet', group_keys=False, sort=True).apply(lambda x: x.sort_values(ascending=False).first())
            # ox_df[:] = ox_df.groupby('sheet', group_keys=True).sum()

        if type(ox_df) == pd.DataFrame:
            ox: pd.DataFrame = ox_df.apply(lambda x: x * at_types["at-type"])
        else:
            ox: pd.DataFrame = at_types.apply(lambda x: x * ox_df)
        if sum_dict is True:
            ox: np.ndarray = (
                ox.groupby("sheet").sum().aggregate("unique", axis="columns")
            )
            idx: pd.Index = (
                ox_df.sort_index().index.get_level_values("sheet").unique()
            )
            ox_dict: dict = dict(zip(idx, ox))
            ox_dict: dict = dict(
                map(
                    lambda x: (x, ox_dict[x] / occupancies[x]),
                    occupancies.keys(),
                )
            )
        else:
            ox_dict: pd.DataFrame = ox.groupby("sheet").apply(
                lambda x: x / occupancies[x.name]
            )
        return at_types, ox_dict

    @_get_oxidation_numbers.register(float)
    @_get_oxidation_numbers.register(int)
    @staticmethod
    def _(
        occupancies: Union[float, int],
        df: Union[pd.DataFrame, pd.Series],
        tot_charge: Optional = None,
        sum_dict: bool = True,
    ) -> Tuple[pd.DataFrame, NDArray]:
        """Get oxidation numbers from unit cell composition and occupancies"""
        ox_dict = UCData._get_ox_dict()
        ox_df: pd.DataFrame = df.copy()
        ox_df.sort_index(level="sheet", sort_remaining=True, inplace=True)
        try:
            ox_df = ox_df.loc[~(ox_df == 0).all(1), :]
        except ValueError:
            ox_df = ox_df.loc[~(ox_df == 0)]
        if tot_charge is not None:
            ox_df = ox_df.loc[:, tot_charge == 0]
        at_types: pd.DataFrame = ox_df.index.get_level_values(
            "at-type"
        ).to_frame()
        at_types.index = ox_df.index
        try:
            at_types.drop(("fe_tot"), inplace=True)
        except KeyError:
            pass
        at_types = at_types.applymap(lambda x: ox_dict[x])
        if type(ox_df) == pd.DataFrame:
            ox: pd.DataFrame = ox_df.apply(lambda x: x * at_types["at-type"])
        else:
            ox: pd.DataFrame = at_types.apply(lambda x: x * ox_df)
        if sum_dict is True:
            ox: np.ndarray = ox.sum().aggregate("unique")
            ox_val = ox / occupancies
        else:
            ox_val = ox.apply(lambda x: x / occupancies)
        return at_types, ox_val[0]

    @staticmethod
    @cache
    def _get_ox_dict() -> Dict[Union[Literal["T"], Literal["O"]], int]:
        """Get dictionary of oxidation numbers for tetrahedral (T) and octahedral (O) elements.
        :return: dictionary of oxidation numbers for tetrahedral (T) and octahedral (O) elements
        :rtype: Dict[str, int]"""
        logger.debug(f"UCS: {UCS.resolve()}")
        with open(UCS / "clay_charges.yaml", "r") as file:
            ox_dict: dict = yaml.safe_load(file)
        return ox_dict


class TargetClayComposition:
    """Class for processing target clay composition."""

    sheet_grouper = pd.Grouper(level="sheet", sort=False)

    def __init__(
        self,
        name,
        csv_file: Union[str, File],
        uc_data: UCData,
        occ_tol,
        sel_priority,
        charge_priority,
        occ_correction_threshold=0.0,
        zero_threshold=0.001,
        manual_setup=True,
    ):
        """Class for processing target clay composition.
        :param name: name of target clay composition
        :type name: str
        :param csv_file: csv file containing target clay composition
        :type csv_file: Union[str, File]
        :param uc_data: unit cell data
        :type uc_data: UCData
        :param occ_tol: occupancy tolerance
        :type occ_tol: float
        :param sel_priority: selection priority
        :type sel_priority: Union[Literal['charge'], Literal['occupancy']]
        :param charge_priority: charge priority
        :type charge_priority: Union[Literal['charge'], Literal['occupancy']]
        :param occ_correction_threshold: occupancy correction threshold
        :type occ_correction_threshold: float
        :param zero_threshold: zero threshold
        :type zero_threshold: float
        :param manual_setup: manual setup
        :type manual_setup: bool
        :return: None"""
        self.name: str = name
        self.manual_setup = manual_setup
        self.zero_threshold = zero_threshold
        self.match_file: File = File(csv_file, check=True)
        self.uc_data: UCData = uc_data
        self.uc_df: pd.DataFrame = self.uc_data.df
        self.occ_tol = occ_tol
        self.sel_priority = sel_priority
        self.charge_priority = charge_priority
        self.occ_corr_threshold = float(occ_correction_threshold)
        self.idx_sel = self.uc_data.idx_sel
        logger.info(get_subheader("Processing target clay composition"))
        match_df, match_idx = self.__get_match_df(csv_file)
        ion_idx = tuple(
            ("I", ion_name) for ion_name in get_ion_charges().keys()
        )
        ion_idx = pd.MultiIndex.from_tuples(
            ion_idx, names=self.uc_df.index.names
        )
        match_idx = (match_idx).union(ion_idx)
        match_idx = pd.MultiIndex.from_tuples(
            match_idx, names=self.uc_df.index.names
        )
        self._df = match_df.reindex(match_idx)
        self._df = self._df.loc[:, self.name]
        self.correct_charged_occupancies(priority=self.sel_priority)
        self.correct_uncharged_occupancies()
        self._df = self._df.reindex(match_idx)
        clay_df = self.clay_df.copy()
        self._df.loc[clay_df.index] = clay_df.where(
            clay_df > zero_threshold, np.NaN
        )

        self.split_fe_occupancies()
        self.__ion_df = None
        self.set_ion_numbers()

    def __str__(self):
        return f"{self.__class__.__name__}({self.name!r})"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name!r})"

    def update_charges(
        self, df: Union[pd.DataFrame, pd.Series]
    ) -> Union[pd.DataFrame, pd.Series]:
        """Update charges in matching composition `df` with charges from `self.df`.
        :param df: matching composition `pd.DataFrame` or `pd.Series`
        :type df: Union[pd.DataFrame, pd.Series]
        :return: matching composition `pd.DataFrame` or `pd.Series` with updated charges
        :rtype: Union[pd.DataFrame, pd.Series]"""
        charges = pd.Series(self.get_charges(df), name=self.name)
        charges = charges.reindex_like(df.xs("C"))
        df.loc["C"].update(charges)
        return df

    @cached_property
    def clayff_at_types(self):
        """Get dictionary of clayff atom types and corresponding elements.
        :return: dictionary of clayff atom types and corresponding elements
        :rtype: Dict[str, str]"""
        clayff_at_types = YAMLFile(CLAYFF_AT_TYPES)
        return clayff_at_types.data

    @cached_property
    def clayff_elements(self) -> Dict[str, str]:
        """Get dictionary of elements and corresponding clayff atom types.
        :return: dictionary of elements and corresponding clayff atom types
        :rtype: Dict[str, str]"""
        reverse_at_types = {}
        for k1, v1 in self.clayff_at_types.items():
            reverse_at_types[k1] = {v: k for (k, v) in v1.items()}
        return reverse_at_types

    def clayff_to_element(self, at_type: str) -> str:
        """Get element from clayff atom type
        :param at_type: clayff atom type
        :type at_type: str
        :return: element
        :rtype: str"""
        return self.clayff_elements[at_type]

    def __get_match_df(self, csv_file: Union[str, File]) -> pd.DataFrame:
        """Get `pd.Dataframe` that matches a target clay composition specified in a csv file.
        :param csv_file: csv file containing target clay composition
        :return: `pd.Dataframe` containing target clay composition
        :rtype: pd.DataFrame"""
        match_df = self.__read_match_df(csv_file)
        match_df = pd.DataFrame(match_df[self.name], columns=[self.name])
        match_df = self.update_charges(match_df)
        match_df.dropna(inplace=True, how="all")
        match_df["new-at-type"] = match_df.index.get_level_values(
            "at-type"
        ).values
        match_df.loc[self.idx_sel, "new-at-type"] = (
            match_df.loc[self.idx_sel, "new-at-type"]
            .groupby("sheet", group_keys=True)
            .apply(lambda x: self.clayff_at_types[x.name])
        )
        nan_at_types = match_df["new-at-type"][match_df["new-at-type"].isna()]
        if nan_at_types.tolist():
            nan_at_type_str = ", ".join(
                list(
                    map(
                        lambda idx: f"\t{idx[0]:^5} - {idx[1]:^7}: {match_df.loc[idx, self.name]:>9.2f}",
                        nan_at_types.index.tolist(),
                    )
                )
            )
            logger.finfo(
                f"Removing invalid atom types:\n\n\tsheet - at-type: occupancy\n{nan_at_type_str}\n",
                replace_whitespace=False,
                expand_tabs=False,
            )
            ion_charges = get_ion_charges()
            for group, series in nan_at_types.groupby("sheet"):
                at_charges = (
                    self.uc_data.atomic_charges.loc[
                        (
                            group,
                            match_df.loc[group, "new-at-type"].dropna().values,
                        ),
                        "at-type",
                    ]
                    - self.uc_data.oxidation_numbers[group]
                )
                match_occs = (
                    match_df.xs(group, level="sheet", drop_level=False)
                    .dropna(subset="new-at-type")
                    .set_index("new-at-type", append=True)
                    .reset_index("at-type", drop=True)
                    .rename_axis(index={"new-at-type": "at-type"})[self.name]
                )
                new_charge: pd.Series = at_charges.multiply(match_occs)
                if not new_charge.hasnans:
                    target_charge = match_df.loc[("C", group), self.name]
                    if np.isclose(new_charge.sum(), target_charge):
                        charge_adjust = "n"
                    else:
                        charge_adjust = "m"
                else:
                    charge_adjust = None
                charge_adjust = select_input_option(
                    self,
                    query=f"Adjust {group} charge? [y]es/[n]o/[m]anual (default y, current value: {match_df.loc[('C', group), self.name]:.2f})\n",
                    options=["y", "n", "", "m"],
                    result=charge_adjust,
                    result_map={"y": "m", "n": "n", "": "m", "m": "y"},
                )
                if charge_adjust == "y":
                    new_charge: float = get_checked_input(
                        query="Enter new charge value:",
                        result_type=float,
                        check_value=r"[0-9.]+",
                    )
                    match_df = self.set_charge(
                        df=match_df, key=group, new_charge=new_charge
                    )
                    try:
                        match_df = self.reduce_charge(
                            df=match_df, key="tot", remove_charge=new_charge
                        )
                    except ValueError:
                        pass
                    logger.finfo(
                        f"Setting new {group} charge to {new_charge:.2f}\n",
                        initial_linebreak=True,
                    )
                elif charge_adjust == "m":
                    logger.finfo(f"Will try to guess new charge.")
                    for idx, value in match_df.loc[
                        series.index, self.name
                    ].items():
                        sheet, at_type = idx
                        try:
                            at_charge = ion_charges[at_type]
                        except KeyError:
                            try:
                                at_charge = re.search(
                                    r"[A-Z][a-z]*?([0-9]+?)", at_type
                                ).group(1)
                            except AttributeError:
                                at_charge = get_checked_input(
                                    query=(
                                        f"Could not guess {at_type} charge.\n"
                                        f"Enter {at_type} charge value: "
                                    ),
                                    check_value=r"[0-9]+?",
                                    result_type=int,
                                )
                                logger.finfo(
                                    f"Assuming {at_type} charge of {at_charge}."
                                )
                        remove_charge = (
                            at_charge - self.uc_data.oxidation_numbers[group]
                        ) * value
                        match_df = self.reduce_charge(
                            df=match_df, key=group, remove_charge=remove_charge
                        )
                        try:
                            match_df = self.reduce_charge(
                                df=match_df,
                                key="tot",
                                remove_charge=remove_charge,
                            )
                        except ValueError:
                            pass
        match_df.dropna(inplace=True, subset="new-at-type")
        match_df = match_df.set_index("new-at-type", append=True).reset_index(
            "at-type", drop=True
        )
        match_df.index = match_df.index.rename(self.uc_df.index.names)
        match_idx = self.__get_match_idx(match_df)
        return match_df, match_idx

    @property
    def df(self):
        df = self._df.dropna().sort_index(
            ascending=False, level="sheet", sort_remaining=True
        )
        # try:
        #     df = df.drop('itp_charge')
        # except KeyError:
        #     pass
        return df

    def reduce_charge(
        self,
        df: pd.DataFrame,
        key: Union[Literal["T"], Literal["O"], Literal["tot"]],
        remove_charge: float,
    ):
        df.loc[("C", key), self.name] -= remove_charge
        return df

    def set_charge(
        self,
        df: pd.DataFrame,
        key: Union[Literal["T"], Literal["O"], Literal["tot"]],
        new_charge: float,
    ):
        df.loc[("C", key), self.name] = new_charge
        return df

    @property
    def clay_df(self):
        return self._df.loc[["T", "O"]].sort_index(
            ascending=False, level="sheet", sort_remaining=True
        )

    @property
    def ion_df(self) -> pd.DataFrame:
        return self.__ion_df.sort_index()

    def split_fe_occupancies(self):
        o_charge = self.get_o_charge()
        try:
            missing_o = self._df.xs(
                ("O", "fe_tot"), level=("sheet", "at-type")
            ).values[0]
            not_fe_occ = self.occupancies["O"] - missing_o
            _, ox_state_not_fe = UCData._get_oxidation_numbers(
                not_fe_occ, self._df.xs("O", level="sheet")
            )
            chg_fe = o_charge + (
                (self.uc_data.oxidation_numbers["O"] - ox_state_not_fe)
                * not_fe_occ
            )
            if np.isclose(chg_fe, 0):
                self._df.loc[:, "feo"] = missing_o
            else:
                self._df.loc[:, "fe2"] = np.abs(chg_fe)
                self._df.loc[:, "feo"] = missing_o - self._df.at[("O", "fe2")]
            self._df.drop(("O", "fe_tot"), inplace=True)
            # self._df.where(self._df.loc['O'] != 0, np.NaN, inplace=True)
            # charge_delta = dict(map(lambda x: (x, self._uc_data.oxidation_numbers[x] - ox_states[x]), ox_states.keys()))
            logger.finfo(
                f"Splitting total iron content ({missing_o:.4f}) to match charge.\n",
                initial_linebreak=True,
            )
            assert np.isclose(
                self.occupancies["O"], self.uc_data.occupancies["O"]
            ), f'Found incorrect occupancies of {self.occupancies["O"]}, expected {self.uc_data.occupancies["O"]}'
        except KeyError:
            pass
        sheet_df = self._df.copy()
        self.print_df_composition(
            sheet_df.loc[["T", "O"], :].dropna(), fill=""
        )
        accept = select_input_option(
            self,
            query="\nAccept clay composition? [y]es/[e]xit (Default y)\n",
            options=["y", "e", ""],
            result=None,
            result_map={"y": "y", "e": "n", "": "y"},
        )
        if accept == "n":
            self.__abort()
        return sheet_df

    @staticmethod
    def __abort():
        logger.finfo(
            "Composition not accepted. Aborting model construction.",
            initial_linebreak=True,
        )
        sys.exit(0)

    def _get_charges(
        self, key: Literal[Union[Literal["tot"], Literal["T"], Literal["O"]]]
    ) -> float:
        return self.get_charges(self._df.xs("C"))[key]

    get_total_charge = partialmethod(_get_charges, key="tot")
    get_t_charge = partialmethod(_get_charges, key="T")
    get_o_charge = partialmethod(_get_charges, key="O")

    def get_charges(self, charge_df: pd.Series):
        if isinstance(charge_df, pd.DataFrame):
            charge_df = charge_df[self.name]
        try:
            charge_df = charge_df.xs("C")
        except KeyError:
            pass
        sheet_charges = charge_df.copy()
        charge_dict = sheet_charges.to_dict()
        # for sheet, charge in charge_df.items():
        #     if not charge_df.loc[sheet].isnan():
        #         charge_dict[sheet] = charge
        try:
            tot_charge = sheet_charges.pop("tot")
        except KeyError:
            tot_charge = np.NAN
        if charge_df.hasnans or np.isnan(tot_charge):
            # missing charge specifications
            if np.isnan(tot_charge) and len(sheet_charges) != len(
                sheet_charges.dropna()
            ):
                assert not charge_df.drop(
                    "tot"
                ).hasnans, "No charges specified"
            # only T/O charges given
            elif np.isnan(tot_charge):
                charge_dict["tot"] = sheet_charges.sum()
            elif len(sheet_charges) != len(sheet_charges.dropna()):
                sheet_charges[sheet_charges.isna] = (
                    tot_charge - sheet_charges.dropna()
                )
                charge_dict.update(sheet_charges.to_dict())
        else:
            charge_diff = sheet_charges.sum() - tot_charge
            if np.isclose(charge_diff, 0.0000):
                pass
            elif charge_diff < self.occ_corr_threshold:
                charge_dict["tot"] = sheet_charges.sum()
            else:
                logger.error(
                    f"Sheet charges ({sheet_charges.sum()}) "
                    f"do not sum to specified total charge ({tot_charge})\n"
                    "Please specify valid charges!"
                )
                self.__abort()
        return charge_dict

    @property
    def oxidation_states(self):
        return UCData._get_oxidation_numbers(
            self.occupancies, self._df.loc[["T", "O"]], sum_dict=False
        )[1]

    @property
    def atom_types(self):
        return UCData._get_oxidation_numbers(
            self.occupancies, self._df.loc[["T", "O"]], sum_dict=False
        )[0]

    @property
    def occupancies(self):
        return UCData._get_occupancies(self._df.loc[["T", "O"]].dropna())

    def sheet_df_decorator(f):
        @wraps(f)
        def wrapper(self, **kwargs):  # , idx_sel=["T", "O"], **kwargs):
            sheet_df: pd.Series = self._df.loc[self.idx_sel, :].copy()
            sheet_df = sheet_df.loc[sheet_df != 0].dropna()
            return f(self, sheet_df, **kwargs)  # , idx_sel, **kwargs)

        return wrapper

    @sheet_df_decorator
    def __get_sheet_df(
        self, sheet_df: pd.DataFrame
    ) -> pd.DataFrame:  # , idx_sel=["T", "O"]):
        charge_substitutions = self.atom_types
        charge_substitutions["charge"] = charge_substitutions.groupby(
            "sheet"
        ).apply(lambda x: x - self.uc_data.oxidation_numbers[x.name])
        charge_substitutions["at-type"] = np.NAN
        charge_substitutions["at-type"].update(sheet_df)
        if self.__has_at_type("fe_tot"):
            if self.__has_at_type("feo") or self.__has_at_type("fe2"):
                logger.error(
                    f"Invalid octahedral iron values!\n"
                    f"Specify either total Fe OR Fe3 and Fe2!"
                )
                self.__abort()
            fe_row_no_charge = pd.DataFrame(
                [[self.df.loc["O", "fe_tot"], 0]],
                index=pd.MultiIndex.from_tuples([("O", "fe_tot")]),
                columns=charge_substitutions.columns,
            )
            fe_row_charged = pd.DataFrame(
                [[self.df.loc["O", "fe_tot"], -1]],
                index=pd.MultiIndex.from_tuples([("O", "fe_tot")]),
                columns=charge_substitutions.columns,
            )
            charge_substitutions = pd.concat(
                [charge_substitutions, fe_row_no_charge, fe_row_charged]
            )
        return charge_substitutions

    def __has_at_type(self, at_type):
        return at_type in self.df.index.get_level_values("at-type")

    @property
    def non_charged_sheet_df(self):  # , idx_sel=["T", "O"]):
        sheet_df = self.__get_sheet_df()  # (self.idx_sel)#=idx_sel)
        return sheet_df[sheet_df["charge"] == 0].dropna()

    @property
    def charged_sheet_df(self):  # , idx_sel=["T", "O"]):
        sheet_df = self.__get_sheet_df()  # (self.idx_sel)#=idx_sel)
        return sheet_df[sheet_df["charge"] != 0].dropna()

    @sheet_df_decorator
    def correct_charged_occupancies(
        self,
        sheet_df: pd.DataFrame,
        # idx_sel=["T", "O"],
        priority: Optional[
            Union[Literal["charges"], Literal["occupancies"]]
        ] = None,
    ) -> Union[pd.DataFrame, pd.Series, NoReturn]:
        priorities = ["charges", "occupancies"]
        priorities.remove(priority)
        non_priority_sel = next(iter(priorities))
        priority_sel = priority
        charged_occ = self.charged_sheet_df  # (self.idx_sel)#=idx_sel)

        non_charged_occ = self.non_charged_sheet_df  # (self.idx_sel)#=idx_sel)

        occ_intersection = charged_occ.index.intersection(
            non_charged_occ.index
        )
        charged_dict = {"charged": charged_occ, "non-charged": non_charged_occ}
        if occ_intersection.tolist():
            for at_type_duplicate in occ_intersection:
                charged_size = (
                    charged_dict["charged"].loc[at_type_duplicate[0]].size
                )
                # non_charged_size = (
                #     charged_dict["non-charged"].loc[at_type_duplicate[0]].size
                # )
                # size_dict = {
                #     "charged": charged_size,
                #     "non-charged": non_charged_size,
                # }
                if charged_size in [1, 2]:
                    if self.sel_priority == "charge":
                        charge_diff = charged_dict["charged"].loc[
                            at_type_duplicate
                        ]
                        target_charge = self._get_charges(at_type_duplicate[0])
                        if charged_size == 1:
                            charged_dict["charged"].loc[at_type_duplicate] = (
                                charge_diff * target_charge
                            )
                        elif charged_size == 2:
                            missing_charge = (
                                target_charge
                                - charged_dict["charged"]
                                .drop(at_type_duplicate)
                                .product()
                            )
                            charged_size["charged"].loc[at_type_duplicate] = (
                                missing_charge / charge_diff["charge"]
                            )
                    else:
                        charged_dict["charged"].loc[
                            at_type_duplicate, "at-type"
                        ] = 0.0
                else:
                    charged_dict["charged"].loc[
                        at_type_duplicate, "at-type"
                    ] = 0.0
        charged_non_zero = (
            charged_dict["charged"]
            .where(charged_dict["charged"]["at-type"] != 0.0, np.NaN)
            .dropna()
        )
        if len(charged_non_zero) > 1:
            charged_dict["charged"] = charged_non_zero
        charged_occ_sum = (
            charged_dict["charged"].product(axis=1).groupby("sheet").sum()
        )
        charged_occ_check = charged_occ_sum.groupby("sheet").apply(
            lambda x: self._get_charges(x.name) - x
        )
        non_zero_charge_occ = np.greater(charged_occ_check, 0.0)
        if (non_zero_charge_occ).any():
            logger.finfo(
                f"Sheet charges ({charged_occ_sum.sum():2.4f}) "
                f"do not sum to specified total charge ({self.get_total_charge():2.4f})\n"
            )
            target_charge = {}
            subst_charges = charged_occ_sum[non_zero_charge_occ]
            deviation = subst_charges.to_dict()
            for dev_occ in deviation.keys():
                target_charge[dev_occ] = self._get_charges(key=dev_occ)
            deviation = self.__deviation_string(deviation=deviation)
            target_charge = self.__deviation_string(target_charge)
            logger.error(
                f"Charge from specified substitution occupancy ({deviation})\n"
                f"exceeds specified charge values ({target_charge})!\n"
            )
            if (charged_occ_check > self.occ_tol).any():
                priority_sel = select_input_option(
                    self,
                    "\nAdjust [c]harges or [o]ccupancies? (exit with e)\n",
                    options=["c", "o", "e"],
                    result_map={"c": "occupancies", "o": "charges", "e": "e"},
                )
            else:
                logger.finfo(
                    f"Adjusting {non_priority_sel} to match specified "
                    f"{priority_sel}."
                )
            if priority_sel == "occupancies":
                new_charges = self._df.loc["C"].copy()
                new_charges.update(subst_charges)
                new_charges["tot"] = new_charges.loc[self.idx_sel].sum()
                self._df.loc["C"].update(new_charges)
                new_charge_str = self.__deviation_string(
                    new_charges, join_str="\n\t"
                )
                logger.finfo(f"New charges: \n\t{new_charge_str}")
            elif priority_sel == "charges":
                new_charged_occs = (
                    charged_dict["charged"]["at-type"]
                    .groupby("sheet")
                    .apply(lambda x: x - (charged_occ_check[x.name] / x.size))
                )
                self.print_df_composition(
                    old_composition=sheet_df.loc[new_charged_occs.index],
                    sheet_df=new_charged_occs,
                )
                sheet_df.update(new_charged_occs)
                self._df.loc[sheet_df.index] = sheet_df.loc[sheet_df.index]
            else:
                self.__abort()

    @property
    def occ_correction_df(self):
        occ_correction_df = self.non_charged_sheet_df["at-type"]
        occ_correction_df = occ_correction_df.where(
            occ_correction_df > self.occ_corr_threshold, np.NaN
        ).dropna()
        return occ_correction_df

    @sheet_df_decorator
    def correct_uncharged_occupancies(
        self, sheet_df: pd.DataFrame
    ) -> None:  # , idx_sel=["T", "O"]):
        correct_uc_occ: pd.Series = pd.Series(self.uc_data.occupancies)
        input_uc_occ: pd.Series = pd.Series(self.occupancies)
        check_occ: pd.Series = input_uc_occ - correct_uc_occ
        check_occ.dropna(inplace=True)
        logger.finfo("Getting sheet occupancies:", initial_linebreak=True)
        if not np.isclose(check_occ.values, 0.0).all():
            for sheet, occ in check_occ.iteritems():
                logger.finfo(
                    f"\tFound {sheet!r} sheet occupancies of {input_uc_occ[sheet]:.4f}/{correct_uc_occ[sheet]:.4f} ({occ:+.4f})"
                )
            logger.finfo(
                "Adjusting values to match expected occupancies:",
                initial_linebreak=True,
            )
            corrected_occupancies = self.occ_correction_df.groupby(
                "sheet"
            ).apply(lambda x: x - check_occ.at[x.name] / x.count())
            sheet_df.update(corrected_occupancies)
            accept = None
            old_composition = self.df.copy()
            self._df.update(sheet_df)
            while accept != "y":
                self._df.update(sheet_df)
                new_occ = pd.Series(self.occupancies)
                new_check_df: pd.Series = new_occ - correct_uc_occ
                new_check_df.dropna(inplace=True)
                new_zeros = np.isclose(new_check_df, 0.0)
                if not (new_zeros).all():
                    deviation = new_check_df[~new_zeros].to_dict()
                    deviation = self.__deviation_string(deviation=deviation)
                    logger.warning(
                        "\nNew occupancies do not match expected values!"
                    )
                    if (new_check_df.abs() >= self.occ_tol).any():
                        logger.error(
                            f"New occupancy deviation ({deviation}) exceeds limit of {self.occ_tol:.2f}!"
                        )
                        print("\nPlease enter enter valid occupancies!")
                        exit_dict = {"y": "n", "n": "e"}
                        accept = select_input_option(
                            self,
                            query="Continue? [y]es/[n]o\n",
                            options={"y", "n"},
                            result=None,
                            result_map=exit_dict,
                        )
                    else:
                        logger.finfo(
                            f"Deviation ({deviation}) is within acceptance limit of {self.occ_tol:.2f}.\n\n"
                            "Correcting selected composition."
                        )
                        sheet_df.update(
                            self.occ_correction_df.groupby("sheet").apply(
                                lambda x: x
                                - new_check_df.at[x.name] / x.count()
                            )
                        )

                    # continue
                if accept is None:
                    common_idx = old_composition.index.intersection(
                        sheet_df.index
                    )
                    self.print_df_composition(
                        sheet_df=sheet_df,  # .loc[corrected_occupancies.index],
                        old_composition=old_composition,  # '.loc[
                        # corrected_occupancies.index
                        # ],
                        fill="\t",
                        print_all=True,
                    )
                accept = select_input_option(
                    self,
                    query="\nAccept new composition? [y]es/[n]o/[e]xit (default y)\n",
                    options=["y", "e", "n", ""],
                    result=accept,
                    result_map={"y": "y", "e": "e", "n": "n", "": "y"},
                )
                if accept == "n":
                    accept = None
                    sheet_df = self.__set_new_value(
                        old_composition,
                        self.idx_sel,
                        check_occ,
                        zero_threshold=self.zero_threshold,
                    )
                if accept == "e":
                    self.__abort()
            self.correct_charged_occupancies(
                # idx_sel=idx_sel,
                priority=self.sel_priority
            )
        else:
            logger.finfo(
                "\tFound correct occupancies:\n" "\t\tsheet: occupancy"
            )
            for sheet, occ in check_occ.iteritems():
                logger.finfo(f"\t\t{sheet!r:^5}: {input_uc_occ[sheet]:>9.0f}")

    @staticmethod
    def __set_new_value(
        old_composition: pd.Series,
        idx_sel: pd.MultiIndex,
        check_occ: Dict[str, int],
        zero_threshold: Union[float, int],
    ) -> pd.Series:
        sheet_df: pd.Series = old_composition.copy()
        sheet_df = sheet_df.loc[idx_sel, :]
        sheet_df = sheet_df.loc[sheet_df > zero_threshold]
        for k, v in check_occ.items():
            if v != 0:
                for atom, occ in sheet_df.loc[k, :].iteritems():
                    accept_input = None
                    while accept_input is None:
                        new_occ = input(
                            f"Enter new value for {k!r} - {atom!r}: ({occ:2.4f}) -> "
                        )
                        if new_occ != "":
                            try:
                                new_occ = float(new_occ)
                            except TypeError:
                                logger.finfo(
                                    f"Invalid value {new_occ} selected!"
                                )
                            if new_occ >= 0.0:
                                sheet_df.loc[k, atom] = float(new_occ)
                                print(f"\tSelecting {float(new_occ):2.4f}")
                                accept_input = True
                            else:
                                logger.finfo(
                                    f"Invalid value {new_occ:2.4f} selected!"
                                )
                        else:
                            print(f"\tKeeping {float(occ):2.4f}")
                            accept_input = True
        return sheet_df

    @staticmethod
    def __deviation_string(deviation: dict, join_str=", ", sep=": ") -> str:
        return f"{join_str}".join(
            [f"{k!r}{sep}{v:2.3f}" for k, v in deviation.items()]
        )

    @staticmethod
    def print_df_composition(
        sheet_df, old_composition=None, fill="", print_all=False
    ):
        if old_composition is None:
            logger.finfo(
                f"{fill}Will use the following target composition:",
                initial_linebreak=True,
            )
            logger.finfo(f"{fill}\tsheet - atom type : occupancies")
        else:
            logger.finfo(
                f"{fill}old occupancies -> new occupancies per unit cell:"
            )
            logger.finfo(
                f"{fill}\tsheet - atom type : {'occupancies':<16} (difference)"
            )
        for idx, occ in sheet_df.iteritems():
            sheet, atom = idx
            try:
                old_val = old_composition[idx]
                if not np.isclose(occ, old_val):
                    logger.finfo(
                        f"{fill}\t{sheet!r:5} - {atom!r:^10}: {old_val:2.4f} -> {occ:2.4f} ({occ - old_val:+2.4f})"
                    )
                elif print_all:
                    raise TypeError
            except TypeError:
                logger.finfo(f"{fill}\t{sheet!r:5} - {atom!r:^10}: {occ:2.4f}")

    correct_t_occupancies = partialmethod(
        correct_uncharged_occupancies, idx_sel=["T"]
    )
    correct_o_occupancies = partialmethod(
        correct_uncharged_occupancies, idx_sel=["O"]
    )

    def _write(
        self,
        outpath: Union[Dir, File, PosixPath],
        fmt: Optional[str] = None,
        backup=False,
    ):
        if type(outpath) == str:
            outpath = PathFactory(outpath)
        if fmt is None:
            if outpath.suffix == "":
                raise ValueError("No file format specified")
            else:
                fmt = outpath.suffix
        fmt = f'.{fmt.lstrip(".")}'
        if outpath.suffix == "":
            outpath = outpath / f"{self.name}_exp_df{fmt}"
        tmpfile = tempfile.NamedTemporaryFile(suffix=fmt, prefix=outpath.stem)
        if fmt == ".csv":
            print_df = self.df.copy()
            print_clay_df = print_df.loc[["T", "O"]]
            self.df.to_csv(tmpfile.name, float_format="%.4f")
        elif fmt == ".p":
            with open(tmpfile.name, "wb") as file:
                pkl.dump(self.df, file)
        else:
            raise ValueError(
                f"Invalid format specification {fmt!r}\n"
                f'Expected {".csv"!r} or {".p"!r}'
            )
        if not outpath.parent.is_dir():
            os.makedirs(outpath.parent)
        logger.finfo(
            f"Writing new target clay composition to {outpath.name!r}",
            initial_linebreak=True,
        )
        if backup:
            backup_files(new_filename=outpath, old_filename=tmpfile.name)
        else:
            shutil.copy(tmpfile.name, outpath)
        assert outpath.is_file(), f"Error writing file {tmpfile.name!r}"

    write_csv = partialmethod(_write, fmt=".csv")
    write_pkl = partialmethod(_write, fmt=".csv")

    def set_ion_numbers(self):
        ion_probs: pd.Series = self._df.loc["I"].copy()
        total_charge = self.get_total_charge()
        ion_df = self.get_ion_numbers(ion_probs, total_charge)
        self.__ion_df = ion_df.sort_index(kind="mergesort")

    @staticmethod
    def get_ion_numbers(ion_probs, total_charge):
        """Read ion types and ratios into pd DataFrame."""
        ion_df = ion_probs.to_frame(name="probs")
        ion_charge_dict = get_ion_charges()
        ion_df["charges"] = ion_df.index.to_series().apply(
            lambda x: ion_charge_dict[x]
        )
        ion_sum = ion_probs.sum()
        if not np.isclose(ion_sum, 1.00):
            ion_probs = ion_probs / ion_sum
        assert np.isclose(
            ion_probs.sum(), 1.00
        ), "Ion species probabilities need to sum to 1"
        ion_df.dropna(inplace=True, subset="probs")
        ion_df.where(
            np.sign(ion_df) != np.sign(total_charge),
            level="charge",
            inplace=True,
        )
        ion_df.where(
            ~np.isclose(ion_df.values, 0.0), level="probs", inplace=True
        )
        return ion_df

    def __get_match_idx(self, match_df):
        match_idx = (self.uc_df.index.to_flat_index()).union(
            match_df.dropna().index.to_flat_index()
        )
        return match_idx

    def __read_match_df(self, csv_file):
        """Read match DataFrame from csv file."""

        match_df: pd.DataFrame = pd.read_csv(csv_file)
        match_df["sheet"].ffill(inplace=True)
        match_cols = match_df.columns.values
        match_cols[:2] = self.uc_df.index.names
        match_df.columns = match_cols
        match_df.set_index(self.uc_df.index.names, inplace=True)
        match_df.sort_index(level="sheet", sort_remaining=True, inplace=True)
        return match_df


class ClayComposition(ABC):
    """Base class for clay composition."""

    def __init__(self, sheet_n_ucs: int, uc_data: UCData, name: str):
        """Initialize clay composition class.
        :param sheet_n_ucs: number of unit cells in each sheet
        :type sheet_n_ucs: int
        :param uc_data: unit cell data
        :type uc_data: UCData
        :param name: name of unit cell type
        :type name: str"""
        self._uc_data = uc_data
        self.sheet_n_ucs = int(sheet_n_ucs)
        self.name = name

    def __str__(self):
        return f"{self.__class__.__name__}({self.name!r})"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name!r})"

    @cached_property
    def match_charge(self) -> pd.Series:
        """Return charge of matched unit cell composition."""
        uc_data = self._uc_data.atomic_charges.copy()
        match_data = self.match_composition.copy().to_frame(name="occ")
        for sheet, charges in uc_data.groupby("sheet", group_keys=True):
            uc_data.loc[sheet] = (
                charges - self._uc_data.oxidation_numbers[sheet]
            )
        match_data["charge"] = uc_data
        match_data = match_data.aggregate("prod", axis=1)
        match_charge = match_data.groupby("sheet").sum()
        match_charge["tot"] = match_charge.sum()
        match_charge.name = "match_charge"
        return match_charge

    @staticmethod
    def print_uc_composition(
        uc_list: Union[NDArray, List[str]],
        uc_df: pd.DataFrame,
        fill: str = "",
        charges: Optional[pd.Series] = None,
    ):
        """Print unit cell groups and their occupancies.
        :param uc_list: dict of unit cell groups
        :type uc_list: Dict[int, List[str]]
        :param uc_df: unit cell DataFrame
        :type uc_df: pd.DataFrame
        :param fill: fill string for printing
        :type fill: str
        :return: None
        """
        uc_array = np.array(uc_list)
        if not uc_array.dtype == str:
            uc_array = np.array([f"{int(x):03d}" for x in uc_array])
        uc_group_df = uc_df.loc[:, uc_array]
        if charges is not None:
            charge_header = " total charge"
        else:
            charge_header = ""
        logger.finfo(
            f'{fill}{"UC index":<10}{"atom type":<15} UC occupancies{charge_header}'
        )
        for uc_id, uc_series in (
            uc_group_df[uc_group_df != 0].dropna(how="all").fillna(0).items()
        ):
            if charges is not None:
                charge_str = f"{fill}{'':28} {float(charges[uc_id]):<+2.0f}"
            else:
                charge_str = ""
            logger.finfo(
                f"{fill}{uc_id:10}{charge_str}", initial_linebreak=False
            )
            for idx, values in uc_series.sort_index(
                ascending=False, sort_remaining=True
            ).items():
                values = int(values)
                sheet, atype = idx
                composition_string = f"{int(values):>3}"
                logger.finfo(
                    f"{fill}{'':10}{sheet!r:<3} - {atype!r:^7} {composition_string}"
                )
        logger.finfo("\n")

    def get_atype_diff_dask(atype_weights, target_values) -> Tuple[float, int]:
        """Return difference between target and actual composition.
        :param atype_weights: actual composition
        :type atype_weights: dask.array
        :param target_values: target composition
        :type target_values: NDArray[float]
        """
        diff_array = atype_weights - target_values
        diff_array = dask.array.linalg.norm(diff_array, axis=1)
        dist = diff_array.min().compute()
        match_idx = dask.array.argwhere(
            dask.array.equal(diff_array, dist)
        ).compute()[0]
        return dist, match_idx

    @cached_property
    @abstractmethod
    def _unique_uc_match_dict(self):
        """Return dict of unique unit cell compositions and their weights."""
        pass

    @cached_property
    @abstractmethod
    def _uc_match_dict(self):
        """Return dict of unit cell compositions and their weights."""
        pass

    def uc_dimensions(self) -> NDArray[float]:
        """Return unit cell dimensions.
        :return: unit cell dimensions
        :rtype: NDArray[float]
        """
        return self._uc_data.dimensions

    @staticmethod
    def print_groups(
        group_dict: Dict[int, List[str]], uc_df: pd.DataFrame, fill: str = ""
    ):
        """Print unit cell groups and their occupancies.
        :param group_dict: dict of unit cell groups
        :type group_dict: Dict[int, List[str]]
        :param uc_df: unit cell DataFrame
        :type uc_df: pd.DataFrame
        :param fill: fill string for printing
        :type fill: str
        :return: None
        """
        for group_id, group_ucs in group_dict.items():
            uc_group_df = uc_df.loc[:, group_ucs]
            logger.finfo(
                f"{fill}Group {group_id}:", initial_linebreak=True
            )  # {uc_group_df.columns}')
            logger.finfo(f'{fill}{"":10}\tUC occupancies')
            try:
                uc_list = "  ".join(
                    list(map(lambda v: f"{v:>3}", uc_group_df.columns))
                )
                logger.finfo(f'{fill}{"UC index":<10}\t{uc_list}')
            except AttributeError:
                logger.finfo(f'{fill}{"UC index":<10}\t{uc_group_df.name}')
            logger.finfo(f'{fill}{"atom type":<10}')
            try:
                for idx, values in uc_group_df.sort_index(
                    ascending=False, sort_remaining=True
                ).iterrows():
                    values = values.astype(int)
                    if not (values == 0).all():
                        sheet, atype = idx
                        composition_string = "  ".join(
                            list(map(lambda v: f"{v:>3}", values))
                        )
                        logger.finfo(
                            f"{fill}{sheet!r:<3} - {atype!r:^4}\t{composition_string}"
                        )
            except AttributeError:
                for idx, values in uc_group_df.sort_index(
                    ascending=False, sort_remaining=True
                ).items():
                    values = int(values)
                    if values != 0:
                        sheet, atype = idx
                        composition_string = f"{int(values):>3}"
                        logger.finfo(
                            f"{fill}{sheet!r:<3} - {atype!r:^4}\t{composition_string}"
                        )

    def _abort(self, status):
        sys.exit(status)

    @cached_property
    def uc_weights(self):
        match_uc_index = pd.Index(
            [
                f"{uc_id:03d}"
                for uc_id in self._uc_match_dict["uc_ids"].astype(int)
            ]
        )
        return pd.Series(
            self._uc_match_dict["uc_weights"].astype(int),
            index=match_uc_index,
            name="uc_weights",
        )

    @cached_property
    def uc_ids(self) -> NDArray[str]:
        return self._uc_match_dict["uc_ids"].values

    def uc_charges(self) -> NDArray[int]:
        return self._uc_match_dict["charges"].values

    @property
    @abstractmethod
    def match_composition(self):
        pass

    @property
    def target_df(self):
        pass


class UCClayComposition(ClayComposition):
    def __init__(
        self,
        uc_index_ratios: Dict[int, float],
        uc_data: UCData,
        sheet_n_ucs,
        name,
    ):
        super().__init__(uc_data=uc_data, sheet_n_ucs=sheet_n_ucs, name=name)
        logger.info(get_subheader("Manually specified unit cells"))
        self.__uc_index_ratios = uc_index_ratios
        self.get_uc_group()

    @cached_property
    def _uc_match_dict(self) -> dict:
        return self._unique_uc_match_dict

    def get_atype_weights(self, uc_df, uc_ids, uc_weights):
        atype_combinations = uc_df[[*list(uc_ids)]].astype(float).T.values
        combinations_iter = np.nditer(
            [np.atleast_2d(uc_weights), atype_combinations, None],
            flags=["external_loop"],
            op_axes=[[1, -1, 0], [0, 1, -1], None],
        )
        for cell, element, weight in combinations_iter:
            weight[...] = cell * element
        atype_weights = combinations_iter.operands[2] / self.sheet_n_ucs
        atype_weights = np.add.reduce(atype_weights, axis=0)
        return atype_weights

    def __get_id_weights(self):
        uc_ids = []
        uc_weights = []
        target_weights = []
        for entry_id, (uc_id, ratio) in enumerate(
            self.__uc_index_ratios.items()
        ):
            uc_ids.append(uc_id)
            n_ucs = ratio * self.sheet_n_ucs
            target_weights.append(n_ucs)
            uc_weights.append(np.rint(n_ucs).astype(int))
            if (
                entry_id == len(self.__uc_index_ratios) - 1
                and np.sum(uc_weights, dtype=int) != self.sheet_n_ucs
            ):
                missing_ucs = self.sheet_n_ucs - np.sum(uc_weights).astype(int)
                uc_weights[-np.abs(missing_ucs) :] += np.sign(missing_ucs) * 1
        return np.array(uc_ids), np.array(uc_weights), np.array(target_weights)

    @cached_property
    def _unique_uc_match_dict(self) -> dict:
        uc_ids, uc_weights, target_uc_weights = self.__get_id_weights()
        match_dict = dict(
            map(
                lambda k: (k, np.NaN),
                [
                    "n_ucs",
                    "uc_ids",
                    "uc_weights",
                    "composition",
                    "dist",
                    "charges",
                ],
            )
        )
        uc_df = self._uc_data.df.copy()
        uc_df.columns = uc_df.columns.astype(int)
        atype_weights = self.get_atype_weights(
            uc_df=uc_df, uc_ids=uc_ids, uc_weights=uc_weights
        )
        target_atype_weights = self.get_atype_weights(
            uc_df=uc_df, uc_ids=uc_ids, uc_weights=target_uc_weights
        )
        diff_array = np.subtract(
            atype_weights.T, np.squeeze(target_atype_weights)
        )
        diff_array = np.linalg.norm(diff_array.astype(np.float128), axis=1)
        dist = np.amin(diff_array)
        match_dict["n_ucs"] = len(self.__uc_index_ratios)
        match_dict["uc_ids"] = np.array(uc_ids)
        match_dict["uc_weights"] = np.array(uc_weights)
        match_dict["composition"] = np.squeeze(np.round(atype_weights.T, 4))
        match_dict["target_composition"] = np.squeeze(
            np.round(target_atype_weights.T, 4)
        )
        match_dict["dist"] = np.round(dist, 4)
        # charges =
        logger.finfo(
            f"Best match found with {match_dict['n_ucs']} unique unit cells "
            + f"(total occupancy deviation {match_dict['dist'].round(3):+.3f})",  # \n\nIndices:",
            initial_linebreak=True,
        )
        logger.finfo(f"Selected unit cell compositions:")
        charges = self._uc_data.tot_charge[
            [f"{uc_id:03d}" for uc_id in match_dict["uc_ids"]]
        ]
        self.print_uc_composition(
            match_dict["uc_ids"], self.uc_df, fill="\t", charges=charges
        )
        match_dict["charges"] = np.round(charges.values.flatten(), 1).astype(
            np.int_
        )
        return match_dict

    @property
    def uc_df(self):
        return self._uc_data.df

    @property
    def target_df(self):
        target_comp = self._uc_match_dict["target_composition"]
        target_df = pd.Series(
            target_comp, index=self.uc_df.index, name=self.name
        )
        return target_df.sort_index(
            ascending=False, level="sheet", sort_remaining=True
        ).dropna()

    @property
    def match_composition(self):
        match_comp = self._uc_match_dict["composition"]
        match_df = pd.Series(
            match_comp, index=self.uc_df.index, name=self.name
        )
        return match_df.sort_index(
            ascending=False, level="sheet", sort_remaining=True
        ).dropna()

    def get_uc_group(self):
        uc_group = None
        uc_groups = self._uc_data.uc_groups
        uc_groups_reversed = {}
        for group_id, uc_ids in uc_groups.items():
            uc_groups_reversed.update({uc_id: group_id for uc_id in uc_ids})
        uc_ids = []
        try:
            for uc_num, uc_ratio in self.__uc_index_ratios.items():
                uc_id = f"{uc_num:03d}"
                uc_ids.append(uc_id)
                if uc_group is None:
                    uc_group = uc_groups_reversed[uc_id]
                    logger.finfo(
                        "Composition of selected unit cell group base:",
                        initial_linebreak=True,
                    )
                    self.print_groups(
                        {uc_group: self._uc_data.base_ucs.get(uc_group)},
                        self._uc_data.df,
                        fill="\t",
                    )
                else:
                    if uc_group == uc_groups_reversed[uc_id]:
                        pass
                    else:
                        raise KeyError(f"Invalid unit cell index {uc_id}")
        except KeyError:
            logger.error(f"Invalid unit cell id {uc_id!r}!")
            self._abort(1)
        else:
            self._uc_data.select_group(uc_group)
            self._uc_data.select_ucs(uc_ids)
            # self._uc_data.check_ucs()


class MatchClayComposition(ClayComposition):
    def __init__(
        self,
        target_composition: TargetClayComposition,
        sheet_n_ucs: int,
        ignore_threshold: float = 0.0,
        manual_setup: bool = True,
        debug_run: bool = False,
        max_ucs: Optional[int] = None,
        array_chunk_size: int = 10000,
        max_dist: float = 0.005,
    ):
        super().__init__(
            sheet_n_ucs=sheet_n_ucs,
            uc_data=target_composition.uc_data,
            name=target_composition.name,
        )
        self.debug_run = debug_run
        logger.info(get_subheader("Selecting unit cells"))
        self.manual_setup = manual_setup
        self.array_chunk_size = array_chunk_size
        self.ignore_threshold = ignore_threshold
        self.max_dist = max_dist
        self.target_comp: TargetClayComposition = target_composition
        self.__target_df: pd.DataFrame = target_composition.clay_df
        self._max_ucs = max_ucs
        self.drop_unused_ucs()
        accept = None
        self.match_composition, self.target_df
        logger.finfo("Unit cell combination has the following composition:")
        TargetClayComposition.print_df_composition(
            self.match_composition, self.target_df, fill="\t", print_all=True
        )
        accept = select_input_option(
            self,
            query="\nAccept matched clay composition? [y]es/[e]xit (Default y)\n",
            options=["y", "e", ""],
            result=accept,
            result_map={"y": "y", "e": "n", "": "y"},
        )
        if accept == "n":
            self.__abort("comp")

    @cached_property
    def factorials(self):
        return [int(math.factorial(n)) for n in range(self.sheet_n_ucs)]

    def get_n_combs(self, k):
        return int(
            self.factorials[self.sheet_n_ucs - 1]
            / (self.factorials[k - 1] * self.factorials[self.sheet_n_ucs - k])
        )

    @cached_property
    def _uc_weights_file(self):
        return self._get_tmp_zarr(prefix="uc_weights")

    def _get_uc_weights_zarr(self, n_ucs):
        n_combs = self.get_n_combs(n_ucs)
        fname = self._uc_weights_file
        shape = (n_combs, n_ucs)
        try:
            if self._uc_weights_zarr.shape != shape:
                raise AttributeError
        except AttributeError:
            self._uc_weights_zarr = zarr.zeros(
                store=fname,
                shape=(n_combs, n_ucs),
                dtype=np.int16,
                chunks=(self.array_chunk_size, None),
                overwrite=True,
            )
        else:
            self._uc_weights_zarr[:] = 0
        return self._uc_weights_zarr

    def get_comb_atype_weights(self, uc_number_combination, uc_occupancies):
        return uc_number_combination.map_blocks(
            self.compute_atype_weights, uc_occupancies, self.sheet_n_ucs
        ).compute()

    def compute_dask_njit(self, da, other, result):
        return result.map_blocks(
            self.compute_njit,
            da,
            other,
            self.sheet_n_ucs,
            dtype=np.float32,
            chunks=(da.chunks[0], other.shape[0]),
        )

    @staticmethod
    @njit(nogil=True, parallel=True)
    def compute_njit(result, arr, other, div):
        a, c = result.shape
        result[:] = 0
        for ai in prange(a):
            for ci in prange(c):
                for bi in prange(arr.shape[1]):
                    result[ai, ci] += arr[ai, bi] * other[ci, bi]
                result[ai, ci] /= div
        return result

    @staticmethod
    @njit(
        ["void(int32, int32[:, :])", "void(int64, int64[:, :])"], parallel=True
    )
    def compute_combinations(sheet_n_ucs, chunk):
        m, n = chunk.shape
        for q in prange(m):
            for eid, z in enumerate(
                zip([-1, *chunk[q, :-1]], [*chunk[q, :-1], sheet_n_ucs - 1])
            ):
                chunk[q, eid] = z[1] - z[0]

    def drop_unused_ucs(self):
        all_ucs_df = self._uc_data.df
        target_df = self.__target_df.dropna().copy()
        combined_idx = (all_ucs_df.index.to_flat_index()).union(
            target_df.index.to_flat_index()
        )
        combined_idx = pd.MultiIndex.from_tuples(
            combined_idx, names=target_df.index.names
        )
        all_ucs_df = all_ucs_df.reindex(combined_idx)
        match_group_found = False
        while not match_group_found:
            target_df = target_df.reindex(combined_idx)
            unused_target_atype_mask = self.__get_nan_xor_zero_mask(target_df)
            accepted_group = {}
            accepted_base = {}
            missing_group_at_types = {}
            base_ucs = self._uc_data.base_ucs

            for group_id, group_uc_ids in self._uc_data.group_iter():
                uc_group_df = all_ucs_df[group_uc_ids]
                # discard all unit cells with non-zero values where target composition has zeros
                uc_group_df = uc_group_df.loc[
                    :, (uc_group_df[unused_target_atype_mask] == 0).all()
                ]
                # check that the group has non-zero values for all atom types in the target composition
                unused_uc_atype_mask = (
                    self.__get_nan_xor_zero_mask(uc_group_df)
                ).all(axis=1)
                missing_uc_at_types = (
                    uc_group_df[unused_uc_atype_mask]
                ).index.difference(target_df[unused_target_atype_mask].index)
                if len(missing_uc_at_types) == 0:
                    accepted_group[group_id] = list(uc_group_df.columns)
                    accepted_base[group_id] = base_ucs[group_id]
                else:
                    missing_group_at_types[group_id] = missing_uc_at_types
            accept = None
            if len(accepted_group) > 1:
                logger.finfo(f"Found {len(accepted_group)} unit cell groups.")
                logger.finfo("Getting best group for target composition.")
                base_dict = dict(
                    map(
                        lambda i: (i[0], all_ucs_df.loc[:, i[1]]),
                        accepted_base.items(),
                    )
                )
                base_df = pd.DataFrame({"target": target_df, **base_dict})
                base_df = (
                    base_df.where(base_df != 0, np.NaN)
                    .dropna(how="all")
                    .fillna(0)
                )
                diff_df = (
                    base_df[[0, 1]]
                    .apply(lambda x: np.abs(x - base_df["target"]))
                    .sum()
                )
                selected_group_id = diff_df[
                    diff_df == diff_df.min()
                ].index.to_list()[0]
                logger.finfo(
                    f"Selected group {selected_group_id} with base composition:"
                )
                self.print_groups(
                    {selected_group_id: accepted_base.get(selected_group_id)},
                    all_ucs_df,
                    fill="\t",
                )
                print_all_ucs = select_input_option(
                    self,
                    "\n\nPrint remaining group UC compositions? [y]es/[n]o (Default n)\n",
                    result=None,
                    options=["y", "n", ""],
                    result_map={
                        "y": lambda: self.print_groups(
                            {
                                selected_group_id: accepted_group.get(
                                    selected_group_id
                                )
                            },
                            all_ucs_df,
                            fill="\t",
                        ),
                        "n": lambda: None,
                        "": lambda: None,
                    },
                )
                print_all_ucs()
                accept = select_input_option(
                    self,
                    "\nAccept selection [y] or choose manually [m]? (Default y)",
                    options=["y", "m", ""],
                    result=None,
                    result_map={
                        "y": str(selected_group_id),
                        "m": None,
                        "": str(selected_group_id),
                    },
                )
                select_keys = list(
                    map(lambda key: f"{key}", accepted_group.keys())
                )
                if accept is None:
                    logger.finfo(
                        "Available UC group base compositions:",
                        initial_linebreak=True,
                    )
                    self.print_groups(accepted_base, all_ucs_df, fill="\t")
                    print_all_ucs = select_input_option(
                        self,
                        "\n\nPrint remaining group UC compositions? [y]es/[n]o (Default n)\n",
                        result=None,
                        options=["y", "n", ""],
                        result_map={
                            "y": lambda: self.print_groups(
                                accepted_group, all_ucs_df, fill="\t"
                            ),
                            "n": lambda: None,
                            "": lambda: None,
                        },
                    )
                    print_all_ucs()
                accept = select_input_option(
                    self,
                    query=f"\nSelect unit cell group? [{'/'.join(select_keys)}] (exit with e)\n",
                    options=[*select_keys, "e"],
                    result=accept,
                    result_map={
                        **dict(map(lambda x: (x, x), select_keys)),
                        "e": "n",
                    },
                )
                if accept in select_keys:
                    accept = int(accept)
                    logger.finfo(
                        f"Selected group {accept}", initial_linebreak=True
                    )
                    accepted_group = {accept: accepted_group.get(accept)}
                    accepted_base = {accept: accepted_base.get(accept)}
                    accept = "y"
                match_group_found = True
            if len(accepted_group) == 1:
                selected_ucs_df = all_ucs_df.loc[
                    :, accepted_group[next(iter(accepted_group.keys()))]
                ]
                logger.finfo(
                    "Composition of matching unit cell group base:",
                    initial_linebreak=True,
                )
                if accept is None:
                    self.print_groups(accepted_base, all_ucs_df, fill="\t")
                print_all_ucs = select_input_option(
                    self,
                    "\n\nPrint remaining group UC compositions? [y]es/[n]o (Default y)\n",
                    result=accept,
                    options=["y", "n", "", next(iter(accepted_group.keys()))],
                    result_map={
                        "y": lambda: self.print_groups(
                            accepted_group, selected_ucs_df, fill="\t"
                        ),
                        "n": lambda: None,
                        "": lambda: None,
                        next(iter(accepted_group.keys())): None,
                    },
                )
                print_all_ucs()
                accept = select_input_option(
                    self,
                    query="\nAccept unit cell group? [y]es/[e]xit (Default y)\n",
                    result=accept,
                    options=["y", "e", ""],
                    result_map={"y": "y", "e": "n", "": "y"},
                )
                match_group_found = True
            elif len(accepted_group) == 0:
                missing_at_types = np.ravel(
                    [
                        missing.get_level_values("at-type").to_list()
                        for missing in missing_group_at_types.values()
                    ]
                )
                ignore = True
                missing_at_types: list = missing_at_types.tolist()
                for missing_at_type in missing_at_types.copy():
                    if (
                        self.target_df.xs(missing_at_type, level="at-type")
                        < self.ignore_threshold
                    ).all():
                        logger.finfo(
                            f'{missing_at_type!r} not found in unit cells, resetting occupancy of {self.target_df.xs(missing_at_type, level="at-type").values[0]} to 0.00 (ignore threshold = {self.ignore_threshold:.2f})\n'
                        )
                        self.target_comp._df.loc[
                            pd.IndexSlice[:, missing_at_type]
                        ] = 0.0
                        missing_at_types.remove(missing_at_type)
                    else:
                        ignore = False
                    if not ignore:
                        missing_str = ", ".join(missing_at_types)
                        raise ValueError(
                            f"Some of the required atom types ({missing_str}) of the target composition are missing in the unit cells!\n"
                        )
                    else:
                        logger.finfo("Correcting new occupancies.")
                        self.target_comp.correct_uncharged_occupancies()
                        self.__target_df.update(self.target_comp.df)
                        self.__target_df.where(
                            self.target_df != 0, np.NaN, inplace=True
                        )
                        target_df = self.__target_df.dropna().copy()

            else:
                raise RuntimeError("Should not reach this point!")
        if accept == "n":
            self.__abort("uc")
        else:
            self._uc_data.select_group(next(iter(accepted_group.keys())))
            self._uc_data.select_ucs(
                accepted_group[next(iter(accepted_group.keys()))]
            )
            assert self.target_df.index.equals(
                self.uc_df.index
            ), "Target composition and unit cell data must share index"
            # self._uc_data.check_ucs()

    @staticmethod
    def __abort(
        reason: Literal[Union[Literal["uc"], Literal["comp"]]], description=""
    ):
        reason_dict = {
            "uc": "No unit cell group accepted,",
            "comp": "Clay composition not accepted.",
        }
        logger.finfo(
            f"{reason_dict[reason]}\n{description}Aborting model construction."
        )
        sys.exit(0)

    @staticmethod
    def __get_nan_xor_zero_mask(df):
        return np.logical_xor(np.isnan(df), df == 0)

    @property
    def uc_df(self):
        return self._uc_data.df.reindex(index=self.target_df.index)

    @property
    def target_df(self):
        return self.__target_df.sort_index(
            ascending=False, level="sheet", sort_remaining=True
        ).dropna()

    @property
    def max_ucs(self):
        if self._max_ucs is None:
            self.max_ucs = None
        return self._max_ucs

    @max_ucs.setter
    def max_ucs(self, max_ucs):
        n_all_ucs = len(self.unique_uc_array)
        if not self._max_ucs:
            self._max_ucs = n_all_ucs
        if not self.debug_run and not max_ucs:
            return
        elif max_ucs:
            n_uc_max = min(int(max_ucs), self._max_ucs)
            if self._max_ucs != n_uc_max:
                logger.finfo(
                    f"Maximum number of unique unit cells in sheet set to {n_uc_max}"
                )
        else:
            n_uc_max = min(2, n_all_ucs)
            logger.finfo(get_debugheader(f"max n_ucs = {n_uc_max}"))
        self._max_ucs = n_uc_max

    def _get_zarr_storage(self, name, **kwargs):
        return zarr.storage.DirectoryStore(BUILDER_DATA / f"{name}", **kwargs)

    def init_weight_arrays(self, target_composition):
        n_atypes = len(target_composition)
        arr_size = get_arr_bytes(
            (self.get_n_combs(self.max_ucs), np.max([n_atypes, self.max_ucs])),
            arr_dtype=np.float32,
        )
        max_ucs = self.max_ucs
        while arr_size > MAX_STORE_ARR_SIZE:
            max_ucs -= 1
            arr_size = get_arr_bytes(
                (self.get_n_combs(max_ucs), np.max([n_atypes, max_ucs])),
                arr_dtype=np.float32,
            )
        if max_ucs != self.max_ucs:
            self.max_ucs = max_ucs
        n_combs = self.get_n_combs(self.max_ucs)
        for array_name, shape_1, arr_type in zip(
            ["uc_weights", "atype_weights"],
            [self.max_ucs, n_atypes],
            [np.int32, np.float32],
        ):
            try:
                out_zarr = zarr.open(
                    store=self._get_zarr_storage(array_name), mode="r+"
                )
                if np.all(out_zarr.shape == (n_combs, shape_1)):
                    pass
                elif np.any(out_zarr.shape < (n_combs, shape_1)):
                    for arr_file in out_zarr.store.listdir():
                        os.remove(f"{out_zarr.store.path}/{arr_file}")
                    raise PathNotFoundError(f"Existing array too small!")
            except PathNotFoundError as e:
                logger.debug(f"{e}")
                out_zarr = zarr.zeros(
                    store=self._get_zarr_storage(array_name),
                    dtype=arr_type,
                    chunks=(self.array_chunk_size, None),
                    shape=(self.get_n_combs(self.max_ucs), shape_1),
                )
            finally:
                setattr(self, f"_{array_name}_zarr", out_zarr)
                out_dask = dask.array.from_zarr(
                    out_zarr.store.path, chunks=out_zarr.chunks
                )
                out_dask[:] = 0
                if np.all(out_zarr.shape != (n_combs, shape_1)):
                    out_dask = out_dask[:n_combs, :shape_1]
                setattr(self, f"_{array_name}_dask", out_dask)

    def get_weight_array_slices(self, n_ucs):
        n_combs = self.get_n_combs(n_ucs)
        self.uc_weights_dask = self._uc_weights_dask[:n_combs, :n_ucs]
        self.atype_weights_dask = self._atype_weights_dask[:n_combs, :]
        self.atype_weights_dask[:] = 0
        self.uc_weights_dask[:] = 0

    def get_uc_weights_d(self, n_ucs):
        combs_dask = self.uc_weights_dask
        combs_dask[:] = 0
        if combs_dask.shape[0] > 1:
            combs_dask[:, :-1] = [
                x
                for x in itertools.combinations(
                    range(self.sheet_n_ucs - 1), n_ucs - 1
                )
            ]
        combs_dask = combs_dask.map_blocks(
            self.compute_combinations_x, self.sheet_n_ucs
        )
        return combs_dask

    @staticmethod
    @njit("int32[:,:](int32[:, :], int32)", parallel=True, nogil=True)
    def compute_combinations_x(chunk, N):
        a, b = chunk.shape
        c = N - 1
        for ai in prange(a):
            x = np.array([-1, *chunk[ai, :-1], c])
            for bi in prange(b):
                chunk[ai, bi] = x[bi + 1] - x[bi]
        return chunk

    def get_restrained_uc_combinations_1(
        self, n_ucs: int, fixed_uc_idxs: np.array
    ):
        # Filtering out the uc_array values that are in fixed_uc_idxs
        mask = ~np.isin(self.unique_uc_array, fixed_uc_idxs)
        remaining_uc_ids = self.unique_uc_array[mask]

        n_remaining = n_ucs - len(fixed_uc_idxs)
        return [
            sorted([*fixed_uc_idxs, *uc_ids])
            for uc_ids in itertools.combinations(remaining_uc_ids, n_remaining)
        ]

    @cached_property
    def _unique_uc_match_dict(self) -> dict:
        max_it = 3
        uc_df = self.uc_df.copy()
        uc_df.columns = uc_df.columns.astype(int)
        match_dict = dict(
            map(
                lambda k: (k, np.NaN),
                [
                    "n_ucs",
                    "uc_ids",
                    "uc_weights",
                    "composition",
                    "dist",
                    "charges",
                ],
            )
        )
        min_dist = np.inf
        logger.info(
            get_subheader(
                "Getting matching unit cell combination for target composition"
            )
        )
        uc_id_sel = []
        target_values = np.squeeze(self.target_df.astype(np.float32).values)
        self.init_weight_arrays(target_values)
        n_ucs = 0
        match_found = False
        worse_matches = 0
        while not match_found and n_ucs < self.max_ucs:
            n_ucs += 1
            self.get_weight_array_slices(n_ucs)
            better_choice_than_prev = False
            logger.finfo(
                f"Getting combinations for {n_ucs}/{self.max_ucs} unique unit cells",
                initial_linebreak=True,
            )
            uc_id_combinations = self.get_restrained_uc_combinations_1(
                n_ucs, uc_id_sel
            )
            uc_number_combinations = self.get_uc_weights_d(n_ucs)
            for uc_ids in tqdm(
                uc_id_combinations,
                total=len(uc_id_combinations),
                ncols=LINE_LENGTH,
                unit_scale=len(uc_number_combinations),
                bar_format="\t{l_bar}{bar}| {n_fmt}/{total} combinations, {elapsed_s:3.0f} s elapsed",
            ):
                self.atype_weights_dask[:] = 0
                atype_weights = self.compute_dask_njit(
                    uc_number_combinations,
                    uc_df[[*list(uc_ids)]].astype(np.float32).values,
                    self.atype_weights_dask,
                )
                dist, match_idx = self.get_atype_diff_dask(
                    atype_weights, target_values
                )
                if min_dist >= dist:
                    match_dict.update(
                        {
                            "n_ucs": n_ucs,
                            "uc_ids": uc_ids,
                            "uc_weights": uc_number_combinations[match_idx],
                            "composition": atype_weights[match_idx],
                            "dist": dist,
                        }
                    )
                    better_choice_than_prev = True
                    min_dist = dist
            # end if result == target composition
            if min_dist == 0.0:
                match_found = True
                logger.finfo(
                    f"Found match with zero deviation from target composition.\nStopping here.",
                    initial_linebreak=True,
                )
            # better match found in previous combinations
            if better_choice_than_prev:
                uc_id_sel = match_dict["uc_ids"]
            # end if previous match was better and the previous was below acceptance tolerance
            elif min_dist <= self.max_dist:
                logger.finfo(
                    f"Found better match with fewer unit cells.\nStopping here.",
                    initial_linebreak=True,
                )
                match_found = True
            # add result to list of matches that made the result worse if maximum number of worse iterations not reached
            elif worse_matches <= max_it:
                logger.finfo(
                    f"Found better match with fewer unit cells but deviation ({min_dist:1.4f}) "
                    f"from target composition greater than accepted threshold of {self.max_dist:1.4f}.\n"
                    "Searching for for better match.",
                    initial_linebreak=True,
                )
                worse_matches += 1
            # end if maximum number of worse iterations reached even if not within threshold
            else:
                logger.finfo(
                    f"The last {max_it} iterations did not provide an improved matching.\n"
                    f"The average unit cell composition still deviates by a total of {min_dist:1.4f} from target stoichiometry.\n"
                    "Stopping here.",
                    initial_linebreak=True,
                )
                match_found = select_input_option(
                    instance_or_manual_setup=True,
                    query="\nContinue with current match? [y]es/[e]xit (Default y)\n",
                    options=["y", "e", ""],
                    result=match_found,
                    result_map={"y": True, "e": False, "": True},
                )
                if not match_found:
                    self.__abort(
                        "uc",
                        "Could not find a unit cell composition that matches the target composition better than the accepted threshold of {self.max_dist:1.4f}.",
                    )
        logger.finfo(
            f"Best match found with {match_dict['n_ucs']} unique unit cells "
            + f"(total occupancy deviation {match_dict['dist'].round(3):+.3f})",  # \n\nIndices:",
            initial_linebreak=True,
        )
        logger.finfo(f"Selected unit cell compositions:")
        charges = self._uc_data.tot_charge[
            [f"{uc_id:03d}" for uc_id in match_dict["uc_ids"]]
        ]
        self.print_uc_composition(
            match_dict["uc_ids"], self.uc_df, fill="\t", charges=charges
        )
        match_dict["composition"] = np.round(
            np.squeeze(match_dict["composition"].compute()), 4
        )
        match_dict["uc_weights"] = np.round(
            np.squeeze(match_dict["uc_weights"].compute()), 4
        )
        match_dict["dist"] = np.round(np.squeeze(match_dict["dist"]), 4)
        match_dict["charges"] = np.round(charges.values.flatten(), 1).astype(
            np.int_
        )
        return match_dict

    @staticmethod
    @njit(nogil=True, parallel=True)
    def get_ax1_norm(arr):
        result = np.zeros((arr.shape[0]), dtype=np.float32)
        for ai in prange(arr.shape[0]):
            for bi in prange(arr.shape[1]):
                result[ai] += arr[ai, bi] ** 2
            result[ai] = math.sqrt(result[ai])
        return result

    @staticmethod
    @njit
    def sel_min_idx(arr, min_dist):
        for ai in prange(arr.shape[0]):
            if np.equal(arr[ai], min_dist):
                return np.array([ai])

    def get_atype_diff_dask(self, atype_weights, target_values):
        diff_array = atype_weights - target_values
        diff_array = diff_array.map_blocks(self.get_ax1_norm, drop_axis=1)
        dist = diff_array.min().compute()
        equal_idx = dask.array.equal(diff_array, dist)
        match_idx = dask.array.argwhere(equal_idx).compute()[0]
        return dist, match_idx

    @property
    def match_composition(self):
        match_values = pd.Series(
            self._unique_uc_match_dict["composition"],
            index=self.target_df.index,
            name="match_composition",
        )
        return match_values

    @property
    def match_diff(self) -> float:
        return self._unique_uc_match_dict["dist"]

    @cached_property
    def duplicate_ucs(self) -> NDArray:
        if not (self._uc_data.path / "duplicate_ucs.pkl").is_file():
            uc_ids = self.uc_df.columns.values.astype(int)
            uc_compositions = self.uc_df.values
            duplicate_ucs = {}
            uc_index, uc_inverse, uc_count = np.unique(
                uc_compositions,
                return_index=True,
                return_inverse=True,
                return_counts=True,
                axis=1,
            )[1:]
            uc_index = uc_index[uc_count > 1]
            for idx in np.sort(uc_index):
                uc_df_idx_sel = uc_ids[idx]
                duplicate_ucs[uc_df_idx_sel] = uc_ids[
                    uc_inverse == uc_inverse[idx]
                ]
            with open(self._uc_data.path / "duplicate_ucs.pkl", "wb") as f:
                pkl.dump(duplicate_ucs, f)
        else:
            with open(self._uc_data.path / "duplicate_ucs.pkl", "rb") as f:
                duplicate_ucs = pkl.load(f)
        return duplicate_ucs

    @staticmethod
    def _select_duplicate_ucs(id_sel, duplicate_ids) -> List[str]:
        return id_sel[0] + 1  # for i in range(1, len(duplicate_ids)))

    @cached_property
    def _uc_match_dict(self) -> dict:
        uc_match_dict = self._unique_uc_match_dict.copy()
        unique_uc_match_ids = np.array(uc_match_dict["uc_ids"]).astype(int)
        unique_uc_occs = uc_match_dict["uc_weights"]
        unique_uc_charges = uc_match_dict["charges"]
        for unique_id, duplicate_ids in self.duplicate_ucs.items():
            if np.in1d(unique_id, unique_uc_match_ids).any():
                _, _, sel_ids = np.intersect1d(
                    duplicate_ids,
                    unique_uc_match_ids,
                    return_indices=True,
                    assume_unique=True,
                )
                uc_match_dict["n_ucs"] += 1
                new_occs: NDArray = np.full(
                    len(duplicate_ids), unique_uc_occs[sel_ids]
                )
                new_charges = np.full(
                    len(duplicate_ids), unique_uc_charges[sel_ids]
                )
                if unique_id == unique_uc_match_ids[-1]:
                    unique_uc_match_ids: NDArray = np.append(
                        unique_uc_match_ids,
                        duplicate_ids[(duplicate_ids != unique_id)],
                    )
                    unique_uc_occs: NDArray = np.append(
                        unique_uc_occs, new_occs[1:]
                    )
                else:
                    unique_uc_match_ids: NDArray = np.insert(
                        unique_uc_match_ids,
                        self._select_duplicate_ucs(sel_ids, duplicate_ids),
                        duplicate_ids[duplicate_ids != unique_id],
                    )
                    unique_uc_occs: NDArray = np.insert(
                        unique_uc_occs,
                        self._select_duplicate_ucs(sel_ids, duplicate_ids),
                        new_occs[1:],
                    )
                    unique_uc_charges = NDArray = np.insert(
                        unique_uc_charges,
                        self._select_duplicate_ucs(sel_ids, duplicate_ids),
                        new_charges[1:],
                    )
                uc_occ_splitting_remainder = unique_uc_occs[sel_ids] % len(
                    duplicate_ids
                )
                unique_uc_occs[
                    sel_ids[0] : sel_ids[0] + len(duplicate_ids)
                ] //= len(duplicate_ids)
                unique_uc_occs[sel_ids] += uc_occ_splitting_remainder
        uc_match_dict["uc_weights"] = unique_uc_occs
        uc_match_dict["uc_ids"] = unique_uc_match_ids
        uc_match_dict["charges"] = unique_uc_charges
        return uc_match_dict

    @cached_property
    def unique_uc_array(self) -> NDArray[int]:
        # uc_array = self.uc_df.T.drop_duplicates().index# .values.astype(int)
        accepted_charges = self._uc_data.tot_charge
        target_charge = self.target_comp.get_total_charge()
        accepted_charges = accepted_charges[
            abs(accepted_charges) <= np.ceil(np.abs(target_charge)) + 0.5
        ]
        accepted_charges = accepted_charges[
            abs(accepted_charges) >= np.floor(np.abs(target_charge)) - 0.5
        ]
        logger.info(
            f"Will use unit cells with charges between {np.min(accepted_charges):.1f} and {np.max(accepted_charges):.1f}."
        )
        uc_array = (
            self.uc_df.T.drop_duplicates()
            .index.intersection(accepted_charges.index)
            .values.astype(int)
        )
        return uc_array

    def _get_uc_weights_arrays(self):
        self._uc_weights_zarr = self._get_uc_weights_zarr(self.max_ucs)
        self._uc_weights_da = dask.array.from_zarr(
            self._uc_weights_zarr.store.path, chunks=(1000, None)
        )

    def uc_weights_da(self, n_ucs):
        if not hasattr(self, "_uc_weights_da"):
            self._get_uc_weights_arrays()
        sel = self._uc_weights_da[: self.get_n_combs(n_ucs), :n_ucs]
        sel[:] = 0
        return sel

    @staticmethod
    def get_uc_weight_list(N, k):
        n_combs = int(
            math.factorial(N - 1)
            / (math.factorial(k - 1) * math.factorial(N - k))
        )
        all_combs = itertools.combinations(  # dask.bag.from_sequence(
            range(N - 1), k - 1
        )
        for q in tqdm(
            all_combs,
            total=n_combs,
            ncols=LINE_LENGTH,
            bar_format="\t{l_bar}{bar}| {n_fmt}/{total_fmt} combinations, {elapsed_s:3.0f} s elapsed",
        ):
            yield [j - i for i, j in zip((-1,) + q, q + (N - 1,))]

    def get_uc_combinations(self, n_ucs):
        return itertools.combinations(self.unique_uc_array, n_ucs)

    def get_restrained_uc_combinations(self, n_ucs, fixed_uc_idxs):
        mask = ~np.isin(self.unique_uc_array, fixed_uc_idxs)
        remaining_uc_ids = self.unique_uc_array[mask]

        n_remaining = n_ucs - len(fixed_uc_idxs)

        return [
            sorted([*fixed_uc_idxs, *uc_ids])
            for uc_ids in itertools.combinations(remaining_uc_ids, n_remaining)
        ]

    def _write(
        self,
        outpath: Union[Dir, File, PosixPath],
        fmt: Optional[str] = None,
        backup=False,
    ):
        if type(outpath) == str:
            outpath = PathFactory(outpath)
        if fmt is None:
            if outpath.suffix == "":
                raise ValueError("No file format specified")
            else:
                fmt = outpath.suffix
        fmt = f'.{fmt.lstrip(".")}'
        if outpath.suffix == "":
            outpath = outpath / f"{self.name}_match_df{fmt}"
        tmpfile = tempfile.NamedTemporaryFile(suffix=fmt, prefix=outpath.stem)
        if fmt == ".csv":
            output_format = self.match_composition
            self.match_composition.to_csv(tmpfile.name, float_format="%.4f")
        elif fmt == ".p":
            with open(tmpfile.name, "wb") as file:
                pkl.dump(self.match_composition)
        else:
            raise ValueError(
                f"Invalid format specification {fmt!r}\n"
                f'Expected {".csv"!r} or {".p"!r}'
            )
        if not outpath.parent.is_dir():
            os.makedirs(outpath.parent)
        logger.finfo(
            f"Writing new match clay composition to {str(outpath)!r}",
            initial_linebreak=True,
        )
        if backup:
            backup_files(new_filename=outpath, old_filename=tmpfile.name)
        else:
            shutil.copy(tmpfile.name, outpath)
        assert outpath.is_file()

    write_csv = partialmethod(_write, fmt=".csv")
    write_pkl = partialmethod(_write, fmt=".csv")


class Ions(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass


class InterlayerIons(Ions):
    def __init__(
        self,
        tot_charge: int,
        ion_ratios: pd.DataFrame,
        n_ucs: int,
        monovalent=["Na", "Cl"],
        neutral=False,
    ):
        self.clay_charge = np.rint(tot_charge * n_ucs)
        ion_charges = get_ion_charges()
        if type(ion_ratios) == dict:
            ion_ratios = pd.DataFrame.from_dict(
                ion_ratios, orient="index", columns=["probs"]
            )
            ion_ratios["charges"] = np.NAN
            ion_ratios["charges"] = ion_ratios.index.to_series().apply(
                lambda x: ion_charges[x]
            )
        if neutral:
            ion_ratios["probs"][
                np.sign(ion_ratios["charges"]) == np.sign(tot_charge)
            ] = 0
        if np.isclose(self.clay_charge, 0.0):
            monovalent, charge = monovalent[0], ion_charges[monovalent[0]]
        else:
            monovalent, charge = tuple(
                *[
                    (ion, ion_charges[ion])
                    for ion in monovalent
                    if np.sign(ion_charges[ion]) != np.sign(self.clay_charge)
                ]
            )
        ion_ratios = ion_ratios[
            np.sign(ion_ratios["charges"]) != np.sign(tot_charge)
        ].dropna(how="all")
        if ion_ratios["probs"].dropna().empty:
            ion_ratios["probs"].fillna(1.0, inplace=True)
        self._df = ion_ratios.copy()
        if charge not in self._df["charges"].values:
            mono_df = pd.DataFrame(
                [[0, charge]],
                columns=self._df.columns,
                index=pd.Index([monovalent], name=self._df.index.name),
            )
            self._df = pd.concat([self._df, mono_df])
        elif monovalent not in self._df.index:
            monovalent = self._df[self._df["charges"] == charge].index[0]
        if np.isclose(self._df["probs"].sum(), 0.00):
            self._df["probs"] = 1.0 / len(self._df["probs"])
        elif not np.isclose(self._df["probs"].sum(), 1.00):
            self._df["probs"] = self._df["probs"].divide(
                np.sum(self._df["probs"])
            )
        assert np.isclose(
            self._df["probs"].sum(), 1.00
        ), "Interlayer ion probabilities do not sum to 1.00"
        self._df["numbers"] = 0
        self.get_ion_numbers(monovalent=monovalent)

    def __repr__(self):
        df_str = ", ".join(
            [f"{k}: {v}" for k, v in self.df["numbers"].items()]
        )
        return f"{self.__class__.__name__}({df_str})"

    def get_ion_numbers(self, monovalent):
        df = self._df[~np.isclose(self._df["charges"], 0.00)]
        avg_charge = df.loc[:, ["probs", "charges"]].prod(axis=1).sum()
        multiplication_factor = np.abs(np.divide(self.clay_charge, avg_charge))
        ion_charge = -np.sign(self.clay_charge) * (
            np.abs(self.clay_charge) + 1
        )
        multiplication_factor_mod = 0
        while np.abs(ion_charge) > np.abs(self.clay_charge):
            multiplication_factor -= multiplication_factor_mod
            df["numbers"] = np.rint(
                np.multiply(df["probs"], multiplication_factor)
            )
            ion_charge = (df["numbers"] * df["charges"]).sum()
            multiplication_factor_mod += 0.05
        if not np.isclose(-ion_charge, self.clay_charge):
            df.loc[monovalent, "numbers"] += np.abs(self.clay_charge) - np.abs(
                ion_charge
            )
        df.dropna(subset=["charges", "numbers"], inplace=True)
        df = df.convert_dtypes()
        self._df = df

    @property
    def df(self):
        return self._df[["charges", "numbers"]]

    @property
    def numbers(self):
        return self.df["numbers"].dropna().apply(np.rint).astype(int).to_dict()


class BulkIons(Ions):
    def __init__(self, ion_con_dict, default_ions):
        ion_charges = get_ion_charges()
        ion_idx = pd.Index(
            np.unique([*ion_con_dict.keys(), *default_ions.keys()]),
            name="itype",
        )
        self.df = pd.DataFrame(columns=["charge", "conc"], index=ion_idx)
        default_ions = pd.Series(default_ions)
        bulk_ion_sel = pd.Series(ion_con_dict)
        default_ions = default_ions[
            default_ions.index.difference(bulk_ion_sel.index)
        ]
        self.df["conc"].update(default_ions)
        self.df["conc"].update(bulk_ion_sel)
        self.df["charge"] = [ion_charges[ion] for ion in self.df.index.values]
        self.df["neutralise"] = True
        select_min_charge = lambda charges: np.abs(charges) == np.min(
            np.abs(charges)
        )
        for ion_slice in [self.df["charge"] > 0, self.df["charge"] < 0]:
            ion_slice_df = self.df[ion_slice]
            ion_slice_df["neutralise"] = ion_slice_df["neutralise"].where(
                select_min_charge(ion_slice_df["charge"]), False
            )
            self.df["neutralise"].update(ion_slice_df["neutralise"])

    def __repr__(self):
        df_str = ", ".join(
            [f"{k}: {v:.2f}" for k, v in self.df["conc"].items()]
        )
        return f"{self.__class__.__name__}({df_str})"

    @property
    def neutralise_ions(self):
        return self.df[self.df["neutralise"]][["charge", "conc"]]

    @property
    def conc(self):
        return self.df[self.df["conc"] > 0][["charge", "conc"]]

    @property
    def tot_conc(self):
        return self.pions["conc"].sum() + self.nions["conc"].sum()

    @property
    def pions(self):
        return self.df[self.df["charge"] > 0].dropna(how="all", subset="conc")

    @property
    def nions(self):
        return self.df[self.df["charge"] < 0].dropna(how="all", subset="conc")
