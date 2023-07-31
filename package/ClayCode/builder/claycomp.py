#!/usr/bin/env python3

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
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from ClayCode.builder.utils import get_checked_input, select_input_option
from ClayCode.core.classes import Dir, File, ITPFile, PathFactory, YAMLFile
from ClayCode.core.consts import LINE_LENGTH, UCS
from ClayCode.core.lib import get_ion_charges
from ClayCode.core.utils import backup_files, get_debugheader, get_subheader
from numpy._typing import NDArray
from tqdm import tqdm

__all__ = ["TargetClayComposition"]

logger = logging.getLogger(__name__)


class UnitCell(ITPFile):
    @property
    def idx(self):
        return self.stem[2:]

    @property
    def clay_type(self):
        return self.parent.name

    @property
    def uc_stem(self):
        return self.stem[:2]

    @cached_property
    def atom_df(self):
        atoms = self.get_parameter("atoms")
        return atoms.df

    @cached_property
    def charge(self):
        return self.atom_df["charge"].sum().round(6)


class UCData(Dir):
    _suffices = [".gro", ".itp"]
    _sheet_grouper = pd.Grouper(level="sheet", sort=False)

    def __init__(self, path: Dir, uc_stem=None, ff=None):
        from ClayCode.core.classes import ForceField

        if uc_stem is None:
            self.uc_stem: str = self.name[-2:]
        else:
            self.uc_stem: str = uc_stem
        logger.info(get_subheader("Getting unit cell data"))
        self.ff: ForceField = ForceField(ff)
        self.__uc_idxs: list = list(map(lambda x: str(x[-2:]), self.available))
        self.uc_idxs = self.__uc_idxs.copy()
        self.atomtypes: pd.DataFrame = self.ff["atomtypes"].df
        self.__full_df: pd.DataFrame = None
        self.__df: pd.DataFrame = None
        self.__get_full_df()
        self.__get_df()
        self.__atomic_charges = None
        self.group_id = None
        self.__gro_groups = None
        self.__itp_groups = None
        self.__dimensions = None
        self.__uc_groups = None
        self.get_uc_groups()
        logger.finfo(f"Found {self.n_groups} {self.name!r} unit cell groups:")
        self.get_uc_group_base_compositions()
        self.idx_sel = (
            self.df.index.get_level_values("sheet").unique().to_list()
        )

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
        logger.finfo(
            "Aborting model construction.",
            initial_linebreak=True,
        )
        sys.exit(0)

    def get_uc_group_base_compositions(self):
        for n_group, group_ids in self.group_iter():
            group_id_str = f", {self.uc_stem}".join(group_ids)
            logger.finfo(
                kwd_str=f"\tGroup {n_group:2d}: ",
                message=f"{self.uc_stem}{group_id_str}\n",
            )
            self.__base_ucs[n_group] = self.__get_base_ucs(group_ids)

    def get_uc_groups(self):
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
        extract_id = lambda file: file.stem[-2:]
        for uc in sorted(self.gro_filelist):
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
                if np.less(bbox_height_new, bbox_height[box_dims[dim_str]]):
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
        self.__base_ucs = {}

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
        uc_df = self.__df.loc[:, uc_ids]
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
    def __gro_df(self):
        gro_df = pd.DataFrame(
            index=np.arange(1, np.max(self.n_atoms) + 1),
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
        for gro in self.gro_filelist:
            n_atoms = self.n_atoms.filter(regex=gro.stem[-2:]).values[0]
            gro_df.update(
                pd.read_csv(
                    gro,
                    index_col=[0, 2],
                    skiprows=2,
                    header=None,
                    sep="\s+",
                    names=gro_df.columns,
                    nrows=n_atoms,
                    comment=";",
                    skip_blank_lines=True,
                )
            )
        # gro_df.reset_index(level='uc-id', inplace=True)
        # gro_df['uc-id'] = gro_df['uc-id'].apply(lambda uc_id: uc_id[-2:])
        # gro_df.set_index('uc-id', inplace=True, append=True)
        # gro_df.index = gro_df.index.reorder_levels(['uc-id', 'atom-id'])
        return gro_df
        # gro_data.index.names = gro_df.index.names
        # gro_df.update(gro_data)

    @property
    def gro_df(self):
        regex = "|".join([f"{self.uc_stem}{uc_id}" for uc_id in self.uc_idxs])
        return (
            self.__gro_df.reset_index("atom-id")
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
        except KeyError as e:
            e(f"{group_id} is an invalid group id!")

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
        return self.__full_df.sort_index(
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
            self.__df.loc[:, self.uc_idxs]
            .sort_index(ascending=False, level="sheet", sort_remaining=True)
            .sort_index(axis=1)
        )

    def __get_full_df(self):
        idx = self.atomtypes.iloc[:, 0]
        cols = [*self.uc_idxs, "charge", "sheet"]
        self.__full_df = pd.DataFrame(
            index=idx, columns=cols, dtype=np.float64
        )
        self.__full_df["charge"].update(
            self.atomtypes.set_index("at-type")["charge"]
        )
        self.__get_df_sheet_annotations()
        self.__full_df["sheet"].fillna("X", inplace=True)
        self.__full_df.fillna(0, inplace=True)
        for uc in self.uc_list:
            atoms = uc["atoms"].df
            self.__full_df[f"{uc.idx}"].update(atoms.value_counts("at-type"))
        self.__full_df.set_index("sheet", append=True, inplace=True)
        self.__full_df.sort_index(inplace=True, level=1, sort_remaining=True)
        self.__full_df.index = self.__full_df.index.reorder_levels(
            ["sheet", "at-type"]
        )

    def __get_df_sheet_annotations(self):
        old_index = self.__full_df.index
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
        self.__full_df["sheet"].update(new_index[0])

    def __get_df(self):
        self.__df = self.full_df.reset_index("at-type").filter(
            regex=r"^(?![X].*)", axis=0
        )
        self.__df = (
            self.__df.reset_index()
            .set_index(["sheet", "at-type"])
            .sort_index(axis=1)
        )

    @cached_property
    def uc_list(self) -> List[UnitCell]:
        uc_list = [UnitCell(itp) for itp in self.itp_filelist]
        return uc_list

    @cached_property
    def occupancies(self) -> Dict[str, int]:
        return self._get_occupancies(self.df)

    @cached_property
    def tot_charge(self) -> pd.Series:
        charge = self.full_df.apply(
            lambda x: x * self.full_df["charge"], raw=True
        )
        total_charge = (
            charge.loc[:, self.uc_idxs].sum().astype(np.float64).round(2)
        )
        total_charge.name = "charge"
        return total_charge

    @cached_property
    def n_atoms(self):
        return self.full_df.filter(regex="[0-9]+").sum(axis=0)

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
        return self.itp_filelist.extract_fstems()

    def __str__(self):
        return f"{self.__class__.__name__}({self.name!r})"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name!r})"

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
    def atomic_charges(self):
        return self._get_oxidation_numbers(
            self.occupancies, self.df, self.tot_charge
        )[0]

    @singledispatchmethod
    @staticmethod
    def _get_oxidation_numbers(
        occupancies,
        df: Union[pd.DataFrame, pd.Series],
        tot_charge: Optional = None,
        sum_dict: bool = True,
    ) -> Dict[str, int]:
        pass

    @_get_oxidation_numbers.register(dict)
    @staticmethod
    def _(
        occupancies: Dict[str, int],
        df: Union[pd.DataFrame, pd.Series],
        tot_charge: Optional = None,
        sum_dict: bool = True,
    ) -> Dict[str, int]:
        """Get oxidation numbers from unit cell composition and occupancies"""
        ox_dict = UCData._get_ox_dict()
        # df = df.loc[['T','O']]
        ox_df: pd.DataFrame = df.copy()
        ox_df.sort_index(level="sheet", sort_remaining=True, inplace=True)
        try:
            ox_df = ox_df.loc[~(ox_df == 0).all(1), :]
        except ValueError:
            ox_df = ox_df.loc[~(ox_df == 0)]
        if tot_charge is not None:
            _ox_df = ox_df.loc[:, tot_charge == 0]
            if _ox_df.empty:
                _ox_df = ox_df[ox_df == ox_df.max()].dropna()
                _ox_df[:] = ox_df.groupby("sheet", group_keys=True).sum()
            ox_df = _ox_df
        at_types: pd.DataFrame = ox_df.index.get_level_values(
            "at-type"
        ).to_frame()
        at_types.index = ox_df.index
        try:
            at_types.drop(("O", "fe_tot"), inplace=True)
        except KeyError:
            pass
        at_types = at_types.applymap(lambda x: ox_dict[x])
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
            ox_dict = ox.groupby("sheet").apply(
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
    ) -> int:
        """Get oxidation numbers from unit cell composition and occupancies"""
        # df = df.loc[['T','O']]
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
            # ox_dict: dict = dict(zip(idx, ox))
            ox_val = ox / occupancies
            # ox_dict: dict = dict(map(lambda x: (x, ox_dict[x] / occupancies[x]), occupancies.keys()))
        else:
            ox_val = ox.apply(lambda x: x / occupancies)
        return at_types, ox_val[0]

    @staticmethod
    @cache
    def _get_ox_dict():
        logger.debug(f"UCS: {UCS.resolve()}")
        with open(UCS / "clay_charges.yaml", "r") as file:
            ox_dict: dict = yaml.safe_load(file)
        return ox_dict


class TargetClayComposition:
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
        self.__ion_df: pd.DataFrame = None
        self.set_ion_numbers()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name!r})"

    def __str__(self):
        def __repr__(self):
            return f"{self.__class__.__name__}({self.name!r})"

    def update_charges(
        self, df: Union[pd.DataFrame, pd.Series]
    ) -> Union[pd.DataFrame, pd.Series]:
        charges = pd.Series(self.get_charges(df), name=self.name)
        charges = charges.reindex_like(df.xs("C"))
        df.loc["C"].update(charges)
        return df

    @cached_property
    def clayff_at_types(self):
        clayff_at_types = YAMLFile(UCS / "clay_at_types.yaml")
        return clayff_at_types.data

    @cached_property
    def clayff_elements(self):
        reverse_at_types = {v: k for (k, v) in self.clayff_at_types.items()}
        return reverse_at_types

    def clayff_to_element(self, at_type):
        return self.clayff_elements[at_type]

    def __get_match_df(self, csv_file):
        match_df = self.__read_match_df(csv_file)
        match_df = pd.DataFrame(match_df[self.name], columns=[self.name])
        match_df = self.update_charges(match_df)
        match_df.dropna(inplace=True, how="all")
        # with open(UCS / "clay_at_types.yaml", "r") as file:
        #     clayff_at_types: dict = yaml.safe_load(file)
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
                new_charge: pd.Series = at_charges.multiply(
                    match_occs
                )  # - ((self.uc_data.occupancies[group] - match_df.loc[series.index, self.name].sum())
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
                    new_charge = get_checked_input(
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
        return self._df.dropna().sort_index(
            ascending=False, level="sheet", sort_remaining=True
        )

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
    def __get_sheet_df(self, sheet_df):  # , idx_sel=["T", "O"]):
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
        sheet_df,
        # idx_sel=["T", "O"],
        priority: Optional[
            Union[Literal["charges"], Literal["occupancies"]]
        ] = None,
    ):
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
                non_charged_size = (
                    charged_dict["non-charged"].loc[at_type_duplicate[0]].size
                )
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
            # charged_size = len(charged_non_zero)
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
                # if self.occupancies != self.uc_data.occupancies:
                #     new_uncharged_occs = (
                #         charged_dict["non-charged"]["at-type"]
                #         .groupby("sheet")
                #         .apply(lambda x: x + charged_occ_check[x.name] / x.size)
                #     )
                # sheet_df.update(new_uncharged_occs)
                # self.print_df_composition(
                #     old_composition=self.df.loc[self.idx_sel],
                #     sheet_df=sheet_df,
                # )
                # self._df.loc[sheet_df.index] = sheet_df.loc[sheet_df.index]
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
    ):  # , idx_sel=["T", "O"]):
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
            # self.occ_correction_df
            # self.non_charged_sheet_df(idx_sel=idx_sel)
            # .dropna()["at-type"]
            # .groupby("sheet").apply(
            #     lambda x: x - check_occ.at[x.name] / x.count()
            # )
            # )
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
            f"\nWriting new target clay composition to {outpath.name!r}\n",
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
        match_df: pd.DataFrame = pd.read_csv(csv_file)
        match_df["sheet"].ffill(inplace=True)
        match_cols = match_df.columns.values
        match_cols[:2] = self.uc_df.index.names
        match_df.columns = match_cols
        match_df.set_index(self.uc_df.index.names, inplace=True)
        match_df.sort_index(level="sheet", sort_remaining=True, inplace=True)
        return match_df


class ClayComposition(ABC):
    def __init__(self, sheet_n_ucs: int, uc_data: UCData, name: str):
        self._uc_data = uc_data
        self.sheet_n_ucs = int(sheet_n_ucs)
        self.name = name

    def __str__(self):
        return f"{self.__class__.__name__}({self.name!r})"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name!r})"

    @cached_property
    def match_charge(self):
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

    @cached_property
    @abstractmethod
    def _unique_uc_match_dict(self) -> dict:
        pass

    @cached_property
    @abstractmethod
    def _uc_match_dict(self):
        pass

    def uc_dimensions(self):
        return self._uc_data.dimensions

    @staticmethod
    def print_groups(group_dict, uc_df, fill=""):
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
                f"{uc_id:02d}"
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

        # self.df = pd.DataFrame.from_dict(uc_index_ratios)
        # for k, v in uc_index_ratios.items():
        #     logger.info(f"\t{k}")

    @cached_property
    def _uc_match_dict(self) -> dict:
        return self._unique_uc_match_dict

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
                ["n_ucs", "uc_ids", "uc_weights", "composition", "dist"],
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
        logger.finfo(
            f"Selected combination has {match_dict['n_ucs']} unique unit cells "
            f'(total occupancy deviation {match_dict["dist"]:+.4f})',
            initial_linebreak=True,
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
                uc_id = f"{uc_num:02d}"
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
            self._uc_data.check_ucs()


class MatchClayComposition(ClayComposition):
    def __init__(
        self,
        target_composition: TargetClayComposition,
        sheet_n_ucs: int,
        ignore_threshold: float = 0.0,
        manual_setup: bool = True,
        debug_run: bool = False,
    ):
        super().__init__(
            sheet_n_ucs=sheet_n_ucs,
            uc_data=target_composition.uc_data,
            name=target_composition.name,
        )
        self.debug_run = debug_run
        logger.info(get_subheader("Selecting unit cells"))
        self.manual_setup = manual_setup
        self.ignore_threshold = ignore_threshold
        self.target_comp: TargetClayComposition = target_composition
        self.__target_df: pd.DataFrame = target_composition.clay_df
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

    def drop_unused_ucs(self):
        all_ucs_df = self._uc_data.df
        target_df = self.__target_df.dropna().copy()  # .reset_index('sheet')
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
                    :, (uc_group_df[unused_target_atype_mask] == 0).any(axis=0)
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
                # combined_mask = np.logical_and(unused_uc_atype_mask, unused_target_atype_mask)
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
                base_df = base_df.where(base_df != 0, np.NaN).dropna(how="all")
                # base_df.append(base_df[[1,2]].apply()
                diff_df = (
                    base_df[[0, 1]]
                    .apply(lambda x: np.abs(x - base_df["target"]))
                    .sum()
                )
                selected_group_id = diff_df[
                    diff_df == diff_df.min()
                ].index.to_list()[0]
                # selected_group = accepted_group[selected_group_id]
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
                    # self.print_groups(accepted_group, all_ucs_df, fill="\t")
                    # self.print_groups(accepted_group, all_ucs_df)
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
                        # self.__target_df.dropna(inplace=True)
                        # print(self.target_df)

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
            self._uc_data.check_ucs()

    @staticmethod
    def __abort(reason: Literal[Union[Literal["uc"], Literal["comp"]]]):
        reason_dict = {
            "uc": "No unit cell group accepted,",
            "comp": "Clay composition not accepted.",
        }
        logger.finfo(f"{reason_dict[reason]} Aborting model construction.")
        sys.exit(0)

    @staticmethod
    def __get_nan_xor_zero_mask(df):
        return np.logical_xor(np.isnan(df), df == 0)

    @property
    def uc_df(self):
        return self._uc_data.df.reindex(
            index=self.target_df.index
        )  # .sort_index(ascending=False, level='sheet', sort_remaining=True).dropna()

    @property
    def target_df(self):
        return self.__target_df.sort_index(
            ascending=False, level="sheet", sort_remaining=True
        ).dropna()

    @property
    def max_n_uc(self) -> int:
        if not self.debug_run:
            n_uc_max = len(self.unique_uc_array) + 1
        else:
            n_uc_max = 3
            logger.info(get_debugheader(f"max n_ucs = {n_uc_max - 1}"))
        return n_uc_max

    @cached_property
    def _unique_uc_match_dict(self) -> dict:
        uc_df = self.uc_df.copy()
        uc_df.columns = uc_df.columns.astype(int)
        match_dict = dict(
            map(
                lambda k: (k, np.NaN),
                ["n_ucs", "uc_ids", "uc_weights", "composition", "dist"],
            )
        )
        min_dist = np.inf
        logger.info(
            get_subheader(
                "Getting matching unit cell combination for target composition"
            )
        )

        for n_ucs in range(2, self.max_n_uc):
            logger.finfo(
                f"Getting combinations for {n_ucs} unique unit cells",
                initial_linebreak=True,
            )
            uc_id_combinations = self.get_uc_combinations(n_ucs)
            occ_combinations = self.get_sheet_uc_weights(n_ucs)
            for uc_ids in uc_id_combinations:
                atype_weights = self.get_atype_weights(
                    uc_df=uc_df, uc_ids=uc_ids, uc_weights=occ_combinations
                )
                diff_array = np.subtract(
                    atype_weights.T, np.squeeze(self.target_df.values)
                )
                diff_array = np.linalg.norm(
                    diff_array.astype(np.float128), axis=1
                )
                dist = np.amin(diff_array)
                match_idx = np.argwhere(np.equal(diff_array, dist))
                if min_dist > dist:
                    match_dict.update(
                        {
                            "n_ucs": n_ucs,
                            "uc_ids": uc_ids,
                            "uc_weights": np.squeeze(
                                occ_combinations[match_idx]
                            ),
                            "composition": np.squeeze(
                                np.round(atype_weights.T[match_idx], 4)
                            ),
                            "dist": np.round(dist, 4),
                        }
                    )
                    min_dist = dist
        logger.finfo(
            f"Best match found with {match_dict['n_ucs']} unique unit cells "
            f'(total occupancy deviation {match_dict["dist"]:+.4f})\n',
            initial_linebreak=True,
        )
        return match_dict

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
        return duplicate_ucs

    @staticmethod
    def __select_duplicate_ucs(id_sel, duplicate_ids) -> List[str]:
        return list(id_sel[0] + i for i in range(1, len(duplicate_ids)))

    @cached_property
    def _uc_match_dict(self) -> dict:
        uc_match_dict = self._unique_uc_match_dict.copy()
        unique_uc_match_ids = np.array(uc_match_dict["uc_ids"]).astype(int)
        unique_uc_occs = uc_match_dict["uc_weights"]
        for unique_id, duplicate_ids in self.duplicate_ucs.items():
            if np.in1d(unique_id, unique_uc_match_ids).any():
                _, _, sel_ids = np.intersect1d(
                    duplicate_ids,
                    unique_uc_match_ids,
                    return_indices=True,
                    assume_unique=True,
                )
                uc_match_dict["n_ucs"] += 1
                new_occs = np.full(len(duplicate_ids), unique_uc_occs[sel_ids])
                if unique_id == unique_uc_match_ids[-1]:
                    unique_uc_match_ids = np.append(
                        unique_uc_match_ids,
                        duplicate_ids[(duplicate_ids != unique_id)],
                    )
                    unique_uc_occs = np.append(unique_uc_occs, new_occs[1:])
                else:
                    unique_uc_match_ids = np.insert(
                        unique_uc_match_ids,
                        self.__select_duplicate_ucs(sel_ids, duplicate_ids),
                        duplicate_ids[duplicate_ids != unique_id],
                    )
                    unique_uc_occs = np.insert(
                        unique_uc_occs,
                        self.__select_duplicate_ucs(sel_ids, duplicate_ids),
                        new_occs[1:],
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
        return uc_match_dict

    @cached_property
    def unique_uc_array(self) -> NDArray[int]:
        unique_uc_ids = self.uc_df.T.drop_duplicates().index.values.astype(int)
        return unique_uc_ids

    def get_sheet_uc_weights(self, n_ucs: int):
        """
        Returns a list of lists with the combinations of 2, 3, sheet_n_ucs given the
        columns.
        """
        sheet_uc_combinations = np.asarray(
            [x for x in self.get_uc_combination_list(self.sheet_n_ucs, n_ucs)]
        )
        return sheet_uc_combinations

    @staticmethod
    def get_uc_combination_list(N, k):
        n_combs = int(
            math.factorial(N - 1)
            / (math.factorial(k - 1) * math.factorial(N - k))
        )
        for q in tqdm(
            itertools.combinations(range(N - 1), k - 1), total=n_combs
        ):
            yield [j - i for i, j in zip((-1,) + q, q + (N - 1,))]

    def get_uc_combinations(self, n_ucs):
        return itertools.combinations(self.unique_uc_array, n_ucs)

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
            f"Writing new match clay composition to {str(outpath)!r}\n",
            initial_linebreak=True,
        )
        if backup:
            backup_files(new_filename=outpath, old_filename=tmpfile.name)
        else:
            shutil.copy(tmpfile.name, outpath)
        assert outpath.is_file()

    write_csv = partialmethod(_write, fmt=".csv")
    write_pkl = partialmethod(_write, fmt=".csv")


class InterlayerIons:
    def __init__(
        self,
        tot_charge: int,
        ion_ratios: pd.DataFrame,
        n_ucs: int,
        monovalent=["Na", "Cl"],
    ):
        self.clay_charge = np.rint(tot_charge * n_ucs)
        ion_charges = get_ion_charges()
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
            np.sign(ion_ratios["charges"]) != tot_charge
        ].dropna(how="all")
        if ion_ratios["probs"].dropna().empty:
            ion_ratios["probs"].fillna(1.0, inplace=True)
        self.__df = ion_ratios.copy()
        if monovalent not in self.__df.index:
            mono_df = pd.DataFrame(
                [[0, charge]],
                columns=self.__df.columns,
                index=pd.Index([monovalent], name=self.__df.index.name),
            )

            self.__df = pd.concat([self.__df, mono_df])
        if not np.isclose(self.__df["probs"].sum(), 1.00):
            self.__df["probs"] = self.__df["probs"].divide(
                np.sum(self.__df["probs"])
            )
        assert np.isclose(
            self.__df["probs"].sum(), 1.00
        ), "Interlayer ion probabilities do not sum to 1.00"
        self.__df["numbers"] = 0
        self.get_ion_numbers(monovalent=monovalent)

    def get_ion_numbers(self, monovalent):
        df = self.__df[~np.isclose(self.__df["charges"], 0.00)]
        avg_charge = df.loc[:, ["probs", "charges"]].prod(axis=1).sum()
        multiplication_factor = np.abs(np.divide(self.clay_charge, avg_charge))
        df["numbers"] = np.rint(
            np.multiply(df["probs"], multiplication_factor)
        )
        ion_charge = (df["numbers"] * df["charges"]).sum()
        if not np.isclose(-ion_charge, self.clay_charge):
            df.loc[monovalent, "numbers"] += np.abs(self.clay_charge) - np.abs(
                ion_charge
            )
        df.dropna(subset=["charges", "numbers"], inplace=True)
        df = df.convert_dtypes()
        self.__df = df

    @property
    def df(self):
        return self.__df[["charges", "numbers"]]

    @property
    def numbers(self):
        return self.df["numbers"].dropna().apply(np.rint).astype(int).to_dict()


class BulkIons:
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
