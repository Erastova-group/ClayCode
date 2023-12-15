#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from __future__ import annotations

import copy
import itertools
import logging
import os
import re
import shutil
from collections import UserDict, UserList
from functools import cached_property
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import nglview
import numpy as np
import pandas as pd
from caseless_dictionary import CaselessDict
from ClayCode.builder.claycomp import UCData
from ClayCode.builder.utils import select_input_option
from ClayCode.core.cctypes import PathType
from ClayCode.core.classes import (
    CSVFile,
    Dir,
    File,
    ForceField,
    GROFile,
    ITPFile,
    YAMLFile,
)
from ClayCode.core.lib import get_system_charges
from ClayCode.data.consts import (
    ALL_UCS,
    CLAY_FF,
    CLAYFF_AT_CHARGES,
    CLAYFF_AT_TYPES,
    ITP_KWDS,
    UC_CHARGE_OCC,
    UCS,
    USER_CHARGE_OCC,
    USER_UCS,
)
from MDAnalysis.topology import tables
from MDAnalysis.topology.guessers import guess_bonds

__all__ = ["UCData"]

logger = logging.getLogger(__name__)

CLAY_ATYPES = YAMLFile(CLAYFF_AT_TYPES)
CLAY_ACHARGES = YAMLFile(CLAYFF_AT_CHARGES)


class UCWriter:
    """Class to write substituted unit cell coordinates and topologies from a GRO file.
    :param gro_file: GRO file to be used as base for unit cell generation.
    :type gro_file: Union[PathType, str]
    :param uc_type: Name of the unit cell type.
    :type uc_type: str
    :param odir: Output directory for unit cell files.
    :type odir: Optional[Union[PathType, str]]
    :param ff: Force field to be used for unit cell generation.
    :type ff: Union[PathType, str]
    """

    sheet_atype_regex_dict = {
        "T": r"[A-Z]*T[0-9]",
        "O": r"[AFLMC]*[A-GI-Z][O2][0-9][0-9]",
        "ob": r"O[XB]*[0-9][0-9]",
        "oh": r"OH*[0-9]",
        "ho": r"HO*[0-9]",
    }
    at_charges_dict = {
        atype[:3].upper(): charge
        for atype, charge in CLAY_ACHARGES.data.items()
    }

    def __init__(
        self,
        gro_file: Union[PathType, str],
        uc_type: str,
        odir: Optional[Union[PathType, str]],
        ff: Union[PathType, str] = CLAY_FF,
        default_solv: Optional[bool] = None,
        substitutions: Optional[Dict[str, Union[str, Dict[str, str]]]] = None,
    ):
        if odir is None:
            odir = UCS
        self.__gro = GROFile(gro_file)
        self._gro = copy.deepcopy(self.__gro)
        self.path = Dir(odir) / uc_type
        self.name = uc_type
        self.ff = ForceField(ff)
        self.default_solv = default_solv
        self.substitutions = substitutions
        self.single_substituted = {"T": [], "O": []}
        self.double_substituted = {"T": [], "O": []}
        self._uc_id = None
        self._substitutable = {}
        try:
            self.init_uc_folder()
        except FileNotFoundError:
            pass

    @property
    def gro(self):
        return self._gro

    @gro.setter
    def gro(self, new_gro_name):
        self._gro = GROFile(new_gro_name)

    @cached_property
    def clay_atom_names_dict(self) -> Dict[str, Dict[str, str]]:
        """Dictionary of clay atom types."""
        atom_names_dict = {
            sheet: {
                at_name: atype
                for at_name, atype in CLAY_ATYPES.data[sheet].items()
            }
            for sheet in CLAY_ATYPES.data
        }
        atom_names_dict["O"]["Fe"] = "feo"
        return atom_names_dict

    def _init_path(self) -> None:
        """Create the output directory if it does not exist."""
        os.makedirs(self.path, exist_ok=True)

    def add_new_uc(
        self, uc_name: str, universe
    ) -> Union[Tuple[GROFile, ITPFile], Tuple[None, None]]:
        """Add a new unit cell to the database.
        :param uc_name: Name of the unit cell type.
        :type uc_name: str
        :param universe: Unit cell universe.
        :type universe: MDAnalysis.Universe
        """
        new_gro = self._get_gro(uc_name, universe)
        new_itp = self._get_itp(uc_name)
        if new_itp is None:
            new_gro = None
        return new_gro, new_itp

    def write_new_uc(
        self, uc_name: str, default_solv_name: Optional[str] = None
    ) -> None:
        """Write a new unit cell to the database.
        :param uc_name: Name of the unit cell type.
        :type uc_name: str
        :param default_solv: Default solvation of the unit cell type.
        :type default_solv: Optional[bool]
        """
        self._init_path()
        self._get_gro(uc_name)
        self.init_uc_folder()
        self._get_itp(uc_name)
        uc_data = UCData(self.path, ff=self.ff, write=False, reset=True)
        occ = uc_data.occupancies
        charge = uc_data.oxidation_numbers
        all_charge_occ_df = UC_CHARGE_OCC
        user_charge_occ_df = USER_CHARGE_OCC
        # user_charge_occ_df = charge_occ_df
        self.default_solv = select_input_option(
            instance_or_manual_setup=True,
            query="Set the default solvation for this unit cell type to 'True'? [y]es/[n]o\n",
            options=["y", "n", True, False],
            result=self.default_solv,
            result_map={"y": True, "n": False, True: True, False: False},
        )
        if self.name in all_charge_occ_df.index.get_level_values("value"):
            uc_specs = all_charge_occ_df.xs(self.name, level="value", axis=0)
            try:
                assert (
                    uc_specs.loc[["T", "O"], "occ"] == [occ["T"], occ["O"]]
                ).all(), (
                    "Specified occupancies do not match unit cell occupancies"
                )
                assert (
                    uc_specs.loc[["T", "O"], "charge"]
                    == [charge["T"], charge["O"]]
                ).all(), "Specified oxidation numbers do not match unit cell oxidation numbers"
                assert (
                    uc_specs.loc[["T", "O"], "solv"] == self.default_solv
                ).all(), "Specified default solvation does not match selected unit cell default solvation"
            except KeyError:
                assert (
                    uc_specs.loc[["O"], "occ"] == occ["O"]
                ).all(), (
                    "Specified occupancies do not match unit cell occupancies"
                )
                assert (
                    uc_specs.loc[["O"], "charge"] == charge["O"]
                ).all(), "Specified oxidation numbers do not match unit cell oxidation numbers"
                assert (
                    uc_specs.loc[["O"], "solv"] == self.default_solv
                ).all(), "Specified default solvation does not match selected unit cell default solvation"
        else:
            new_entry = pd.DataFrame(
                index=pd.MultiIndex.from_product(
                    [
                        all_charge_occ_df.index.get_level_values(
                            "sheet"
                        ).unique(),
                        [self.name],
                    ],
                    names=all_charge_occ_df.index.names,
                ),
                columns=all_charge_occ_df.columns,
            )
            try:
                new_entry.loc[pd.IndexSlice[["T", "O"], self.name], "occ"] = [
                    occ["T"],
                    occ["O"],
                ]
                new_entry.loc[
                    pd.IndexSlice[["T", "O"], self.name], "charge"
                ] = [
                    charge["T"],
                    charge["O"],
                ]
                new_entry.loc[
                    pd.IndexSlice[["T", "O"], self.name], "solv"
                ] = self.default_solv
            except KeyError:
                new_entry.loc[pd.IndexSlice[["O"], self.name], "occ"] = [
                    occ["O"]
                ]
                new_entry.loc[pd.IndexSlice[["O"], self.name], "charge"] = [
                    charge["O"]
                ]
                new_entry.loc[
                    pd.IndexSlice[["O"], self.name], "solv"
                ] = self.default_solv
            try:
                user_charge_occ_df = user_charge_occ_df.append(new_entry)
            except AttributeError:
                user_charge_occ_df = new_entry
            logger.finfo(
                f"Adding new {self.name} unit cell specifications to database."
            )
            user_charge_occ_df.convert_dtypes().reset_index().to_csv(
                USER_UCS / "charge_occ.csv", index=False
            )

    def _get_gro(self, uc_name: str, universe=None) -> GROFile:
        """Write GRO file.
        :param uc_name: Name of the unit cell type.
        :type uc_name: str
        :param universe: Unit cell universe.
        :type universe: MDAnalysis.Universe
        """
        if universe is None:
            u = self._gro.universe
        else:
            u = universe
        resnames = np.unique(u.residues.resnames)
        if not (len(resnames) == 1 and uc_name in resnames):
            for res in u.residues:
                res.resname = f"{uc_name}"
        gro_file = self.path / f"{uc_name}.gro"
        gro = GROFile(gro_file)
        gro.universe = u
        gro.write()
        self._gro = gro
        return gro

    def _get_itp(
        self,
        uc_name: str,
        max_charge=5,  # , default_itp: Union[PathType, str] = DEFAULT_ITP
    ) -> Union[ITPFile, None]:
        """Write the ITP file that corresponds to the gro file of the same name.
        Uses force field parameters from :py:data::`ClayCode.data.FF.ClayFF_Fe` database.
        Saves the final file into :py:attr::`UCWriter.odir`
        :param uc_name: Name of the unit cell type.
        :type uc_name: str
        """
        n_atoms = self.gro.n_atoms
        itp_file = self.path / f"{uc_name}.itp"
        if itp_file.is_file():
            os.remove(itp_file)
        itp_file.touch()
        itp_file = ITPFile(itp_file, check=False)
        # itp_file.string = "[ moleculetype ]\n" "[ atoms ]\n" "[ bonds ]\n"
        # itp_file.write()
        itp_file["moleculetype"] = re.sub(
            r"NAME", uc_name, "NAME\t1", flags=re.MULTILINE
        )
        uc_id = self.get_id(uc_name)
        gro_df = self.get_uc_gro(uc_id).df
        itp_kwds = copy.deepcopy(ITP_KWDS)
        atom_cols = itp_kwds["atoms"]
        atom_cols.remove("id")
        itp_atom_df = pd.DataFrame(
            index=pd.Index(gro_df["atom-id"], name="id"), columns=atom_cols
        )
        itp_atom_df["at-name"] = gro_df["at-type"].values
        itp_atom_df["res-name"] = gro_df.index.values
        itp_atom_df["res-name"] = itp_atom_df["res-name"].apply(
            lambda x: re.sub(r"\d(?=[A-Z])", "", x)
        )
        itp_atom_df["res-number"] = 1
        itp_atom_df["charge-nr"] = itp_atom_df.index.values
        search_pattern = "|".join(self.at_charges_dict.keys())
        itp_atom_df["at-type"] = (
            itp_atom_df["at-name"]
            .apply(
                lambda x: re.sub(
                    rf"({search_pattern})[0-9]+", r"\1", x
                ).lower()
            )
            .values
        )
        itp_atom_df["at-type"] = itp_atom_df["at-type"].apply(
            lambda x: f"{x}s" if re.match("o[bx][tos]", x) else x
        )
        ff_df = self.ff["atomtypes"].df.set_index("at-type")
        for prm in ["charge", "mass"]:
            try:
                itp_atom_df[prm] = itp_atom_df["at-type"].apply(
                    lambda x: ff_df.loc[x][prm]
                )
            except KeyError as e:
                logger.debug(e)
        tot_charge = itp_atom_df["charge"].sum()
        if abs(tot_charge) > abs(max_charge) + 1e-1:
            logger.finfo(
                f"Total absolute charge of {abs(tot_charge):.3f} is greater than {abs(max_charge)}"
            )
            os.remove(self.gro)
            os.remove(itp_file)
            self.uc_id = self._uc_id
            return None
        charge_diff = np.float32(tot_charge - np.rint(tot_charge))
        charge_masks = itertools.cycle(
            [
                "o[hbx]",
                "o[bx]",
                "ohs",
                "ob",
                "oh",
                "obts",
                "obos",
                "oxts",
                "oxos",
                "ho",
                "feo|ao",
                "mgo|fe2|cao",
                "lio",
                "at|fet",
                "st",
            ]
        )
        while not np.isclose(charge_diff, 0.0, atol=1e-10):
            logger.debug(f"Total charge of {tot_charge:.3f} is not an integer")
            charge_mask = next(charge_masks)
            new_charge_mask = (
                itp_atom_df["at-type"]
                .apply(lambda x: x if re.match(charge_mask, x) else pd.NA)
                .dropna()
            )
            n_ob: pd.Series = new_charge_mask.value_counts().sum()
            new_charges = itp_atom_df.loc[new_charge_mask.index, "charge"]
            new_charges = new_charges.apply(
                lambda x: x - (charge_diff / n_ob)
            ).round(6)
            itp_atom_df["charge"].update(new_charges)
            logger.debug(
                f"Corrected total charge to {itp_atom_df['charge'].sum():.3f}"
            )
            tot_charge = itp_atom_df["charge"].sum()
            charge_diff = np.float32(tot_charge - np.rint(tot_charge))
        itp_file["atoms"] = itp_atom_df
        bond_df = (
            itp_atom_df["at-type"]
            .apply(lambda x: x if re.match("oh[xs]*", x) else pd.NA)
            .dropna()
        )
        n_bonds = bond_df.value_counts()[0]
        bond_df = itp_atom_df.loc[bond_df.index]
        itp_bond_df = pd.DataFrame(columns=ITP_KWDS["bonds"])
        at_name_pattern = re.compile(r"(OH[SX]*)([0-9]+)", flags=re.IGNORECASE)
        ff_df = self.ff["bondtypes"].df.set_index("ai")
        for at_id, oh in bond_df.iterrows():
            at_type = oh["at-type"]
            o_match = at_name_pattern.match(oh["at-name"])
            at_name, o_id = o_match.group(1), o_match.group(2)
            ho = itp_atom_df.loc[itp_atom_df["at-name"] == f"HO{o_id}"]
            h_id = ho.index.values[0]
            bond_prms = ff_df.loc[at_type]
            itp_bond_df.loc[at_id] = [
                at_id,
                h_id,
                1,
                bond_prms["b0"],
                bond_prms["kb"],
            ]
        itp_bond_df = itp_bond_df.convert_dtypes()
        itp_bond_df.set_index("ai", inplace=True)
        itp_file["bonds"] = itp_bond_df
        self.itp = itp_file
        self.itp.write()
        return itp_file

    def _substitute(self, base_id: Optional[str] = None) -> None:
        if base_id is None:
            base_id = self.base_id
        for k, v in self.substitutions.items():
            v = self._split_sub_str(v)
            if isinstance(v, list):
                for v1 in v:
                    self.add_substitution(
                        base_uc_id=base_id, sub_sheet=k, substitution=v1
                    )
            else:
                self.add_substitution(
                    base_uc_id=base_id, sub_sheet=k, substitution=v
                )

    def add_single_substitutions(self, base_id: Optional[str] = None) -> None:
        """Generate new unit cell by adding substitutions to base unit cell."""
        if base_id is None:
            base_id = self.base_id
        for k, v in self.substitutions.items():
            v = self._split_sub_str(v)
            if isinstance(v, list):
                for v1 in v:
                    self.add_substitution(
                        base_uc_id=base_id, sub_sheet=k, substitution=v1
                    )
            else:
                self.add_substitution(
                    base_uc_id=base_id, sub_sheet=k, substitution=v
                )
        # uc_df = UCData(self.path, uc_stem=self.uc_stem, ff=self.ff, write=False).df

        # self._substitute()
        #     self.single_substituted[self.base_id].append(substitution)
        # for k, v in self.substitutions.items():
        #     v = self._split_sub_str(v)
        #     if isinstance(v, list):
        #         for v1 in v:
        #             self.single_substituted[k].append(self.add_substitution(base_uc_id=self.base_id, sub_sheet=k, substitution=v1))
        #     else:
        #         self.single_substituted[k].append(self.add_substitution(base_uc_id=self.base_id, sub_sheet=k, substitution=v))

    def add_double_substitutions(self) -> None:
        logger.finfo(f"Adding double substitutions to {self.name}")
        substituted_ids = self.sub_info.loc[
            self.sub_info["n_substitutions"].astype(int) == 1, "uc_id"
        ]
        for uc_id in substituted_ids:
            self.add_single_substitutions(base_id=uc_id)

    def add_triple_substitutions(self) -> None:
        logger.finfo(f"Adding triple substitutions to {self.name}")
        substituted_ids = self.sub_info.loc[
            self.sub_info["n_substitutions"].astype(int) == 2, "uc_id"
        ]
        for uc_id in substituted_ids:
            self.add_single_substitutions(base_id=uc_id)

    @staticmethod
    def _split_sub_str(sub_str: str) -> List[Tuple[str, str]]:
        """Split a '->' separated string into a tuple of two strings.
        :param sub_str: string to split
        :type sub_str: str
        :return: tuple of two strings
        :return type: List[Tuple[str, str]]
        """
        sub_str = re.sub(r"\s*->\s*", "->", sub_str).strip(" ").split(" ")
        return [tuple(s.split("->")) for s in sub_str]

    @property
    def sub_info(
        self,
    ) -> Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, str]]]]]:
        """Information about unit cell substitutions.
        Info format:
            - Index of the unit cell.
            - Sheet type to be substituted ([T]etrahedral, [O]ctahedral).
            - Substituted atoms
            - Unit cell charge
            - Location of the T substitution ([t]op, [b]ottom).
        """
        # self._info_header = "# uc_id\tsubstitution_sheet\tsubstitution_loc\tsubstitution\tcharge\n"
        if not hasattr(self, "_info_file"):
            info_file = CSVFile(self.path / "info.csv", check=False)
            if info_file.is_file():
                pass
            else:
                pass
                # sub_info = pd.DataFrame(columns=['uc_id', 'substitution_sheet', 'substitution_loc', 'substitution', 'charge'])
                # sub_info.set_index(['uc_id', 'substitution_sheet', 'substitution'], inplace=True)
            self._info_file: CSVFile = info_file
        return self._info_file.data

    def add_sub_info(
        self,
        uc_id,
        sub_sheet,
        sub_atoms,
        sub_charge,
        sheet_loc=None,
        max_charge=3,
    ):
        info_dict = self.sub_info
        if info_dict is None:
            info_dict = pd.DataFrame(
                columns=[
                    "uc_id",
                    "base_id",
                    "substitution_sheet",
                    "substitutions",
                    "substitution_loc",
                    "charge",
                    "n_substitutions",
                ],
                dtype=str,
            )
        # if sheet_loc is not None:
        parent_id, new_id = uc_id.split(" -> ")
        info_dict.loc[len(info_dict)] = [
            new_id,
            parent_id,
            sub_sheet,
            sub_atoms,
            sheet_loc,
            f"{sub_charge:.1f}",
            None,
        ]
        n_subs = 1
        while parent_id != self.base_id:
            old_id = parent_id
            subs = info_dict.loc[len(info_dict) - 1, "substitutions"]
            parent_subs = info_dict.loc[
                info_dict["uc_id"] == old_id, "substitutions"
            ].values[0]
            new_subs = parent_subs.split(";")
            n_subs = len(new_subs)
            uc_charge = f'{(float(info_dict.loc[len(info_dict) - 1, "charge"]) + info_dict.loc[info_dict["uc_id"] == old_id, "charge"].astype(float).values[0]):.1f}'
            info_dict.loc[len(info_dict) - 1, "charge"] = uc_charge
            new_subs.append(subs)
            subs = ";".join(sorted(new_subs))
            info_dict.loc[len(info_dict) - 1, "substitutions"] = subs
            parent_id = info_dict[info_dict["uc_id"] == old_id][
                "base_id"
            ].values[0]
            info_dict.loc[len(info_dict) - 1, "base_id"] = parent_id
            n_subs += 1
        info_dict.loc[len(info_dict) - 1, "n_substitutions"] = n_subs
        # print(n_subs, len(np.unique([info.split(' -> ') for info in info_dict.loc[len(info_dict) - 1, "substitutions"].split(";")])))
        if (
            info_dict.duplicated(
                subset=[
                    "substitutions",
                ]
            ).any()
            or abs(float(info_dict.loc[len(info_dict) - 1, "charge"]))
            > max_charge + 0.1
            or len(
                np.unique(
                    [
                        info.split(" -> ")
                        for info in info_dict.loc[
                            len(info_dict) - 1, "substitutions"
                        ].split(";")
                    ]
                )
            )
            != 2 * n_subs
        ):
            logger.debug(
                f"new unit cell is duplicate or charge is too big.\nRemoving last uc {new_id}."
            )
            # os.remove(self.gro)
            # os.remove(self.itp)
            info_dict.drop(len(info_dict) - 1, inplace=True)
            self.uc_id = self._uc_id
        else:
            #         info_dict[uc_id][sub_sheet][sub_atoms][sub_charge] = sheet_loc
            #     except KeyError:
            #         try:
            #             info_dict[uc_id][sub_sheet][sub_atoms] = {sub_charge: sheet_loc}
            #         except KeyError:
            #             try:
            #                 info_dict[uc_id][sub_sheet] = {sub_atoms: {sub_charge: sheet_loc}}
            #             except KeyError:
            #                 info_dict[uc_id] = {sub_sheet: {sub_atoms: {sub_charge: sheet_loc}}}
            # else:
            #     try:
            #         info_dict[uc_id][sub_sheet][sub_atoms] = sub_charge
            #     except KeyError:
            #         try:
            #             info_dict[uc_id][sub_sheet] = {sub_atoms: sub_charge}
            #         except KeyError:
            #             info_dict[uc_id] = {sub_sheet: {sub_atoms: sub_charge}}
            self._info_file.data = info_dict
            logger.debug(
                f"Added substitution info for {uc_id}, {sub_sheet}, {sub_atoms}, {sheet_loc}, {sub_charge:.7f}, {n_subs}"
            )
            logger.finfo(
                f"Wrote new unit cell substitution of {self.uc_stem}{self.uc_id} to {self.uc_stem}{uc_id.split(' -> ')[1]} with {sub_atoms} (charge = {sub_charge:.2f})"
            )
            return uc_id

    def add_substitution(
        self,
        base_uc_id: str,
        sub_sheet: Union[Literal["T"], Literal["O"]],
        substitution: Tuple[str, str],
        uc_id: Optional[int] = None,
    ):
        """
        Add isomorphic substitutions to base_uc unit cell. Replaces sub_dict keys by :py:obj::`sub_dict` values.
        Saves the final file into :py:attr::`UCWriter.odir`.
        If given, the final UC will have index :py:object::`uc_id`.
        Otherwise, the new index will be guessed from existing unit cells in :py:attr::`UCWriter.odir`.
        :param base_uc_id: Name of the base unit cell type.
        :type base_uc_id: str
        :param sub_sheet: Sheet type to be substituted.
        :type sub_sheet: Union[Literal["T"], Literal["O"]]
        :param substitution: Dictionary of substitutions to be made.
        :type substitution: Dict[str, str]
        :param uc_id: Index of the new unit cell.
        :type uc_id: Optional[int]
        """
        substitution = list(substitution)
        for atom_id, sub_atom in enumerate(substitution):
            if sub_atom not in self.clay_atom_names_dict[sub_sheet].values():
                try:
                    substitution[atom_id] = self.clay_atom_names_dict[
                        sub_sheet
                    ][sub_atom]
                except KeyError:
                    logger.finfo(
                        f"Invalid atom type {sub_atom}!\n"
                        f"Valid atom names for {sub_sheet} are: "
                        + ", ".join(
                            [
                                *CLAY_ATYPES[sub_sheet].keys(),
                                *CLAY_ATYPES[sub_sheet].values(),
                            ]
                        )
                    )
        self.write_single_substituted_ucs(
            sub_dict={substitution[0].upper(): substitution[1].upper()},
            uc_id=base_uc_id,
        )

    @property
    def uc_id(self):
        if self._uc_id is None:
            self._uc_id = self.base_id
        logger.debug(f"Setting uc id to {self._uc_id}")
        return self._uc_id

    @uc_id.setter
    def uc_id(self, uc_id: Union[int, str]):
        self._uc_id = f"{int(uc_id):03d}"
        self.gro = self._gro.with_stem(f"{self.uc_stem}{self.uc_id}")
        self.get_substitution_atoms(o_bonds={"T": 4, "O": 6})

    @property
    def substitutable(self):
        if not self._substitutable:
            self.get_substitution_atoms()
        return self._substitutable

    def get_substitution_atoms(self, o_bonds={"T": 4, "O": 6}):
        """Get substitutable atoms from unit cell.
        :param o_bonds: Number of bonds for oxygen atoms.
        :type o_bonds: Dict[str, int]
        """
        sheets = self.get_sheets(self.uc_id)
        subs = {"T": {}, "O": {}}
        u = sheets.pop("u")
        # vdwradii = {at_name: 1 for at_name in u.atoms.types}
        vdwradii = tables.vdwradii.copy()
        vdwradii["O"] = 2
        vdwradii.update(
            {at_name: 2 for at_name in np.unique(sheets["O"]["other"].types)}
        )
        vdwradii.update(
            {
                at_name: 1
                for at_name in u.atoms.types
                if at_name not in vdwradii
            }
        )
        sheets["O"].pop("oh")
        sheets["O"].pop("ho")
        bonds = guess_bonds(
            atoms=u.atoms,
            coords=u.atoms.positions,
            box=u.dimensions,
            vdwradii=vdwradii,
        )
        u.add_bonds(bonds)
        for sheet_type, sheet_dict in sheets.items():
            for t_sheet, atoms in sheet_dict.items():
                subs[sheet_type][t_sheet] = []
                for atom in atoms:
                    if atom.name[0] not in ["H", "O"]:
                        no_pbc = atom.bonds.bonds(pbc=False)
                        pbc = atom.bonds.bonds(pbc=True)
                        if (np.isclose(no_pbc, pbc, atol=5e-2).all()) and (
                            len(no_pbc) == o_bonds[sheet_type]
                        ):
                            bonded: list = atom.bonds.atom1 + atom.bonds.atom2
                            bonded -= atom
                            subs[sheet_type][t_sheet].append((atom, bonded))
        self._substitutable[self.uc_id] = subs

    def write_single_substituted_ucs(self, sub_dict, uc_id=None):
        """Write single substituted unit cells.
        :param sub_dict: Dictionary of substitutions to be made.
        :type sub_dict: Dict[str, str]
        :param uc_id: Index of the new unit cell.
        :type uc_id: Optional[int]
        """
        if uc_id is not None:
            self.uc_id = uc_id
        o_sub_dict = {
            "T": {
                "OB": "OBT",
                "OX": "OXO",
                "OXO": "OXO",
                "OBO": "OBS",
                "OBO": "OBS",
                "OBS": "OBS",
            },
            "O": {
                "OB": "OBO",
                "OX": "OXO",
                "OH": "OHS",
                "OHX": "OHX",
                "OBO": "OBS",
                "OXO": "OXO",
            },
        }

        if not hasattr(self, "substitutable"):
            logger.finfo(
                "Getting substitutable atoms with default number of bonds"
            )
            self.get_substitution_atoms()
        sub_atoms = self.substitutable[self.uc_id]
        search_pattern_sheet = "|".join(
            [k for k in self.at_charges_dict.keys() if k[0] not in ["O", "H"]]
        )
        search_pattern_o = "|".join(
            [k for k in self.at_charges_dict.keys() if k[0] == "O"]
        )
        name_match_pattern = re.compile(rf"({search_pattern_sheet})([0-9]+)")
        o_match_pattern = re.compile(rf"({search_pattern_o})([0-9]+)")
        sheets = {}
        base_names = self.gro.universe.atoms.names
        for sub_sheet, sheet_dict in sub_atoms.items():
            sheets[sub_sheet] = {}
            for sheet_part, sheet_part_dict in sheet_dict.items():
                sheets[sub_sheet][sheet_part] = {}
                sub_ucs = 0
                for atom, other in sheet_part_dict:
                    logger.debug(f"Trying to substitute {atom.name}")
                    atom_names = atom.universe.atoms.names.copy()
                    if atom.name != base_names[atom.ix]:
                        logger.debug(f"# Atom {atom.name} already substituted")
                        continue
                    atom_names = copy.deepcopy(atom_names)
                    at_name_match = name_match_pattern.match(atom.name)
                    try:
                        sub_name = f"{sub_dict[at_name_match.group(1)]}{at_name_match.group(2)}"
                        atom_names[atom.ix] = sub_name
                    except KeyError:
                        pass
                    else:
                        sub_charge = int(
                            -self.at_charges_dict[at_name_match.group(1)]
                            + self.at_charges_dict[
                                sub_dict[at_name_match.group(1)]
                            ]
                        )
                        if sub_charge == -1:
                            for bonded in other:
                                other_name_match = o_match_pattern.match(
                                    bonded.name
                                )
                                try:
                                    atom_names[
                                        bonded.ix
                                    ] = f"{o_sub_dict[sub_sheet][other_name_match.group(1)]}{other_name_match.group(2)}"
                                except KeyError:
                                    logger.debug(
                                        f"Not substituting {other_name_match.group(1)}.\nToo close to other substitution"
                                    )
                                    continue
                        elif sub_charge > 0:
                            raise ValueError(
                                f"Expected lower charge after substitution!"
                            )
                        sub_ucs += 1
                        sheets[sub_sheet][sheet_part][sub_ucs] = atom_names
                        last_uc = sorted(
                            self.path.glob(rf"{self.uc_stem}[0-9]*")
                        )[-1]
                        last_uc_id = self.get_id(last_uc.stem)
                        new_uc_id = f"{last_uc_id + 1:03d}"
                        new_u = atom.universe.copy()
                        new_u.atoms.names = atom_names
                        new_name = self.get_uc_gro(new_uc_id)
                        # new_gro, new_itp = self.add_new_uc(
                        #     uc_name=new_name.stem, universe=new_u
                        # )
                        added = self.add_sub_info(
                            f"{self.uc_id} -> {new_uc_id}",
                            sub_sheet,
                            f"{atom.name} -> {sub_name}",
                            sub_charge,
                            sheet_part,
                        )
                        # if new_gro is not None:
                        if added is not None:
                            new_gro, new_itp = self.add_new_uc(
                                uc_name=new_name.stem, universe=new_u
                            )
                            # charge = new_itp["atoms"].df["charge"].sum()
                            logger.debug(
                                f"Wrote new unit cell substitution of {self.uc_stem}{self.uc_id} to {self.uc_stem}{new_uc_id} with {atom.name} -> {sub_name} (charge = {sub_charge:.2f})"
                            )
                            # self.add_sub_info(
                            #     f"{self.uc_id} -> {new_uc_id}",
                            #     sub_sheet,
                            #     f"{atom.name} -> {sub_name}",
                            #     charge,
                            #     sheet_part,
                            # )

    def get_id(self, uc_stem: str) -> int:
        """Get the unit cell index from the unit cell name.
        :param uc_stem: Name of the unit cell.
        :type uc_stem: str
        :return: Index of the unit cell.
        :rtype: int
        """
        return int(re.match(f"{self.uc_stem}(\d+)", uc_stem).group(1))

    def get_uc_gro(self, uc_id: int) -> GROFile:
        """Get the unit cell GRO file from the unit cell index.
        :param uc_id: Index of the unit cell.
        :type uc_id: int
        :return: Unit cell GRO file.
        :rtype: GROFile"""
        return GROFile(self.path / f"{self.uc_stem}{int(uc_id):03d}.gro")

    def get_sheets(
        self, uc_id: Optional[int] = None
    ) -> Dict[str, Dict[str, Dict[str, GROFile]]]:
        """Get the unit cell sheets from the unit cell index.
        :param uc_id: Index of the unit cell.
        :type uc_id: Optional[int]
        :return: Dictionary of unit cell sheets.
        :rtype: Dict[str, Dict[str, Dict[str, GROFile]]]
        """
        if uc_id is None:
            u = self.gro.universe
        else:
            u = self.get_uc_gro(uc_id).universe
        t_sheet = u.select_atoms(
            f"name {self.sheet_atype_regex_dict['T']} {self.sheet_atype_regex_dict['ob']}"
        )
        o_sheet = u.atoms - t_sheet
        o_oh = u.select_atoms(f"name {self.sheet_atype_regex_dict['oh']}")
        o_ho = u.select_atoms(f"name {self.sheet_atype_regex_dict['ho']}")
        o_other = o_sheet - o_oh - o_ho
        o_pos = np.max((o_sheet - o_oh).positions[:, 2])
        t_top = t_sheet.select_atoms(f"prop z > {o_pos}")
        t_bottom = t_sheet - t_top
        return {
            "u": u,
            "T": {"t": t_top, "b": t_bottom},
            "O": {"oh": o_oh, "ho": o_ho, "other": o_other + o_oh},
        }

    def init_uc_folder(self, base_id=None):
        if base_id is None:
            uc_stem = self.gro.stem
        try:
            base_id = int(re.search(r"\d+", f"{uc_stem}").group(0)[-3:])
        except AttributeError:
            self.base_id = None
        else:
            self.base_id = f"{base_id:03d}"
            self.uc_stem = self.gro.stem[: -len(self.base_id)]

    def delete_uc_type(self):
        remove = select_input_option(
            instance_or_manual_setup=True,
            query=f"Remove {self.name} unit cell type? [y]es/[n]o (default y)\n",
            result=None,
            options=["y", "n"],
            result_map={"y": True, "n": False, "": True},
        )
        if remove:
            charge_occ_df = (
                pd.read_csv(UCS / "charge_occ.csv")
                .fillna(method="ffill")
                .set_index(["sheet", "value"])
            )
            charge_occ_df.drop(index=self.name, inplace=True, level=1)
            charge_occ_df.convert_dtypes().reset_index().to_csv(
                UCS / "charge_occ.csv", index=False
            )
            shutil.rmtree(self.path)

    def remove_substituted_ucs(self):
        substituted_ucs = ...


class UCModify:
    def __init__(self, gro_filename: PathType):
        self.universe = GROFile(gro_filename).universe

    def get_sheets(
        self,
    ) -> Tuple[Type["AtomGroup"], Type["AtomGroup"], Type["AtomGroup"]]:
        u = self.universe
        t_sheet = u.select_atoms("name [A-Z]*T* OB* OX*")
        o_oh = u.select_atoms("name HO* OH*")
        o_sheet = u.atoms - t_sheet
        o_pos = np.max((o_sheet - o_oh).positions[:, 2])
        top_t = t_sheet.select_atoms(f"prop z > {o_pos}")
        bottom_t = t_sheet - top_t
        return top_t, bottom_t, o_sheet

    def show_mda(self, ag=None, labeltype="atomname"):
        if ag is None:
            ag = self.universe.atoms
        v = nglview.show_mdanalysis(ag)
        v.add_representation(
            "label",
            color="black",
            labelType=labeltype,
            yOffset=0.1,
            xOffset=0.1,
            zOffset=0.5,
            showBackground=True,
            backgroundColor="white",
        )
        v.add_representation("unitcell")
        return v


class Substitutions(UserDict):
    def __init__(
        self, atom: Optional[str] = None, sub_atom: Optional[str] = None
    ):
        self.data = {}
        if atom is not None and sub_atom is not None:
            self.add_substitution(atom, sub_atom)

    def __repr__(self):
        return "\n".join(
            [f"{atom} -> {sub}" for atom, sub in self.data.items()]
        )

    def add_substitution(self, atom: str, sub_atom: str):
        self.data[atom] = sub_atom

    def __eq__(self, other):
        equal = True
        for k, v in self.data.items():
            for k1, v1 in other.data.items():
                if k == k1:
                    if v != v1:
                        equal = False
                        break
        return equal
