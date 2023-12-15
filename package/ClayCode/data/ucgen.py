import copy
import os
import re
import shutil
from typing import Dict, Literal, Optional, Union

import numpy as np
import pandas as pd
from ClayCode.builder.claycomp import UCData
from ClayCode.builder.utils import select_input_option
from ClayCode.core.cctypes import PathType
from ClayCode.core.classes import Dir, ForceField, GROFile, ITPFile
from ClayCode.core.consts import ITP_KWDS, UCS
from ClayCode.data.consts import CLAY_ACHARGES, CLAY_FF
from MDAnalysis.topology import tables
from MDAnalysis.topology.guessers import guess_bonds

# DEFAULT_ITP = UCS / "default.itp"

# class SubstitutionAtom(TopologyAttr):
#     def __init__(self, atom: Atom, other, sheet=None):
#         super().__init__(self, values=(atom, other))
#
#
#     @property
#     def bonded(self):
#         return self.values[0]
#
#     @property
#     def atom(self):
#         return self.values[1]


class UCWriter:
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
    ):
        if odir is None:
            odir = UCS
        self._gro = GROFile(gro_file)
        self.gro = copy.deepcopy(self._gro)
        self.path = Dir(odir) / uc_type
        self.name = uc_type
        self.ff = ForceField(ff)
        try:
            self.init_uc_folder()
        except FileNotFoundError:
            pass

    def _init_path(self):
        os.makedirs(self.path, exist_ok=True)

    def add_new_uc(self, uc_name: str, universe):
        self._get_gro(uc_name, universe)
        self._get_itp(uc_name)

    def write_new_uc(self, uc_name: str, default_solv: Optional[bool] = None):
        self._init_path()
        self._get_gro(uc_name)
        self.init_uc_folder()
        self._get_itp(uc_name)
        uc_data = UCData(self.path, ff=self.ff)
        occ = uc_data.occupancies
        charge = uc_data.oxidation_numbers
        charge_occ_df = (
            pd.read_csv(UCS / "charge_occ.csv")
            .fillna(method="ffill")
            .set_index(["sheet", "value"])
        )
        default_solv = select_input_option(
            instance_or_manual_setup=True,
            query="Set the default solvation for this unit cell type to 'True'? [y]es/[n]o\n",
            options=["y", "n", True, False],
            result=default_solv,
            result_map={"y": True, "n": False},
        )
        if self.name in charge_occ_df.index.get_level_values("value"):
            uc_specs = charge_occ_df.xs(self.name, level="value", axis=0)
            assert (
                uc_specs.loc[["T", "O"], "occ"] == [occ["T"], occ["O"]]
            ).all(), "Specified occupancies do not match unit cell occupancies"
            assert (
                uc_specs.loc[["T", "O"], "charge"]
                == [charge["T"], charge["O"]]
            ).all(), "Specified oxidation numbers do not match unit cell oxidation numbers"
            assert (
                uc_specs.loc[["T", "O"], "solv"] == default_solv
            ).all(), "Specified default solvation does not match selected unit cell default solvation"
        else:
            new_entry = pd.DataFrame(
                index=pd.MultiIndex.from_product(
                    [
                        charge_occ_df.index.get_level_values("sheet").unique(),
                        [self.name],
                    ],
                    names=charge_occ_df.index.names,
                ),
                columns=charge_occ_df.columns,
            )
            new_entry.loc[pd.IndexSlice[["T", "O"], self.name], "occ"] = [
                occ["T"],
                occ["O"],
            ]
            new_entry.loc[pd.IndexSlice[["T", "O"], self.name], "charge"] = [
                charge["T"],
                charge["O"],
            ]
            new_entry.loc[
                pd.IndexSlice[["T", "O"], self.name], "solv"
            ] = default_solv
            charge_occ_df = charge_occ_df.append(new_entry)
            print(
                f"Adding new {self.name} unit cell specifications to database."
            )
            charge_occ_df.convert_dtypes().reset_index().to_csv(
                UCS / "charge_occ.csv", index=False
            )

    def _get_gro(self, uc_name: str, universe=None):
        """Write GRO file:
        :param uc_name"""
        if universe is None:
            u = self._gro.universe
        else:
            u = universe
        resnames = np.unique(u.residues.resnames)
        if not (len(resnames) == 1 and uc_name in resnames):
            for res in u.residues:
                res.resname = f"{uc_name}"
        gro_file = self.path / f"{uc_name}.gro"
        self.gro = GROFile(gro_file)
        self.gro.universe = u
        self.gro.write()

    def _get_itp(
        self, uc_name: str  # , default_itp: Union[PathType, str] = DEFAULT_ITP
    ):
        """Write the ITP file that corresponds to the gro file of the same name.
        Uses force field parameters from :py:data::`ClayCode.data.FF.ClayFF_Fe` database.
        Saves the final file into :py:attr::`UCWriter.odir`
        :param uc_name:
        :type uc_name:
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
            itp_atom_df[prm] = itp_atom_df["at-type"].apply(
                lambda x: ff_df.loc[x][prm]
            )
        tot_charge = itp_atom_df["charge"].sum()
        charge_diff = tot_charge - np.rint(tot_charge)
        if not np.isclose(charge_diff, 0.0, atol=1e-5):
            new_charge_mask = (
                itp_atom_df["at-type"]
                .apply(lambda x: x if re.match("o[hbx]*", x) else pd.NA)
                .dropna()
            )
            n_ob: pd.Series = new_charge_mask.value_counts().sum()
            new_charges = itp_atom_df.loc[new_charge_mask.index, "charge"]
            new_charges = new_charges.apply(lambda x: x - charge_diff / n_ob)
            itp_atom_df["charge"].update(new_charges)
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

    def add_substitution(
        self,
        base_uc_id: str,
        sub_sheet: Union[Literal["T"], Literal["O"]],
        sub_dict: Dict[str, str],
        uc_id: Optional[int] = None,
    ):
        """
        Add isomorphic substitutions to base_uc unit cell. Replaces sub_dict keys by :py:obj::`sub_dict` values.
        Saves the final file into :py:attr::`UCWriter.odir`.
        If given, the final UC will have index :py:object::`uc_id`.
        Otherwise, the new index will be guessed from existing unit cells in :py:attr::`UCWriter.odir`.
        :param base_uc:
        :type base_uc:
        :param sub_sheet:
        :type sub_sheet:
        :param sub_dict:
        :type sub_dict:
        :return:
        :rtype:
        """
        ...

    def get_substitution_atoms(self, o_bonds={"T": 4, "O": 6}):
        sheets = self.get_sheets()
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
        # u.atoms.guess_bonds(vdwradii=vdwradii)
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
                    if not atom.name[0] in ["H", "O"]:
                        no_pbc = atom.bonds.bonds(pbc=False)
                        pbc = atom.bonds.bonds(pbc=True)
                        if (
                            len(atom.bonds[no_pbc.round(6) == pbc.round(6)])
                            == o_bonds[sheet_type]
                        ):
                            bonded: list = atom.bonds.atom1 + atom.bonds.atom2
                            bonded -= atom
                            subs[sheet_type][t_sheet].append((atom, bonded))
        self.substitutable = subs

    def write_single_substituted_ucs(self, sub_dict, uc_id=None):
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
            print("Getting substitutable atoms with default number of bonds")
            self.get_substitution_atoms()
        sub_atoms = self.substitutable
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
                    print(atom.name)
                    atom_names = atom.universe.atoms.names.copy()
                    if atom.name != base_names[atom.ix]:
                        print(f"# Atom {atom.name} already substituted")
                        continue
                    atom_names = copy.deepcopy(atom_names)
                    at_name_match = name_match_pattern.match(atom.name)
                    try:
                        atom_names[
                            atom.ix
                        ] = f"{sub_dict[at_name_match.group(1)]}{at_name_match.group(2)}"
                    except KeyError:
                        pass
                    else:
                        if int(
                            self.at_charges_dict[at_name_match.group(1)]
                        ) > int(
                            self.at_charges_dict[
                                sub_dict[at_name_match.group(1)]
                            ]
                        ):
                            for bonded in other:
                                other_name_match = o_match_pattern.match(
                                    bonded.name
                                )
                                atom_names[
                                    bonded.ix
                                ] = f"{o_sub_dict[sub_sheet][other_name_match.group(1)]}{other_name_match.group(2)}"
                        elif int(
                            self.at_charges_dict[at_name_match.group(1)]
                        ) < int(
                            self.at_charges_dict[
                                sub_dict[at_name_match.group(1)]
                            ]
                        ):
                            raise ValueError(
                                f"Expected lower charge after substitution!"
                            )
                        sub_ucs += 1
                        sheets[sub_sheet][sheet_part][sub_ucs] = atom_names
                        last_uc = sorted(
                            self.path.glob(rf"{self.uc_stem}[0-9]*")
                        )[-1]
                        last_uc_id = self.get_id(last_uc.stem)
                        new_uc_id = f"{last_uc_id + 1:02d}"
                        new_u = atom.universe.copy()
                        new_u.atoms.names = atom_names
                        new_name = self.get_uc_gro(new_uc_id)
                        uw.add_new_uc(uc_name=new_name.stem, universe=new_u)
                        print(
                            f'Wrote new unit cell substitution of {self.gro.stem} to {new_name.stem} with {atom.name} -> {f"{sub_dict[at_name_match.group(1)]}{at_name_match.group(2)}"}'
                        )
                        # print(new_uc_id, atom_names)

    def get_id(self, uc_stem):
        return int(re.match(f"{self.uc_stem}(\d+)", uc_stem).group(1))

    def get_uc_gro(self, uc_id):
        return GROFile(self.path / f"{self.uc_stem}{int(uc_id):02d}.gro")

    def get_sheets(self, uc_id=None):
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
            base_id = int(re.search(r"\d+", f"{uc_stem}").group(0)[-2:])
        except AttributeError:
            self.base_id = None
        else:
            self.base_id = f"{base_id:02d}"
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
