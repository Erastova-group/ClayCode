import os
import re
from typing import Dict, Literal, Optional, Union

import numpy as np
import pandas as pd
from ClayCode.builder.claycomp import UCClayComposition, UCData, UnitCell
from ClayCode.builder.utils import select_input_option
from ClayCode.core.classes import ForceField, GROFile, ITPFile
from ClayCode.core.consts import FF, ITP_KWDS, KWD_DICT, UCS
from ClayCode.core.types import AnyPathType

# DEFAULT_ITP = UCS / "default.itp"
CLAY_FF = FF / "ClayFF_Fe"


class UCWriter:
    def __init__(
        self,
        gro_file: Union[AnyPathType, str],
        uc_type: str,
        odir: Union[AnyPathType, str],
        ff: Union[AnyPathType, str] = CLAY_FF,
    ):
        self._gro = GROFile(gro_file)
        self.path = odir / uc_type
        self.name = uc_type
        self.ff = ForceField(ff)

    def _init_path(self):
        os.makedirs(self.path, exist_ok=True)

    def write_new_uc(self, uc_name: str, default_solv: Optional[bool] = None):
        self._init_path()
        self._get_gro(uc_name)
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
            uc_specs = charge_occ_df.xs(self.name, level="value")
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
            charge_occ_df.loc[pd.IndexSlice[["T", "O"], self.name], "occ"] = [
                occ["T"],
                occ["O"],
            ]
            charge_occ_df.loc[
                pd.IndexSlice[["T", "O"], self.name], "charge"
            ] = [charge["T"], charge["O"]]
            charge_occ_df.loc[
                pd.IndexSlice[["T", "O"], self.name], "solv"
            ] = default_solv
            print(
                f"Adding new {self.name} unit cell spcifications to database."
            )
            charge_occ_df.convert_dtypes().reset_index().to_csv(
                UCS / "charge_occ.csv", index=False
            )

    def _get_gro(self, uc_name: str):
        """Write GRO file into the"""
        u = self._gro.universe
        resnames = np.unique(u.residues.resnames)
        if not (len(resnames) == 1 and uc_name in resnames):
            for res in u.residues:
                res.resname = f"{uc_name}"
        gro_file = self.path / f"{uc_name}.gro"
        self.gro = GROFile(gro_file)
        self.gro.universe = u
        self.gro.write()

    def _get_itp(
        self,
        uc_name: str,  # , default_itp: Union[AnyPathType, str] = DEFAULT_ITP
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
            r"NAME",
            uc_name,
            "NAME\t1",
            flags=re.MULTILINE,
        )
        gro_df = self.gro.df
        atom_cols = ITP_KWDS["atoms"]
        atom_cols.remove("id")
        itp_atom_df = pd.DataFrame(
            index=pd.Index(gro_df["atom-id"], name="id"),
            columns=atom_cols,
        )
        itp_atom_df["at-name"] = gro_df["at-type"].values
        itp_atom_df["res-name"] = gro_df.index.values
        itp_atom_df["res-name"] = itp_atom_df["res-name"].apply(
            lambda x: re.sub(r"\d(?=[A-Z])", "", x)
        )
        itp_atom_df["res-number"] = 1
        itp_atom_df["charge-nr"] = itp_atom_df.index.values
        itp_atom_df["at-type"] = (
            itp_atom_df["at-name"]
            .apply(lambda x: re.sub(r"([A-Z]+)[0-9]+", r"\1", x).lower())
            .values
        )
        itp_atom_df["at-type"] = itp_atom_df["at-type"].apply(
            lambda x: f"{x}s" if re.match("o[bx][os]", x) else x
        )
        ff_df = self.ff["atomtypes"].df.set_index("at-type")
        for prm in ["charge", "mass"]:
            itp_atom_df[prm] = itp_atom_df["at-type"].apply(
                lambda x: ff_df.loc[x][prm]
            )
        tot_charge = itp_atom_df["charge"].sum()
        if not np.isclose(tot_charge, 0.0, atol=1e-5):
            new_charge_mask = (
                itp_atom_df["at-type"]
                .apply(lambda x: x if re.match("o[hbx]*", x) else pd.NA)
                .dropna()
            )
            n_ob: pd.Series = new_charge_mask.value_counts().sum()
            new_charges = itp_atom_df.loc[new_charge_mask, "charge"]
            new_charges = new_charges.apply(lambda x: x - tot_charge / n_ob)
            itp_atom_df["charges"].update(new_charges)
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
        Otherwise, the new index will guessed from existing unit cells in :py:attr::`UCWriter.odir`.
        :param base_uc:
        :type base_uc:
        :param sub_sheet:
        :type sub_sheet:
        :param sub_dict:
        :type sub_dict:
        :return:
        :rtype:
        """
        uc_files = self.path.itp_filelist
        if uc_id is None:
            last_uc_file = sorted(uc_files)[-1].stem
            uc_id = int(re.search(r"\d+", uc_id).group(0))


uw = UCWriter("/storage/hectorite.gro", uc_type="T21", odir=UCS)
uw.write_new_uc("T200")
