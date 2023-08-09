import os
import re
import shutil

import numpy as np
import pandas as pd
from ClayCode.builder.claycomp import UCClayComposition, UCData, UnitCell
from ClayCode.core.classes import ForceField, GROFile, ITPFile
from ClayCode.core.consts import FF, UCS

DEFAULT_ITP = UCS / "default.itp"
CLAY_FF = FF / "ClayFF_Fe"


class UCWriter:
    def __init__(self, gro_file, uc_type, odir, ff=CLAY_FF):
        self.gro = GROFile(gro_file)
        self.path = odir / uc_type
        self.name = uc_type
        self.ff = ForceField(ff)

    def _init_path(self):
        os.makedirs(self.path, exist_ok=True)

    def write_new_uc(self, uc_name, **kwargs):
        self._init_path()
        self._get_gro(uc_name)
        self._get_itp(uc_name)

    def _get_gro(self, uc_name):
        u = self.gro.universe
        resnames = np.unique(u.residues.resnames)
        if not (len(resnames) == 1 and uc_name in resnames):
            for res in u.residues:
                res.resname = f"{uc_name}"
        self.gro.universe = u
        gro_file = self.path / f"{uc_name}.gro"
        shutil.copy(self.gro, gro_file)
        self.gro = GROFile(gro_file)
        self.gro.write()

    def _get_itp(self, uc_name, default_itp=DEFAULT_ITP):
        n_atoms = self.gro.n_atoms
        itp_file = self.path / f"{uc_name}.itp"
        shutil.copy(default_itp, itp_file)
        itp_file = ITPFile(itp_file)
        itp_file["moleculetype"] = re.sub(
            r"NAME",
            uc_name,
            itp_file["moleculetype"].string,
            flags=re.MULTILINE,
        )
        gro_df = self.gro.df
        itp_atom_df = pd.DataFrame(
            index=pd.Index(gro_df["atom-id"], name="id"),
            columns=itp_file["atoms"].df.columns[1:],
        )
        itp_atom_df["at-name"] = gro_df["at-type"].values
        itp_atom_df["res-name"] = gro_df.index.values
        itp_atom_df["res-name"] = itp_atom_df["res-name"].apply(
            lambda x: re.sub(r"1([A-Z]+[0-9]*)", "\1", x)
        )
        itp_atom_df["res-number"] = 1
        itp_atom_df["charge-nr"] = itp_atom_df.index.values
        itp_atom_df["at-type"] = (
            itp_atom_df["at-name"]
            .apply(lambda x: re.sub(r"([A-Z]+)[0-9]+", r"\1", x).lower())
            .values
        )
        itp_atom_df["at-type"] = itp_atom_df["at-type"].apply(
            lambda x: f"{x}s" if re.match("ob[ots]", x) else x
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
                .apply(
                    lambda x: x if re.match("oh*|ot|ob[tso]*s", x) else pd.NA
                )
                .dropna()
            )
            n_ob: pd.Series = new_charge_mask.value_counts().sum()
            new_charges = itp_atom_df.loc[new_charge_mask, "charge"]
            new_charges = new_charges.apply(lambda x: x - tot_charge / n_ob)
            itp_atom_df["charges"].update(new_charges)
        itp_file["atoms"] = itp_atom_df
        bond_df = (
            itp_atom_df["at-type"]
            .apply(lambda x: x if re.match("oh[ts]*", x) else pd.NA)
            .dropna()
        )
        n_bonds = bond_df.value_counts()[0]
        bond_df = itp_atom_df.loc[bond_df.index]
        itp_bond_df = pd.DataFrame(
            columns=["i", "j", "funct", "length", "force.c."]
        )
        at_name_pattern = re.compile(r"(OH[TS]*)([0-9]+)", flags=re.IGNORECASE)
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
        itp_bond_df.set_index("i", inplace=True)
        itp_file["bonds"] = itp_bond_df

        print(itp_file.string)


uw = UCWriter("/storage/hectorite.gro", uc_type="T21", odir=UCS)
uw.write_new_uc("T200")
