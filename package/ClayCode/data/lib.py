from __future__ import annotations

import logging
import os
import pathlib as pl
import pickle as pkl
import re
import sys
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Union, cast

import numpy as np
import pandas as pd
from ClayCode import Dir
from MDAnalysis import Universe

logger = logging.getLogger(__name__)

from ClayCode.data.consts import AA, DATA, FF, UCS, USER_UCS

PRM_INFO_DICT = {
    "n_atoms": cast(
        Callable[[Universe], Dict[str, int]],
        lambda u: dict(
            [
                (r, u.select_atoms(f"moltype {r}").n_atoms)
                for r in u.atoms.moltypes
            ]
        ),
    ),
    "charges": cast(
        Callable[[Universe], Dict[str, float]],
        lambda u: dict(
            zip(u.atoms.moltypes, np.round(u.atoms.residues.charges, 4))
        ),
    ),
    "masses": cast(
        Callable[[Universe], Dict[str, float]],
        lambda u: dict(
            zip(u.atoms.moltypes, np.round(u.atoms.residues.masses, 4))
        ),
    ),
}


def get_mol_prms(
    prm_str: str,
    itp_file: Union[str, pl.Path],
    include_dir: Union[str, pl.Path] = FF,
    write=False,
    force_update=False,
) -> dict:
    """Get parameters of a given type for a molecule.

    :param prm_str:
    :type prm_str:
    :param itp_file:
    :type itp_file:
    :param include_dir:
    :type include_dir:
    :param write:
    :type write:
    :param force_update:
    :type force_update:
    :return:
    :rtype:
    """
    dict_func = PRM_INFO_DICT[prm_str]
    residue_itp = Path(itp_file)
    prop_file = DATA / f"{residue_itp.stem}_{prm_str}.p"
    if (force_update is True) or (not prop_file.is_file()):
        atom_u = Universe(
            str(residue_itp),
            topology_format="ITP",
            include_dir=str(include_dir),
            infer_system=True,
        )
        prop_dict = dict_func(atom_u)
        if write is True:
            if not prop_file.parent.is_dir():
                os.makedirs(itp_file.parent)
            with open(prop_file, "wb") as prop_file:
                pkl.dump(prop_dict, prop_file)
    else:
        with open(prop_file, "rb") as prop_file:
            prop_dict = pkl.load(prop_file)
    return prop_dict


get_mol_n_atoms = partial(get_mol_prms, prm_str="n_atoms")
get_mol_charges = partial(get_mol_prms, prm_str="charges")
get_mol_masses = partial(get_mol_prms, prm_str="masses")
PRM_METHODS = {
    "charges": get_mol_charges,
    "n_atoms": get_mol_n_atoms,
    "masses": get_mol_masses,
}
ion_itp = FF / "Ions.ff/ions.itp"
get_ion_charges = partial(get_mol_charges, itp_file=ion_itp)
get_ion_n_atoms = partial(get_mol_n_atoms, itp_file=ion_itp)
get_ion_masses = partial(get_mol_masses, itp_file=ion_itp)


def get_ion_prms(prm_str: str, **kwargs):
    """Get parameters of a given type for ions.
    :param prm_str: parameter to get
    :type prm_str: str
    :param force_update: force update of parameters
    :type force_update: bool
    :param write: write parameters to file
    :type write: bool
    :param name: name of parameter file
    :type name: Optional[str]
    :return: parameter dictionary
    :rtype: Dict[str, Union[float, int]]"""
    if prm_str == "charges":
        prm_dict = get_ion_charges(**kwargs)
    elif prm_str == "n_atoms":
        prm_dict = get_ion_n_atoms(**kwargs)
    elif prm_str == "masses":
        prm_dict = get_ion_masses(**kwargs)
    else:
        raise KeyError(f"Unexpected parameter: {prm_str!r}")
    return prm_dict


def get_ion_list() -> List[str]:
    """Get list of ion resnames from ions.itp file
    :return: list of ion resnames
    :rtype: List[str]
    """
    ions = get_ion_charges()
    ion_list = list(ions.keys())
    return ion_list


def get_clay_prms(prm_str: str, uc_name: str, uc_path=UCS, force_update=False):
    """Get parameters of a given type for clay unit cells.
    :param prm_str: parameter to get
    :type prm_str: str
    :param uc_name: name of unit cell
    :type uc_name: str
    :param force_update: force update of parameters
    :type force_update: bool
    :param write: write parameters to file
    :type write: bool
    :param name: name of parameter file
    :type name: Optional[str]
    :return: parameter dictionary
    :rtype: Dict[str, Union[float, int]]
    """
    prm_func = PRM_METHODS[prm_str]
    prm_file = DATA / f"{uc_name.upper()}_{prm_str}.pkl"
    if not prm_file.is_file() or force_update is True:
        charge_dict = {}
        for uc_path in [UCS, USER_UCS]:
            uc_files = uc_path.glob(rf"{uc_name}/*[0-9].itp")
            for uc_file in uc_files:
                uc_charge = prm_func(
                    itp_file=uc_file, write=False, force_update=force_update
                )
                charge_dict.update(uc_charge)
    else:
        with open(prm_file, "rb") as prm_file:
            charge_dict = pkl.read(prm_file)
    return charge_dict


get_clay_charges = partial(get_clay_prms, prm_str="charges")
get_clay_n_atoms = partial(get_clay_prms, prm_str="n_atoms")
get_clay_masses = partial(get_clay_prms, prm_str="masses")


def get_sol_prms(
    prm_str: str,
    sol_path=FF / "ClayFF_Fe.ff",
    include_dir: Union[str, pl.Path] = FF,
    force_update=False,
):
    """Get parameters of a given type for solvent molecules.
    :param prm_str: parameter to get
    :type prm_str: str
    :param force_update: force update of parameters
    :type force_update: bool
    :param write: write parameters to file
    :type write: bool
    :param name: name of parameter file
    :type name: Optional[str]
    :return: parameter dictionary
    :rtype: Dict[str, Union[float, int]]"""
    prm_func = PRM_METHODS[prm_str]
    charge_file = DATA / f"SOL_{prm_str}.pkl"
    if not charge_file.is_file() or (force_update is True):
        charge_dict = {}
        sol_fnames = ["interlayer_spc", "spc"]
        for file in sol_fnames:
            itp = f"{sol_path}/{file}.itp"
            sol_charge = prm_func(
                itp_file=itp,
                write=False,
                include_dir=include_dir,
                force_update=force_update,
            )
            charge_dict.update(sol_charge)
    else:
        with open(charge_file, "rb") as charge_file:
            charge_dict = pkl.read(charge_file)
    return charge_dict


get_sol_charges = partial(get_sol_prms, prm_str="charge")
get_sol_n_atoms = partial(get_sol_prms, prm_str="n_atoms")
get_sol_masses = partial(get_sol_prms, prm_str="masses")


def get_aa_prms(prm_str: str, aa_name: str, aa_path=AA, force_update=False):
    prm_func = PRM_METHODS[prm_str]
    charge_file = DATA / f"{aa_name.upper()}_{prm_str}.pkl"
    if not charge_file.is_file() or force_update is True:
        charge_dict = {}
        aa_dirs = aa_path.glob(rf"pK[1-9]/{aa_name.upper()}[1-9].itp")
        for aa_file in aa_dirs:
            aa_charge = prm_func(
                itp_file=aa_file, write=False, force_update=force_update
            )
            charge_dict.update(aa_charge)
    else:
        with open(charge_file, "rb") as charge_file:
            charge_dict = pkl.read(charge_file)
    return charge_dict


get_aa_charges = partial(get_aa_prms, prm_str="charges")
get_aa_n_atoms = partial(get_aa_prms, prm_str="n_atoms")
get_aa_masses = partial(get_aa_prms, prm_str="masses")


def get_all_prms(
    prm_str, force_update=True, write=True, name=None
) -> Dict[str, Union[float, int]]:
    """Get parameters of a given type for all systems.
    :param prm_str: parameter to get
    :type prm_str: str
    :param force_update: force update of parameters
    :type force_update: bool
    :param write: write parameters to file
    :type write: bool
    :param name: name of parameter file
    :type name: Optional[str]
    :return: parameter dictionary
    :rtype: Dict[str, Union[float, int]]
    """
    if name is not None:
        namestr = f"{name}_"
    else:
        namestr = ""
    charge_file = DATA / f"{namestr}{prm_str}.pkl"
    if not charge_file.is_file() or force_update is True:
        ion_dict = get_ion_prms(prm_str=prm_str, force_update=force_update)
        aa_types = [
            "ala",
            "arg",
            "asn",
            "asp",
            "ctl",
            "cys",
            "gln",
            "glu",
            "gly",
            "his",
            "ile",
            "leu",
            "lys",
            "met",
            "phe",
            "pro",
            "ser",
            "thr",
            "trp",
            "tyr",
            "val",
        ]
        aa_dict = {}
        for aa in aa_types:
            aa_dict.update(
                get_aa_prms(
                    prm_str=prm_str, aa_name=aa, force_update=force_update
                )
            )
        clay_types = Dir(UCS).dirlist.filter(r"[A-Z]*[0-9]*")
        clay_types += Dir(USER_UCS).dirlist.filter(r"[A-Z]*[0-9]*")
        clay_dict = {}
        try:
            for uc in clay_types:
                if not uc.name.startswith("."):
                    uc = uc.stem
                    clay_dict.update(
                        get_clay_prms(
                            prm_str=prm_str,
                            uc_name=uc,
                            force_update=force_update,
                        )
                    )
        except Exception as e:
            logger.ferror(
                f"{e}\nError getting {uc.name} unit cell parameters."
            )
            sys.exit(1)
        sol_dict = get_sol_prms(prm_str=prm_str, force_update=force_update)
        charge_dict = {**ion_dict, **clay_dict, **sol_dict, **aa_dict}
        if write is True:
            with open(charge_file, "wb") as file:
                pkl.dump(charge_dict, file)
    else:
        with open(charge_file, "rb") as file:
            charge_dict = pkl.load(file)
    return charge_dict


get_all_charges = partial(get_all_prms, prm_str="charges")
get_all_n_atoms = partial(get_all_prms, prm_str="n_atoms")
get_all_masses = partial(get_all_prms, prm_str="masses")


def get_system_prms(
    prm_str, crds: Union[str, Path, Universe], write=True, force_update=True
) -> Union[str, pd.Series, None]:
    """Get system parameters (charges, atoms per molecule) for a coordinate file.
    :param prm_str: parameter to get
    :type prm_str: str
    :param crds: coordinate file
    :type crds: Union[str, Path, Universe]
    :param write: write parameters to file
    :type write: bool
    :param force_update: force update of parameters
    :type force_update: bool
    :return: system parameters
    :rtype: Union[str, pd.Series, None]
    """
    if type(crds) == Universe:
        u = crds
        name = "universe"
    else:
        try:
            u = Universe(str(crds))
            name = Path(crds).stem
        except ValueError:
            logger.ferror(f"Could not create Universe from {crds}")
            return None
    prm_df = pd.Series(
        get_all_prms(prm_str, write=write, force_update=force_update),
        name=name,
    )
    if prm_str == "charges":
        residue_df = pd.Series(
            u.residues.resnames, name="residues", dtype="str"
        )
        residue_df = residue_df.aggregate("value_counts")
        prm_df = pd.concat(
            {prm_str: prm_df, "counts": residue_df}, axis=1, join="inner"
        )
        prm_df["sum"] = prm_df.apply("product", axis=1).astype(int)
        sys_prms = prm_df["sum"].sum().astype(int)
    elif prm_str == "n_atoms":
        sys_prms = prm_df
    return sys_prms


get_system_charges = partial(get_system_prms, prm_str="charges")
get_system_n_atoms = partial(get_system_prms, prm_str="n_atoms")
get_system_masses = partial(get_system_prms, prm_str="masses")


def add_mols_to_top(
    topin: Union[str, pl.Path],
    topout: Union[str, pl.Path],
    insert: Union[None, str, pl.Path],
    n_mols: int,
    include_dir,
):
    """Add specified number of `insert` molecules to topology.
    :param topin: input topology filename
    :type topin: Union[str, pl.Path]
    :param topout: output topology filename
    :type topout: Union[str, pl.Path]
    :param insert: molecule to insert
    :type insert: Union[None, str, pl.Path]
    :param n_mols: number of molecules to insert
    :type n_mols: int
    :param include_dir: directory containing force field files
    :type include_dir: Union[str, pl.Path]
    :return: None"""
    if n_mols != 0:
        itp_path = pl.Path(insert).parent.resolve()
        itp = pl.Path(insert).stem
        insert = Universe(
            str(itp_path / f"{itp}.itp"),
            topology_format="ITP",
            include_dir=include_dir,
        )
    else:
        itp_path = pl.Path(topin).parent
    topin = pl.Path(topin)
    with open(topin, "r") as topfile:
        topstr = topfile.read().rstrip()
    topmatch = re.search(
        r"\[ system \].*", topstr, flags=re.MULTILINE | re.DOTALL
    ).group(0)
    with open(itp_path.parent.parent / "FF/tophead.itp", "r") as tophead:
        tophead = tophead.read()
    with open(topout, "w") as topfile:
        if n_mols != 0:
            topfile.write(
                "\n".join(
                    [
                        tophead,
                        topmatch,
                        f"\n{insert.residues.moltypes[0]}  {n_mols}\n",
                    ]
                )
            )
        else:
            topfile.write("\n".join([tophead, topmatch]))
