#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
r""":mod:`ClayCode.core.lib` --- Clay-specific utilities
======================================================
This module provides utility functions for the ClayCode package.
"""
from __future__ import annotations

import logging
import pathlib as pl
import re
import shutil
import tempfile
from functools import update_wrapper, wraps
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import MDAnalysis as mda
import MDAnalysis.coordinates
import numpy as np
from ClayCode import ITPFile, TOPFile
from ClayCode.core.cctypes import AtomGroup, PathType
from ClayCode.core.classes import ForceField, GROFile, Parameter, YAMLFile
from ClayCode.core.consts import SOL, SOL_DENSITY
from ClayCode.core.gmx import gmx_command_wrapper

# )
from ClayCode.core.utils import backup_files, temp_file_wrapper
from ClayCode.data.consts import CLAYFF_AT_TYPES, FF, MDP
from ClayCode.data.lib import (
    get_aa_charges,
    get_aa_masses,
    get_aa_n_atoms,
    get_all_charges,
    get_all_masses,
    get_all_n_atoms,
    get_clay_charges,
    get_clay_masses,
    get_clay_n_atoms,
    get_ion_charges,
    get_ion_list,
    get_ion_masses,
    get_ion_n_atoms,
    get_mol_charges,
    get_mol_n_atoms,
    get_sol_charges,
    get_sol_masses,
    get_sol_n_atoms,
    get_system_charges,
    get_system_masses,
    get_system_n_atoms,
    ion_itp,
)
from MDAnalysis import Universe

# from ClayCode.core.lib import (
# add_resnum,
# get_ion_charges,
# logger,
# rename_il_solvent,

logging.getLogger("MDAnalysis.topology.TPRparser").setLevel(
    level=logging.WARNING
)

logger = logging.getLogger(__name__)

__name__ = "lib"
__all__ = [
    "select_solvent",
]


def fix_gro_residues(crdin: Union[Path, str], crdout: Union[Path, str]):
    u = mda.Universe(crdin)
    if np.unique(u.residues.resnum).tolist() == [1]:
        u = add_resnum(crdin=crdin, crdout=crdout)
    if "iSL" not in u.residues.resnames:
        u = rename_il_solvent(crdin=crdout, crdout=crdout)
    return u


def get_ag_numbers_info(ag: AtomGroup) -> Tuple[int, int]:
    return "Selected {} atoms in {} residues.".format(
        ag.n_atoms, ag.n_residues
    )


def select_clay(
    universe: MDAnalysis.Universe,
    ff: Optional[ForceField] = None,
    atomtypes: Optional[List[str]] = None,
) -> MDAnalysis.AtomGroup:
    """Select clay atoms based on atom names in force field.
    :param universe: MDAnalysis universe
    :type universe: MDAnalysis.Universe
    :param ff: force field
    :type ff: ForceField
    :return: clay atoms
    :rtype: MDAnalysis.AtomGroup
    """
    if atomtypes is None and ff is not None:
        atomtypes = (
            ff["clay"]["atomtypes"]["at-type"]
            .apply(lambda x: x[:3].upper())
            .tolist()
        )
    elif atomtypes is None and ff is None:
        atomtypes = []
        for sheet_types in YAMLFile(CLAYFF_AT_TYPES).data.values():
            for atype in sheet_types.values():
                if len(atype.split("_")) == 1:
                    atomtypes.append(atype[:3].upper())
    else:
        atomtypes = [at.upper() for at in atomtypes]
    clay_atoms = universe.select_atoms(
        f'name {"* ".join(atomtypes)}*'
    ).residues.atoms
    return clay_atoms


def generate_restraints(
    atom_group: AtomGroup,
    outfilename: Union[str, PathType],
    fc: Union[List[int], int] = 1000,
    add_to_file: bool = False,
):
    if not Path(outfilename).is_file() or not add_to_file:
        with open(outfilename, "w") as outfile:
            outfile.write("[ position_restraints ]\n")
            outfile.write(
                '; {"i":6}  {"funct":5} {"fx":6} {"fy":6} {"fz":6}\n'
            )
    with open(outfilename, "a") as outfile:
        if type(fc) == int:
            fc = [fc]
        if len(fc) == 1:
            fc = fc * 3
        else:
            assert len(fc) in [1, 3], ValueError(
                f"Expected 1 or 3 force constants, got {len(fc)}"
            )
        for atom in atom_group:
            outfile.write(
                f'{atom.index + 1:6} {1:5} {" ".join([f"{fci:6}" for fci in fc])}\n'
            )


def select_outside_clay_stack(
    atom_group: MDAnalysis.AtomGroup, clay: MDAnalysis.AtomGroup, extra=1
):
    atom_group = atom_group.select_atoms(
        f" prop z > {np.max(clay.positions[:, 2]) + extra} or"
        f" prop z < {np.min(clay.positions[:, 2]) - extra}"
    )
    logger.debug(
        f"'atom_group': Selected {atom_group.n_atoms} atoms of names: {np.unique(atom_group.names)} "
        f"(residues: {np.unique(atom_group.resnames)})"
    )
    return atom_group


def search_ndx_group(ndx_str: str, sel_name: str):
    try:
        sel_match = re.search(
            f"\[ {sel_name} \]", ndx_str, flags=re.MULTILINE | re.DOTALL
        ).group(0)
        match = True
    except AttributeError:
        match = False
    return match


def save_selection(
    outname: Union[str, Path],
    atom_groups: List[mda.AtomGroup],
    ndx=False,
    traj=".trr",
    pdbqt=False,
    backup=False,
):
    """Save atom groups to coordinates, trajectory and optionally PDBQT file.
    :param outname: output filename
    :type outname: Union[str, Path]
    :param atom_groups: list of atom groups
    :type atom_groups: List[mda.AtomGroup]
    :param ndx: write index file, defaults to False
    :type ndx: bool, optional
    :param traj: trajectory file extension, defaults to ".trr"
    :type traj: str, optional
    :param pdbqt: write PDBQT file, defaults to False
    :type pdbqt: bool, optional
    :param backup: backup existing files, defaults to False
    :type backup: bool, optional
    """
    ocoords = Path(outname).with_suffix(".gro").resolve()
    opdb = Path(outname).with_suffix(".pdbqt").resolve()
    logger.finfo(f"Writing coordinates and trajectory ({traj!r})")
    if ndx is True:
        ondx = Path(outname).with_suffix(".ndx").resolve()
        if ondx.is_file():
            with open(ondx, "r") as ndx_file:
                ndx_str = ndx_file.read()
        else:
            ndx_str = ""
    outsel = atom_groups[0]
    if ndx is True:
        group_name = "clay"
        atom_name = np.unique(atom_groups[0].atoms.names)[0][:2]
        group_name += f"_{atom_name}"
        if not search_ndx_group(ndx_str=ndx_str, sel_name=group_name):
            atom_groups[0].write(ondx, name=group_name, mode="a")
    for ag in atom_groups[1:]:
        if ndx is True:
            group_name = np.unique(ag.residues.resnames)[0]
            group_name = re.match("[a-zA-Z]*", group_name).group(0)
            atom_name = np.unique(ag.atoms.names)[0][:2]
            group_name += f"_{atom_name}"
            if not search_ndx_group(ndx_str=ndx_str, sel_name=group_name):
                ag.write(ondx, name=group_name, mode="a")
        outsel += ag
        logger.finfo(
            f"New trajectory from {len(atom_groups)} groups with {outsel.n_atoms} total atoms"
        )
    logger.finfo(f"1. {ocoords!r}")
    if backup:
        backup_files(new_filename=ocoords)
    outsel.write(str(ocoords))
    if pdbqt is True:
        logger.finfo(f"2. {opdb!r}")
        if backup:
            backup_files(new_filename=opdb)
        outsel.write(str(opdb), frames=outsel.universe.trajectory[-1::1])
    if type(traj) != list:
        traj = [traj]
    for t in traj:
        otraj = outname.with_suffix(t).resolve()
        logger.finfo(f"3. {otraj!r}")
        if backup:
            backup_files(otraj)
        outsel.write(str(otraj), frames="all")


# def check_traj(
#     instance: analysis_class, check_len: Union[int, Literal[False]]
# ) -> None:
#     """Check length of trajectory in analysis class instance.
#     :param instance: analysis class instance
#     :type instance: analysis_class
#     :param check_len: expected number of trajectory frames, defaults to False
#     :type check_len: Union[int, Literal[False]]
#     :raises SystemExit: Error if trajectory length != check_len
#     """
#     logger.debug(f"Checking trajectory length: {check_len}")
#     if type(check_len) == int:
#         if instance._universe.trajectory.n_frames != check_len:
#             raise SystemExit(
#                 "Wrong number of frames: "
#                 f"{instance._universe.trajectory.n_frames}"
#             )


# def process_box(instance: analysis_class) -> None:
#     """Assign distance minimisation function in orthogonal or triclinic box.
#
#     Correct x, x2, z interatomic distances for periodic boundary conditions
#     in orthogonal box inplace
#                                          O*
#        +--------------+       +---------/----+
#        |       S      |       |       S      |
#        |        \     |  -->  |              |
#        |         \    |       |              |
#        |          O   |       |              |
#        +--------------+       +--------------+
#
#     :param instance: analysis class instance
#     :type instance: analysis_class
#     """
#     box = instance._universe.dimensions
#     if np.all(box[3:] == 90.0):
#         instance._process_distances = process_orthogonal
#         instance._process_axes = process_orthogonal_axes
#     else:
#         instance._process_distances = process_triclinic
#         instance._process_axes = process_triclinic_axes
#

# def process_orthogonal_axes(
#     distances: NDArray[np.float64],
#     dimensions: NDArray[np.float64],
#     axes: List[int],
# ) -> None:
#     """
#     Correct x, x2, z interatomic distances for periodic boundary conditions
#     in orthogonal box inplace
#
#     :param axes:
#     :type axes:
#     :param distances: interatomic distance array of shape (n, m, 3)
#     :type distances: NDArray[np.float64]
#     :param dimensions: simulation box dimension array of shape (6,)
#     :type dimensions: NDArray[np.float64]
#     :return: no return
#     :rtype: NoReturn
#     """
#     assert (
#         distances.shape[-1] == len(axes) or distances.ndim == 2
#     ), f"Shape of distance array ({distances.shape[-1]}) does not match selected axes {axes}"
#     for idx, dist in np.ma.ndenumerate(distances):
#         distances[idx] -= dimensions[:3][axes] * np.rint(
#             dist / dimensions[:3][axes]
#         )


# def process_orthogonal(
#     distances: NDArray[np.float64], dimensions: NDArray[np.float64]
# ) -> None:
#     """
#     Correct x, x2, z interatomic distances for periodic boundary conditions
#     in orthogonal box inplace
#
#     :param distances: interatomic distance array of shape (n, m, 3)
#     :type distances: NDArray[np.float64]
#     :param dimensions: simulation box dimension array of shape (6,)
#     :type dimensions: NDArray[np.float64]
#     :return: no return
#     :rtype: NoReturn
#     """
#     distances -= dimensions[:3] * np.rint(distances / dimensions[:3])


# def process_triclinic_axes(
#     distances: NDArray[np.float64],
#     dimensions: NDArray[np.float64],
#     axes: List[int],
# ) -> None:
#     """
#     Correct x, x2, z interatomic distances for periodic boundary conditions
#     in triclinic box inplace
#
#     :param axes:
#     :type axes:
#     :param distances: interatomic distance array of shape (n, m, 3)
#     :type distances: NDArray[np.float64]
#     :param dimensions: simulation box dimension array of shape (6,)
#     :type dimensions: NDArray[np.float64]
#     :return: no return
#     :rtype: NoReturn
#     """
#     box = triclinic_vectors(dimensions)
#     assert distances.shape[-1] >= len(
#         axes
#     ), f"Shape of distance array ({distances.shape[-1]}) does not match selected axes {axes}"
#     logger.debug(
#         distances / np.diagonal(box)[..., axes],
#         np.rint(distances / np.diagonal(box)[..., axes]),
#     )
#     distances -= np.diagonal(box)[..., axes] * np.rint(
#         distances / np.diagonal(box)[..., axes]
#     )


# def process_triclinic(
#     distances: NDArray[np.float64], dimensions: NDArray[np.float64]
# ) -> None:
#     """
#     Correct x, x2, z interatomic distances for periodic boundary conditions
#     in triclinic box inplace
#
#     :param distances: interatomic distance array of shape (n, m, 3)
#     :type distances: NDArray[np.float64]
#     :param dimensions: simulation box dimension array of shape (6,)
#     :type dimensions: NDArray[np.float64]
#     :return: no return
#     :rtype: NoReturn
#     """
#     box = triclinic_vectors(dimensions)
#     distances -= np.diagonal(box) * np.rint(distances / np.diagonal(box))


# def select_cyzone(
#     distances: MaskedArray,
#     max_z_dist: float,
#     xy_rad: float,
#     mask_array: MaskedArray,
# ) -> None:
#     """
#     Select all distances corresponding to atoms within a cylindrical volume
#     of dimensions +- max_z_dist and radius xy_rad
#     :param distances: masked interatomic distance array of shape (n, m, 3)
#     :type distances: MaskedArray[np.float64]
#     :param max_z_dist: absolute value for cutoff in z direction
#     :type max_z_dist: float
#     :param xy_rad: absolute value for radius in xy plane
#     :type xy_rad: float
#     :param mask_array: array for temporary mask storage of shape (n, m)
#     :type mask_array: MaskedArray[np.float64]
#     :return: no return
#     :rtype: NoReturn
#     """
#     z_col = distances[:, :, 2]
#     z_col.mask = np.abs(z_col) > max_z_dist
#     distances.mask = np.broadcast_to(
#         z_col.mask[:, :, np.newaxis], distances.shape
#     )
#     np.ma.sum(distances[:, :, [0, 1]].__pow__(2), axis=2, out=mask_array)
#     mask_array.harden_mask()
#     mask_array.mask = mask_array > xy_rad.__pow__(2)
#     np.copyto(distances.mask, mask_array.mask[:, :, np.newaxis])


# def exclude_xyz_cutoff(distances: NDArray[np.int64], cutoff: float) -> None:
#     """
#     Select all distances corresponding to atoms within a box
#     with length 2* cutoff
#     :param distances: masked interatomic distance array of shape (n, m, 3)
#     :type distances: NDArray[np.float64]
#     :param cutoff: absolute value for maximum distance
#     :type cutoff: float
#     :return: no return
#     :rtype: NoReturn
#     """
#     mask = np.any(np.abs(distances) >= cutoff, axis=2)
#     np.copyto(distances.mask, mask[:, :, np.newaxis])


# def exclude_z_cutoff(distances: NDArray[np.int64], cutoff: float) -> None:
#     """
#     Select all distances corresponding to atoms within a box
#     with length 2* cutoff
#     :param distances: masked interatomic distance array of shape (n, m, 3)
#     :type distances: NDArray[np.float64]
#     :param cutoff: absolute value for maximum distance
#     :type cutoff: float
#     :return: no return
#     :rtype: NoReturn
#     """
#     mask = np.abs(distances[..., 2]) > cutoff
#     distances.mask += np.broadcast_to(mask[..., np.newaxis], distances.shape)


# def get_dist(
#     ag_pos: NDArray[np.float64],
#     ref_pos: NDArray[np.float64],
#     distances: NDArray[np.float64],
#     box: NDArray[np.float64],
# ) -> NoReturn:
#     """Calculate minimum elementwise x, x2, z distances
#     of selection atom positions to reference atom positions in box.
#     Output array shape(len(ag_pos), len(ref_pos), 3)
#     :param ag_pos: atom group positions of shape (n_atoms, 3)
#     :type ag_pos: NDArray[np.float64]
#     :param ref_pos: atom group positions of shape (n_atoms, 3)
#     :type ref_pos: NDArray[np.float64]
#     distances: result array of shape (len(ag_pos), len(ref_pos), 3)
#     :type distances: NDArray[np.float64]
#     :param box: Timestep dimensions array of shape (6, )
#     :type box: NDArray[np.float64]
#     """
#     for atom_id, atom_pos in enumerate(ag_pos):
#         distances[atom_id, :, :] = minimize_vectors(atom_pos - ref_pos, box)


# def get_self_dist(
#     ag_pos: NDArray[np.float64], distances: NDArray[np.float64]
# ) -> NoReturn:
#     """Calculate minimum elementwise x, x2, z distances
#     of selection atom positions to reference atom positions in box.
#     Output array shape(len(ag_pos), len(ref_pos), 3)
#     :param ag_pos: atom group positions of shape (n_atoms, 3)
#     :type ag_pos: NDArray[np.float64]
#     distances: result array of shape (len(ag_pos), len(ref_pos), 3)
#     :type distances: NDArray[np.float64]
#     """
#     for atom_id, atom_pos in enumerate(ag_pos):
#         distances[atom_id, ...] = np.where(
#             np.ix_(ag_pos[..., 0]) != atom_id, atom_pos - ag_pos, 0
#         )


def select_solvent(
    center_ag: str, solvent_ag: MDAnalysis.core.groups.AtomGroup, radius: float
) -> AtomGroup:
    """Select solvent OW* atoms within sphere of
    specified radius around atom group
    :param center_ag: solvated atom group
    :type center_ag: MDAnalysis.core.groups.AtomGroup
    :param solvent_ag: solvent atom group
    :type center_ag: MDAnalysis.core.groups.AtomGroup
    :param radius: sphere radius
    :type radius: float
    :return: subsection of solvent_ag
    :rtype: MDAnalysis.core.groups.AtomGroup
    :"""
    return solvent_ag.select_atoms(
        f"name OW* and {radius} around global center_ag", updating=True
    )


def update_universe(f):
    """Decorator to update universe after function call"""
    wraps(f)

    def wrapper(
        crdname: str, crdout: Union[str, Path], **kwargs
    ) -> mda.Universe:
        if type(crdname) != Universe:
            u = mda.Universe(str(crdname))
            f(u=u, crdout=crdout, **kwargs)
            u = mda.Universe(str(crdout))
        else:
            u = crdname
        return u

    return wrapper


def get_n_mols(
    conc: Union[float, int],
    u: Universe,
    solvent: str = SOL,
    density: float = SOL_DENSITY,  # g/dm^3
):
    """Calculate number of molecules to add to reach target concentration.
    :param conc: target concentration in mol/L
    :type conc: Union[float, int]
    :param u: MDAnalysis universe
    :type u: Universe
    :param solvent: solvent residue name, defaults to SOL
    :type solvent: str, optional
    :param density: solvent density in g/dm^3, defaults to SOL_DENSITY
    :type density: float, optional
    :return: number of molecules to add"""
    sol = u.select_atoms(f"resname {solvent}")
    m = np.sum(sol.masses)  # g
    V = m / density  # L
    n_mols = conc * V
    n_mols = np.round(n_mols).astype(int)
    logger.finfo(
        "Calculating molecule numbers:\n"
        f"Target concentration = {conc:.3f} mol L-1\n"
        f"Bulk volume = {V:.2f} A3\n"
        f"Density = {density:.2f} g L-1\n"
        f"Molecules to add = {n_mols}\n"
    )
    return n_mols


def write_insert_dat(
    n_mols: Union[int, float], save: Union[str, Literal[False]], posz=0
):
    pos = np.zeros((int(n_mols), 3), dtype=np.float16)
    if posz != 0:
        pos[:, 2] = float(posz)
    if save:
        save = pl.Path(save)
        if save.suffix != ".dat":
            save = str(save.resolve()) + ".dat"
        logger.debug(f"Saving {n_mols} insert positions to {save}")
        np.savetxt(save, pos, fmt="%4.3f", delimiter=" ", newline="\n")
        with open(save, "r") as file:
            r = file.read()


def get_n_ucs(universe):
    """Get number of unit cells per sheet.
    :param universe: MDAnalysis universe
    :type universe: MDAnalysis.Universe
    :return: number of unit cells
    :rtype: int"""
    isl = universe.select_atoms("resname iSL")
    ions = universe.select_atoms(f"resname {' '.join(get_ion_list())}")


@temp_file_wrapper
def add_new_ff_to_top(
    topin: Union[str, pl.Path],
    topout: Union[str, pl.Path],
    force_field: Optional[ForceField] = None,
):
    logger.finfo(f"Adding new force field to {topin}")
    temp_top = ""
    topfile = TOPFile(topin)
    combined_top = TOPFile(topout)
    try:
        top_itp_files = force_field.itp_filelist
        for itp in top_itp_files:
            with open(itp, "r") as itpfile:
                temp_top += itpfile.read() + "\n\n"
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmpfile:
            tmpfile.write(temp_top)
            temp_top = TOPFile(tmpfile.name)
    except AttributeError:
        temp_top = ITPFile(force_field)
    except TypeError:
        logger.fdebug("No files provided.")
        return
    with open(combined_top, "w") as outfile:
        for kwd in topfile._kwd_dict.keys():
            if kwd in Parameter.kwd_list:
                logger.finfo(f"Checking {kwd}")
                df = topfile[kwd]
                logger.finfo(f"Data has {len(df.df)} entries")
                if df is None:
                    logger.finfo(f"kwd not in top file: {kwd}")
                    df = temp_top[kwd]
                    logger.finfo(f"Data has {len(df.df)} entries")
                    if df is None:
                        logger.finfo(f"kwd not in temp top file: {kwd}")
                        continue
                else:
                    if kwd == "defaults" or temp_top[kwd] is None:
                        logger.finfo(f"Skipping {kwd} addition")
                        pass
                    else:
                        logger.finfo(f"Adding {kwd}")
                        df = df + temp_top[kwd]
                        logger.finfo(f"Data has {len(df.df)} entries")
                logger.finfo(
                    f"Adding {len(df.string.splitlines())} lines to {kwd}\n"
                )
                outfile.write(f"\n[ {kwd} ]\n{df.string}\n\n")
            else:
                logger.finfo(f"Stopping at {kwd}")
                break
    mol_section = f"\n{topfile.get_section('moleculetype')}\n{temp_top.get_section('moleculetype')}\n"
    sys_section = f"\n{topfile.get_section('system')}\n"
    # combined_top.write()
    with open(combined_top, "a") as outfile:
        outfile.write(mol_section)
        outfile.write(sys_section)
    assert Path(combined_top).exists()


@temp_file_wrapper
def add_mol_list_to_top(
    topin: Union[str, pl.Path],
    topout: Union[str, pl.Path],
    insert_list: List[str],
):
    logger.debug(insert_list)
    with open(topin, "r") as topfile:
        topstr = topfile.read().rstrip()
    topmatch = re.search(
        r"\[ system \].*", topstr, flags=re.MULTILINE | re.DOTALL
    ).group(0)
    tophead = re.match(
        r"(.*)\[ system \].*", topstr, flags=re.MULTILINE | re.DOTALL
    ).group(1)
    if len(insert_list) != 0:
        topstr = "\n".join([tophead, topmatch, *insert_list])
    else:
        topstr = "\n".join([tophead, topmatch])
    with open(topout, "w") as topfile:
        topfile.write(topstr)
    assert Path(topout).exists()


@gmx_command_wrapper
@temp_file_wrapper
def add_ions_n_mols(
    odir: Path,
    crdin: Path,
    topin: Path,
    ion: str,
    n_atoms: int,
    gmx_commands,
    charge=None,
) -> int:
    """
    Add a selected number of ions.
    :param odir: output directory
    :type odir: Path
    :param crdin: input coordinates filename
    :type crdin: GROFile
    :param topin: input topology filename
    :type topin: TOPFile
    :param ion: ion atom type
    :type ion: str
    :param n_atoms: number of ions to insert
    :type n_atoms: int
    :param charge: ion type charge
    :type charge: Optional[int]
    :return: number of inserted ions
    :rtype: int
    """
    logger.debug(f"adding {n_atoms} {ion} ions to {crdin}")
    odir = Path(odir).resolve()
    assert odir.is_dir()
    tpr = odir / "add_ions.tpr"
    ndx = odir / "add_ions.ndx"
    gmx_commands.run_gmx_make_ndx(f=crdin, o=ndx)
    if charge is None:
        charge = int(get_ion_charges()[ion])
    if charge < 0:
        nname = ion
        nn = n_atoms
        nq = charge
        pname = "Na"
        pq = 1
        np = 0
    elif charge > 0:
        pname = ion
        np = n_atoms
        pq = charge
        nname = "Cl"
        nq = -1
        nn = 0
    if ndx.is_file():
        gmx_commands.run_gmx_grompp(
            c=crdin,
            p=topin,
            o=tpr,  # pp=topout,
            po=tpr.with_suffix(".mdp"),
            v="",
            maxwarn=1,
            run_type="GENION",
        )
        err, out = gmx_commands.run_gmx_genion_add_n_ions(
            s=tpr,
            p=topin,
            o=crdin,
            n=ndx,
            pname=pname,
            np=np,
            pq=pq,
            nname=nname,
            nq=nq,
            nn=nn,
        )
        logger.debug(f"{GROFile(crdin).universe.atoms.n_atoms} atoms")
        replaced = re.findall(
            "Replacing solvent molecule", err, flags=re.MULTILINE
        )
        logger.debug(
            f"Replaced {len(replaced)} SOL molecules with {ion} in {crdin.name!r}"
        )  # add_resnum(crdin=crdin, crdout=crdin)  # rename_il_solvent(crdin=crdin, crdout=crdin)
    else:
        raise RuntimeError(
            f"No index file {ndx.name} created!"
        )  # replaced = []
    return len(replaced)


@gmx_command_wrapper
@temp_file_wrapper
def add_ions_neutral(
    odir: Path,
    crdin: Path,
    topin: Path,
    topout: Path,
    nion: str,
    pion: str,
    gmx_commands,
    nq=None,
    pq=None,
) -> str:
    """
    Neutralise system charge with selected anion and cation types.
    :param odir: output directory
    :type odir: Path
    :param crdin: input coordinates filename
    :type crdin: GROFile
    :param crdout: output corrdiantes filename
    :type crdout: GROFile
    :param topin: input topology filename
    :type topin: TOPFile
    :param topout: output topology filename
    :type topout: TOPFile
    :param nion: anion name
    :type nion: str
    :param pion: cation name
    :type pion: str
    :param nq: anion charge
    :type nq: int
    :param pq: cation charge
    :type pq: int
    :return: number of inserted ions
    :rtype: int
    """
    logger.debug(f"Neutralising with {pion} and {nion}")
    odir = Path(odir).resolve()
    assert odir.is_dir()
    tpr = odir / "neutral.tpr"
    ndx = odir / "neutral.ndx"
    gmx_commands.run_gmx_make_ndx(f=crdin, o=ndx)
    if ndx.is_file():
        gmx_commands.run_gmx_grompp(
            c=crdin,
            p=topin,
            o=tpr,
            po=tpr.with_suffix(".mdp"),
            v="",
            maxwarn=1,
            run_type="GENION",
            pp=topout,
        )
        if nq is None:
            nq = int(get_ion_charges()[nion])
        if pq is None:
            pq = int(get_ion_charges()[pion])
        logger.debug("gmx grompp completed successfully.")
        err, out = gmx_commands.run_gmx_genion_neutralise(
            s=tpr,
            p=topin,
            o=crdin,
            n=ndx,
            pname=pion,
            pq=int(pq),
            nname=nion,
            nq=int(nq),
        )
        replaced = re.findall(
            "Replacing solvent molecule", err, flags=re.MULTILINE
        )
        logger.debug(
            f"Replaced {len(replaced)} SOL molecules in {crdin.name!r}"
        )  # add_resnum(crdin=crdout, crdout=crdout)  # rename_il_solvent(crdin=crdout, crdout=crdout)  # shutil.move('processed.top', topin)
    else:
        logger.ferror(f"No index file {ndx.name} created!")
        replaced = []
    return len(replaced)


@update_universe
def _remove_excess_gro_ions(
    u: MDAnalysis.Universe,
    crdout: Union[Path, str],
    n_ions: int,
    ion_type: str,
) -> Universe:
    """Remove excess ions from GRO file.
    :param u: MDAnalysis Universe
    :type u: Universe
    :param crdout: output coordinates filename
    :type crdout: Union[Path, str]
    :param n_ions: number of ions to remove
    :type n_ions: int
    :param ion_type: ion type
    :type ion_type: str
    :return: MDAnalysis Universe
    """
    last_sol_id = u.select_atoms("resname SOL")[-1].index
    ions = u.select_atoms(f"resname {ion_type}")
    remove_ions = u.atoms.select_atoms(
        f"index {last_sol_id + 1} - " f"{ions.indices[-1] - n_ions}"
    )
    u.atoms -= remove_ions
    logger.debug(f"Removing {remove_ions.n_atoms} " f"{ion_type} atoms")
    u.atoms.write(str(crdout))
    logger.debug(f"Writing new coordinates to {Path(crdout).resolve()!r}")
    return u


@temp_file_wrapper
def _remove_excess_top_ions(
    topin: Union[Path, str],
    topout: Union[Path, str],
    n_ions: int,
    ion_type: str,
) -> None:
    """Remove excess ions from topology file.
    :param topin: input topology filename
    :type topin: Union[Path, str]
    :param topout: output topology filename
    :type topout: Union[Path, str]
    :param n_ions: number of ions to remove
    :type n_ions: int
    :param ion_type: ion type
    :type ion_type: str
    """
    with open(topin, "r") as topfile:
        topstr = topfile.read()
    ion_matches = re.search(
        rf".*system.*({ion_type}\s+\d)*.*({ion_type}\s+{n_ions}.*)",
        topstr,
        flags=re.MULTILINE | re.DOTALL,
    )
    sol_matches = re.search(
        r"(.*system.*SOL\s+\d+\n).*", topstr, flags=re.MULTILINE | re.DOTALL
    )
    topstr = sol_matches.group(1) + ion_matches.group(2)
    with open(topout, "w") as topfile:
        logger.debug(f"Writing new topology to {Path(topout).resolve()!r}.")
        topfile.write(topstr)


@temp_file_wrapper
def remove_excess_ions(crdin, topin, crdout, topout, n_ions, ion_type) -> None:
    _remove_excess_top_ions(
        topin=topin, topout=topout, n_ions=n_ions, ion_type=ion_type
    )
    _remove_excess_gro_ions(
        crdname=crdin, crdout=crdout, n_ions=n_ions, ion_type=ion_type
    )


@temp_file_wrapper
def rename_il_solvent(
    crdin: MDAnalysis.Universe, crdout: Union[Path, str]
) -> Universe:
    u = Universe(str(crdin))
    if "isl" not in list(
        map(lambda n: n.lower(), np.unique(u.residues.resnames))
    ):
        logger.debug("Renaming interlayer SOL to iSL")
        isl: MDAnalysis.AtomGroup = u.select_atoms("resname SOL").residues
        try:
            idx: int = isl[np.ediff1d(isl.resnums, to_end=1) != 1][-1].resnum
        except IndexError:
            logger.finfo("No interlayer solvation")
        isl: MDAnalysis.AtomGroup = isl.atoms.select_atoms(f"resnum 0 - {idx}")
        isl.residues.resnames = "iSL"
        if type(crdout) != Path:
            crdout = Path(crdout)
        crdout = str(crdout.resolve())
        u.atoms.write(crdout)
    else:
        logger.debug("No interlayer SOL to rename")
        if str(Path(crdin).resolve()) != str(Path(crdout.resolve())):
            logger.debug(f"Overwriting {crdin.name!r}")
            shutil.move(crdin, crdout)
    return u


@temp_file_wrapper
def add_resnum(
    crdin: Union[Path, str],
    crdout: Union[Path, str],
    res_n_atoms: Optional[int] = None,
) -> Universe:
    """Add residue numbers to GRO file.
    :param crdin: input coordinates filename
    :type crdin: GROFile, str
    :param crdout: output corrdiantes filename
    :type crdout: GROFile
    :return: MDAnalysis Universe
    :rtype: Universe
    """
    u = Universe(str(crdin))
    logger.debug(f"Adding residue numbers to:\n{str(crdin)!r}")
    if res_n_atoms is None:
        res_n_atoms = get_system_n_atoms(crds=u, write=False)
    atoms: MDAnalysis.AtomGroup = u.atoms
    # for i in np.unique(atoms.residues.resnames):
    #     logger.debug(f"Found residues: {i} - {res_n_atoms[i]} atoms")
    res_idx = 1
    first_idx = 0
    last_idx = 0
    resids = []
    while last_idx < atoms.n_atoms:
        resname = atoms[last_idx].residue.resname
        n_atoms = res_n_atoms[resname]
        last_idx = first_idx + n_atoms
        first_idx = last_idx
        resids.extend(np.full(n_atoms, res_idx).tolist())
        res_idx += 1
    logger.debug(f"Added {len(resids)} residues")
    resids = list(map(lambda resid: f"{resid:5d}", resids))
    if type(crdout) != Path:
        crdout = Path(crdout)
    crdout = str(crdout.resolve())
    pattern = re.compile(r"^\s*\d+")
    with open(crdin, "r") as crdfile:
        crdlines = crdfile.readlines()
    crdlines = [line for line in crdlines if re.match(r"\s*\n", line) is None]
    new_lines = crdlines[:2]
    for linenum, line in enumerate(crdlines[2:-1]):
        line = re.sub(pattern, resids[linenum][:5], line)
        new_lines.append(line)
    new_lines.append(crdlines[-1])
    with open(crdout, "w") as crdfile:
        logger.debug(f"Writing coordinates to {str(crdout)!r}")
        for line in new_lines:
            crdfile.write(line)
    logger.debug(f"{crdfile.name!r} written")
    u = MDAnalysis.Universe(str(crdout))
    return u


update_wrapper(get_mol_n_atoms, "n_atoms")

update_wrapper(get_mol_charges, "charges")

update_wrapper(get_mol_charges, "masses")

update_wrapper(get_ion_charges, ion_itp)

update_wrapper(get_ion_n_atoms, ion_itp)

update_wrapper(get_ion_masses, ion_itp)

update_wrapper(get_clay_charges, "charges")

update_wrapper(get_clay_n_atoms, "n_atoms")

update_wrapper(get_clay_masses, "masses")

update_wrapper(get_sol_charges, "charges")

update_wrapper(get_sol_n_atoms, "n_atoms")

update_wrapper(get_sol_masses, "masses")

update_wrapper(get_aa_charges, "charges")

update_wrapper(get_aa_n_atoms, "n_atoms")

update_wrapper(get_aa_masses, "masses")

update_wrapper(get_all_charges, "charges")

update_wrapper(get_all_n_atoms, "n_atoms")

update_wrapper(get_all_masses, "masses")

update_wrapper(get_system_charges, "charges")

update_wrapper(get_system_n_atoms, "n_atoms")

update_wrapper(get_system_masses, "masses")


def remove_replaced_SOL(
    topin: Union[str, pl.Path],
    topout: Union[str, pl.Path],
    n_mols: int,
    debug_mode=False,
) -> None:
    """Remove specified number of SOL molecules from topology.
    :param topin: input topology filename
    :type topin: str
    :param topout: output topology filename
    :type topout: str
    :param n_mols: number of SOL molecules to remove
    :type n_mols: int
    :return: None"""
    if n_mols > 0:
        with open(topin, "r") as topfile:
            topstr = topfile.read()

        substr = r"(SOL\s*)([0-9]*)"

        pattern = rf"{substr}(?!.*{substr})"

        try:
            topmatch = re.search(
                pattern, topstr, flags=re.MULTILINE | re.DOTALL
            ).group(2)
        except AttributeError:
            raise ValueError("No solvent found")
        except IndexError:
            raise IndexError("Not enough interlayer solvent groups found!")
        n_sol = int(topmatch) - n_mols
        logger.fdebug(
            debug_mode, "Removing {n_mols} SOL residues from topology."
        )

        if n_sol < 0:
            raise ValueError(f"Invalid number of solvent residues: {n_sol}")

        else:
            topstr = re.sub(
                pattern, rf"\1 {n_sol}", topstr, flags=re.MULTILINE | re.DOTALL
            )

            with open(topout, "w") as topfile:
                logger.fdebug(
                    debug_mode,
                    f"New topology {topout.name!r} has {n_sol} SOL molecules.",
                )
                topfile.write(topstr)


@update_universe
def center_clay_universe(
    u: mda.Universe, crdout: Union[str, Path], uc_name: Optional[str]
) -> None:
    """Center clay unit cell in box and wrap coordinates
    :param u: MDAnalysis Universe
    :type u: mda.Universe
    :param crdout: output coordinates filename
    :type crdout: str
    :param uc_name: unit cell name
    :type uc_name: str
    :return: None"""
    from MDAnalysis.transformations.translate import center_in_box
    from MDAnalysis.transformations.wrap import wrap

    if uc_name is None:
        clay = u.select_atoms("not resname SOL iSL" + " ".join(get_ion_list()))
    else:
        clay = u.select_atoms(f"resname {uc_name}*")
    for ts in u.trajectory:
        ts = center_in_box(clay, wrap=True)(ts)
        ts = wrap(u.atoms)(ts)
    u.atoms.write(crdout)


@update_universe
def center_clay(
    u: Universe,
    crdout: Union[Path, str],
    uc_name: Optional[str],
    other_resnames=" ".join(get_ion_list()),
):
    """Center clay unit cell in box and wrap coordinates
    :param u: MDAnalysis Universe
    :type u: mda.Universe
    :param crdout: output coordinates filename
    :type crdout: str
    :param uc_name: unit cell name
    :type uc_name: str
    :return: None"""
    from MDAnalysis.transformations.translate import center_in_box
    from MDAnalysis.transformations.wrap import wrap

    if uc_name is None:
        clay = u.select_atoms(f"not resname SOL iSL {other_resnames}")
    else:
        clay = u.select_atoms(f"resname {uc_name}*")
    for ts in u.trajectory:
        ts = center_in_box(clay, wrap=True)(ts)
        ts = wrap(u.atoms)(ts)
    u.atoms.write(crdout)


@update_universe
def remove_ag(
    u: mda.Universe,
    crdout: str,
    selstr: str,
    last: Union[bool, int],
    first: Union[bool, int],
    debug_mode=False,
) -> None:
    sel = u.select_atoms(selstr)
    logger.fdebug(
        f"Before: {u.atoms.n_atoms}. "
        f"Removing first {first} last {last} {np.unique(sel.residues.resnames)}"
    )
    if first is not False:
        if last is not False:
            raise ValueError(
                "Not possible to select first and last ends of atom group at the same time"
            )
    elif last is not False:
        first = -last
        logger.fdebug("last not false", first)
    else:
        first = 0
    u.atoms -= sel[first:]
    logger.fdebug(debug_mode, f"After: {u.atoms.n_atoms}")
    u.atoms.write(crdout)


@gmx_command_wrapper
@temp_file_wrapper
def add_ions_conc(
    odir: Path,
    crdin: Path,
    crdout: Path,
    topin: Path,
    topout: Path,
    ion: str,
    ion_charge: float,
    conc: float,
    gmx_commands,
    debug_mode=False,
):
    """Add ions to system at specified concentration.
    :param odir: output directory
    :type odir: Path
    :param crdin: input coordinate filename
    :type crdin: Path
    :param crdout: output coordinate filename
    :type crdout: Path
    :param topin: input topology filename
    :type topin: Path
    :param topout: output topology filename
    :type topout: Path
    :param ion: ion name
    :type ion: str
    :param ion_charge: ion charge
    :type ion_charge: float
    :param conc: ion concentration
    :type conc: float
    :return: number of replaced molecules
    :rtype: int
    """
    logger.fdebug(debug_mode, "Adding {conc} mol/L {ion}")
    odir = Path(odir).resolve()
    assert odir.is_dir()
    tpr = odir / "conc.tpr"
    ndx = odir / "conc.ndx"
    shutil.copy(crdin, crdout)
    gmx_commands.run_gmx_make_ndx(f=crdout, o=ndx)
    if ndx.is_file():
        gmx_commands.run_gmx_grompp(
            c=crdout,
            p=topin,
            pp=topout,
            o=tpr,
            po=tpr.with_suffix(".mdp"),
            v="",
            maxwarn=1,
            # renum="",
            run_type="GENION",
        )
        err, out = gmx_commands.run_gmx_genion_conc(
            s=tpr,
            p=topin,
            o=crdout,
            n=ndx,
            iname=ion,
            iq=ion_charge,
            conc=conc,
        )
        replaced = re.findall(
            "Replacing solvent molecule", err, flags=re.MULTILINE
        )
        logger.finfo(
            f"Replaced {len(replaced)} SOL molecules in {crdin.name!r}"
        )
    else:
        logger.ferror(f"No index file {ndx.name} created!")
        replaced = []
    return len(replaced)


def check_insert_numbers(
    add_repl: Literal["Added", "Replaced"], searchstr: str
) -> int:
    """Get number of inserted/replaced molecules.

    :param add_repl: specify if molecules search for added or replaced molecules
    :type add_repl: 'Added' or 'Replaced'
    :param searchstr: string to be searched
    :type searchstr: str
    :return: number of added or replaced molecules
    :rtype: int or None if no match found

    """
    return int(
        re.search(
            rf".*{add_repl} (\d+)", searchstr, flags=re.MULTILINE | re.DOTALL
        ).group(1)
    )


# @gmx_command_wrapper
@temp_file_wrapper
def run_em(  # mdp: str,
    crdin: Union[str, Path],
    topin: Union[str, Path],
    topout: Union[str, Path],
    odir: Path,
    gmx_commands,
    outname: str = "em",
    mdp_prms: Optional[Dict[str, str]] = None,
    freeze_grps: Optional[Union[List[str], str], bool] = None,
    freeze_dims: Optional[
        Union[List[Union[Literal["Y"], Literal["N"]]], bool]
    ] = ["Y", "Y", "Y"],
    ndx=None,
    debug_mode=False,
    constraints=False,
) -> Union[str, None]:
    """
    Run an energy minimisation using gmx and
    return convergence information if successful.
    :param mdp_prms: mdp parameters
    :type mdp_prms: Optional[Dict[str, str]]
    :param freeze_dims: Freeze dimensions
    :type freeze_dims: Optional[Union[List[Union[Literal["Y"], Literal["N"]]], bool]]
    :param freeze_grps: Freeze groups
    :type freeze_grps: Optional[Union[List[Union[Literal["Y"], Literal["N"]]], bool]]
    :param mdp: mdp parameter file path
    :type mdp: Union[Path, str]
    :param crdin: Input coordinate file name
    :type crdin: Union[Path, str]
    :param topin: In put topology file name
    :type topin: Union[Path, str]
    :param tpr: Output tpr file name
    :type tpr: Union[Path, str]
    :param odir: Output directory path
    :type odir: Path
    :param outname: Default stem for output files
    :type outname: str
    :param gmx_commands: gmx_commands object
    :type gmx_commands: GMXCommands
    :return: Convergence information message
    :rtype: Union[str, None]
    """
    logger.debug("# MINIMISING ENERGY")
    outname = (Path(odir) / outname).resolve()
    if topout is None:
        topout = outname.with_suffix(".top")
    elif type(topout) == str:
        topout = (Path(odir) / topout).with_suffix(".top")
    if topin.resolve() == topout.resolve():
        tmp_top = tempfile.NamedTemporaryFile(
            prefix=topin.stem, suffix=".top", dir=odir
        )
        topout = tmp_top.name
        logger.debug(f"Creating temporary output file {tmp_top}")
        otop_copy = True
    else:
        otop_copy = False
    tpr = outname.with_suffix(".tpr")
    gmx_commands.run_gmx_grompp(
        c=crdin,
        p=topin,
        o=tpr,
        pp=topout,
        v="",
        po=outname.with_suffix(".mdp"),
        mdp_prms=mdp_prms,
        run_type="EM",
        freeze_grps=freeze_grps,
        freeze_dims=freeze_dims,
        renum="",
        n=ndx,
        define=constraints,
    )
    error, em, out = gmx_commands.run_gmx_mdrun(s=tpr, deffnm=outname, v="")
    if error is None:
        conv = re.search(
            r"converged to Fmax < (\d+) in (\d+) steps",
            em,
            flags=re.MULTILINE | re.DOTALL,
        )
        if conv is None:
            logger.ferror("Energy minimisation run not converged!\n")
            logger.finfo(em)  # logger.finfo(out)
        else:
            fmax, n_steps = conv.groups()
            final_str = (
                re.search(
                    r"(?<=writing lowest energy coordinates.\n).*(?=GROMACS reminds you)",
                    em,
                    flags=re.DOTALL | re.MULTILINE,
                )
                .group(0)
                .strip("\n")
            )
            logger.info(f"{final_str}\n")
            logger.fdebug(debug_mode, "Output written to {outname.name!r}")
            conv = (
                f"Fmax: {fmax}, reached in {n_steps} steps."
                f"Output written to {outname!r}\n"
            )
            conv = re.search(
                r"GROMACS reminds you.*?(?=\n)", em, re.MULTILINE
            ).group(0)
        if otop_copy is True:
            shutil.copy(topout, topin)
    else:
        conv = False
    return conv


@temp_file_wrapper
@gmx_command_wrapper
def neutralise_system(
    odir: Path,
    crdin: Path,
    topin: Path,
    topout: Path,
    nion: str,
    pion: str,
    gmx_commands,
    # mdp_template: Path = MDP / "genion.mdp",
):
    logger.debug("neutralise_system")
    # mdp = MDP / "genion.mdp"
    # assert mdp_template.exists(), f"{mdp_template.resolve()} does not exist"
    odir = Path(odir).resolve()
    assert odir.is_dir()
    # make_opath = lambda p: odir / f"{p.stem}.{p.suffix}"
    tpr = odir / "neutral.tpr"
    ndx = odir / "neutral.ndx"
    # isl = grep_file(crdin, 'iSL')
    gmx = gmx_commands.run_gmx_make_ndx(f=crdin, o=ndx)
    if ndx.is_file():
        # if topin.resolve() == topout.resolve():
        #     topout = topout.parent / f"{topout.stem}_n.top"
        #     otop_copy = True
        # else:
        #     otop_copy = False
        _, out = gmx_commands.run_gmx_grompp(
            f=MDP / "genion.mdp",
            c=crdin,
            p=topin,
            o=tpr,
            pp=topout,
            po=tpr.with_suffix(".mdp"),
            v="",
            maxwarn=1,
            # renum="",
        )
        # isl = grep_file(crdin, 'iSL')
        err = re.search(r"error", out)
        if err is not None:
            logger.error(f"gmx grompp raised an error!")
            replaced = []
        else:
            logger.debug(f"gmx grompp completed successfully.")
            out = gmx_commands.run_gmx_genion_neutralise(
                s=tpr,
                p=topout,
                o=crdin,
                n=ndx,
                pname=pion,
                pq=int(get_ion_charges()[pion]),
                nname=nion,
                nq=int(get_ion_charges()[nion]),
            )
            if not topout.is_file():
                logger.error(f"gmx genion raised an error!")
            else:
                logger.info(f"gmx genion completed successfully.")
                # add_resnum(crdname=crdin, crdout=crdin)
                # rename_il_solvent(crdname=crdin, crdout=crdin)
                # isl = grep_file(crdin, 'iSL')
                # if otop_copy is True:
                #     shutil.move(topout, topin)
            replaced = re.findall(
                "Replacing solvent molecule", out.stderr, flags=re.MULTILINE
            )
            logger.info(f"{crdin.name!r} add numbers, rename il solv")
            add_resnum(crdin=crdin, crdout=crdin)
            rename_il_solvent(crdin=crdin, crdout=crdin)
    else:
        logger.error(f"No index file {ndx.name} created!")
        replaced = []
    return len(replaced)
