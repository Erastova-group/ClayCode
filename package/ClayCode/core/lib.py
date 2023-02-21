#!/usr/bin/env python3
import logging
import os
import pathlib as pl
import re
import shutil
import sys
import tempfile
from functools import partial, update_wrapper
from pathlib import Path, PosixPath
import pickle as pkl
from typing import (
    NoReturn,
    Union,
    List,
    Optional,
    Literal,
    TypeVar,
    overload,
    Tuple,
    Callable,
    Dict,
    Any,
    cast,
    Sequence,
)

import MDAnalysis
import MDAnalysis as mda
import MDAnalysis.coordinates
import numpy as np
import pandas as pd
from MDAnalysis import Universe
from MDAnalysis.lib.distances import minimize_vectors
from MDAnalysis.lib.mdamath import triclinic_vectors
from numpy.typing import NDArray

from ClayCode.config._consts import SOL, SOL_DENSITY, IONS, MDP, FF, DATA, AA, UCS
from ClayCode.core import gmx
# from ClayCode.analysisbase import analysis_class

# from ClayAnalysis.gmx import run_gmx_convert_tpr
from ClayAnalysis.utils import change_suffix, grep_file

tpr_logger = logging.getLogger("MDAnalysis.topology.TPRparser").setLevel(
    level=logging.WARNING
)

MaskedArray = TypeVar("MaskedArray")
AtomGroup = TypeVar("AtomGroup")

__name__ = "lib"
__all__ = [
    "process_orthogonal",
    "process_triclinic",
    "select_cyzone",
    "get_dist",
    "select_solvent",
    "process_box",
    "exclude_xyz_cutoff",
    "check_traj",
    "run_analysis",
    "get_selections",
    "get_mol_prms",
    "get_system_charges",
    "process_orthogonal_axes",
    "process_triclinic_axes",
]

logger = logging.getLogger("lib")

def init_temp_inout(
    inf: Union[str, Path],
    outf: Union[str, Path],
    new_tmp_dict: Dict[Union[Literal["crd"], Literal["top"]], bool],
    which=Union[Literal["crd"], Literal["top"]],
) -> Tuple[
    Union[str, Path],
    Union[str, Path],
    Dict[Union[Literal["crd"], Literal["top"]], bool],
]:
    inp = Path(inf)
    outp = Path(outf)
    if inp == outp:
        temp_outp = tempfile.NamedTemporaryFile(
            suffix=outp.suffix, prefix=outp.stem, dir=outp.parent
        )
        # outp = inp.with_stem(inp.stem + "_tmp")
        outp = Path(temp_outp.name)
        new_tmp_dict[which] = True
        logger.info(f"Creating temporary output file {outp.name!r}")
    else:
        temp_outp = None
    if type(inf) == str:
        inp = str(inp.resolve())
    if type(outf) == str:
        temp_outp = str(outp.resolve())
    return inp, outp, new_tmp_dict, temp_outp


def fix_gro_residues(crdin: Union[Path, str], crdout: Union[Path, str]):
    u = mda.Universe(crdin)
    if np.unique(u.residues.resnums).tolist() == [1]:
        u = add_resnum(crdin=crdin, crdout=crdout)
    if "iSL" not in u.residues.resnames:
        u = rename_il_solvent(crdin=crdout, crdout=crdout)
    return u


def temp_file_wrapper(f: Callable):
    def wrapper(**kwargs):
        kwargs_dict = locals()["kwargs"]
        fargs_dict = {}
        new_tmp = {}
        for ftype in ["crd", "top"]:
            if f"{ftype}in" in kwargs_dict.keys():
                fargs_dict[f"{ftype}in"] = kwargs_dict[f"{ftype}in"]
            if f"{ftype}out" in kwargs_dict.keys():
                (
                    fargs_dict[f"{ftype}in"],
                    fargs_dict[f"{ftype}out"],
                    new_tmp,
                    temp_outp,
                ) = init_temp_inout(
                    kwargs_dict[f"{ftype}in"],
                    kwargs_dict[f"{ftype}out"],
                    new_tmp_dict=new_tmp,
                    which=ftype,
                )
            elif f"{ftype}out" in kwargs_dict.keys():
                fargs_dict[f"{ftype}out"] = kwargs_dict[f"{ftype}out"]
        for k, v in fargs_dict.items():
            locals()["kwargs"][k] = v
        r = f(**kwargs_dict)
        for ftype, new in new_tmp.items():
            if new is True:
                infile = Path(fargs_dict[f"{ftype}in"])
                outfile = Path(fargs_dict[f"{ftype}out"])
                assert outfile.exists(), f"No file generated!"
                shutil.copy(outfile, infile)
                logger.info(f"Renaming {outfile.name!r} to {infile.name!r}")
        return r

    return wrapper


def run_analysis(instance: analysis_class, start: int, stop: int, step: int):
    """Run MDAnalysis analysis.
    :param start: First frame, defaults to None
    :type start: int, optional
    :param stop: Last frame, defaults to None
    :type stop: int, optional
    :param step: Iteration step, defaults to None
    :type step: int, optional
    """
    kwarg_dict = {}
    for k, v in {"start": start, "step": step, "stop": stop}.items():
        if v is not None:
            kwarg_dict[k] = v
    instance.run(**kwarg_dict)


@overload
def get_selections(
    infiles: Sequence[Union[str, Path, PosixPath]],
    sel: Sequence[str],
    clay_type: str,
    other: Sequence[str],
    in_memory: bool,
) -> Tuple[AtomGroup, AtomGroup, AtomGroup]:
    ...


@overload
def get_selections(
    infiles: Sequence[Union[str, Path, PosixPath]],
    sel: Sequence[str],
    clay_type: str,
    other: None,
    in_memory: bool,
) -> Tuple[AtomGroup, AtomGroup]:
    ...


def get_selections(
    infiles, sel, clay_type, other=None, in_memory=False
):  # , save_new=True):
    """Get MDAnalysis atom groups for clay, first and optional second selection.
    :param in_memory: store trajectory to memory
    :type in_memory: bool
    :param clay_type: Unit cell type
    :type clay_type: str
    :param infiles: Coordinate and trajectory files
    :type infiles: Sequence[Union[str, Path, PosixPath]]
    :param sel: selection keywords as [resname] or [resname, atom type] or 'not SOL'
    :type sel: Sequence[str]
    :param other: selection keywords as [resname] or [resname, atom type], defaults to None
    :type other: Sequence[str], optional
    # :raises ValueError: lengths of sel or other != in [1, 2]
    # :return sel: atom group for sel
    # :rtype sel: MDAnalysis.core.groups.AtomGroup
    # :return clay: atom group for clay
    # :rtype clay: MDAnalysis.core.groups.AtomGroup
    # :return other: atom group for other, optional
    # :rtype other: MDAnalysis.core.groups.AtomGroup
    """
    infiles = [str(Path(infile).absolute()) for infile in infiles]
    for file in infiles:
        logger.info(f"Reading: {file!r}")
    u = MDAnalysis.Universe(*infiles, in_memory=in_memory)
    # only resname specified
    if len(sel) == 1:
        sel = u.select_atoms(f"resname {sel[0]}")
    # rename and atom type specified
    elif len(sel) == 2:
        # expand search string for terminal O atom types
        if sel[1] == "OT*":
            sel[1] = "OT* O OXT"
        sel = u.select_atoms(f"resname {sel[0]}* and name {sel[1]}")
    else:
        raise ValueError('Expected 1 or 2 arguments for "sel"')
    if other is None:
        pass
    elif len(other) == 1:
        logger.debug(f"other: {other}")
        other = u.select_atoms(f"resname {other[0]}")
    elif len(other) == 2:
        logger.debug(f"other: {other}")
        if other[1] == "OT*":
            other[1] = "OT* O OXT"
        other = u.select_atoms(f"resname {other[0]}* and name {other[1]}")
    else:
        raise ValueError('Expected 1 or 2 arguments for "other"')
    clay = u.select_atoms(f"resname {clay_type}* and name OB* o*")
    logger.info(
        f"'clay': Selected {clay.n_atoms} atoms of "
        f"{clay.n_residues} {clay_type!r} unit cells"
    )

    sel = select_outside_clay_stack(sel, clay)

    # Clay + two other atom groups selected
    if other is not None:
        other = select_outside_clay_stack(other, clay)
        return sel, clay, other

    # Only clay + one other atom group selected
    else:
        return sel, clay

def select_outside_clay_stack(atom_group: MDAnalysis.AtomGroup,
                              clay: MDAnalysis.AtomGroup
                              ):
    atom_group = atom_group.select_atoms(
        f" prop z >= {np.max(clay.positions[:, 2] - 1)} or"
        f" prop z <= {np.min(clay.positions[:, 2] + 1)}"
    )
    logger.info(
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
):
    ocoords = Path(outname).with_suffix(".gro").resolve()
    opdb = Path(outname).with_suffix(".pdbqt").resolve()
    logger.info(f"Writing coordinates and trajectory ({traj!r})")
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
    # logger.info(f'ag : {outsel.n_atoms} {outsel.atoms}')
    for ag in atom_groups[1:]:
        if ndx is True:
            group_name = np.unique(ag.residues.resnames)[0]
            group_name = re.match("[a-zA-Z]*", group_name).group(0)
            atom_name = np.unique(ag.atoms.names)[0][:2]
            group_name += f"_{atom_name}"
            if not search_ndx_group(ndx_str=ndx_str, sel_name=group_name):
                ag.write(ondx, name=group_name, mode="a")
        outsel += ag
        logger.info(
            f"New trajectory from {len(atom_groups)} groups with {outsel.n_atoms} total atoms"
        )
    logger.info(f"1. {ocoords!r}")
    outsel.write(str(ocoords))
    if pdbqt is True:
        logger.info(f"2. {opdb!r}")
        outsel.write(str(opdb), frames=outsel.universe.trajectory[-1::1])
    if type(traj) != list:
        traj = [traj]
    for t in traj:
        otraj = outname.with_suffix(t).resolve()
        logger.info(f"3. {otraj!r}")
        outsel.write(str(otraj), frames="all")
        # logger.info(f'4. {otrr}')
        # outsel.write(str(otrr), frames='all')


def check_traj(instance: analysis_class, check_len: Union[int, Literal[False]]) -> None:
    """Check length of trajectory in analysis class instance.
    :param instance: analysis class instance
    :type instance: analysis_class
    :param check_len: expected number of trajectory frames, defaults to False
    :type check_len: Union[int, Literal[False]]
    :raises SystemExit: Error if trajectory length != check_len
    """
    logger.debug(f"Checking trajectory length: {check_len}")
    if type(check_len) == int:
        if instance._universe.trajectory.n_frames != check_len:
            raise SystemExit(
                "Wrong number of frames: " f"{instance._universe.trajectory.n_frames}"
            )


def process_box(instance: analysis_class) -> None:
    """Assign distance minimisation function in orthogonal or triclinic box.

    Correct x, x2, z interatomic distances for periodic boundary conditions
    in orthogonal box inplace
                                         O*
       +--------------+       +---------/----+
       |       S      |       |       S      |
       |        \     |  -->  |              |
       |         \    |       |              |
       |          O   |       |              |
       +--------------+       +--------------+

    :param instance: analysis class instance
    :type instance: analysis_class
    """
    box = instance._universe.dimensions
    if np.all(box[3:] == 90.0):
        instance._process_distances = process_orthogonal
        instance._process_axes = process_orthogonal_axes
    else:
        instance._process_distances = process_triclinic
        instance._process_axes = process_triclinic_axes


def process_orthogonal_axes(
    distances: NDArray[np.float64], dimensions: NDArray[np.float64], axes: List[int]
) -> None:
    """
    Correct x, x2, z interatomic distances for periodic boundary conditions
    in orthogonal box inplace

    :param axes:
    :type axes:
    :param distances: interatomic distance array of shape (n, m, 3)
    :type distances: NDArray[np.float64]
    :param dimensions: simulation box dimension array of shape (6,)
    :type dimensions: NDArray[np.float64]
    :return: no return
    :rtype: NoReturn
    """
    assert (
        distances.shape[-1] == len(axes) or distances.ndim == 2
    ), f"Shape of distance array ({distances.shape[-1]}) does not match selected axes {axes}"
    # logger.info(distances / dimensions[:3][axes], np.rint(distances / dimensions[:3][axes]))
    for idx, dist in np.ma.ndenumerate(distances):
        distances[idx] -= dimensions[:3][axes] * np.rint(dist / dimensions[:3][axes])


def process_orthogonal(
    distances: NDArray[np.float64], dimensions: NDArray[np.float64]
) -> None:
    """
    Correct x, x2, z interatomic distances for periodic boundary conditions
    in orthogonal box inplace

    :param distances: interatomic distance array of shape (n, m, 3)
    :type distances: NDArray[np.float64]
    :param dimensions: simulation box dimension array of shape (6,)
    :type dimensions: NDArray[np.float64]
    :return: no return
    :rtype: NoReturn
    """
    # logger.info(distances[:3] / dimensions[:3], np.rint(distances / dimensions[:3]))
    # for idx, dist in np.ma.ndenumerate(distances[:, :, 0]):
    # old_dist = distances.copy()
    distances -= dimensions[:3] * np.rint(distances / dimensions[:3])
    # print(distances[distances != old_dist])

def process_triclinic_axes(
    distances: NDArray[np.float64], dimensions: NDArray[np.float64], axes: List[int]
) -> None:
    """
    Correct x, x2, z interatomic distances for periodic boundary conditions
    in triclinic box inplace

    :param axes:
    :type axes:
    :param distances: interatomic distance array of shape (n, m, 3)
    :type distances: NDArray[np.float64]
    :param dimensions: simulation box dimension array of shape (6,)
    :type dimensions: NDArray[np.float64]
    :return: no return
    :rtype: NoReturn
    """
    box = triclinic_vectors(dimensions)
    assert distances.shape[-1] >= len(
        axes
    ), f"Shape of distance array ({distances.shape[-1]}) does not match selected axes {axes}"
    logger.info(
        distances / np.diagonal(box)[..., axes],
        np.rint(distances / np.diagonal(box)[..., axes]),
    )
    distances -= np.diagonal(box)[..., axes] * np.rint(
        distances / np.diagonal(box)[..., axes]
    )


def process_triclinic(
    distances: NDArray[np.float64], dimensions: NDArray[np.float64]
) -> None:
    """
    Correct x, x2, z interatomic distances for periodic boundary conditions
    in triclinic box inplace

    :param distances: interatomic distance array of shape (n, m, 3)
    :type distances: NDArray[np.float64]
    :param dimensions: simulation box dimension array of shape (6,)
    :type dimensions: NDArray[np.float64]
    :return: no return
    :rtype: NoReturn
    """
    box = triclinic_vectors(dimensions)
    # logger.info(distances / np.diagonal(box), np.rint(distances / np.diagonal(box)))
    distances -= np.diagonal(box) * np.rint(distances / np.diagonal(box))


def select_cyzone(
    distances: MaskedArray, z_dist: float, xy_rad: float, mask_array: MaskedArray
) -> None:
    """
    Select all distances corresponding to atoms within a cylindrical volume
    of dimensions +- z_dist and radius xy_rad
    :param distances: masked interatomic distance array of shape (n, m, 3)
    :type distances: MaskedArray[np.float64]
    :param z_dist: absolute value for cutoff in z direction
    :type z_dist: float
    :param xy_rad: absolute value for radius in xy plane
    :type xy_rad: float
    :param mask_array: array for temporary mask storage of shape (n, m)
    :type mask_array: MaskedArray[np.float64]
    :return: no return
    :rtype: NoReturn
    """
    z_col = distances[:, :, 2]
    z_col.mask = np.abs(distances[:, :, 2]) > z_dist
    distances.mask = np.broadcast_to(z_col.mask[:, :, np.newaxis], distances.shape)
    np.ma.sum(distances[:, :, [0, 1]].__pow__(2), axis=2, out=mask_array)
    # print(np.min(mask_array))
    # print(xy_rad.__pow__(2))
    mask_array.harden_mask()
    mask_array.mask = mask_array > xy_rad.__pow__(2)
    np.copyto(distances.mask, mask_array.mask[:, :, np.newaxis])
    ...

def exclude_xyz_cutoff(
    distances: NDArray[np.int64],
    cutoff: float,
) -> None:
    """
    Select all distances corresponding to atoms within a box
    with length 2* cutoff
    :param distances: masked interatomic distance array of shape (n, m, 3)
    :type distances: NDArray[np.float64]
    :param cutoff: absolute value for maximum distance
    :type cutoff: float
    :return: no return
    :rtype: NoReturn
    """
    mask = np.any(np.abs(distances) >= cutoff, axis=2)
    np.copyto(distances.mask, mask[:, :, np.newaxis])
    # distances.mask += np.abs(distances) > cutoff
    # distances.mask = np.bitwise_and.accumulate(distances.mask, axis=2)


def exclude_z_cutoff(
    distances: NDArray[np.int64],
    cutoff: float,
) -> None:
    """
    Select all distances corresponding to atoms within a box
    with length 2* cutoff
    :param distances: masked interatomic distance array of shape (n, m, 3)
    :type distances: NDArray[np.float64]
    :param cutoff: absolute value for maximum distance
    :type cutoff: float
    :return: no return
    :rtype: NoReturn
    """
    mask = np.abs(distances[..., 2]) > cutoff
    distances.mask += np.broadcast_to(mask[..., np.newaxis], distances.shape)

def get_dist(
    ag_pos: NDArray[np.float64],
    ref_pos: NDArray[np.float64],
    distances: NDArray[np.float64],
    box: NDArray[np.float64],
) -> NoReturn:
    """Calculate minimum elementwise x, x2, z distances
    of selection atom positions to reference atom positions in box.
    Output array shape(len(ag_pos), len(ref_pos), 3)
    :param ag_pos: atom group positions of shape (n_atoms, 3)
    :type ag_pos: NDArray[np.float64]
    :param ref_pos: atom group positions of shape (n_atoms, 3)
    :type ref_pos: NDArray[np.float64]
    distances: result array of shape (len(ag_pos), len(ref_pos), 3)
    :type distances: NDArray[np.float64]
    :param box: Timestep dimensions array of shape (6, )
    :type box: NDArray[np.float64]
    """
    for atom_id, atom_pos in enumerate(ag_pos):
        distances[atom_id, :, :] = minimize_vectors(atom_pos - ref_pos, box)


def get_self_dist(
    ag_pos: NDArray[np.float64],
    distances: NDArray[np.float64],
) -> NoReturn:
    """Calculate minimum elementwise x, x2, z distances
    of selection atom positions to reference atom positions in box.
    Output array shape(len(ag_pos), len(ref_pos), 3)
    :param ag_pos: atom group positions of shape (n_atoms, 3)
    :type ag_pos: NDArray[np.float64]
    distances: result array of shape (len(ag_pos), len(ref_pos), 3)
    :type distances: NDArray[np.float64]
    :param box: Timestep dimensions array of shape (6, )
    :type box: NDArray[np.float64]
    """
    for atom_id, atom_pos in enumerate(ag_pos):
        distances[atom_id, ...] = np.where(
            np.ix_(ag_pos[..., 0]) != atom_id, atom_pos - ag_pos, 0
        )


# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
# def _minimize_vectors_ortho(cython.floating[:, :] vectors not None, cython.floating[:] box not None,
#                             cython.floating[:, :] output not None):
#     cdef int i, n
#     cdef cython.floating box_inverse[3]
#     cdef cython.floating[:] box_inverse_view
#
#     box_inverse[0] = 1.0 / box[0]
#     box_inverse[1] = 1.0 / box[1]
#     box_inverse[2] = 1.0 / box[2]
#
#     box_inverse_view = box_inverse
#
#     n = len(vectors)
#     with nogil:
#         for i in range(n):
#             output[i, 0] = vectors[i, 0]
#             output[i, 1] = vectors[i, 1]
#             output[i, 2] = vectors[i, 2]
#             _minimum_image_orthogonal(output[i, :], box, box_inverse_view)

# https://github.com/MDAnalysis/mdanalysis/blob/develop/package/MDAnalysis/lib/c_distances.pyx

# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef inline void _minimum_image_orthogonal(cython.floating[:] dx,
#                                            cython.floating[:] box,
#                                            cython.floating[:] inverse_box) nogil:
#     """Minimize dx to be the shortest vector
#     Parameters
#     ----------
#     dx : numpy.array, shape (3,)
#       vector to minimize
#     box : numpy.array, shape (3,)
#       box length in each dimension
#     inverse_box : numpy.array, shape (3,)
#       inverse of box
#     Operates in-place on dx!
#     """
#     cdef int i
#     cdef cython.floating s
#
#     for i in range(3):
#         if box[i] > 0:
#             s = inverse_box[i] * dx[i]
#             dx[i] = box[i] * (s - cround(s))


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
    def wrapper(crdname: str, crdout: Union[str, Path], **kwargs) -> mda.Universe:
        u = mda.Universe(str(crdname))
        f(u=u, crdout=crdout, **kwargs)
        u = mda.Universe(str(crdout))
        return u

    return wrapper


def get_n_mols(
    conc: Union[float, int],
    u: Universe,
    solvent: str = SOL,
    density: float = SOL_DENSITY,  # g/dm^3
):
    sol = u.select_atoms(f"resname {solvent}")
    m = np.sum(sol.masses)  # g
    V = m / density  # L
    n_mols = conc * V
    n_mols = np.round(n_mols).astype(int)
    logger.info(
        "Calculating molecule numbers:\n"
        f"Target concentration = {conc:.3f} mol L-1\n"
        f"Bulk volume = {V:.2f} A3\n"
        f"Density = {density:.2f} g L-1\n"
        f"Molecules to add = {n_mols}\n"
    )
    return n_mols


def write_insert_dat(n_mols: Union[int, float], save: Union[str, Literal[False]]):
    pos = np.zeros((int(n_mols), 3), dtype=np.float16)
    if save:
        save = pl.Path(save)
        if save.suffix != ".dat":
            save = str(save.resolve()) + ".dat"
        logger.debug(f"Saving {n_mols} insert positions to {save}")
        np.savetxt(save, pos, fmt="%4.3f", delimiter=" ", newline="\n")
        with open(save, "r") as file:
            r = file.read()


@update_universe
def center_clay(u: Universe, crdout: Union[Path, str], uc_name: Optional[str]):
    from MDAnalysis.transformations.translate import center_in_box
    from MDAnalysis.transformations.wrap import wrap

    if uc_name is None:
        clay = u.select_atoms("not resname SOL iSL" + " ".join(IONS))
    else:
        clay = u.select_atoms(f"resname {uc_name}*")
    for ts in u.trajectory:
        ts = center_in_box(clay, wrap=True)(ts)
        ts = wrap(u.atoms)(ts)
    u.atoms.write(crdout)


@temp_file_wrapper
def add_mol_list_to_top(
    topin: Union[str, pl.Path],
    topout: Union[str, pl.Path],
    insert_list: List[str],
    ff_path: Union[pl.Path, str] = FF,
):
    logger.debug(insert_list)
    with open(topin, "r") as topfile:
        topstr = topfile.read().rstrip()
    topmatch = re.search(
        r"\[ system \].*", topstr, flags=re.MULTILINE | re.DOTALL
    ).group(0)
    ff_path = pl.Path(ff_path)
    if not ff_path.is_dir():
        raise FileNotFoundError(
            f"Specified force field path: {ff_path.resolve()!r} does not exist!"
        )
    with open(ff_path / f"new_tophead.itp", "r") as tophead:
        tophead = tophead.read()
    if len(insert_list) != 0:
        topstr = "\n".join([tophead, topmatch, *insert_list])
    else:
        topstr = "\n".join([tophead, topmatch])
    with open(topout, "w") as topfile:
        topfile.write(topstr)
    assert Path(topout).exists()


@temp_file_wrapper
def neutralise_system(
    odir: Path, crdin: Path, topin: Path, topout: Path, nion: str, pion: str
):
    logger.debug("neutralise_system")
    mdp = MDP / "genion.mdp"
    assert mdp.exists(), f"{mdp.resolve()} does not exist"
    odir = Path(odir).resolve()
    assert odir.is_dir()
    # make_opath = lambda p: odir / f"{p.stem}.{p.suffix}"
    tpr = odir / "neutral.tpr"
    ndx = odir / "neutral.ndx"
    # isl = grep_file(crdin, 'iSL')
    gmx.run_gmx_make_ndx(f=crdin, o=ndx)
    if ndx.is_file():
        # if topin.resolve() == topout.resolve():
        #     topout = topout.parent / f"{topout.stem}_n.top"
        #     otop_copy = True
        # else:
        #     otop_copy = False
        _, out = gmx.run_gmx_grompp(
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
            out = gmx.run_gmx_genion_neutralise(
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


@update_universe
def _remove_excess_gro_ions(
    u: MDAnalysis.Universe, crdout: Union[Path, str], n_ions: int, ion_type: str
) -> None:
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
    topin: Union[Path, str], topout: Union[Path, str], n_ions: int, ion_type: str
) -> None:
    with open(topin, "r") as topfile:
        topstr = topfile.read()
    ion_matches = re.search(
        rf".*system.*({ion_type}\s+\d)*.*({ion_type}\s+{n_ions}.*)",
        topstr,
        flags=re.MULTILINE | re.DOTALL,
    )
    sol_matches = re.search(
        rf"(.*system.*SOL\s+\d+\n).*", topstr, flags=re.MULTILINE | re.DOTALL
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
def rename_il_solvent(crdin: MDAnalysis.Universe, crdout: Union[Path, str]) -> None:
    u = Universe(str(crdin))
    if "isl" not in list(map(lambda n: n.lower(), np.unique(u.residues.resnames))):
        logger.info(f"Renaming interlayer SOL to iSL")
        isl: MDAnalysis.AtomGroup = u.select_atoms("resname SOL").residues
        idx: int = isl[np.ediff1d(isl.resnums, to_end=1) != 1][-1].resnum
        isl: MDAnalysis.AtomGroup = isl.atoms.select_atoms(f"resnum 0 - {idx}")
        isl.residues.resnames = "iSL"
        if type(crdout) != Path:
            crdout = Path(crdout)
        crdout = str(crdout.resolve())
        u.atoms.write(crdout)
    else:
        logger.info(f"No interlayer SOL to rename")
        if str(Path(crdin).resolve()) != str(Path(crdout.resolve())):
            logger.info(f"Overwriting {crdin.name!r}")
            shutil.move(crdin, crdout)
    return u


@temp_file_wrapper
def add_resnum(crdin: Union[Path, str], crdout: Union[Path, str]) -> Universe:
    u = Universe(str(crdin))
    # print(str(crdin))
    logger.info(f"Adding residue numbers to:\n{crdin.resolve()!r}")
    res_n_atoms = get_system_n_atoms(crds=u, write=False)
    atoms: MDAnalysis.AtomGroup = u.atoms
    for i in np.unique(atoms.residues.resnames):
        logger.info(f"Found residues: {i} - {res_n_atoms[i]} atoms")
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
    logger.info(f"added {len(resids)} residues")
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
        line = re.sub(pattern, resids[linenum], line)
        new_lines.append(line)
    new_lines.append(crdlines[-1])
    with open(crdout, "w") as crdfile:
        logger.debug(f"Writing coordinates to {str(crdout)!r}")
        for line in new_lines:
            crdfile.write(line)
    logger.info(f"{crdfile.name!r} written")
    u = MDAnalysis.Universe(str(crdout))
    return u

    # new_resnums = []
    # for res_idx, res in enumerate(atoms):
    #     new_resnums.append(np.full(res_n_atoms[res.resname], res_idx+1, dtype=int))
    # print(res, res_n_atoms[res.resname])


# class PrmInfo:
#     def __init__(self,
#                  name: Literal[Union['charges', 'n_atoms']],
#                  include_dir: Union[str, Path] = FF,
#                 write=False,
#                 force_update=False,
#                  ):
#         self.name =


PRM_INFO_DICT = {
    "n_atoms": cast(
        Callable[[Universe], Dict[str, int]],
        lambda u: dict(
            [(r, u.select_atoms(f"moltype {r}").n_atoms) for r in u.atoms.moltypes]
        ),
    ),
    "charges": cast(
        Callable[[Universe], Dict[str, float]],
        lambda u: dict(zip(u.atoms.moltypes, np.round(u.atoms.residues.charges, 4))),
    ),
}


def get_mol_prms(
    prm_str: str,
    itp_file: Union[str, pl.Path],
    include_dir: Union[str, pl.Path] = FF,
    write=False,
    force_update=False,
) -> dict:
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
            prop_dict = pkl.read(prop_file)
    return prop_dict


get_mol_n_atoms = partial(get_mol_prms, prm_str="n_atoms")
update_wrapper(get_mol_n_atoms, "n_atoms")

get_mol_charges = partial(get_mol_prms, prm_str="charges")
update_wrapper(get_mol_charges, "charges")

PRM_METHODS = {"charges": get_mol_charges, "n_atoms": get_mol_n_atoms}
# def get_residue_n_atoms(
#     residue_itp: Union[str, pl.Path],
#     include_dir: Union[str, pl.Path] = FF,
#     write=False,
#     force_update=False,
# ):
#     residue_itp = Path(residue_itp)
#     n_atoms_file = DATA / f"{residue_itp.stem}_n_atoms.p"
#     if not n_atoms_file.is_file() or force_update is True:
#         atom_u = Universe(
#             str(residue_itp),
#             topology_format="ITP",
#             include_dir=str(include_dir),
#             infer_system=True,
#         )
#         n_atoms_dict = dict([(r, atom_u.select_atoms(f'moltype {r}').n_atoms) for r in atom_u.atoms.moltypes])
#         if write is True:
#             with open(n_atoms_file, "wb") as n_atoms_file:
#                 pkl.dump(n_atoms_dict, n_atoms_file)
#     else:
#         with open(n_atoms_file, "rb") as n_atoms_file:
#             n_atoms_dict = pkl.read(n_atoms_file)
#     return n_atoms_dict

# def get_atom_type_charges(
#     atom_itp: Union[str, pl.Path],
#     include_dir: Union[str, pl.Path] = FF,
#     write=False,
#     force_update=False,
# ):
#     atom_itp = Path(atom_itp)
#     charge_file = DATA / f"{atom_itp.stem}_charges.p"
#     if not charge_file.is_file() or force_update is True:
#         atom_u = Universe(
#             str(atom_itp),
#             topology_format="ITP",
#             include_dir=str(include_dir),
#             infer_system=True,
#         )
#         charge_dict = dict(
#             zip(atom_u.atoms.moltypes, np.round(atom_u.atoms.residues.charges, 4))
#         )
#         if write == True:
#             with open(charge_file, "wb") as charge_file:
#                 pkl.dump(charge_dict, charge_file)
#     else:
#         with open(charge_file, "rb") as charge_file:
#             charge_dict = pkl.read(charge_file)
#     return charge_dict

ion_itp = FF / "Ion_Test.ff/ions.itp"
get_ion_charges = partial(get_mol_charges, itp_file=ion_itp)
update_wrapper(get_ion_charges, ion_itp)

get_ion_n_atoms = partial(get_mol_n_atoms, itp_file=ion_itp)
update_wrapper(get_ion_charges, ion_itp)


def get_ion_prms(prm_str: str, **kwargs):
    if prm_str == "charges":
        prm_dict = get_ion_charges(**kwargs)
    elif prm_str == "n_atoms":
        prm_dict = get_ion_n_atoms(**kwargs)
    else:
        raise KeyError(f"Unexpected parameter: {prm_str!r}")
    return prm_dict


# get_clay_charges = partial(get_atom_type_charges, atom_itp=FF/"ClayFF_Fe.ff/ffnonbonded.itp")


def get_clay_prms(prm_str: str, uc_name: str, uc_path=UCS, force_update=False):
    prm_func = PRM_METHODS[prm_str]
    prm_file = DATA / f"{uc_name.upper()}_{prm_str}.pkl"
    if not prm_file.is_file() or force_update is True:
        charge_dict = {}
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
update_wrapper(get_clay_charges, "charges")

get_clay_n_atoms = partial(get_clay_prms, prm_str="n_atoms")
update_wrapper(get_clay_n_atoms, "n_atoms")


def get_sol_prms(
    prm_str: str,
    sol_path=FF / "ClayFF_Fe.ff",
    include_dir: Union[str, pl.Path] = FF,
    force_update=False,
):
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
update_wrapper(get_sol_charges, "charges")

get_sol_n_atoms = partial(get_sol_prms, prm_str="n_atoms")
update_wrapper(get_sol_n_atoms, "n_atoms")


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
update_wrapper(get_aa_charges, "charges")

get_aa_n_atoms = partial(get_aa_prms, prm_str="n_atoms")
update_wrapper(get_aa_n_atoms, "n_atoms")


def get_all_prms(prm_str, force_update=True, write=True, name=None):
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
                get_aa_prms(prm_str=prm_str, aa_name=aa, force_update=force_update)
            )
        clay_types = ["D21"]
        clay_dict = {}
        for uc in clay_types:
            clay_dict.update(
                get_clay_prms(prm_str=prm_str, uc_name=uc, force_update=force_update)
            )
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
update_wrapper(get_all_charges, "charges")

get_all_n_atoms = partial(get_all_prms, prm_str="n_atoms")
update_wrapper(get_all_n_atoms, "n_atoms")


def get_system_prms(
    prm_str, crds: Union[str, Path, Universe], write=True, force_update=True
) -> Union[str, pd.Series, None]:
    if type(crds) == Universe:
        u = crds
        name = "universe"
    else:
        try:
            u = Universe(str(crds))
            name = Path(crds).stem
        except ValueError:
            logger.error(f"Could not create Universe from {crds}")
            return None
    prm_df = pd.Series(
        get_all_prms(prm_str, write=write, force_update=force_update), name=name
    )
    if prm_str == "charges":
        residue_df = pd.Series(u.residues.resnames, name="residues", dtype="str")
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
update_wrapper(get_system_charges, "charges")

get_system_n_atoms = partial(get_system_prms, prm_str="n_atoms")
update_wrapper(get_system_n_atoms, "n_atoms")


# def neutralise_charge(
#         charge,
#         crdin: Union[str, pl.Path],
#         crdout: Union[str, pl.Path],
#         topin: Union[str, pl.Path],
#         topout: Union[str, pl.Path],
#         ion_itp: Union[str, pl.Path],
#         include_dir: Union[str, pl.Path],
#         pion: Optional[str] = "Na",
#         nion: Optional[str] = "Cl",
# ):
#     if charge == 0:
#         return
#     else:
#         logger.debug("\n# NEUTRALISING EXCESS CHARGE:\n")
#         ion_u = Universe(
#             str(ion_itp),
#             topology_format="ITP",
#             include_dir=str(include_dir),
#             infer_system=True,
#         )
#         bulk_ions = [pion, nion]
#         ion_sel = [
#             ion_u.select_atoms(f"resname {pion}"),
#             ion_u.select_atoms(f"resname {nion}"),
#         ]
#         ion_charges = list(np.sum(ion[0].charge) for ion in ion_sel)
#         pion = {k: v for k, v in zip(bulk_ions, ion_charges) if v > 0}
#         nion = {k: v for k, v in zip(bulk_ions, ion_charges) if v < 0}
#         n_ions = [0, 0]
#         if len(pion) > 1 or len(nion) > 1:
#             raise KeyError(
#                 f"expected one value for positive/negative bulk ion, found {len(pion)}/{len(nion)}"
#             )
#         if charge < 0:
#             ion_id = 0
#         else:
#             ion_id = 1
#         with open(topin, "r") as topfile:
#             topstr = topfile.read()
#         substr = rf"({bulk_ions[ion_id]}\s*)([0-9]*)"
#         pattern = rf"{substr}(?!.*{substr})"
#         topmatch = re.search(pattern, topstr, flags=re.MULTILINE | re.DOTALL).group(2)
#         logger.debug(f"Found {topmatch} {bulk_ions[ion_id]} ions")
#         add_ions = int(abs(charge // ion_charges[ion_id]))
#         n_ions[ion_id] = int(add_ions)
#         logger.debug(f"Adding {add_ions} {bulk_ions[ion_id]} to topology.")
#         if ion_charges[ion_id] != 1:
#             remainder = int((charge % ion_charges[ion_id]))
#             if remainder != 0:
#                 n_ions[ion_id] += ion_id
#                 logger.debug("Remaining non-zero charge.")
#                 other_id = np.abs(ion_id - 1)
#                 if np.abs(ion_charges[other_id]) != 1:
#                     bulk_ions[other_id] = "Cl"
#                     ion_charges[other_id] = -1
#                 n_ions[other_id] = np.abs(remainder)
#                 logger.debug(
#                     f"Adding {n_ions[other_id]} atoms of {bulk_ions[other_id]}."
#                 )
#         ndx = "replace.ndx"
#         tpr = "genion.tpr"
#         if topin is not topout:
#             shutil.copyfile(topin, topout)
#         gmx.run_gmx_select(s=crdin, f=crdin, select="'SOL'", on=ndx)
#         _, grompp = gmx.run_gmx_grompp(
#             f="genion.mdp", c=crdin, p=topout, o=tpr, maxwarn=1
#         )
#         err = re.search(r"error", grompp)
#         if err is None:
#             crdout, err = gmx.run_gmx_genion(
#                 n=ndx,
#                 s=tpr,
#                 p=topout,
#                 o=crdout,
#                 pname=bulk_ions[0],
#                 np=int(n_ions[0]),
#                 pq=int(ion_charges[0]),
#                 nname=bulk_ions[1],
#                 nn=int(n_ions[1]),
#                 nq=int(ion_charges[1]),
#             )
#             replaced = re.findall("Replacing solvent molecule", err, flags=re.MULTILINE)
#             if len(replaced) == np.sum(n_ions):
#                 logger.debug(
#                     f"Inserted {n_ions[0]} {bulk_ions[0]} and {n_ions[1]} {bulk_ions[1]} ions."
#                 )
#         else:
#             logger.debug(grompp)
#             shutil.rmtree(crdin.parent)
#             logger.debug("\ngmx grompp raised error!")


def add_mols_to_top(
    topin: Union[str, pl.Path],
    topout: Union[str, pl.Path],
    insert: Union[None, str, pl.Path],
    n_mols: int,
    include_dir,
):
    if n_mols != 0:
        itp_path = pl.Path(insert).parent.resolve()
        itp = pl.Path(insert).stem
        # insertgro = Universe(str(itp_path / f'{itp}.crdin'))
        insert = Universe(
            str(itp_path / f"{itp}.itp"), topology_format="ITP", include_dir=include_dir
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
    # topstr_iter = topstr.splitlines()

    # for line in topstr_iter:
    #     match =re.search(r'^\s*[#]\s*include\s*["'+r"']"+r'([A-Za-z0-9./_])["'+r"']" , line)
    #     if match is not None:
    #         top_list.append(match.group(1))
    # for file in top_list:
    #     with open(file, 'r') as topfile:
    #         topstr=topfile.read().rstrip()
    #     topstr = topstr.splitlines()
    #     topstr_iter.extend(topstr)
    # prm_dict=OrderedDict()
    # prms=['defaults', 'atomtypes','bondtypes','pairtypes','angletypes', 'dihedraltypes',
    #       'constrainttypes', 'nonbond_params', 'moleculetype', 'atoms', 'system', 'molecules']
    # for prm in prms:
    #     prm_dict[prm] = []
    # prm_str=get_search_str(prm_dict)
    #
    # match_list=[]
    # kwd=None
    # topstr_iter=iter(topstr_iter)
    # while True:
    #     try:
    #         line=next(topstr_iter)
    #
    #         match=re.search(rf'{prm_str}', line)
    #         if match is None and kwd is None:
    #             pass
    #         elif match is not None:
    #
    #             kwd = match.group(0)
    #
    #             if len(match_list) != 0:
    #                 if last_kwd == 'defaults':
    #                     for item in match_list:
    #                         if item.strip().startswith(';'):
    #                             match_list.remove(item)
    #                 prm_dict[last_kwd].extend([match_list])
    #             match_list = []
    #             last_kwd=kwd
    #             line=next(topstr_iter)
    #         elif match is None and re.search(r'[[][a-zA-Z\s]+[]]', line) is not None:
    #             if kwd is None:
    #                 pass
    #             # else:
    #             #     last_kwd=kwd
    #             match, kwd= None, None
    #         line=line.strip().split(sep='\s')
    #         if kwd is not None and len(line) != 0:#and not line[0].strip().startswith(';'):
    #
    #             # if ';', in line:
    #             #     line = line[:line.index(';')]
    #
    #             line=''.join(line) + '\n'
    #             match_list.extend([line])
    #     except StopIteration:
    #         prm_dict[last_kwd].extend([match_list])
    #         break
    #
    # parameters =['defaults', 'atomtypes', 'bondtypes', 'pairtypes', 'angletypes', 'dihedraltypes',
    #              'constrainttypes', 'nonbond_params']
    # molecules = ['moleculetype', 'atoms']
    with open(topout, "w") as topfile:
        # for prm in parameters:
        #     topfile.write(f'\n[ {prm} ]\n')
        #     for p in prm_dict[prm]:
        #
        #         topfile.writelines(p)
        #     # topfile.write(*prm_dict[prm])
        # for mol_id, mol in enumerate(prm_dict['moleculetype']):
        #
        #     #               f'{prm_dict["moleculetype"][mol_id]}\n'
        #     #               '\n[ atoms ]\n')
        #
        #     topfile.write('\n[ moleculetype ]\n')
        #     for lines in prm_dict["moleculetype"][mol_id]:
        #         topfile.writelines(lines)
        #     topfile.write('\n[ atoms ]\n')
        #     for lines in prm_dict['atoms'][mol_id]:
        #         topfile.writelines(lines)
        # topfile.write('\n[ system ]\n')
        # for lines in prm_dict['system']:
        #     topfile.writelines(lines)
        # topfile.write('\n[ molecules ]\n')
        #
        # for lines in prm_dict['molecules']:
        #
        #     topfile.writelines(lines)
        if n_mols != 0:
            topfile.write(
                "\n".join(
                    [tophead, topmatch, f"\n{insert.residues.moltypes[0]}  {n_mols}\n"]
                )
            )
        else:
            topfile.write("\n".join([tophead, topmatch]))

    # # substr=r'([#]\s*include\s*[."A-Za-z0-9\s]\{-}\n)'
    # # topstr = re.sub(rf'({substr})', '\1' + f'{itp_path}/{itp}.itp', topstr, flags=re.MULTILINE)
    # substr=r'^\s*\[ system \]'
    # topstr = re.sub(rf'({substr})', rf'\n#include "{itp_path}/{itp}.itp"\n\1', topstr,
    #                 flags=re.MULTILINE)
    #
    #
    # topstr += f'\n{insert.residues.moltypes[0]}  {n_mols}\n'
    #
    # with open(topout, 'w') as topfile:
    #     topfile.write(topstr)


# def remove_charge_Cl(
#         topin: Union[str, pl.Path],
#         topout: Union[str, pl.Path],
#         insert: Union[str, pl.Path],
#         n_mols: int,
#         include_dir,
# ):
#     itp_path = pl.Path(insert).parent.resolve()
#     itp = pl.Path(insert).stem
#     insert = Universe(
#         str(itp_path / f"{itp}.top"),
#         topology_format="ITP",
#         include_dir=include_dir,
#         infer_system=True,
#     )
#     insert_charge = np.sum(insert.residues.charges)
#
#     insert_charge *= n_mols
#     insert_charge = np.round(insert_charge, 3)
#
#     if insert_charge < 0:
#         with open(topin, "r") as topfile:
#             topstr = topfile.read()
#
#         substr = r"(Cl\s*)([0-9]*)"
#
#         pattern = rf"{substr}(?!.*{substr})"
#
#         try:
#             topmatch = re.search(pattern, topstr, flags=re.MULTILINE | re.DOTALL).group(
#                 2
#             )
#             n_cl = int(topmatch) + insert_charge
#
#             if n_cl < 0:
#                 pass
#
#             else:
#                 topstr = re.sub(pattern, rf"\1 {n_cl}", flags=re.MULTILINE | re.DOTALL)
#                 with open(topout, "w") as topfile:
#                     topfile.write(topstr)
#
#         except:
#             pass
#


# def remove_charge_ions(topin, topout, charge, ion_itp, include_dir, pion: str, nion: str):
#     if charge == 0:
#         logger.debug("charge", 0)
#         return pion, 0, 0
#     else:
#         logger.debug("charge", charge)
#         ion_u = Universe(
#             str(ion_itp),
#             topology_format="ITP",
#             include_dir=include_dir,
#             infer_system=True,
#         )
#         bulk_ions = (pion, nion)
#         ion_sel = (
#             ion_u.select_atoms(f"resname {pion}"),
#             ion_u.select_atoms(f"resname {nion}"),
#         )
#         ion_charges = tuple(np.sum(ion[0].charge) for ion in ion_sel)
#         pion = {k: v for k, v in zip(bulk_ions, ion_charges) if v > 0}
#         nion = {k: v for k, v in zip(bulk_ions, ion_charges) if v < 0}
#         if len(pion) > 1 or len(nion) > 1:
#             raise KeyError(
#                 f"expected one value for positive/negative bulk ion, found {len(pion)}/{len(nion)}"
#             )
#         if charge > 0:
#             ion_id = 0
#         else:
#             ion_id = 1
#         with open(topin, "r") as topfile:
#             topstr = topfile.read()
#         substr = rf"({bulk_ions[ion_id]}\s*)([0-9]*)"
#         pattern = rf"{substr}(?!.*{substr})"
#         try:
#             topmatch = re.search(pattern, topstr, flags=re.MULTILINE | re.DOTALL).group(
#                 2
#             )
#             logger.debug(f"Found {topmatch} {bulk_ions[ion_id]} ions")
#             remove_ions = int(abs(charge // ion_charges[ion_id]))
#             remainder = int((charge % ion_charges[ion_id]))
#             n_ions = int(int(topmatch) - remove_ions)
#             logger.debug(f"Removing {remove_ions} {bulk_ions[ion_id]} from topology.")
#             if remainder != 0:
#                 cl_substr = rf"(Cl\s*)([0-9]*)"
#                 logger.debug("Remainder not 0")
#                 cl_pattern = rf"{cl_substr}(?!.*{cl_substr})"
#                 cl_match = re.search(
#                     cl_pattern, topstr, flags=re.MULTILINE | re.DOTALL
#                 ).group(2)
#                 topstr = re.sub(
#                     pattern,
#                     rf"\1 {int(cl_match) - remainder}",
#                     topstr,
#                     flags=re.MULTILINE | re.DOTALL,
#                 )
#                 logger.debug(f"Removing {cl_match} {bulk_ions[1]} from topology.")
#                 n_cl = int(cl_match)
#             else:
#                 n_cl = 0
#             topstr = re.sub(
#                 pattern, rf"\1 {n_ions}", topstr, flags=re.MULTILINE | re.DOTALL
#             )
#             with open(topout, "w") as topfile:
#                 logger.debug(
#                     f"Removing {remove_ions} {bulk_ions[ion_id]} from topology {topout.name!r}"
#                 )
#                 topfile.write(topstr)
#
#         except:
#             remove_ions = 0
#             logger.debug("No matching ions to remove")
#             raise KeyError("Not matching ions found")
#         return bulk_ions[ion_id], remove_ions, n_cl


def remove_replaced_SOL(
    topin: Union[str, pl.Path], topout: Union[str, pl.Path], n_mols: int
):
    if n_mols > 0:
        with open(topin, "r") as topfile:
            topstr = topfile.read()

        substr = r"(SOL\s*)([0-9]*)"

        pattern = rf"{substr}(?!.*{substr})"

        try:
            topmatch = re.search(pattern, topstr, flags=re.MULTILINE | re.DOTALL).group(
                2
            )
            n_sol = int(topmatch) - n_mols
            logger.debug(
                f"Removing {n_mols} SOL residues from topology."
            )  # , topmatch)

            if n_sol < 0:
                raise ValueError

            else:
                topstr = re.sub(
                    pattern, rf"\1 {n_sol}", topstr, flags=re.MULTILINE | re.DOTALL
                )

                with open(topout, "w") as topfile:
                    logger.debug(
                        f"New topology {topout.name!r} has {n_sol} SOL molecules."
                    )
                    topfile.write(topstr)
        except:
            raise ValueError


@update_universe
def center_clay_universe(
    u: mda.Universe, crdout: Union[str, Path], uc_name: Optional[str]
) -> None:
    from MDAnalysis.transformations.translate import center_in_box
    from MDAnalysis.transformations.wrap import wrap

    if uc_name is None:
        clay = u.select_atoms("not resname SOL iSL" + " ".join(IONS))
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
) -> None:
    sel = u.select_atoms(selstr)
    logger.debug(
        f"Before: {u.atoms.n_atoms}. "
        f"Removing first {first} last {last} {np.unique(sel.residues.resnames)}"
    )
    if first is not False:
        if last is not False:
            raise ValueError(
                f"Not possible to select first and last ends of atom group at the same time"
            )
    elif last is not False:
        first = -last
        logger.debug("last not false", first)
    else:
        first = 0
    u.atoms -= sel[first:]
    logger.debug(f"After: {u.atoms.n_atoms}")
    u.atoms.write(crdout)
