#!/usr/bin/env python3
import logging
import pickle as pkl
import re
import sys
import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, List, Literal, NoReturn, Optional, Union

import cython
import MDAnalysis as mda
import numpy as np
import pandas as pd
from ClayAnalysis.analysisbase import ClayAnalysisBase
from ClayAnalysis.classes import SimDir
from ClayAnalysis.lib import (
    check_traj,
    get_dist,
    get_selections,
    process_box,
    run_analysis,
    save_selection,
    search_ndx_group,
    select_cyzone,
)
from MDAnalysis.lib.distances import apply_PBC, minimize_vectors
from tqdm import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)

__all__ = ["mod_traj"]

logger = logging.getLogger(Path(__file__).stem)


def mod_traj(
    crdin: Union[Path, str],
    trajin: Optional[Union[Path, str]] = None,
    outstem: Optional[str] = None,
    crdout_suffix: Optional[str] = None,
    trajout_suffix: Optional[str] = None,
    sel: Optional[List[str]] = None,
    clay_type: Optional[List[str]] = None,
    in_memory: bool = False,
    ndx=False,
    overwrite: bool = False,
    start: int = 0,
    step: int = 1,
):
    infiles = []
    crdin = check_path(crdin)
    crdout = get_outname(
        infile=crdin, outstem=outstem, outfile_suffix=crdout_suffix
    )
    fmt = crdin.suffix.strip(".").upper()
    trajout = None
    if ndx is True:
        ondx = Path(crdout).with_suffix(".ndx").resolve()
        if ondx.is_file():
            with open(ondx, "r") as ndx_file:
                ndx_str = ndx_file.read()
        else:
            ndx_str = ""
    if trajin is not None:
        trajin = check_path(trajin)
        trajout = get_outname(
            infile=trajin, outstem=outstem, outfile_suffix=crdout_suffix
        )
        if not trajout.is_file() or overwrite is True:
            infiles.append(str(crdin))
            infiles.append(str(trajin))
            fmt = trajin.suffix.strip(".").upper()
        else:
            raise FileExistsError(f"{trajout} already exists.")
    else:
        logger.info("Continuing without trajectory")
        if not crdout.is_file() or overwrite is True:
            infiles.append(str(crdin))
        else:
            raise FileExistsError(f"{crdout} already exists.")
    print(infiles, fmt)
    u = MDAnalysis.Universe(*infiles, format=fmt, in_memory=in_memory)
    sel_list = []
    for sel_item in sel:
        sel_list.append(u.select_atoms(sel_item))
    if clay_type is not None:
        clay = u.select_atoms(f"resname {clay_type}* and name OB* o*")
        outsel = clay
        logger.info(
            f"Selected {clay.n_atoms} atoms of "
            f"{clay.n_residues} {clay_type!r} unit cells"
        )
        if ndx is True:
            group_name = "clay"
            atom_name = np.unique(clay.atoms.names)[0][:2]
            group_name += f"_{atom_name}"
            if not search_ndx_group(ndx_str=ndx_str, sel_name=group_name):
                clay.write(ondx, name=group_name, mode="a")
        for sel_id, sel_item in enumerate(sel_list):
            sel_list[sel_id] = sel_item.select_atoms(
                f"not resname {clay_type} and"
                f" (prop z >= {np.max(clay.positions[:, 2] - 1)} or"
                f" prop z <= {np.min(clay.positions[:, 2] + 1)})"
            )
    for sel_id, sel_item in enumerate(sel_list):
        outsel += sel_item
        logger.info(
            f"{sel_id+1}.\nSelected {sel_item.n_atoms} atoms of names: {np.unique(sel_item.names)} "
            f"(residues: {np.unique(sel_item.resnames)})"
        )
        if ndx is True:
            group_name = np.unique(sel_item.residues.resnames)[0]
            group_name = re.match("[a-zA-Z]*", group_name).group(0)
            atom_name = np.unique(sel_item.atoms.names)[0][:2]
            group_name += f"_{atom_name}"
            if not search_ndx_group(ndx_str=ndx_str, sel_name=group_name):
                sel_item.write(ondx, name=group_name, mode="a")
    if not crdout.is_file() or overwrite is True:
        try:
            outsel.write(str(crdout), frames=outsel.universe.trajectory[-1::1])
        except:
            outsel.write(str(crdout))
    if trajout is not None:
        outsel.write(str(trajout), frames=outsel.universe.trajectory[::step])
    logger.info("Finished writing all files.")


def check_path(name: Union[str, Path]) -> Path:
    path = Path(name)
    if not path.is_file():
        raise FileNotFoundError(f"No file found for {path.resolve()!r}")
    return Path(path.resolve())


def get_outname(infile: Path, outstem: str, outfile_suffix: str) -> Path:
    if outfile_suffix is None:
        outfile_suffix = infile.suffix
    else:
        outfile_suffix = outfile_suffix.lstrip(".")
        outfile_suffix = f".{outfile_suffix}"
    crdout = infile.with_name(f"{outstem}{outfile_suffix}")
    return crdout


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="trajmod",
        description="Write new trajectories.",
        add_help=True,
        allow_abbrev=False,
    )
    parser.add_argument(
        "-crdin",
        type=str,
        help="Coordinate input file name",
        metavar=("coordinates"),
        dest="crdin",
        required=True,
    )
    parser.add_argument(
        "-trajin",
        type=str,
        help="Trajectory input file name",
        metavar=("trajectory"),
        dest="trajin",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-outstem",
        type=str,
        help="Output file stem",
        dest="outstem",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-sel",
        type=str,
        nargs="+",
        help="Atom type selection strings",
        dest="sel",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-uc",
        type=str,
        help="Clay unit cell type",
        dest="clay_type",
        required=False,
    )
    parser.add_argument(
        "-cfmt",
        type=str,
        help="Output coordinate format",
        dest="crdout_suffix",
        required=False,
    )
    parser.add_argument(
        "-tfmt",
        type=str,
        help="Output trajectory format",
        dest="trajout_suffix",
        required=False,
    )
    parser.add_argument(
        "-ndx",
        default=False,
        help="Write index file.",
        dest="ndx",
        action="store_true",
    )
    parser.add_argument(
        "-start",
        type=int,
        help="First frame",
        dest="start",
        required=False,
        default=0,
    )
    parser.add_argument(
        "-step",
        type=int,
        help="Frame step",
        dest="step",
        required=False,
        default=1,
    )
    parser.add_argument(
        "-overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing files.",
        dest="overwrite",
    )
    parser.add_argument(
        "-mem",
        action="store_true",
        default=False,
        help="Store trajectory in memory.",
        dest="in_memory",
    )
    args = parser.parse_args(sys.argv[1:])
    mod_traj(
        crdin=args.crdin,
        trajin=args.trajin,
        outstem=args.outstem,
        crdout_suffix=args.crdout_suffix,
        trajout_suffix=args.trajout_suffix,
        clay_type=args.clay_type,
        sel=args.sel,
        in_memory=args.in_memory,
        ndx=args.in_memory,
        overwrite=args.overwrite,
        start=args.start,
        step=args.step,
    )
