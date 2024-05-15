#!/usr/bin/env python3

from __future__ import annotations

import logging
import os
import pathlib as pl
import random
import re
import sys
import warnings
from argparse import ArgumentParser
from typing import Any, List, Tuple, Union

import MDAnalysis as mda
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.NullHandler()],
)

warnings.filterwarnings("ignore")

WDIR = pl.Path(".")

parser = ArgumentParser(
    description="Remove water molecules from a simulation run file.",
    # help="Remove water molecules from a simulation run file.",
)
parser.add_argument(
    "-p",
    metavar="run_path",
    type=pl.Path,
    help="Path to the simulation run file.",
    dest="run_path",
)
parser.add_argument(
    "-n",
    metavar="n_wat_to_remove",
    type=int,
    help="Number of water molecules to remove per unit cell or sheet.",
    dest="n_wat_to_remove",
)
parser.add_argument(
    "-d",
    metavar="d_spacing",
    type=float,
    help="Target d-spacing in angstrom.",
    dest="d_spacing",
)
parser.add_argument(
    "-u",
    metavar="n_ucs",
    type=int,
    default=1,
    help="Number of unit cells per sheet that will be "
    "used as multiplication factor for the number "
    "of water molecules to remove.",
    required=False,
    dest="n_ucs",
)


def raise_sys_exit(*optional_printargs: List[str]) -> None:
    """Raise SystemExit with usage message.
    :param optional_printargs: Optional arguments to print.
    """
    if optional_printargs:
        optional_print = "\n".join(optional_printargs)
        logger.info(optional_print)
    raise SystemExit(
        f"Usage: {sys.argv[0]} <run_number> "
        "<run_name> <waters_to_remove> "
        "<target_d_spacing>"
        "<num_ucs>"
    )


def get_sim_fname(
    simnum: int, simname: str, ext: str, path: pl.Path = WDIR
) -> pl.Path:
    """Get the path to a simulation file.
    :param simnum: Simulation number.
    :param simname: Simulation name.
    :param ext: File extension.
    :param path: Path to simulation files.
    :return: Path to simulation file.
    """
    return path / f"{simname}_{simnum:02d}/{simname}_{simnum:02d}.{ext}"


def remove_top_iSL(
    intop: pl.Path, otop: pl.Path, n_wat_to_remove: int
) -> None:
    """Remove iSL residues from a topology file.
    :param intop: Path to input topology file.
    :param otop: Path to output topology file.
    :param n_wat_to_remove: Number of iSL residues to remove.
    """
    with open(intop, "r") as topfile:
        topstr_full = topfile.read()
    topstr_top, topstr = topstr_full.split(r"[ system ]")  # split top file
    topstr = r"[ system ]" + topstr  # add delimiter back into split file
    re_top_iSL = re.compile(
        r"(?<=iSL)(\s+)(\d+)"
    )  # writes a pattern to search for "iSL" and matches what it preceeds into subgroups of one or more spaces and one or more decimal digits
    n_iSL = int(
        re_top_iSL.search(topstr).group(2)
    )  # searches the file for the patterns and returns the second subgroup (the digits)
    # n_IL = len(
    #     re_top_iSL.findall(topstr)
    # )  # returns the length of a list of all non-overlapping matches of the pattern in the split file
    logger.info(f"Topology contains {n_iSL} iSL solvent residues.")
    topstr = re_top_iSL.sub(
        rf"\1 {int(n_iSL - n_wat_to_remove)}", topstr
    )  # find the digits from the pattern and substracts the number of waters removed from all unit cells
    logger.info(f'Removing {n_wat_to_remove} iSL residues from "{intop}".')
    with open(otop, "w") as topfile:
        logger.info(f'Writing new topology to "{otop}".')
        topfile.write(topstr_top + topstr)  # writes an updated top file


def get_paths(inpath: pl.Path) -> Tuple[pl.Path, pl.Path, pl.Path, pl.Path]:
    """Get input and output paths for a given run path."""
    path = inpath.resolve().parents[1]
    logger.info(f"Inspecting files in '{path}':")
    run_name = inpath.stem
    n_run = int(run_name.split("_")[-1])
    new_name = run_name.strip(f"_{n_run:02d}") + f"_{n_run + 1:02d}"
    inpath = path / f"{run_name}/{run_name}"
    outpath = path / f"{new_name}/{new_name}"
    ingro = inpath.with_suffix(".gro")
    outgro = outpath.with_suffix(".gro")
    intop = inpath.with_suffix(".top")
    outtop = outpath.with_suffix(".top")
    if not outpath.parent.is_dir():
        logger.info(f"\tCreating output directory {outpath.parent}")
        os.mkdir(outpath.parent)
    return ingro, outgro, intop, outtop


def remove_waters(
    inpath: pl.Path,
    n_wat_to_remove: int,
    target_spacing: float,
    non_clay_residues: List[str] = [
        "iSL",
        "SOL",
        "Cl",
        "Na",
        "Ca",
        "Mg",
        "K",
        "Ba",
    ],
) -> int | Any:
    """Remove water molecules from a simulation run file.
    :param inpath: Path to the simulation run file.
    :param n_wat_to_remove: Number of water molecules to remove per unit cell or sheet.
    :param target_spacing: Target d-spacing in angstrom.
    :param non_clay_residues: List of residue names that are not clay.
    :return: Average d-spacing of clay sheets.
    """
    ingro, outgro, intop, outtop = get_paths(inpath)
    logger.info(f"\tChecking d-spacing in {ingro.name}:")
    u = mda.Universe(
        str(ingro)
    )  # creates MDA object with all atoms and coordinates
    clay = u.select_atoms("not resname " + " ".join(non_clay_residues))
    sheet_list = []
    remove_list = []
    isl_residues = u.select_atoms(
        "resname iSL"
    ).residues  # creates group of all iSL atoms
    if isl_residues.n_residues == 0:
        logger.info(f"\t\tNo iSL residues found in {ingro.name}.")
    clay_sheet = None
    iSL_sheet = None
    n_iSL = 0
    iSL_count = 0
    for r in u.residues:
        if r.resid not in isl_residues.resids:
            if n_iSL != 0:
                if len(iSL_sheet.resids) >= n_wat_to_remove:
                    isl_ids = random.sample(
                        sorted(iSL_sheet.resids), n_wat_to_remove
                    )
                else:
                    logger.info(
                        f"\t\tNot enough iSL residues ({iSL_count}/{n_wat_to_remove})."
                        f'\n\t\tRemoving all iSL residues from "{ingro}".'
                    )
                    isl_ids = sorted(iSL_sheet.resids)
                    n_wat_to_remove = iSL_sheet.resids.n_residues
                remove_list.extend(isl_ids)
                n_iSL = 0
                iSL_sheet = None
        else:
            n_iSL += 1
            iSL_count = max(n_iSL, iSL_count)
            if iSL_sheet is None:
                iSL_sheet = r
            else:
                iSL_sheet += r
        if r.resid in clay.residues.resids:
            if clay_sheet is None:
                clay_sheet = r
            else:
                clay_sheet += r
        else:
            if clay_sheet is not None and clay_sheet.atoms.n_atoms > 0:
                sheet_list.append(clay_sheet)
                clay_sheet = None
    # )  # adds list of all clay atoms from bottom sheet to list
    sheet_list = list(
        map(lambda x: x.center_of_geometry()[2], sheet_list)
    )  # returns the z coordinate of the centre of geometry of each sheet of clay in the list
    sheet_spacing = np.ediff1d(
        np.array(sheet_list)
    )  # returns the difference between consecutive z centres for each clay sheet
    sheet_spacing = np.mean(sheet_spacing).round(
        2
    )  # returns the mean of the sheet spacing
    logger.info(
        f"\t\tFound average d-spacing of {sheet_spacing} Å.\n\t\t"
        f"Reference value is {d_spacing} Å."
    )
    if np.round(sheet_spacing, 1) > target_spacing + 0.2 and iSL_count != 0:
        removestr = " ".join(
            list(map(lambda x: str(x), remove_list))
        )  # turns resids into strings, and creates one long string of resids separated by a space
        new_u = u.select_atoms(
            f"not resid {removestr}"
        )  # selects all atoms that aren't the chosen resids
        new_u.atoms.write(outgro)  # writes a new gro file of the new universe
        logger.info(
            (
                f'\t\tRemoving iSL residues {removestr} from "{ingro}" and '
                f'writing to "{outgro}".'
            )
        )
        remove_top_iSL(intop, outtop, n_wat_to_remove)
        return 0
    else:
        logger.info(f'\t\tNot removing waters from "{ingro}".')
        return sheet_spacing


if __name__ == "__main__":
    args = parser.parse_args()
    run_path = args.run_path
    n_wat_to_remove = args.n_wat_to_remove
    d_spacing = args.d_spacing
    n_ucs = args.n_ucs
    fhandler = logging.FileHandler(
        run_path.resolve().parent / "remove_waters.log", mode="w"
    )
    logger.addHandler(fhandler)
    logger.info(
        f"Trying to remove {n_wat_to_remove} waters from {n_ucs} unit cells "
        f"to achieve a d-spacing of {d_spacing:.1f} Å.\n"
    )
    spacing = remove_waters(
        run_path, int(np.round(n_wat_to_remove * n_ucs, 0)), d_spacing
    )
    print(spacing)
    fhandler.close()
