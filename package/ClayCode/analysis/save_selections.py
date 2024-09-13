#!/usr/bin/env python3
import logging
import sys
from argparse import ArgumentParser
from pathlib import Path

from ClayAnalysis.classes import SimDir
from ClayAnalysis.gmx import run_gmx_convert_tpr
from ClayAnalysis.lib import get_selections, save_selection

logger = logging.getLogger(Path(__file__).stem)

parser = ArgumentParser("save_selections", add_help=True)

parser.add_argument(
    "-p", help="In/Output directory.", type=str, required=True, dest="path"
)
parser.add_argument(
    "-o",
    help="Output directory.",
    type=str,
    required=False,
    dest="odir",
    default=None,
)
parser.add_argument(
    "-uc",
    type=str,
    help="Clay unit cell type",
    dest="clay_type",
    required=True,
)
parser.add_argument(
    "-sel",
    type=str,
    nargs="+",
    help="Atom type selection",
    dest="sel",
    required=True,
)
parser.add_argument(
    "-other",
    type=str,
    nargs="+",
    help="Other atomtype for distance selection",
    dest="other",
    required=False,
    default=None,
)
if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    path = SimDir(args.path)
    if args.odir is None:
        odir = path
    else:
        odir = Path(args.odir)
    tpr = path.tpr
    trr = path.trr
    outname = f'{path._gro.stem}_{args.sel[-1].strip("*")}'
    outname = (odir / outname).resolve()
    if args.other is not None:
        outname = f'{outname}_{args.other[-1].strip("*")}'
        sel, clay, other = get_selections(
            (tpr, trr),
            sel=args.sel,
            clay_type=args.clay_type,
            other=args.other,
        )
        logger.info(
            f"selection: {args.sel} - {sel.n_atoms} atoms, clay: {args.clay_type} - {clay.n_atoms} atoms"
        )
        save_selection(
            outname=outname,
            atom_groups=[clay, sel, other],
            traj=[".trr", ".xtc"],
        )
    else:
        (sel, clay) = get_selections(
            (tpr, trr), sel=args.sel, clay_type=args.clay_type
        )
        logger.info(
            f"selection: {args.sel} - {sel.n_atoms} atoms, clay: {args.clay_type} - {clay.n_atoms} atoms"
        )
        save_selection(
            outname=path / outname,
            atom_groups=[clay, sel],
            traj=[".trr", ".xtc"],
        )
