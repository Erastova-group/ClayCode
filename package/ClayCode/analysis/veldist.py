#!/usr/bin/env python3
import logging
import sys
import pickle as pkl

import MDAnalysis
import numpy as np
import MDAnalysis as mda
from typing import Union, Optional, NoReturn, Any, List, Literal

from argparse import ArgumentParser
import warnings

from ClayAnalysis.classes import SimDir
from ClayAnalysis.lib import (
    check_traj,
    process_box,
    run_analysis,
    get_selections,
    save_selection,
)
from ClayAnalysis.analysisbase import ClayAnalysisBase
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

__all__ = ["VelDens"]

logger = logging.getLogger(Path(__file__).stem)


class VelDens(ClayAnalysisBase):
    _attrs = ["xyvel", "zvel", "totvel", "surfzvel"]

    def __init__(
        self,
        sysname: str,
        sel: mda.AtomGroup,
        clay: mda.AtomGroup,
        zdist: Union[str, Path],
        n_bins: Optional[int] = None,
        bin_step: Optional[Union[int, float]] = None,
        cutoff: Union[float, int] = 6,
        save: Union[bool, str] = True,
        check_traj_len: Union[Literal[False], int] = False,
        guess_steps: bool = True,
        **basekwargs: Any,
    ) -> NoReturn:
        super(VelDens, self).__init__(sel.universe.trajectory, **basekwargs)
        self._init_data(n_bins=n_bins, bin_step=bin_step, cutoff=cutoff, min=-10)
        self.sysname = sysname
        self._ags = [sel]
        self._universe = self._ags[0].universe
        self.sel = sel
        self.sel_n_atoms = sel.n_atoms
        self.clay = clay
        self.save = save
        if type(zdist) == str:
            zdist = Path(zdist)
        assert zdist.is_file(), f"z-density file {zdist} does not exist"
        self._zdist_file = zdist
        if self.save is False:
            pass
        else:
            if type(self.save) == bool:
                self.save = (
                    f"{self.__class__.__name__.lower()}_"
                    f"{self.sysname}_{self.sel.resnames[0]}"
                )
        check_traj(self, check_traj_len)
        self._guess_steps = guess_steps

    def _prepare(self) -> NoReturn:
        logger.info(
            f"Starting run:\n"
            f"Frames start: {self.start}, "
            f"stop: {self.stop}, "
            f"step: {self.step}\n"
        )
        data = np.load(self._zdist_file)
        self.mask = data["mask"]
        self.zdist = np.ma.masked_array(data["zdist"], mask=self.mask)
        start, stop, step = data["run_prms"]
        if len(np.arange(start=self.start, stop=self.stop, step=self.step)) != len(
            self.zdist
        ):
            logger.info("Selected Trajectory slicing does not match zdens data!")
            if self._guess_steps == True:
                logger.info(
                    "Using slicing from zdens:\n"
                    f"start: {start: d}, "
                    f"stop: {stop:d}, "
                    f"step: {step:d}.\n"
                )
                self._setup_frames(self._trajectory, start, stop, step)
            else:
                raise IndexError(
                    "Slicing error!\n"
                    f"Expected start: {start:d}, "
                    f"stop: {stop:d}, "
                    f"step: {step:d}.\n"
                    f"Found start: {self.start:d}, "
                    f"stop: {self.stop:d}, "
                    f"step: {self.step:d}"
                )
        self._z_cutoff = np.rint(np.max(self.zdist))
        process_box(self)

        self._3dvel_array = np.ma.empty(
            (self.sel.n_atoms, 3),
            dtype=np.float64,
            fill_value=np.nan,
        )
        # self._1dvel_array = np.empty(self.sel_n_atoms, dtype=np.float64)
        # _attrs absolute
        self._abs = [True, True, True, False]

    def _single_frame(self) -> None:
        self._3dvel_array.fill(0)
        # print(self._frame_index, self.n_frames)
        self._3dvel_array.mask = self.mask[self._frame_index]
        self._3dvel_array[:] = self.sel.velocities
        signs = np.sign(self.zdist[self._frame_index])
        self.data["zvel"].timeseries.append(np.abs(self._3dvel_array[:, 2]))
        self.data["surfzvel"].timeseries.append(
            np.multiply(self._3dvel_array[:, 2], signs)
        )
        self.data["xyvel"].timeseries.append(
            np.linalg.norm(self._3dvel_array[:, :2], axis=1)
        )
        self.data["totvel"].timeseries.append(np.linalg.norm(self._3dvel_array, axis=1))

    def _save(self) -> NoReturn:
        if self.save is False:
            pass
        else:
            for v in self.data.values():
                v.save(
                    self.save,
                    sel_names=np.unique(self.sel.names),
                    n_atoms=self.sel_n_atoms,
                    n_frames=self.n_frames,
                )
        logger.info("Done!\n")


parser = ArgumentParser(
    prog="veldist",
    description="Compute velocities of atom selection within "
    "specified cutoff from clay surface.",
    add_help=True,
    allow_abbrev=False,
)
parser.add_argument(
    "-name", type=str, help="System name", dest="sysname", required=True
)

parser.add_argument(
    "-inp",
    type=str,
    help="Input file names",
    nargs=2,
    metavar=("coordinates", "trajectory"),
    dest="infiles",
    required=False,
)
parser.add_argument(
    "-uc", type=str, help="Clay unit cell type", dest="clay_type", required=True
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
    "-zdist", type=str, help="z-dist data filename", dest="zdist", required=True
)
parser.add_argument(
    "-edges",
    type=float,
    nargs="+",
    help="Adsorption shell upper limits",
    required=False,
    dest="edges",
    default=None,
)
parser.add_argument(
    "-n_bins",
    default=None,
    type=int,
    help="Number of bins in histogram",
    dest="n_bins",
)
parser.add_argument(
    "-bin_step",
    type=float,
    default=None,
    help="bin size in histogram",
    dest="bin_step",
)
parser.add_argument(
    "-xyrad",
    type=float,
    default=5,
    help="xy-radius for calculating z-position clay surface",
    dest="xyrad",
)
parser.add_argument(
    "-cutoff", type=float, default=6, help="cutoff in z-direction", dest="cutoff"
)

parser.add_argument(
    "-start", type=int, default=None, help="First frame for analysis.", dest="start"
)
parser.add_argument(
    "-step", type=int, default=None, help="Frame steps for analysis.", dest="step"
)
parser.add_argument(
    "-stop", type=int, default=None, help="Last frame for analysis.", dest="stop"
)
parser.add_argument(
    "-out",
    type=str,
    help="Filename for results pickle.",
    dest="save",
    default=False,
)
parser.add_argument(
    "-guess_steps",
    action="store_true",
    default=False,
    help="Guess start, stop and step based on zdist data.",
    dest="guess_steps",
)

parser.add_argument(
    "-path", default=False, help="File with analysis data paths.", dest="path"
)

if __name__ == "__main__":
    logger.info(f"Using MDAnalysis {MDAnalysis.__version__}")
    args = parser.parse_args(sys.argv[1:])
    traj_format = ".trr"

    if args.path:
        path = SimDir(args.path)
        gro = path.gro  # select_file(path=path, suffix="crdin")
        trr = path.trr  # select_file(path=path, suffix="trr", how="largest")
        logger.info(f"{gro}, {trr}")
        sysname = args.sysname

    else:
        gro, trr = args.infiles
        path = SimDir(Path(gro).parent)
        sysname = args.sysname

    logger.info(f"Found coordinates: {str(gro.resolve())!r}")
    logger.info(f"Found trajectory: {str(trr.resolve())!r}")
    logger.info(f"System name: {sysname!r}")

    outname = f'{gro}_{args.sel[-1].strip("*")}'
    outname = (path / outname).resolve()
    logger.info(f'Output path: {outname!r}')

    # outname = f'{path.gro.stem}_{args.sel[-1].strip("*")}'
    # outname = (path / outname).resolve()
    # pdbqt = lambda: path._get_filelist(ext='.pdbqt')
    # traj = lambda: path._get_filelist(ext=f'.{traj_format}')
    # traj = outname.with_suffix(traj_format)
    # coords = outname.with_suffix(".gro")
    coords = gro
    traj = trr
    # try:
    #     u = mda.Universe(str(coords), str(traj))
    # except:
    # logger.info(f"Using {coords.name} and {traj.name}.")
    # new = Falsede
    # if not traj.is_file() or not coords.is_file():
    #     new = True
    # else:
    try:
        u = mda.Universe(str(coords), str(traj))
        if not u.trajectory.n_frames == 35001:
            new = True
    except:
        new = True
    # if len(traj()) == 0: # or len(pdbqt()) == 0 or
    # if new is True:
    #     logger.info(f"Saving selection coordinates and trajectory.")
    #     sel, clay = get_selections((gro, trr), args.sel, args.clay_type)
    #     save_selection(outname=outname, atom_groups=[clay, sel], traj=traj_format)
    # pdbqt = pdbqt()[0]
    # crds = select_file(path, suffix='crdin')
    # traj = select_file(path, suffix=traj_format.strip('.'))
    logger.info(f"Using {coords} and {traj}.")

    sel, clay = get_selections((coords, traj), args.sel, args.clay_type, in_memory=False)

    # sel, clay = get_selections((crdin, trr), args.sel, args.clay_type)

    if args.save == "True":
        args.save = True
    elif args.save == "False":
        args.save = False
    # if args.write == "True":
    #     args.write = True
    # elif args.write == "False":
    #     args.write = False

    dist = VelDens(
        sysname=args.sysname,
        sel=sel,
        clay=clay,
        n_bins=args.n_bins,
        bin_step=args.bin_step,
        cutoff=args.cutoff,
        zdist=args.zdist,
        save=args.save,
        guess_steps=args.guess_steps,
    )

    run_analysis(dist, start=args.start, stop=args.stop, step=args.step)
