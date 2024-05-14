import logging
import pickle as pkl
import sys
import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, List, Literal, NoReturn, Optional, Union

import MDAnalysis
import numpy as np
from ClayCode.analysis.analysisbase import ClayAnalysisBase
from ClayCode.analysis.lib import (
    check_traj,
    exclude_xyz_cutoff,
    get_dist,
    get_selections,
    process_box,
    run_analysis,
)
from ClayCode.analysis.utils import get_paths
from MDAnalysis.lib.distances import apply_PBC

warnings.filterwarnings("ignore", category=DeprecationWarning)

__all__ = ["RDFDist"]

logger = logging.getLogger(Path(__file__).stem)


class RDFDist(ClayAnalysisBase):
    _attrs = ["rdens"]

    def __init__(
        self,
        sysname: str,
        sel: MDAnalysis.AtomGroup,
        clay: MDAnalysis.AtomGroup,
        zdist: Union[str, Path],
        other: Optional[MDAnalysis.AtomGroup] = None,
        n_bins: Optional[int] = None,
        bin_step: Optional[Union[int, float]] = None,
        cutoff: Optional[Union[float, int]] = None,
        save: Union[bool, str] = True,
        check_traj_len: Union[Literal[False], int] = False,
        guess_steps: bool = True,
        **basekwargs: Any,
    ) -> NoReturn:
        logger.info(sel.universe.trajectory)
        super(RDFDist, self).__init__(sel.universe.trajectory, **basekwargs)
        self._init_data(n_bins=n_bins, bin_step=bin_step, cutoff=cutoff, min=0)
        self._process_distances = None
        self.sysname = sysname
        self.cutoff = float(cutoff)
        self._ags = [sel]
        self._universe = self._ags[0].universe
        self.sel = sel
        self.sel_n_atoms = sel.n_atoms
        self.clay = clay
        if other is None:
            other = sel
        # assert isinstance(sel, AtomGroup)
        # assert isinstance(other, AtomGroup)
        if other == sel:
            self.other = self.sel
            self.self = True
        else:
            self.other = other
            self.self = False
        self.other_n_atoms = other.n_atoms

        if type(zdist) == str:
            zdist = Path(zdist)

        # assert zdist.is_file(), f"z-density file {zdist} does not exist"
        self._zdist_file = zdist
        self.save = save
        if self.save is False:
            pass
        else:
            if type(self.save) == bool:
                self.save = (
                    f"{self.__class__.__name__.lower()}_"
                    f"{self.sysname}_{self.sel.resnames[0]}_{self.other.resnames[0]}"
                )
        check_traj(self, check_traj_len)
        self._guess_steps = guess_steps

    def _prepare(self):
        process_box(self)
        logger.info(
            f"Starting run:\n"
            f"Frames start: {self.start}, "
            f"stop: {self.stop}, "
            f"step: {self.step}\n"
        )

        if self._zdist_file is not None:
            data = np.load(self._zdist_file)
            self.mask = data["mask"]
            self.zdist = np.ma.masked_array(data["zdist"], mask=self.mask)
            start, stop, step = data["run_prms"]
            if len(
                np.arange(start=self.start, stop=self.stop, step=self.step)
            ) != len(self.zdist):
                logger.info(
                    "Selected Trajectory slicing does not match zdens data!"
                )
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
            self._z_cutoff = np.rint(data["zdist"].cutoff)
        else:
            self.mask = False
            self.zdist = None
            self._z_cutoff = None
        self._xyz_cutoff = np.rint(self.cutoff)

        self._dist = np.ma.empty(
            (self.sel.n_atoms, self.other.n_atoms, 3),
            dtype=np.float64,
            fill_value=np.nan,
        )

        self._rad = np.empty(
            (self.sel.n_atoms, self.other.n_atoms),
            # fill_value=np.nan,
            dtype=np.float64,
        )
        self._sel_pos = np.ma.empty(
            (self.sel_n_atoms, 3), fill_value=np.nan, dtype=np.float64
        )
        # self._other_pos = np.ma.empty((self._other_pos, 3),
        #                               dtype=np.float64,
        #                               fill_value=np.nan)
        if self.self is False:
            self._other_pos = np.empty(
                (self.other_n_atoms, 3), dtype=np.float64
            )
            self.diag_mask = False
        else:
            self.other = None
            dist_slice = self._dist_m[..., 0]
            diag_idx = np.diag_indices_from(dist_slice)
            diag_mask = np.ma.getmaskarray(dist_slice)
            diag_mask[diag_idx] = True
            diag_mask = np.broadcast_to(
                diag_mask[..., np.newaxis], self._dist.shape
            )
            self.diag_mask = np.bitwise_or.accumulate(diag_mask, axis=2).copy()

    def _single_frame(self):
        self._sel_pos.fill(0)
        self._sel_pos.soften_mask
        if self.zdist is not None:
            mask = self.mask[self._frame_index]
            self._sel_pos.mask = np.broadcast_to(
                mask[:, np.newaxis], self._sel_pos.shape
            )
            self._sel_pos.harden_mask()
        else:
            self._sel_pos.mask = False
        self._dist.mask = np.broadcast_to(
            self._sel_pos.mask[:, np.newaxis, :], self._dist.shape
        )
        self._dist.fill(np.nan)
        self._rad.fill(np.nan)
        self._dist.fill(np.nan)
        self._sel_pos[:] = self.sel.positions
        self._sel_pos[:] = apply_PBC(self._sel_pos, self._ts.dimensions)

        if self.self is False:
            self._other_pos.fill(0)
            self._other_pos[:] = apply_PBC(
                self.other.positions, self._ts.dimensions
            )
        else:
            new_mask = self.diag_mask + self._dist.mask
            self._dist.mask = new_mask
            self._other_pos = self._sel_pos

        get_dist(
            self._sel_pos, self._other_pos, self._dist, self._ts.dimensions
        )
        self._process_distances(self._dist, self._ts.dimensions)
        exclude_xyz_cutoff(distances=self._dist, cutoff=self.cutoff)
        # mask = np.any(np.abs(self._dist) > self.cutoff, axis=2).mask
        self._rad[:] = np.where(
            ~self._dist[:, :, 0].mask,
            np.linalg.norm(self._dist, axis=2),
            np.nan,
        )

        self._rad[:] = np.where(
            self._rad <= self.cutoff, np.abs(self._rad), np.nan
        )
        self.data["rdens"].timeseries.append(self._rad.copy())

    def _save(self) -> NoReturn:
        if self.save is False:
            pass
        else:
            for v in self.data.values():
                logger.info(f"value: {v}")
                v.save(
                    self.save,
                    rdf=True,
                    sel_names=np.unique(self.sel.names),
                    sel_n_atoms=self.sel.n_atoms,
                    other_names=np.unique(self.other.names),
                    other_n_atoms=self.other.n_atoms,
                    n_frames=self.n_frames,
                )
            # outsel = self.sel + self.clay
            # ocoords = str(change_suffix(self.save, "pdbqt"))
            # otraj = str(change_suffix(self.save, "traj"))
            # outsel.write(
            #     otraj, frames=self._trajectory[self.start : self.stop : self.step]
            # )
            # outsel.write(
            #     ocoords, frames=self._trajectory[self.start : self.stop : self.step][-1]
            # )
            # logger.info(
            #     f"Wrote final coordinates to {ocoords.name} and trajectory to {otraj.name}"
            # )

        logger.info("Done!\n")


#
#
#     pairs, dist = distances.capped_distance(self.sel.positions,
#                                             self.other.positions,
#                                             self._maxrange,
#                                             box=self._ts.dimensions)
#
#     # Maybe exclude same molecule distances
#     if self._exclusion_block is not None:
#         idxA = pairs[:, 0] // self._exclusion_block[0]
#         idxB = pairs[:, 1] // self._exclusion_block[1]
#         mask = np.where(idxA != idxB)[0]
#         dist = dist[mask]
#
#     count, _ = np.histogram(dist, **self.rdf_settings)
#     self.results.count += count
#
#     if self.norm == "rdf":
#         self.volume_cum += self._ts.volume
#
#
# def _conclude(self):
#     norm = self.n_frames
#     if self.norm in ["rdf", "density"]:
#         # Volume in each radial shell
#         vols = np.power(self.results.ads_edges, 3)
#         norm *= 4 / 3 * np.pi * np.diff(vols)
#
#     if self.norm == "rdf":
#         # Number of each selection
#         nA = len(self.sel)
#         nB = len(self.other)
#         N = nA * nB
#
#         # If we had exclusions, take these into account
#         if self._exclusion_block:
#             xA, xB = self._exclusion_block
#             nblocks = nA / xA
#             N -= xA * xB * nblocks
#
#         # Average number density
#         box_vol = self.volume_cum / self.n_frames
#         norm *= N / box_vol
#
#     self.results.rdf = self.results.count / norm

if __name__ == "__main__":
    parser = ArgumentParser(
        prog="rdf",
        description="Compute radial denisty of atom species atoms relative to other.",
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
        "-zdist",
        type=str,
        help="z-dist data filename",
        dest="zdist",
        required=False,
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
        "-check_traj",
        type=int,
        default=False,
        help="Expected trajectory length.",
        dest="check_traj_len",
    )

    parser.add_argument(
        "-cutoff",
        type=float,
        default=None,
        help="cutoff in x,x2,z-direction",
        dest="cutoff",
    )

    # parser.add_argument('-cutoff',
    #                     type=float,
    #                     default=None,
    #                     help='radial cutoff',
    #                     dest='cutoff')

    parser.add_argument(
        "-start",
        type=int,
        default=None,
        help="First frame for analysis.",
        dest="start",
    )
    parser.add_argument(
        "-step",
        type=int,
        default=None,
        help="Frame steps for analysis.",
        dest="step",
    )
    parser.add_argument(
        "-stop",
        type=int,
        default=None,
        help="Last frame for analysis.",
        dest="stop",
    )
    parser.add_argument(
        "-out",
        type=str,
        help="Filename for results pickle.",
        dest="save",
        default=False,
    )

    parser.add_argument(
        "-update",
        action="store_true",
        default=False,
        help="Overwrite existing trajectory and coordinate array data.",
        dest="new",
    )

    parser.add_argument(
        "-path",
        default=False,
        help="File with analysis data paths.",
        dest="path",
    )
    parser.add_argument(
        "-inpname",
        type=str,
        help="Input file names",
        metavar="name_stem",
        dest="inpname",
        required=False,
    )
    if __name__ == "__main__":
        logger.info(f"Using MDAnalysis {MDAnalysis.__version__}")
        logger.info(f"Using numpy {np.__version__}")
        # logger.info(f"Using cython {cython.__version__}")
        args = parser.parse_args(sys.argv[1:])
        traj_format = ".xtc"
    gro, trr, path = get_paths(
        infiles=args.infiles,
        inpname=args.inpname,
        path=args.path,
        traj_suffix="xtc",
    )

    # if args.path:
    #     path = SimDir(args.path)
    #     gro = path.gro  # select_file(path=path, suffix="crdin")
    #     trr = path.trr  # select_file(path=path, suffix="trr", how="largest")
    sysname = args.sysname

    # else:
    #     gro, trr = args.infiles
    #     gro, trr = Path(gro), Path(trr)
    #     path = SimDir(gro.parent)
    #     sysname = args.sysname

    logger.info(f"Found coordinates: {str(gro.resolve())!r}")
    logger.info(f"Found trajectory: {str(trr.resolve())!r}")
    logger.info(f"System name: {sysname!r}")

    if args.save is None:
        outpath = path
    else:
        outpath = Path(args.save)
    if outpath.is_dir():
        outname = f'{gro}_{args.sel[-1].strip("*")}'
        outname = (path / outname).resolve()
    else:
        outname = Path(args.save).resolve()
    logger.finfo(f"{str(outname.resolve())!r}", kwd_str=f"Output path: ")

    logger.info(f"Output path: {outname!r}")

    # if a.r_cutoff is None:
    #     r_cutoff = a.cutoff
    # if len(a.r_cutoff) == 1:
    #     r_cutoff = [a.r_cutoff[0] for c in range(3)]
    # elif len(a.r_cutoff) == 3:
    #     r_cutoff = a.r_cutoff
    # else:
    #     raise ValueError('Expected either 1 or 3 arguments for r_cutoff!')
    #
    # print(r_cutoff)

    coords = gro
    traj = trr
    logger.info(f"Using {coords.name} and {traj.name}.")
    try:
        u = MDAnalysis.Universe(str(coords), str(traj))
        new = False
        if not u.trajectory.n_frames == 35001:
            logger.info("Wrong frame number")
            new = True
    except:
        logger.info("Could not construct universe")
        new = True
    sel, clay, other = get_selections(
        infiles=(coords, traj),
        sel=args.sel,
        other=args.other,
        clay_type=args.clay_type,
        in_memory=False,
        only_surface=False,
        separate_ob_oh_surfaces=False,
    )

    if args.save == "True":
        args.save = True
    elif args.save == "False":
        args.save = False
    # if args.write == "True":
    #     args.write = True
    # elif args.write == "False":
    #     args.write = False
    dist = RDFDist(
        sysname=args.sysname,
        sel=sel,
        clay=clay,
        other=other,
        n_bins=args.n_bins,
        bin_step=args.bin_step,
        cutoff=args.cutoff,
        save=args.save,
        zdist=args.zdist,
        check_traj_len=args.check_traj_len,
    )
    run_analysis(dist, start=args.start, stop=args.stop, step=args.step)
