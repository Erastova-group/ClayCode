#!/usr/bin/env python3
import logging
import pathlib as pl
import sys
import warnings
from argparse import ArgumentParser
from typing import Any, Literal, NoReturn, Optional, Union

import MDAnalysis as mda
import numpy as np
from ClayCode.analysis.analysisbase import ClayAnalysisBase
from ClayCode.analysis.lib import (
    check_traj,
    get_dist,
    get_selections,
    process_box,
    run_analysis,
    select_cyzone,
)
from ClayCode.analysis.utils import get_paths
from ClayCode.core.classes import Dir
from ClayCode.core.utils import get_subheader
from MDAnalysis import Universe
from MDAnalysis.lib.distances import apply_PBC

# from ClayCode.core.lib import (
#     get_dist,
#     get_selections,
#     process_box,
#     run_analysis,
#     select_cyzone,
# )
warnings.filterwarnings("ignore", category=DeprecationWarning)

__all__ = ["ZDens"]

logger = logging.getLogger(__name__)


class ZDens(ClayAnalysisBase):
    # histogram attributes format:
    # --------------------------
    # name: [name, bins, timeseries, hist, hist2d, edges, n_bins, cutoff, bin_step]
    _attrs = ["zdens"]
    _abs = [True]
    _name = "linear z-distance"
    """Calculate absolute densities of atom z-positions relative to clay surface O-atoms.
    """

    def __init__(
        self,
        sysname: str,
        sel: mda.AtomGroup,
        clay: mda.AtomGroup,
        n_bins: Optional[int] = None,
        bin_step: Optional[Union[int, float]] = None,
        xy_rad: Union[float, int] = 3.0,
        cutoff: Union[float, int] = 20.0,
        save: Union[bool, str] = True,
        write: Union[bool, str] = True,
        overwrite: bool = False,
        check_traj_len: Union[Literal[False], int] = False,
        **basekwargs: Any,
    ) -> None:
        """
        :param sysname: system name
        :type sysname: str
        :param sel: adsorbed atom group
        :type sel: MDAnalysis.core.groups.AtomGroup
        :param clay: clay surface O-atoms
        :type clay: MDAnalysis.core.groups.AtomGroup
        :param n_bins: z-distance bins
        :type n_bins: int, defaults to None
        :param bin_step: default z-distance bin step in Angstrom, defaults to None
        :type bin_step: float
        :param xy_rad: cylinder radius for selecting surface O, defaults to 5.0
        :type xy_rad: float
        :param cutoff: cylinder height for selecting surface O, defaults to 20.0
        :type cutoff: float
        :param save:
        :type save:
        :param write:
        :type write:
        :param overwrite:
        :type overwrite:
        :param check_traj_len:
        :type check_traj_len:
        :param basekwargs:
        :type basekwargs:
        """
        super(ZDens, self).__init__(sel.universe.trajectory, **basekwargs)
        self._init_data(n_bins=n_bins, bin_step=bin_step, cutoff=cutoff)
        self._process_distances = None
        self.sysname = sysname
        self._ags = [sel]
        self._universe = self._ags[0].universe
        self.sel = sel
        self.sel_n_atoms = sel.n_atoms
        self.clay = clay
        self.xy_rad = float(xy_rad)
        self.save = save
        self.write = write
        if self.save is False:
            pass
        else:
            try:
                self.save = Dir(self.save)
            except TypeError:
                pass
            if type(self.save) in [bool, Dir]:
                savename = (
                    f"{self.__class__.__name__.lower()}_" f"{self.sysname}"
                )
                try:
                    self.save = self.save / savename
                except TypeError:
                    self.save = savename
        if self.write is not False:
            if self.save is False:
                savename = (
                    f"{self.__class__.__name__.lower()}_" f"{self.sysname}"
                )
                try:
                    self.write = self.write / savename
                except TypeError:
                    self.write = savename
                self.write = pl.Path(self.write).with_suffix(".npz")
            elif type(self.write) == bool:
                self.write = pl.Path(self.save).with_suffix(".npz")
            else:
                logger.error("Should not get here!")
                sys.exit(1)
            if pl.Path(self.write).is_file():
                if overwrite is False:
                    logger.finfo(
                        f"Done!\n{str(self.write)!r} already exists and overwrite not selected."
                    )
                    self._get_new_data = False
                    return
                    # raise FileExistsError(f"{self.write!r} already exists.")
        check_traj(self, check_traj_len)

    def _prepare(self) -> NoReturn:
        process_box(self)
        logger.info(
            f"Starting run:\n"
            f"Frames start: {self.start}, "
            f"stop: {self.stop}, "
            f"step: {self.step}\n"
        )
        self._dist_array = np.ma.empty(
            (self.sel.n_atoms, self.clay.n_atoms, 3),
            dtype=np.float64,
            fill_value=np.nan,
        )
        self._z_dist = np.empty(self.sel.n_atoms, dtype=np.float64)
        self.mask = []
        self._sel = np.empty_like(self._z_dist, dtype=bool)
        self._sel_mask = np.ma.empty_like(
            self._dist_array[:, :, 0], dtype=np.float64
        )

    def _single_frame(self) -> NoReturn:
        self._dist_array.fill(0)
        self._dist_array.mask = False
        self._sel_mask.fill(0)
        self._sel_mask.mask = False
        self._z_dist.fill(0)
        self._sel_mask.soften_mask()
        # Wrap coordinates back into simulation box (use mda apply_PBC)
        sel_pos = apply_PBC(self.sel.positions, self._ts.dimensions)
        clay_pos = apply_PBC(self.clay.positions, self._ts.dimensions)
        # get minimum x, y, z distances between sel and clay in box
        get_dist(sel_pos, clay_pos, self._dist_array, self._ts.dimensions)
        self._process_distances(self._dist_array, self._ts.dimensions)
        # self._dist_array[:] = np.apply_along_axis(lambda x: minimize_vectors(x, self._ts.dimensions), axis=self._dist_array)
        # consider only clay atoms within a cylinder around sel atoms
        select_cyzone(
            distances=self._dist_array,
            xy_rad=self.xy_rad,
            z_dist=self.data["zdens"].cutoff,
            mask_array=self._sel_mask,
        )

        # get minimum z-distance to clay for each sel atom
        self._z_dist = np.min(
            np.abs(self._dist_array[:, :, 2]), axis=1, out=self._z_dist
        )
        # if np.isnan(self._z_dist).any():
        # logger.info(f"{self._z_dist}, {self._z_dist.shape}")
        # logger.info(np.argwhere(self._z_dist[np.isnan(self._z_dist)]))
        # only_data = np.min(
        #     np.abs(self._dist_array.data[:, :, 2]),
        #     axis=1,
        #     out=self._z_dist,
        # )
        # for i in range(len(self._z_dist)):
        # logger.info(f"{i}: {self._z_dist[i]:.3f}, {only_data[i]:.3f}")
        # logger.info(self._z_dist.shape)
        # logger.info(np.isnan(self._z_dist).any())
        self._sel[:] = np.isnan(self._z_dist)
        self.data["zdens"].timeseries.append(self._z_dist.copy())
        self.mask.append(self._sel.copy())

    def _save(self) -> NoReturn:
        if self.save is False:
            pass
        else:
            for v in self.data.values():
                v.save(
                    self.save,
                    sel_names=np.unique(self.sel.names),
                    n_atoms=self.sel.n_atoms,
                    n_frames=self.n_frames,
                )
        if self.write is not False:
            with open(self.write, "wb") as outfile:
                np.savez(
                    outfile,
                    zdist=np.array(self.data["zdens"].timeseries),
                    mask=np.array(self.mask),
                    frames=self.results["frames"],
                    times=self.results["times"],
                    run_prms=np.array([self.start, self.stop, self.step]),
                    cutoff=self.data["zdens"].cutoff,
                    bin_step=self.data["zdens"].bin_step,
                    sel_n_atoms=self.sel.n_atoms,
                )
            assert len(
                np.arange(start=self.start, stop=self.stop, step=self.step)
            ) == len(
                self.data["zdens"].timeseries
            ), "Length of timeseries does not conform to selected start, stop and step!"
            # logger.info(f"{self.start}, {self.stop}, {self.step}")

            logger.finfo(f"Wrote z-dist array to {str(self.write)!r}")
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


parser = ArgumentParser(
    prog="zdens",
    description="Compute z-density relative to clay surface OB atoms.",
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
    "-inpname",
    type=str,
    help="Input file names",
    metavar="name_stem",
    dest="inpname",
    required=False,
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
    default=3,
    help="xy-radius for calculating z-position clay surface",
    dest="xyrad",
)

parser.add_argument(
    "-cutoff",
    type=float,
    default=20,
    help="cutoff in z-direction",
    dest="cutoff",
)

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
    default=True,
)
parser.add_argument(
    "-check_traj",
    type=int,
    default=False,
    help="Expected trajectory length.",
    dest="check_traj_len",
)
parser.add_argument(
    "--write_z",
    type=str,
    default=True,
    help="Binary array output of selection z-distances.",
    dest="write",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    default=False,
    help="Overwrite existing z-distance array data.",
    dest="overwrite",
)
parser.add_argument(
    "--update",
    action="store_true",
    default=False,
    help="Overwrite existing trajectory and coordinate array data.",
    dest="new",
)

parser.add_argument(
    "-path", default=False, help="File with analysis data paths.", dest="path"
)
parser.add_argument(
    "--in_mem",
    default=False,
    action="store_true",
    help="Read trajectory in memory.",
    dest="in_mem",
)

if __name__ == "__main__":
    logger.info(f"Using MDAnalysis {mda.__version__}")
    logger.info(f"Using numpy {np.__version__}")
    args = parser.parse_args(sys.argv[1:])
    traj_format = ".xtc"

    sysname = args.sysname

    gro, trr, path = get_paths(args.path, args.infiles, args.inpname)

    logger.finfo(f"{sysname!r}", kwd_str=f"System name: ")

    if args.save is None:
        outpath = path
    else:
        outpath = pl.Path(args.save)
    if outpath.is_dir():
        outname = f'{gro}_{args.sel[-1].strip("*")}'
        outname = (path / outname).resolve()
    else:
        outname = pl.Path(args.save).resolve()
    logger.finfo(f"{str(outname.resolve())!r}", kwd_str=f"Output path: ")
    # pdbqt = lambda: path._get_filelist(ext='.pdbqt')
    # traj = lambda: path._get_filelist(ext=f'.{traj_format}')
    # traj = outname.with_suffix(traj_format)
    # coords = outname.with_suffix(".gro")
    # try:
    #     u = mda.Universe(str(coords), str(traj))
    # except:

    # if (args.new is True) or (not traj.is_file() or not coords.is_file()):
    #     logger.info("Files missing")
    #     new = True
    # else:
    #     try:
    #         u = mda.Universe(str(coords), str(traj))
    #         new = False
    #         if not u.trajectory.n_frames == 35001:
    #             logger.info("Wrong frame number")
    #             new = True
    #     except:
    #         logger.info("Could not construct universe")
    #         new = True
    # # if len(traj()) == 0: # or len(pdbqt()) == 0 or
    #
    # if new is True:
    #     logger.info(f"Saving selection coordinates and trajectory.")
    #     sel, clay = get_selections((gro, trr), args.sel, args.clay_type)
    #     save_selection(outname=outname, atom_groups=[clay, sel], traj=traj_format)
    # pdbqt = pdbqt()[0]
    # crds = select_file(path, suffix='crdin')
    # traj = select_file(path, suffix=traj_format.strip('.'))
    coords = gro
    traj = trr
    logger.debug(f"Using {coords.name} and {traj.name}.")
    try:
        u = Universe(str(coords), str(traj))
        new = False
        if not args.check_traj_len:
            logger.finfo(
                "Skipping trajectory length check.", initial_linebreak=True
            )
        else:
            if not u.trajectory.n_frames == args.check_traj_len:
                logger.finfo(
                    f"Wrong frame number, found {u.trajectory.n_frames}, expected {args.check_traj_len}!",
                    initial_linebreak=True,
                )
                new = True
            else:
                logger.finfo(
                    f"Trajectory has correct frame number of {args.check_traj_len}.",
                    initial_linebreak=True,
                )
    except:
        logger.info("Could not construct universe!", initial_linebreak=True)
        new = True
    logger.info(get_subheader("Getting atom groups"))
    sel, clay = get_selections(
        infiles=(coords, traj),
        sel=args.sel,
        clay_type=args.clay_type,
        in_memory=args.in_mem,
    )

    if args.save == "True":
        args.save = True
    elif args.save == "False":
        args.save = False
    if args.write == "True":
        args.write = True
    elif args.write == "False":
        args.write = False

    zdens = ZDens(
        sysname=sysname,
        sel=sel,
        clay=clay,
        n_bins=args.n_bins,
        bin_step=args.bin_step,
        xy_rad=args.xyrad,
        cutoff=args.cutoff,
        save=args.save,
        write=args.write,
        overwrite=args.overwrite,
        check_traj_len=args.check_traj_len,
    )

    run_analysis(zdens, start=args.start, stop=args.stop, step=args.step)
