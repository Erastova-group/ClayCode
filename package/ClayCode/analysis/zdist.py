#!/usr/bin/env python3
import logging
import pathlib as pl
import re
import sys
import warnings
from argparse import ArgumentParser
from typing import Any, List, Literal, NoReturn, Optional, Union

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
from numpy._typing import NDArray

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
    _abs = [False]
    _name = "linear z-distance"
    """Calculate absolute densities of atom z-positions relative to clay surface O-atoms.
    """

    def __init__(
        self,
        sysname: str,
        sel: mda.AtomGroup,
        clay_type: mda.AtomGroup,
        dist_selection: Literal[
            "smaller", "larger", "gt_zero", "lt_zero"
        ] = "gt_zero",
        upper_clay_sheet_idxs: Optional[NDArray] = None,
        n_bins: Optional[int] = None,
        bin_step: Optional[Union[int, float]] = None,
        xy_rad: Union[float, int] = 3.0,
        cutoff: Union[float, int] = 20.0,
        # min_cutoff: Union[float, int] = 0,
        save: Union[bool, str] = True,
        write: Union[bool, str] = True,
        overwrite: bool = False,
        check_traj_len: Union[Literal[False], int] = False,
        # use_abs: bool = True,
        **basekwargs: Any,
    ) -> None:
        """
        :param sysname: system name
        :type sysname: str
        :param sel: adsorbed atom group
        :type sel: MDAnalysis.core.groups.AtomGroup
        :param clay_type: clay surface O-atoms
        :type clay_type: MDAnalysis.core.groups.AtomGroup
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
        uc_height = np.max(
            clay_type.residues[0].atoms.positions[:, 2]
        ) - np.min(clay_type.residues[0].atoms.positions[:, 2])
        self.dist_selection = dist_selection.strip(" ")
        self._clay_sheet_ids = None
        # self.use_abs = use_abs
        # logger.info("Will use absolute z-distances.")
        # if use_abs is True:
        #     _min = 0
        # else:
        #     _min = min_cutoff
        if self.dist_selection == "gt_zero":
            self.symmetrical = True
            min = 0
        elif self.dist_selection == "lt_zero":
            self.symmetrical = True
            min = -int(uc_height)
            self._clay_sheet_ids = np.isin(
                clay_type.atoms.resids,
                upper_clay_sheet_idxs,
                assume_unique=False,
                invert=True,
            )
        else:
            self.symmetrical = False
            min = (
                0  # -int(uc_height)  # {"zdens": -int(cutoff), "zdens_abs": 0}
            )
        self._init_data(
            n_bins=n_bins, bin_step=bin_step, cutoff=cutoff, min=min
        )
        self._process_distances = None
        self.sysname = sysname
        self._ags = [sel]
        self._universe = self._ags[0].universe
        self.sel = sel
        self.sel_n_atoms = sel.n_atoms
        self.clay = clay_type
        self.xy_rad = float(xy_rad)
        self.save = save
        self.write = write
        if self.symmetrical is False:
            name_ext = (
                re.search(
                    "OH|OB",
                    "".join(clay_type.atoms.names),
                    flags=re.IGNORECASE,
                ).group(0)
                + "/"
            )
        else:
            name_ext = ""
        if self.save is False:
            pass
        else:
            try:
                self.save = Dir(self.save)
            except TypeError:
                pass
            if type(self.save) in [bool, Dir]:
                savename = (
                    f"{name_ext}{self.__class__.__name__.lower()}_"
                    f"{self.sysname}"
                )
                try:
                    self.save = self.save / savename
                except TypeError:
                    self.save = savename
        if self.write is not False:
            if self.save is False:
                savename = (
                    f"{name_ext}{self.__class__.__name__.lower()}_"
                    f"{self.sysname}"
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
                    return  # raise FileExistsError(f"{self.write!r} already exists.")
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
        # if self.dist_selection == "lt_zero":
        #     self._z_dist_lt_zero = np.empty(self.sel.n_atoms, dtype=np.float64)
        #     self._z_dist_gt_zero = np.empty(self.sel.n_atoms, dtype=np.float64)
        self._z_dist = np.empty(self.sel.n_atoms, dtype=np.float64)
        self.mask = []
        self._sel = np.empty_like(self._z_dist, dtype=bool)
        self._sel_mask = np.ma.empty_like(
            self._dist_array[:, :, 0], dtype=np.float64
        )
        self._get_z_dist = getattr(self, f"_get_{self.dist_selection}_dist")
        self._select_cyzone = getattr(
            self, f"_select_cyzone_{self.dist_selection}"
        )

    def _select_cyzone_gt_zero(self):
        select_cyzone(
            distances=self._dist_array,
            xy_rad=self.xy_rad,
            max_z_dist=self.data["zdens"].cutoff,
            mask_array=self._sel_mask,
            absolute=True,  # self.use_abs,
        )

    def _select_cyzone_lt_zero(self):
        np.negative(
            self._dist_array[:, :, 2],
            out=self._dist_array[:, :, 2],
            where=self._clay_sheet_ids,
        )
        select_cyzone(
            distances=self._dist_array,
            xy_rad=self.xy_rad,
            max_z_dist=self.data["zdens"].cutoff,
            mask_array=self._sel_mask,
            min_z_dist=self.data["zdens"]._min,
        )

    def _select_cyzone_larger(self):
        select_cyzone(
            distances=self._dist_array,
            xy_rad=self.xy_rad,
            max_z_dist=self.data["zdens"].cutoff,
            mask_array=self._sel_mask,
            min_z_dist=self.data["zdens"]._min,
        )

    def _select_cyzone_smaller(self):
        select_cyzone(
            distances=self._dist_array,
            xy_rad=self.xy_rad,
            min_z_dist=-self.data["zdens"].cutoff,
            mask_array=self._sel_mask,
            max_z_dist=-self.data["zdens"]._min,
        )

    def _get_gt_zero_dist(self):
        # self._z_dist[:] = self._dist_array[
        #               np.argwhere(np.min(np.abs(self._dist_array[:, :, 2]), axis=1)),
        #               :,
        #               2,
        #               ]
        np.min(np.abs(self._dist_array[:, :, 2]), axis=1, out=self._z_dist)
        # np.compress(
        #     np.min(np.abs(self._dist_array[:, :, 2]), axis=1),
        #     np.abs(self._dist_array[:, :, 2],
        #     out=self._z_dist,
        # )

    def _get_lt_zero_dist(self):
        self._z_dist[:] = np.ma.filled(
            np.take_along_axis(
                self._dist_array[:, :, 2],
                np.argmin(
                    np.abs(self._dist_array[:, :, 2]),
                    axis=1,
                    keepdims=True,
                ),
                axis=1,
            ).flatten()
        )
        # np.compress(
        #     np.argmin(np.abs(self._dist_array[:, :, 2]), axis=1),
        #     self._dist_array[:, :, 2],
        #     out=self._z_dist,
        # )

    # def _get_lt_zero_dist(self):
    #     dist_mask = self._dist_array.mask
    #     lt_array = np.where(self._dist_array[:, :, 2] < 0, self._dist_array[:, :, 2], np.NaN)
    #     np.compress(
    #         np.min(np.abs(lt_array), axis=1),
    #         -np.sign(self._dist_array[:, :, 2]),
    #         out=self._z_dist_lt_zero,
    #     )
    #     gt_array = np.where(self._dist_array[:, :, 2] > 0, self._dist_array[:, :, 2], np.NaN)
    #     np.compress(
    #         np.min(np.abs(gt_array), axis=1),
    #         np.sign(self._dist_array[:, :, 2]) * self._dist_array[:, :, 2],
    #         out=self._z_dist_gt_zero,
    #     )
    #     np.where(np.abs(np.min(self._z_dist_lt_zero)) < np.abs(np.min(self._z_dist_gt_zero)), self._z_dist_lt_zero, self._z_dist_gt_zero,
    #                             out=self._z_dist)

    def _get_larger_dist(self):
        dist_mask = self._dist_array.mask[:, :, 2]
        dist_mask = np.where(self._dist_array[:, :, 2] < 0, True, dist_mask)
        self._dist_array.mask = np.broadcast_to(
            dist_mask[:, :, np.newaxis], self._dist_array.shape
        )
        # idx_sel = np.argwhere(np.min(np.abs(self._dist_array[:, :, 2]), axis=1))
        # self._z_dist.fill(np.NaN)
        # self._z_dist[idx_sel] = self._dist_array[idx_sel, :, 2]
        # self._z_dist = self._dist_array[
        #               np.argwhere(np.min(np.abs(self._dist_array[:, :, 2]), axis=1)),
        #               :,
        #               2,
        #               ]
        self._z_dist[:] = np.ma.filled(
            np.take_along_axis(
                self._dist_array[:, :, 2],
                np.argmin(
                    np.abs(self._dist_array[:, :, 2]),
                    axis=1,
                    keepdims=True,
                ),
                axis=1,
            ).flatten()
        )

    def _get_smaller_dist(self):
        dist_mask = self._dist_array.mask[:, :, 2]
        dist_mask = np.where(self._dist_array[:, :, 2] > 0, True, dist_mask)
        self._dist_array.mask = np.broadcast_to(
            dist_mask[:, :, np.newaxis], self._dist_array.shape
        )
        self._z_dist[:] = np.ma.filled(
            np.take_along_axis(
                np.negative(self._dist_array[:, :, 2]),
                np.argmin(
                    np.abs(self._dist_array[:, :, 2]), axis=1, keepdims=True
                ),
                axis=1,
            ).flatten()
        )

    def _single_frame(self) -> NoReturn:
        self._dist_array.fill(0)
        self._dist_array.mask = False
        self._sel_mask.fill(0)
        self._sel_mask.soften_mask()
        self._sel_mask.mask = False
        self._sel_mask.soften_mask()
        # Wrap coordinates back into simulation box (use mda apply_PBC)
        sel_pos = apply_PBC(self.sel.positions, self._ts.dimensions)
        clay_pos = apply_PBC(self.clay.positions, self._ts.dimensions)
        # get minimum x, y, z distances between sel and clay in box
        get_dist(sel_pos, clay_pos, self._dist_array, self._ts.dimensions)
        self._process_distances(self._dist_array, self._ts.dimensions)
        # self._dist_array[:] = np.apply_along_axis(lambda x: minimize_vectors(x, self._ts.dimensions), axis=self._dist_array)
        # consider only clay atoms within a cylinder around sel atoms
        self._select_cyzone()
        # get minimum z-distance to clay for each sel atom
        self._get_z_dist()
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

            logger.finfo(
                f"Wrote z-dist array to {str(self.write)!r}"
            )  # outsel = self.sel + self.clay  # ocoords = str(change_suffix(self.save, "pdbqt"))  # otraj = str(change_suffix(self.save, "traj"))  # outsel.write(  #     otraj, frames=self._trajectory[self.start : self.stop : self.step]  # )  # outsel.write(  #     ocoords, frames=self._trajectory[self.start : self.stop : self.step][-1]  # )  # logger.info(  #     f"Wrote final coordinates to {ocoords.name} and trajectory to {otraj.name}"  # )


parser = ArgumentParser(
    prog="zdens",
    description="Compute z-density relative to clay surface OB atoms.",
    add_help=True,
    allow_abbrev=False,
)
parser.add_argument(
    "-sysname", type=str, help="System name", dest="sysname", required=True
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
    "-min",
    type=float,
    default=0,
    help="minimum in z-direction",
    dest="_min",
)
parser.add_argument(
    "--not_abs",
    default=False,
    help="don't use absolute z-distances",
    action="store_true",
    dest="use_abs",
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
parser.add_argument(
    "--asymmetrical_clay",
    default=False,
    action="store_true",
    help="Separate clay surfaces for OB and OH atoms.",
    dest="asymmetrical_clay",
)
parser.add_argument(
    "--lt_zero",
    default=False,
    action="store_true",
    help="Select z-distances less than zero.",
    dest="lt_zero",
)

if __name__ == "__main__":
    logger.info(f"Using MDAnalysis {mda.__version__}")
    logger.info(f"Using numpy {np.__version__}")
    args = parser.parse_args(sys.argv[1:])
    traj_format = "xtc"

    sysname = args.sysname

    gro, trr, path = get_paths(
        args.path, args.infiles, args.inpname, traj_format
    )

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
        only_surface=True,
        separate_ob_oh_surfaces=args.asymmetrical_clay,
        lt_zero=args.lt_zero,
    )

    if args.save == "True":
        args.save = True
    elif args.save == "False":
        args.save = False
    if args.write == "True":
        args.write = True
    elif args.write == "False":
        args.write = False
    if not type(clay) == tuple:
        clay = (clay,)
    upper_clay_sheet_idxs = None
    if not args.asymmetrical_clay and not args.lt_zero:
        dist_type = ["gt_zero"]
    elif not args.asymmetrical_clay and args.lt_zero:
        dist_type = ["lt_zero"]
        upper_clay_sheet_idxs = clay[0].residues.resids
        clay = (clay[0] + clay[1],)
    else:
        dist_selection_dict = {0: "smaller", 1: "larger"}
        mean_clay_z = list(map(lambda c: np.mean(c.positions[:, 2]), clay))
        dist_type = list(
            map(lambda z: dist_selection_dict[z], np.argsort(mean_clay_z))
        )
    for c, dist_selection in zip(clay, dist_type):
        zdens = ZDens(
            sysname=sysname,
            sel=sel,
            clay_type=c,
            dist_selection=dist_selection,
            n_bins=args.n_bins,
            bin_step=args.bin_step,
            xy_rad=args.xyrad,
            cutoff=args.cutoff,
            # min_cutoff=args._min,
            save=args.save,
            write=args.write,
            overwrite=args.overwrite,
            check_traj_len=args.check_traj_len,
            upper_clay_sheet_idxs=upper_clay_sheet_idxs,
            # use_abs=args.use_abs,
        )

        run_analysis(zdens, start=args.start, stop=args.stop, step=args.step)

else:
    # for action in parser._actions:
    # new_option_names = [
    #     option_string.strip("-").upper()
    #     for option_string in action.option_strings
    # ]
    # action.option_strings = new_option_names
    ZDens.parser = parser
