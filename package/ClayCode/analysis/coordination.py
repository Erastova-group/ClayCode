#!/usr/bin/env python3
import logging
import pickle as pkl
import sys
import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, List, Literal, NoReturn, Optional, Union

import MDAnalysis as mda
import numpy as np
from ClayCode.analysis.analysisbase import ClayAnalysisBase
from ClayCode.analysis.consts import PE_DATA
from ClayCode.analysis.lib import (
    check_traj,
    exclude_xyz_cutoff,
    exclude_z_cutoff,
    get_dist,
    get_edge_fname,
    get_paths,
    get_selections,
    process_box,
    read_edge_file,
    run_analysis,
)
from ClayCode.analysis.zdist import ZDens
from ClayCode.builder.utils import get_checked_input, select_input_option
from ClayCode.core.utils import get_subheader
from MDAnalysis import Universe
from MDAnalysis.core.groups import AtomGroup
from MDAnalysis.lib.distances import (
    apply_PBC,
    distance_array,
    self_distance_array,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

__all__ = ["CrdDist"]

logger = logging.getLogger(Path(__file__).stem)


class CrdDist(ClayAnalysisBase):
    _attrs = ["rdf", "groups"]
    _abs = [True, True]

    def __init__(
        self,
        sysname: str,
        sel: AtomGroup,
        clay: AtomGroup,
        other: Optional[AtomGroup] = None,
        n_bins: Optional[int] = None,
        bin_step: Optional[Union[int, float]] = None,
        cutoff: Optional[Union[float, int]] = None,
        edges: Optional[Union[str, Path]] = None,
        zdist: Optional[Union[str, Path]] = None,
        save: Union[bool, str] = True,
        check_traj_len: Union[Literal[False], int] = False,
        guess_steps: bool = False,
        **basekwargs: Any,
    ) -> NoReturn:
        super(CrdDist, self).__init__(sel.universe.trajectory, **basekwargs)
        self.sysname = sysname
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
        if edges is None:
            edge_file = get_edge_fname(
                atom_type=self.sel.resnames[0],
                cutoff=cutoff,
                bins=bin_step,
                name="pe",
            )
        if type(edges) == str:
            edge_file = Path(edges)
        elif type(edges) == Path:
            edge_file = edges
        if not edge_file.is_file():
            edge_file = PE_DATA / edge_file.with_suffix(".p").name
            if edge_file.is_file():
                logger.info(
                    f"Using edge file {str(edge_file.name)!r} from database"
                )
        if not edge_file.is_file():
            logger.info(f"Edge file {str(edge_file.name)!r} does not exist.\n")
            edge_selection = select_input_option(
                instance_or_manual_setup=True,
                query="Use one of the edge files in database? [y]es/[n]o (default yes)\n",
                options=["y", "n", ""],
                result_map={"y": True, "n": False, "": True},
            )
            if edge_selection:
                options = sorted(
                    [f for f in PE_DATA.glob(f"{self.sel.resnames[0]}_*.p")]
                )
                logger.info("Available edge files:")
                for i, f in enumerate(options):
                    logger.finfo(f"{f.name}", kwd_str=f"{i}: ", indent="\t")
                edge_file = get_checked_input(
                    result_type=int,
                    result=edge_file,
                    exit_val="e",
                    check_value="|".join(list(map(str, range(len(options))))),
                    query="Select edge file: (exit with e)\n",
                )
                if edge_file == "e":
                    logger.info("Exiting.")
                    sys.exit(0)
                else:
                    edge_file = options[edge_file]

        assert edge_file.is_file(), f"edge file {edge_file} does not exist"
        self._edges_file = edge_file
        self._edges = read_edge_file(self._edges_file, cutoff, skip=False)
        self._edges = self._edges[self._edges > 0]
        while self._edges[-1] > cutoff:
            self._edges = self._edges[:-1]
        if self._edges[-1] < cutoff:
            self._edges = np.append(self._edges, cutoff)
        self._attrs.extend(
            [f"group_{edge}" for edge in range(len(self._edges))]
        )
        self.zdist = zdist
        bin_step = dict(
            map(
                lambda x: (x, bin_step) if x != "groups" else (x, 1),
                self._attrs,
            )
        )
        cutoff = dict(
            map(
                lambda x: (x, cutoff)
                if x != "groups"
                else (x, len(self._edges)),
                self._attrs,
            )
        )
        verbose = dict(
            map(
                lambda x: (x, True) if x in ["groups", "rdf"] else (x, False),
                self._attrs,
            )
        )
        self._init_data(
            n_bins=n_bins,
            bin_step=bin_step,
            cutoff=cutoff,
            min=0,
            verbose=verbose,
        )
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
        # print(datargs.files, data['edge_file'].shape)
        # cutoff = np.ravel(self.edge_file),
        # cutoff = np.rint(np.max(np.ravel(self.edge_file)))

        # if r_cutoff is None:
        #     self.r_cutoff = np.array([*np.max(self.sel.universe.dimensions[:2]), 5.0])
        # elif type(r_cutoff) in [int, float]:
        #     self.r_cutoff = np.array([float(r_cutoff) for c in range(3)])
        # elif type(r_cutoff) == list and len(r_cutoff) == 3:
        #     self.r_cutoff = np.array(r_cutoff)
        # else:
        #     raise ValueError('Wrong type or length for cutoff!')
        # self.r_cutoff = self.r_cutoff.astype(np.float64)
        # print('r cutoff', self.r_cutoff)
        # self.save = save
        # if self.save is False:
        #     pass
        # else:
        #     if type(self.save) == bool:
        #         self.save = (
        #             f"{self.__class__.__name__.lower()}_"
        #             f"{self.sysname}_{self.sel.resnames[0]}"
        #         )
        # self._other_dist_f = distance_array
        # self._provide_args = lambda: self.sel.positions, self.other.positions

        # check_traj(self, check_traj_len)
        # self._guess_steps = guess_steps

    def _prepare(self) -> NoReturn:
        logger.info(
            f"Starting run:\n"
            f"Frames start: {self.start}, "
            f"stop: {self.stop}, "
            f"step: {self.step}\n"
        )
        zdist = None
        overwrite = False
        while zdist is None:
            zdist = self.zdist
            if type(zdist) == str:
                zdist = Path(zdist)
            if zdist is None or not Path(zdist).is_file():
                zdist = ZDens(
                    sysname=sysname,
                    sel=sel,
                    clay=clay,
                    n_bins=self.data["rdf"].n_bins,
                    bin_step=self.data["rdf"].bin_step,
                    cutoff=self.data["rdf"].cutoff,
                    save=False,
                    write=self.zdist,
                    overwrite=overwrite,
                )
                run_analysis(
                    zdist, start=args.start, stop=args.stop, step=args.step
                )
                zdist = Path(zdist.write)
            zdata = np.load(zdist)
            self.mask = zdata["mask"]
            self.zdata = np.ma.masked_array(zdata["zdist"], mask=self.mask)
            start, stop, step = zdata["run_prms"]
            if len(
                np.arange(start=self.start, stop=self.stop, step=self.step)
            ) != len(self.zdata):
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
                    logger.finfo(
                        "Slicing error!\n"
                        f"Expected start: {start:d}, "
                        f"stop: {stop:d}, "
                        f"step: {step:d}.\n"
                        f"Found start: {self.start:d}, "
                        f"stop: {self.stop:d}, "
                        f"step: {self.step:d}\n"
                        f"Will overwrite {zdist.name!r} with new zdens data.\n"
                    )
                    self.zdist = zdist = None
                    overwrite = True
                    continue
            if self.sel_n_atoms != zdata["sel_n_atoms"]:
                raise ValueError(
                    f"Atom number mismatch between z-data ({zdata['sel_n_atoms']}) and selection atoms ({self.sel.n_atoms})!"
                )
            self._z_cutoff = np.rint(zdata["cutoff"])
        process_box(self)

        self._dist = np.empty(
            (self.sel.n_atoms, self.other.n_atoms, 3),
            dtype=np.float64,
        )

        self._dist_m = np.ma.array(
            self._dist,
            dtype=np.float64,
            fill_value=np.nan,
        )

        self._rad = np.empty(
            (self.sel.n_atoms, self.other.n_atoms),
            dtype=np.float64,
        )
        self._rad_m = np.ma.array(
            self._rad,
            fill_value=np.nan,
            dtype=np.float64,
        )
        # self._z_dist = np.margs.empty(
        #     (self.sel.n_atoms, self.other.n_atoms), fill_value=np.nan, dtype=np.float64
        # )

        # _attrs absolute
        # self._abs = [True, True, False]

        self._sel_pos = np.empty((self.sel_n_atoms, 3), dtype=np.float64)
        self._sel_pos_m = np.ma.array(
            self._sel_pos, fill_value=np.nan, dtype=np.float64
        )
        # self._other_pos = np.margs.empty((self._other_pos, 3),
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

    def _single_frame(self) -> NoReturn:
        self._edge_numbers = np.digitize(
            self.zdata[self._frame_index], self._edges
        )
        self._rad.fill(0)
        self._rad_m.mask = False
        self._dist.fill(0)
        self._dist_m.fill(0)
        self._dist_m.mask = False  # [..., 2] = self.mask[self._frame_index]
        self._sel_pos.fill(0)
        self._sel_pos_m.fill(0)
        self._sel_pos_m.soften_mask()
        self._sel_pos_m.mask = np.broadcast_to(
            self.mask[self._frame_index, :, np.newaxis], self._sel_pos_m.shape
        )
        self._sel_pos_m.harden_mask()
        self._dist_m.mask = np.broadcast_to(
            self._sel_pos_m.mask[:, np.newaxis, :], self._dist_m.shape
        )
        # self._dist.fill(0)
        # self._dist.mask = self.mask[self._frame_index]

        self._sel_pos_m[:] = self.sel.positions
        self._sel_pos_m[:] = apply_PBC(self._sel_pos_m, self._ts.dimensions)

        if self.self is False:
            self._other_pos.fill(0)
            self._other_pos[:] = apply_PBC(
                self.other.positions, self._ts.dimensions
            )
        else:
            self._dist_m.mask += self.diag_mask
            self._other_pos = self._sel_pos
        # if self.self:

        # if self.self is True:
        #     sel = self._dist == 0.0
        #     sel = np.bitwise_or.accumulate(sel, axis=2)
        #     self._dist.mask += sel

        get_dist(
            self._sel_pos_m, self._other_pos, self._dist_m, self._ts.dimensions
        )
        self._process_distances(self._dist_m, self._ts.dimensions)
        new_mask = np.any(
            np.greater(np.abs(self._dist_m), self.data["rdf"].cutoff), axis=2
        )
        self._dist_m.mask += new_mask[:, :, np.newaxis]
        self._rad[:] = np.linalg.norm(self._dist_m, axis=2)
        self._rad[:] = np.where(
            self._rad <= self.data["rdf"].cutoff, self._rad, 0
        )
        self._rad[:] = np.where(self._rad == 0, np.NaN, self._rad)
        # if self.self is True:
        #     self._rad.mask += self._rad == 0.0
        self.data["groups"].timeseries.append(self._edge_numbers)
        self.data["rdf"].timeseries.append(self._rad)
        # # categorise by edge
        # # self._rad_m.mask = np.broadcast_to(
        # self._dist_m.mask += (
        #     np.abs(self._dist) >= self.data["zmtd"].cutoff
        # )
        # self._dist_m.mask = np.broadcast_to(
        #     self._dist.mask[..., 2, np.newaxis], self._dist.shape
        # )
        #
        # exclude_z_cutoff(self._dist, self.data["zmtdist"].cutoff)
        #
        # self.data["zmtdist"].timeseries.append(self._dist[:, :, 2])
        # self._rad[:] = np.linalg.norm(self._dist, axis=2)
        # self._rad.mask = self._rad > self.data["rdf"].cutoff
        # self.data["rdf"].timeseries.append(self._rad)
        # # print(self._rad.shape, self._dist.shape)
        # rdist = np.min(self._rad, axis=1)
        # self.data["rmtdist"].timeseries.append(rdist)
        # print(np.min(self._rad, axis=1).shape)

    def _post_process(self) -> NoReturn:
        logger.finfo(
            f"Grouping RDF data by adsorption shell on clay surface.",
            initial_linebreak=True,
        )
        prev = 0
        for i, edge in enumerate(self._edges):
            self.data[f"group_{i}"].timeseries = np.where(
                np.array(self.data["groups"].timeseries)[:, :, np.newaxis]
                != i,
                np.array(self.data["rdf"].timeseries),
                np.NaN,
            )
            logger.finfo(
                f'"group_{i}"',
                kwd_str=f"{prev:.1f} <= z < {edge:.2f}: ",
                indent="\t",
            )

    def _save(self):
        if self.save is False:
            pass
        else:
            for v in self.data.values():
                v.save(
                    self.save,
                    sel_names=np.unique(self.sel.names),
                    n_atoms=self.sel.n_atoms,
                    n_frames=self.n_frames,
                    other=np.unique(self.other.names),
                    n_other_atoms=self.other.n_atoms,
                )
        print("Done!")


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="coordiantion",
        description="Compute radial distributions between 2 atom types.",
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
        "-zdist",
        type=str,
        help="z-dist data filename",
        dest="zdist",
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
        type=str,
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
        "-path",
        default=False,
        help="File with analysis data paths.",
        dest="path",
    )
    parser.add_argument(
        "--in_mem",
        default=False,
        action="store_true",
        help="Read trajectory in memory.",
        dest="in_mem",
    )


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])

    sysname = args.sysname

    gro, trr, path = get_paths(
        infiles=args.infiles, inpname=args.inpname, path=args.path
    )

    logger.finfo(f"{sysname!r}", kwd_str=f"System name: ")

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

    # if args.r_cutoff is None:
    #     r_cutoff = args.cutoff
    # if len(args.r_cutoff) == 1:
    #     r_cutoff = [args.r_cutoff[0] for c in range(3)]
    # elif len(args.r_cutoff) == 3:
    #     r_cutoff = args.r_cutoff
    # else:
    #     raise ValueError('Expected either 1 or 3 arguments for r_cutoff!')
    #
    # print(r_cutoff)

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
    sel, clay, other = get_selections(
        infiles=(coords, traj),
        sel=args.sel,
        clay_type=args.clay_type,
        other=args.other,
        in_memory=args.in_mem,
    )

    if args.save == "True":
        args.save = True
    elif args.save == "False":
        args.save = False
    # if args.write == "True":
    #     args.write = True
    # elif args.write == "False":
    #     args.write = False

    dist = CrdDist(
        sysname=args.sysname,
        sel=sel,
        clay=clay,
        other=other,
        n_bins=args.n_bins,
        bin_step=args.bin_step,
        cutoff=args.cutoff,
        edges=args.edges,
        zdist=args.zdist,
        save=args.save,
        check_traj_len=args.check_traj_len,
    )
    run_analysis(dist, start=args.start, stop=args.stop, step=args.step)
