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
from lib import (
    check_traj,
    exclude_xyz_cutoff,
    exclude_z_cutoff,
    get_dist,
    get_selections,
    process_box,
    run_analysis,
)
from MDAnalysis.core.groups import AtomGroup
from MDAnalysis.lib.distances import (
    apply_PBC,
    distance_array,
    self_distance_array,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

__all__ = ["RDFDist"]

logger = logging.getLogger(Path(__file__).stem)


class RDFDist(ClayAnalysisBase):
    _attrs = ["rdf", "rmtdist", "zmtdist"]

    def __init__(
        self,
        sysname: str,
        sel: mda.AtomGroup,
        clay: mda.AtomGroup,
        zdist: Union[str, Path],
        other: Optional[mda.AtomGroup] = None,
        n_bins: Optional[int] = None,
        bin_step: Optional[Union[int, float]] = None,
        cutoff: Optional[Union[float, int]] = None,
        save: Union[bool, str] = True,
        check_traj_len: Union[Literal[False], int] = False,
        guess_steps: bool = True,
        **basekwargs: Any,
    ) -> NoReturn:
        super(RDFDist, self).__init__(sel.universe.trajectory, **basekwargs)
        self._init_data(n_bins=n_bins, bin_step=bin_step, cutoff=cutoff, min=0)
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

        if type(zdist) == str:
            zdist = Path(zdist)
        assert zdist.is_file(), f"z-density file {zdist} does not exist"
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
        # print(data.files, data['zdist'].shape)
        # cutoff = np.ravel(self.zdist),
        # cutoff = np.rint(np.max(np.ravel(self.zdist)))

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
        process_box(self)

        self._dist = np.empty(
            (self.sel.n_atoms, self.other.n_atoms, 3), dtype=np.float64
        )

        self._dist_m = np.ma.array(
            self._dist, dtype=np.float64, fill_value=np.nan
        )

        self._rad = np.empty(
            (self.sel.n_atoms, self.other.n_atoms), dtype=np.float64
        )
        self._rad_m = np.ma.array(
            self._rad, fill_value=np.nan, dtype=np.float64
        )
        # self._z_dist = np.ma.empty(
        #     (self.sel.n_atoms, self.other.n_atoms), fill_value=np.nan, dtype=np.float64
        # )

        # _attrs absolute
        self._abs = [True, True, False]

        self._sel_pos = np.empty((self.sel_n_atoms, 3), dtype=np.float64)
        self._sel_pos_m = np.ma.array(
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

    def _single_frame(self) -> NoReturn:
        self._rad.fill(0)
        self._rad_m.mask = False
        self._dist.fill(0)
        self._dist_m.mask = False  # [..., 2] = self.mask[self._frame_index]
        self._sel_pos.fill(0)
        self._sel_pos_m.mask.soften_mask()
        self._sel_pos.mask = self.mask[self._frame_index]
        self._sel_pos.harden_mask()
        self._dist_m.mask = np.broadcast_to(
            self._sel_pos.mask[:, np.newaxis, :], self._dist.shape
        )
        # self._dist.fill(0)
        # self._dist.mask = self.mask[self._frame_index]

        self._sel_pos[:] = self.sel.positions
        self._sel_pos[:] = apply_PBC(self._sel_pos, self._ts.dimensions)

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
            self._sel_pos[..., 2], self._other_pos[..., 2], self._dist[..., 2]
        )
        self._process_axes(self._dist[..., 2], self._ts.dimensions, [2])

        self._dist_m.mask[..., 2] += (
            np.abs(self._dist[..., 2]) >= self.data["zmtd"].cutoff
        )
        self._dist_m.mask = np.broadcast_to(
            self._dist.mask[..., 2, np.newaxis], self._dist.shape
        )

        exclude_z_cutoff(self._dist, self.data["zmtdist"].cutoff)

        self.data["zmtdist"].timeseries.append(self._dist[:, :, 2])
        self._rad[:] = np.linalg.norm(self._dist, axis=2)
        self._rad.mask = self._rad > self.data["rdf"].cutoff
        self.data["rdf"].timeseries.append(self._rad)
        # print(self._rad.shape, self._dist.shape)
        rdist = np.min(self._rad, axis=1)
        self.data["rmtdist"].timeseries.append(rdist)
        # print(np.min(self._rad, axis=1).shape)

        if self.self is True:
            self._rad.mask += self._rad == 0.0

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
        prog="multidist",
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
        required=True,
    )
    parser.add_argument(
        "-zdist",
        type=str,
        help="z-dist data filename",
        dest="zdist",
        required=True,
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
        "-ads_edges",
        type=float,
        nargs="+",
        help="Adsorption shell upper limits",
        required=False,
        dest="ads_edges",
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

    a = parser.parse_args(sys.argv[1:])

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

    sel, clay, other = get_selections(a.infiles, a.sel, a.clay_type, a.other)

    if a.save == "True":
        a.save = True

    dist = RDFDist(
        sysname=a.sysname,
        sel=sel,
        clay=clay,
        other=other,
        n_bins=a.n_bins,
        bin_step=a.bin_step,
        cutoff=a.cutoff,
        save=a.save,
        zdist=a.zdist,
        check_traj_len=a.check_traj_len,
    )
    run_analysis(dist, start=a.start, stop=a.stop, step=a.step)
