import logging
import sys
import pickle as pkl

import MDAnalysis
import cython
import numpy as np
import MDAnalysis as mda
import pandas as pd
from typing import Union, Optional, NoReturn, Any, List, Literal

from argparse import ArgumentParser
import warnings

from MDAnalysis.lib.distances import apply_PBC

from ClayAnalysis.classes import SimDir
from ClayAnalysis.lib import (
    check_traj,
    process_box,
    run_analysis,
    get_dist,
    select_cyzone,
)
from ClayAnalysis.analysisbase import ClayAnalysisBase, AnalysisData
from pathlib import Path
from ClayAnalysis import AA

warnings.filterwarnings("ignore", category=DeprecationWarning)

__all__ = ["AADist"]

logger = logging.getLogger(Path(__file__).stem)


class AADist(ClayAnalysisBase):
    _attrs = ["aadens", "atype"]

    def __init__(
        self,
        sysname: str,
        clay: mda.AtomGroup,
        n_bins: Optional[int] = None,
        bin_step: Optional[Union[int, float]] = None,
        xy_rad: Union[float, int] = 4.0,
        cutoff: Optional[Union[float, int]] = 50.0,
        save: Union[bool, str] = True,
        write: Union[bool, str] = True,
        check_traj_len: Union[Literal[False], int] = False,
        **basekwargs: Any,
    ) -> NoReturn:
        logger.info(clay.universe.trajectory)
        super(AADist, self).__init__(clay.universe.trajectory, **basekwargs)

        self._process_distances: callable = None
        self.sysname: str = sysname
        self.cutoff: float = float(cutoff)

        aas: np.array = pd.read_csv(AA / "aa.csv", index_col=0).index.to_numpy()
        aas = np.vectorize(lambda x: x.upper())(aas)
        aas_str = "* ".join(aas)
        aas_str += "*"

        sel: MDAnalysis.AtomGroup = clay.universe.select_atoms(
            f"resname {aas_str} and not type H*"
        )
        self.sel = sel
        self._ags = [sel]

        self._universe = self._ags[0].universe
        self.sel_n_atoms = self.sel.residues.n_residues
        self.clay = clay
        self.xy_rad = float(xy_rad)
        self._unique_res_group = self.sel.groupby("resnames")
        self._sel_atoms = {
            x: y.groupby("resnums") for x, y in self._unique_res_group.items()
        }
        self.atypes = {}
        amax = 0
        self.acodes = {}
        pos_idx = []
        self.atypes_flat = []
        self.acodes_flat = []
        for res_type, atom_groups in self._sel_atoms.items():
            self.atypes[res_type] = [
                atom.name for atom in list(atom_groups.values())[0]
            ]
            self.acodes[res_type] = amax + np.arange(
                len(np.unique(self.atypes[res_type]))
            )
            amax = np.max(self.acodes[res_type]) + 1
            pos_idx.append(
                pd.MultiIndex.from_product(
                    [
                        [res_type],
                        [
                            atom_group[0].residue.resnum
                            for atom_group in list(atom_groups.values())
                        ],
                        self.acodes[res_type],
                    ]
                ).to_frame()
            )
            self.atypes_flat.append(self.atypes[res_type])
            self.acodes_flat.append((self.acodes[res_type]))
        self.acodes_flat = np.ravel(self.acodes_flat)
        self.atypes_flat = np.ravel(self.atypes_flat)
        pos_idx = pd.concat(pos_idx)
        pos_idx = pd.MultiIndex.from_frame(pos_idx, names=["mol", "resnum", "atype"])

        self.pos_idx = pos_idx
        self._init_data(
            aadens={"n_bins": n_bins, "bin_step": bin_step, "cutoff": cutoff, "min": 0},
            atype={
                "min": 0,
                "bin_step": 1,
                "n_bins": len(self.acodes_flat),
                "cutoff": len(self.acodes_flat),
            },
        )
        self.save = save
        if self.save is False:
            pass
        else:
            if type(self.save) == bool:
                self.save = (
                    f"{self.__class__.__name__.lower()}_"
                    f"{self.sysname}_{self.sel.resnames[0]}"
                )

        self.write = write
        check_traj(self, check_traj_len)

    def _prepare(self):
        self._abs = [True, False]
        idsl = pd.IndexSlice
        process_box(self)
        logger.info(
            f"Starting run:\n"
            f"Frames start: {self.start}, "
            f"stop: {self.stop}, "
            f"step: {self.step}\n"
        )
        self._sel_df = pd.DataFrame(self._sel_atoms)
        self._sel_df = self._sel_df.stack(0)
        self._sel_df.index.names = ["resnum", "mol"]
        self._sel_df = pd.DataFrame({"ag": self._sel_df, "atype": 0, "z": np.NaN})
        self._sel_df.index = self._sel_df.index.reorder_levels([1, 0])
        self._dist_df = pd.DataFrame(
            index=self.pos_idx,
            columns=["z"],
            dtype=np.float64,
        )
        delattr(self, "pos_idx")
        dist_dict = {
            resname: np.ma.empty(
                (list(res_atoms.values())[0].n_atoms, self.clay.n_atoms, 3),
                dtype=np.float64,
                fill_value=np.nan,
            )
            for resname, res_atoms in self._sel_atoms.items()
        }
        setattr(self, "_dist_dict", dist_dict)
        sel_dict = {
            resname: np.ma.empty_like(d_arr[:, :, 0], dtype=np.float32)
            for resname, d_arr in self._dist_dict.items()
        }
        setattr(self, "_sel_dict", sel_dict)
        self.clay_pos = np.empty((self.clay.n_atoms, 3), dtype=np.float64)
        self._sel_df.reset_index("mol", inplace=True)
        self._count_df = pd.Series(index=self._sel_df.index)
        self._count_df[:] = 0
        self._count_df = self._count_df.astype("int16")

    get_arr_pos = np.vectorize(lambda ag: ag.positions)

    def get_pos(self, group):
        pos = self._sel_df.loc[group.name].positions
        pos = apply_PBC(pos, self._ts.dimensions)
        return pos

    def get_distances(self, group):
        resnum = group.name
        ag = group["ag"]
        mol = group["mol"]
        sel_pos = apply_PBC(ag.positions, self._ts.dimensions)
        arr = self._dist_dict[mol].copy()
        mask = self._sel_dict[mol].copy()
        atypes = self.acodes[mol]
        arr.fill(0)
        arr.mask = False
        mask.fill(0)
        mask.mask = False
        get_dist(sel_pos, self.clay_pos, arr, self._ts.dimensions)
        self._process_distances(arr, self._ts.dimensions)
        select_cyzone(
            distances=arr,
            xy_rad=self.xy_rad,
            z_dist=self.data["aadens"].cutoff,
            mask_array=mask,
        )
        indexer = pd.IndexSlice[mol, resnum, atypes]
        df_slice = self._dist_df.loc[indexer]
        df_slice["z"] = np.min(np.abs(arr[:, :, 2]), axis=1)
        df_slice = df_slice[df_slice["z"] == np.min(df_slice["z"])]
        if len(df_slice) != 1:
            logger.info(f"Found {len(df_slice)} equal distances!")
        group["atype"] = np.ravel(df_slice.index.get_level_values("atype").values)[0]
        group["z"] = np.ravel(df_slice.values)[0]
        return group

    def _single_frame(self):
        self._dist_df.values[:] = 0
        self.clay_pos[:] = apply_PBC(self.clay.positions, self._ts.dimensions)
        self._sel_df = self._sel_df.apply(self.get_distances, axis=1)
        self.data["aadens"].timeseries.append(self._sel_df["z"].values)
        self.data["atype"].timeseries.append(self._sel_df["atype"].values)

    def _save(self) -> NoReturn:
        if self.save is False:
            pass
        else:
            for v in self.data.values():
                logger.info(f"value: {v}")
                v.save(
                    self.save,
                    sel_names=np.unique(self.sel.names),
                    sel_n_atoms=self.sel_n_atoms,
                    # sel_n_residues=self.sel_n_residues,
                    atypes_flat=self.atypes_flat,
                    acodes_flat=self.acodes_flat,
                    atypes=self.atypes,
                    acodes=self.acodes,
                    n_frames=self.n_frames,
                )
        logger.info("Done!\n")


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="aadist",
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
        "-uc", type=str, help="Clay unit cell type", dest="clay_type", required=True
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
        "-update",
        action="store_true",
        default=False,
        help="Overwrite existing trajectory and coordinate array data.",
        dest="new",
    )

    parser.add_argument(
        "-path", default=False, help="File with analysis data paths.", dest="path"
    )
    if __name__ == "__main__":
        logger.info(f"Using MDAnalysis {mda.__version__}")
        logger.info(f"Using numpy {np.__version__}")
        logger.info(f"Using cython {cython.__version__}")
        args = parser.parse_args(sys.argv[1:])
        traj_format = ".xtc"

        if args.path:
            path = SimDir(args.path)
            gro = path.gro
            trr = path.trr
            sysname = args.sysname

        else:
            gro, trr = args.infiles
            gro, trr = Path(gro), Path(trr)
            path = SimDir(gro.parent)
            sysname = args.sysname

        logger.info(f"Found coordinates: {str(gro.resolve())!r}")
        logger.info(f"Found trajectory: {str(trr.resolve())!r}")
        logger.info(f"System name: {sysname!r}")

        outname = f"{gro}"
        outname = (path / outname).resolve()
        logger.info(f"Output path: {outname!r}")
    coords = gro
    traj = trr
    logger.info(f"Using {coords.name} and {traj.name}.")
    u = MDAnalysis.Universe(str(coords), str(traj), format="TRR", in_memory=True)
    clay = u.select_atoms(f"resname {args.clay_type}* and name OB* o*")

    if args.save == "True":
        args.save = True
    elif args.save == "False":
        args.save = False
    dist = AADist(
        sysname=args.sysname,
        clay=clay,
        n_bins=args.n_bins,
        bin_step=args.bin_step,
        cutoff=args.cutoff,
        save=args.save,
        check_traj_len=args.check_traj_len,
    )
    run_analysis(dist, start=args.start, stop=args.stop, step=args.step)
