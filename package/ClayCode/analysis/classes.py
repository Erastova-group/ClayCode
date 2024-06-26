from __future__ import annotations

import logging
import os
import pickle as pkl
import re
import sys
from functools import cached_property
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from ClayCode.analysis.analysisbase import AnalysisData
from ClayCode.analysis.consts import PE_DATA
from ClayCode.analysis.utils import get_atom_type_group, make_1d, redirect_tqdm
from ClayCode.core.classes import Dir, File, PathType
from ClayCode.core.consts import LINE_LENGTH
from ClayCode.plots.classes import Bins, Cutoff
from matplotlib import colormaps
from matplotlib import colors as mpc
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Data:
    """
    Class for histogram analysis data processing and plotting.
    Reads files in `indir` that match the naming pattern
    "`namestem`*_`cutoff`_`bins`.dat"
    (or "`namestem`*_`cutoff`_`bins`_`analysis`.dat" if `analysis` != `None`.
    The data is stored in a :class:`pandas.DataFrame`
    :param indir: Data directory
    :type indir: Union[str, Path]
    :param cutoff: Maximum value in the histogram bins
    :type cutoff: Union[int, float]
    :param bins: Histogram bin size
    :type bins: float
    :param ions: List of ion types in solvent
    :type ions: List[Literal['Na', 'K', 'Ca', 'Mg']], optional
    :param atoms: List of atom types in selection, defaults to `None`
    :type atoms: List[Literal['ions', 'OT', 'N', 'CA', 'OW']], optional
    :param other: Optional list of atom types in a second selection,
     defaults to `None`
    :type other: List[Literal['ions', 'OT', 'N', 'CA', 'OW']], optional
    :param clays: List of clay types,
     defaults to `None`
    :type clays: List[Literal['NAu-1', 'NAu-2']], optional
    :param aas: List of amino acid types in lower case 3-letter code,
     defaults to `None`
    :type aas: Optional[List[Literal['ala', 'arg', 'asn', 'asp',
                                   'ctl', 'cys', 'gln', 'glu',
                                   'gly', 'his', 'ile', 'leu',
                                   'lys', 'met', 'phe', 'pro',
                                   'ser', 'thr', 'trp', 'tyr',
                                   'val']]]
    :param load: Load,  defaults to False
    :type load: Union[str, Literal[False], Path], optional
    :param odir: Output directory, defaults to `None`
    :type odir: str, optional
    :param nameparts: number of `_`-separated partes in `namestem`
    :type nameparts: int, defaults to 1
    :param namestem: leading string in naming pattern, optional
    :type namestem: str, defaults to ''
    :param analysis: trailing string in naming pattern, optional
    defaults to `None`
    :type analysis: str, optional
    :param df: :class: `pandas.DataFrame`
    :param atomname: atom names included in filename
    :type atomname: bool, defaults to True
    :param use_rel_data: use relative data
    :type use_rel_data: bool, defaults to False
    :param group_all: group all atoms into one category
    :type group_all: bool, defaults to False
    """

    aas = [
        "ala",
        "arg",
        "asn",
        "asp",
        "ctl",
        "cys",
        "gln",
        "glu",
        "gly",
        "his",
        "ile",
        "leu",
        "lys",
        "met",
        "phe",
        "pro",
        "ser",
        "thr",
        "trp",
        "tyr",
        "val",
    ]

    ions = ["Na", "K", "Ca", "Mg", "Ba"]
    atoms = [
        "ions",
        "OT",
        "N",
        "CA",
        "NH",
        "OH",
        "CR",
        "SR",
        "C",
        "NR",
        "OR",
        "O",
        "S",
    ]
    clays = [
        "NAu-1",
        "NAu-2",
        "SWy-123",
        "SWy-1-simplified",
        "KGa-1",
        "IMt-1",
    ]  # , 'LDH31']

    @redirect_tqdm
    def __init__(
        self,
        indir: Union[str, Path],
        cutoff: Union[int, float],
        bins: float,
        ions: List[Literal["Na", "K", "Ca", "Mg", "Ba"]] = None,
        atoms: List[Literal["ions", "OT", "N", "CA", "OW"]] = None,
        other: List[Literal["ions", "OT", "N", "CA", "OW"]] = None,
        clays: List[Literal["NAu-1", "NAu-2"]] = None,
        aas: List[
            Literal[
                "ala",
                "arg",
                "asn",
                "asp",
                "ctl",
                "cys",
                "gln",
                "glu",
                "gly",
                "his",
                "ile",
                "leu",
                "lys",
                "met",
                "phe",
                "pro",
                "ser",
                "thr",
                "trp",
                "tyr",
                "val",
            ]
        ] = None,
        load: Union[str, Literal[False], Path] = False,
        odir: Optional[str] = None,
        nameparts: int = 1,
        namestem: str = "",
        analysis: Optional[str] = None,
        atomname=True,
        use_rel_data=False,
        group_all=False,
        max_line_width=LINE_LENGTH,
    ):
        """Constructor method"""
        logger.info(f"Initialising {self.__class__.__name__}")
        self.infilelist: list = []
        self.bins: Bins = Bins(bins)
        self.cutoff: float = Cutoff(cutoff)
        self.analysis: Union[str, None] = analysis
        if use_rel_data is True:
            self._arr_col = 2
        else:
            self._arr_col = 1

        if type(indir) != Path:
            indir = Path(indir)

        self._indir = indir

        if self.analysis is None:
            logger.info(
                rf"Getting {namestem}*_"
                rf"{self.cutoff}_"
                rf"{self.bins}.dat from {str(indir.resolve())!r}"
            )
            self.filelist: List[Path] = sorted(
                list(
                    indir.glob(
                        rf"{namestem}*_" rf"{self.cutoff}_" rf"{self.bins}.dat"
                    )
                )
            )
        else:
            logger.info(
                rf"Getting {namestem}*_"
                rf"{self.cutoff}_"
                rf"{self.bins}_"
                rf"{analysis}.dat from {str(indir.resolve())!r}"
            )
            self.filelist: List[Path] = sorted(
                list(
                    indir.glob(
                        rf"{namestem}*_"
                        rf"{self.cutoff}_"
                        rf"{self.bins}_"
                        rf"{self.analysis}.dat"
                    )
                )
            )
        logger.info(f"Found {len(self.filelist)} files.")

        if load is not False:
            load = Path(load.resolve())
            self.df: pd.DataFrame = pkl.load(load)
            logger.info(f"Using data from {load!r}")
        else:
            if ions is None:
                ions = self.__class__.ions
                logger.info(
                    f"ions not specified, using default {self.__class__.ions}"
                )
            else:
                logger.info(f"Using custom {ions} for ions")
            if atoms is None:
                atoms = self.__class__.atoms
                logger.info(
                    f"atoms not specified, using default {self.__class__.atoms}"
                )
            else:
                logger.info(f"Using custom {atoms} for atoms")
            if atomname is False:
                assert len(atoms) == 1, "Expected one atom category"
                self.atomnames = atoms
                logger.finfo(f"Setting atomnames to {self.atomnames}")
            else:
                self.atomnames = None
            if aas is None:
                aas = self.__class__.aas
                logger.info(
                    f"aas not specified, using default {self.__class__.aas}"
                )
            else:
                logger.info(f"Using custom {aas} for aas")
            if clays is None:
                clays = self.__class__.clays
                logger.info(
                    f"clays not specified, using default {self.__class__.clays}"
                )
            else:
                logger.info(f"Using custom {clays} for clays")

            # f = self.filelist[0]
            # print(f)
            x = None
            for fi, f in enumerate(self.filelist):
                new_x = pd.read_csv(
                    f, delimiter="\s+", comment="#", header=None
                ).to_numpy()
                new_x = new_x[:, 0]
                if not np.isclose(
                    np.ceil(new_x[-1]), self.cutoff.num
                ) or not np.any(np.isclose(np.diff(new_x), self.bins.num)):
                    self.filelist.pop(fi)
                    logger.info(
                        f"Bins ({np.unique(np.diff(new_x).round(2))}) and cutoff {np.ceil(new_x[-1])} do not match requirements!\n"
                        f"Removing {f.name!r} from filelist"
                    )
                    continue
                if x is not None:
                    if np.min(new_x) < np.min(x) or np.max(new_x) > np.max(x):
                        x = np.union1d(x, new_x)
                else:
                    x = new_x

            cols = pd.Index(clays, name="clays")

            if other is not None:
                if other is True:
                    other = atoms
                    other.append("OW")
                idx = pd.MultiIndex.from_product(
                    [ions, aas, atoms, other, x],
                    names=["ions", "aas", "atoms", "other", "x"],
                )
                self.other: List[str] = other
                logger.info(f"Setting second atom selection to {self.other}")
            else:
                idx = pd.MultiIndex.from_product(
                    [ions, aas, atoms, x], names=["ions", "aas", "atoms", "x"]
                )
                self.other: None = None
            self.df: pd.DataFrame = pd.DataFrame(index=idx, columns=cols)

            self._get_data(nameparts)

        self.df.dropna(inplace=True, how="all", axis=0)
        self.df.dropna(inplace=True, how="all", axis=1)
        # self.not_ions = [atom for atom in self.df.index.get_level_values('atoms').unique() if (atom != 'ions' and atom not in self.df.index.get_level_values('ions').unique())]
        if group_all is True:
            mod_at = self.df.reset_index("atoms")["atoms"]
            mod_at.update(
                mod_at[mod_at != "ions"].apply(
                    lambda x: x if type(x) == str else x[0]
                )
            )  # groupby(self.df.index.drop('atoms').names)['atoms'].
            self.df.reset_index("atoms", inplace=True)
            self.df["atoms"].update(mod_at)
            self.df.set_index("atoms", append=True, inplace=True)
            self.df = self.df.reorder_levels(idx.names)
            self.df = self.df.groupby(self.df.index.names).sum()
            self.name += "_grouped"
            self.group_all = "_grouped"
        else:
            self.group_all = ""
        self.df = self.df.groupby(self.df.index.names).sum()
        self.not_ions = [
            atom
            for atom in self.df.index.get_level_values("atoms").unique()
            if (
                atom != "ions"
                and atom
                not in self.df.index.get_level_values("ions")
                .unique()
                .to_numpy()
            )
        ]

        setattr(self, self.df.columns.name, list(self.df.columns))
        self._process_atom_type_idx(idx)
        self._init_xbin_idx()

        for iid, i in enumerate(self.df.index.names):
            value: List[Union[str, float]] = (
                self.df.index._get_level_values(level=iid).unique().tolist()
            )
            logger.info(
                f"Setting {i} to {np.array2string(np.array(value), precision=2, separator=', ', suppress_small=True)}"
            )
            setattr(self, i, value)

        if odir is not None:
            self.odir: Path = Path(odir)
        else:
            self.odir: Path = Path(".").cwd()

        logger.info(f"Output directory set to {str(self.odir.resolve())!r}\n")
        self._bin_df = pd.DataFrame(columns=self.df.columns)

        self._edges = {}
        self._peaks = {}

    @redirect_tqdm
    def _process_atom_type_idx(self, idx: pd.MultiIndex) -> None:
        """Process atom type index. Replaces 'ions' with ion type and
        adds '_atoms' level.
        :param idx: index
        :type idx: pd.MultiIndex"""
        self.df.reset_index(level=["ions", "atoms"], inplace=True)
        self.df["_atoms"] = self.df["atoms"].where(
            self.df["atoms"] != "ions", self.df["ions"], axis=0
        )
        self.df.set_index(
            ["ions", "atoms", "_atoms"], inplace=True, append=True
        )
        self.df.index = self.df.index.reorder_levels([*idx.names, "_atoms"])
        self._atoms = self.df.index.get_level_values("_atoms").tolist()

    @redirect_tqdm
    def _init_xbin_idx(self) -> None:
        """Initialise x_bins index."""
        self.df["x_bins"] = np.NaN
        self.df.set_index(["x_bins"], inplace=True, append=True)

    @redirect_tqdm
    def _get_data(self, nameparts: int) -> None:
        """Get data from files in `filelist` and store in :class:`pandas.DataFrame`
        :param nameparts: number of `_`-separated parts in filename stem
        :type nameparts: int"""
        idsl = pd.IndexSlice
        for f in tqdm(self.filelist, desc="filelist"):
            namesplit = f.stem.split("_")
            if self.analysis is not None:
                namesplit.pop(-1)
            else:
                self.analysis = "zdist"
            name = namesplit[:nameparts]
            namesplit = namesplit[nameparts:]
            if self.other is not None:
                # other = namesplit[5]
                other = namesplit.pop(5)
                if other in self.ions:
                    other = "ions"
            try:
                if self.atomnames is None:
                    clay, ion, aa, pH, atom, cutoff, bins = namesplit
                else:
                    clay, ion, aa, pH, cutoff, bins = namesplit
                    atom = self.atomnames
                assert cutoff == self.cutoff
                assert bins == self.bins
                array = pd.read_csv(
                    f, delimiter="\s+", comment="#", header=None, index_col=0
                )
                try:
                    self.df.loc[
                        idsl[ion, aa, atom, array.index.values], clay
                    ] = array.values[
                        :, self._arr_col
                    ]  # print(array[:, self._arr_col].sum(), clay, ion)
                except ValueError:
                    try:
                        self.df.loc[
                            idsl[ion, aa, atom, other, array.index.values],
                            clay,
                        ] = array.values[:, self._arr_col]
                    except NameError:
                        pass
                except IndexError:
                    try:
                        self.df.loc[
                            idsl[ion, aa, atom, array.index.values], clay
                        ] = array.values[:, 1]
                    except ValueError:
                        self.df.loc[
                            idsl[ion, aa, atom, other, array.index.values],
                            clay,
                        ] = array.values[:, 1]
                except KeyError:
                    pass
            except IndexError as e:
                logger.info(e)
                logger.info(f"Encountered IndexError while getting data")
            except ValueError as e:
                logger.info(e)
                logger.info(f"Encountered ValueError while getting data")
        self.name = "_".join(name)

    def __repr__(self):
        return self.df[self.clays].dropna().__repr__()

    @property
    def densdiff(self) -> pd.DataFrame:
        """Density difference between different clay types.
        :return: density difference
        :rtype: pd.DataFrame"""
        try:
            return self.df["diff"].dropna()
        except KeyError:
            self._get_densdiff()
            return self.df["diff"].dropna()

    def _get_densdiff(self) -> None:
        """Calculate density difference between different clay types."""
        self.df["diff"] = -self.df.diff(axis=1)[self.df.columns[-1]]

    def plot(
        self,
        x: Literal["clays", "aas", "ions", "atoms", "other"],
        y: Literal["clays", "ions", "aas", "atoms", "other"],
        select: Literal["clays", "ions", "aas", "atoms", "other"],
        rowlabel: str = "y",
        columnlabel: str = "x",
        figsize=None,
        dpi=None,
        diff=False,
        xmax=50,
        ymax=50,
        save=False,
        xlim=None,
        ylim=None,
        odir=".",
        plot_table=None,
    ):
        aas_classes = [
            ["arg", "lys", "his"],
            ["glu", "gln"],
            ["cys"],
            ["gly"],
            ["pro"],
            ["ala", "val", "ile", "leu", "met"],
            ["phe", "tyr", "trp"],
            ["ser", "thr", "asp", "gln"],
        ]
        ions_classes = [["Na", "Ca"], ["Ca", "Mg"]]
        atoms_classes = [["ions"], ["N"], ["OT"], ["CA"]]
        clays_classes = [["NAu-1"], ["NAu-2"]]
        cmaps_seq = ["Purples", "Blues", "Greens", "Oranges", "Reds"]
        cmaps_single = ["Dark2"]
        sel_list = ("clays", "ions", "aas", "atoms")
        # for color, attr in zip([''], sel_list):
        #     cmaps_dict[attr] = {}
        # cm.get_cmap()
        cmap_dict = {"clays": []}

        title_dict = {
            "clays": "Clay type",
            "ions": "Ion type",
            "aas": "Amino acid",
            "atoms": "Atom type",
            "other": "Other atom type",
        }

        sel_list = ["clays", "ions", "aas", "atoms"]

        if self.other != None:
            sel_list.append("other")

        separate = [s for s in sel_list if (s != x and s != y and s != select)]
        idx = pd.Index([s for s in sel_list if (s != x and s not in separate)])

        sep = pd.Index(separate)

        vx = getattr(self, x)
        # print(vx)

        if diff == True:
            vx = "/".join(vx)
            lx = 1
        else:
            lx = len(vx)

        vy = getattr(self, y)

        ly = len(vy)
        # print(ly)

        yid = np.ravel(np.where(np.array(idx) == y))[0]
        # print(yid)

        label_key = idx.difference(
            pd.Index([x, y, *separate]), sort=False
        ).values[0]

        label_id = idx.get_loc(key=label_key)

        # label_classes = locals()[f'{label_key}_classes']
        # cmap_dict = {}
        # single_id = 0
        # seq_id = 0
        # for category in label_classes:
        #     if len(category) == 1:
        #         cmap = matplotlib.cycler('color', cm.Dark2.colors)
        #         single_id += 1
        #     else:
        #         cmap = getattr(cm, cmaps_seq[seq_id])(np.linspace(0, 1, len(category)))
        #
        #         cmap = matplotlib.cycler('color', cmap)
        #         # cmap = cmap(np.linspace(0, 1, len(category))).colors
        #             #                                viridis(np.linspace(0,1,N)))
        #             # cm.get_cmap(cmaps_seq[seq_id], len(category))
        #         seq_id += 1
        #     for item_id, item in enumerate(category):
        #         cmap_dict[item] = cmap.__getitem__(item_id)
        #
        n_plots = len(sep)

        x_dict = dict(zip(vx, np.arange(lx)))

        if diff == True:
            diffstr = "diff"
            sel = "diff"
            self._get_densdiff()
        else:
            sel = self.clays
            diffstr = ""

        plot_df = self.df[sel].copy()
        plot_df.index = plot_df.index.droplevel(["_atoms", "x_bins"])
        plot_df.reset_index().set_index([*idx, "x"])
        # print(plot_df.head(5))

        if figsize == None:
            figsize = tuple(
                [
                    5 * lx if (10 * lx) < xmax else xmax,
                    5 * ly if (5 * ly) < ymax else ymax,
                ]
            )

        if dpi == None:
            dpi = 300

        iters = np.array(
            np.meshgrid(*[getattr(self, idxit) for idxit in idx])
        ).T.reshape(-1, len(idx))

        logger.info(f"Printing plots for {sep}\nColumns: {vx}\nRows: {vy}")

        label_mod = lambda l: ", ".join(
            [li.upper() if namei == "aas" else li for li, namei in l]
        )

        sep_it = np.array(
            np.meshgrid(*[getattr(self, idxit) for idxit in sep])
        ).T.reshape(-1, len(sep))
        # print(vx, vy, lx, ly)

        for pl in sep_it:
            # print(pl)
            # try:
            #     fig.clear()
            # except:
            #     pass
            y_dict = dict(zip(vy, np.arange(ly)))
            if separate == "atoms" and pl != "":
                ...

            legends_list = [(a, b) for a in range(ly) for b in range(lx)]

            legends = dict(
                zip(legends_list, [[] for a in range(len(legends_list))])
            )

            # if type(pl) in [list, tuple, np.ndarray]:
            #     viewlist = []
            #     for p in pl:
            #         viewlist.append(plot_df.xs((p), level=separate, axis=0))
            #
            #     sepview = pd.concat(viewlist)
            #     plsave = 'ions'
            #
            # else:
            sepview = plot_df.xs((pl), level=separate, axis=0)
            plsave = pl

            fig, ax = plt.subplots(
                nrows=ly,
                ncols=lx,
                figsize=figsize,
                sharey=True,
                dpi=dpi,
                constrained_layout=True,
            )

    # def _get_edge_fname(
    #     self,
    #     atom_type: str,
    #     other: Optional[str],
    #     name: Union[Literal["pe"], Literal["edge"]] = "pe",
    # ):
    #     if other is not None:
    #         other = f"{other}_"
    #     else:
    #         other = ""
    # # fname = Path.cwd() / f"edge_data/edges_{atom_type}_{self.cutoff}_{self.bins}.p"
    #                 / f"{self.name}_{diffstr}_{x}_{y}_{plsave}_{self.cutoff}_{self.bins}.png"
    #             )
    #     else:
    #         plt.show()
    #     self.fig.clear()

    def _get_edge_fname(
        self,
        atom_type: str,
        other: Optional[str],
        name: Union[Literal["pe"], Literal["edge"]] = "pe",
    ):
        return get_edge_fname(atom_type, name, other, PE_DATA)

    def _get_edges(
        self,
        height: Union[float, int] = 0.01,
        distance: Union[float, int] = 2,
        width: int = 1,
        wlen: int = 11,
        peak_cutoff: Union[int, float] = 10,
        prominence: float = 0.005,
        atom_type="all",
        other=None,
        **kwargs,
    ) -> List[float]:
        """Identifies ads_edges and maxima of position density peaks.
        Peak and edge identification based on the ``scipy.signal`` :func:`find_peaks`
        :param height: Required height of peaks.
        :type height: Union[float, int]
        :param distance: Required minimal horizontal distance (>= 1) in samples between
        neighbouring peaks.
        :type distance: Union[float, int]
        :param width: Required width of peaks in samples.
        :type width: int
        :param wlen: A window length in samples that optionally limits the evaluated area
        for each peak to a subset of the evaluated sequence.
        :type wlen: int
        :return: list of peak edges
        :rtype: List[float]"""
        from ClayAnalysis.peaks import Peaks
        from sklearn.neighbors import KernelDensity

        p = Peaks(self)
        edge_df: pd.DataFrame = self.df.copy()
        if other is None:
            logger.info(
                f'Found atom types {edge_df.index.unique("_atoms").tolist()}'
            )
            edge_df.index = edge_df.index.droplevel(["ions", "atoms"])
            # logger.info(edge_df.groupby(["_atoms", "x"]).count())
            edge_df = edge_df.groupby(["_atoms", "x"]).sum()
            # Take sum for all columns (clay types)
            edge_df = edge_df.aggregate("sum", axis=1)
            if atom_type != "all":
                atom_types = [atom_type]
            else:
                atom_types = edge_df.index.unique(level="_atoms").tolist()
        else:
            logger.info(
                f'Found atom types {edge_df.index.unique("_atoms").tolist()}'
            )
            logger.info(
                f'Found atom types {edge_df.index.unique("other").tolist()}'
            )
            edge_df.index = edge_df.index.droplevel(["ions", "atoms"])
            edge_df = edge_df.groupby(["other", "_atoms", "x"]).sum()

        logger.info(f"Getting peaks edges for {atom_types}")
        for atom_type in atom_types:
            outname = self._get_edge_fname(
                atom_type=atom_type, other=other, name="pe"
            )
            if not outname.is_file():
                if atom_type == "OT":
                    atom_peak_cutoff = peak_cutoff + 5
                else:
                    atom_peak_cutoff = peak_cutoff
                df_slice = edge_df.xs(
                    atom_type, level="_atoms", drop_level=False
                )
                expanded_x = np.expand_dims(
                    df_slice.index.get_level_values("x"), axis=1
                )
                kde: KernelDensity = KernelDensity(
                    kernel="tophat", bandwidth=0.01
                ).fit(X=expanded_x, sample_weight=df_slice.to_numpy())
                score = kde.score_samples(expanded_x)
                score -= np.amin(score, where=~np.isinf(score), initial=0)
                score = np.where(
                    np.logical_or(
                        np.isinf(score),
                        df_slice.index.get_level_values("x")
                        >= atom_peak_cutoff,
                    ),
                    0,
                    score,
                )
                cut = np.argwhere(score != 0)
                plt.plot(
                    df_slice.index.get_level_values("x")[cut],
                    10 * score[cut] / np.sum(score[cut]),
                    linestyle="dotted",
                )
                # score = np.where(df_slice.index.get_level_values('x') > peak_cutoff, 0, score)
                peak_prominence = prominence * score
                # plt.plot(x_vals, score)

                peaks, peak_dict = find_peaks(
                    score,
                    # df_slice.to_numpy(),
                    # height=height,
                    distance=distance,
                    width=width,
                    wlen=wlen,
                    prominence=peak_prominence,
                )
                plt.plot(
                    df_slice.index.get_level_values("x"),
                    df_slice.to_numpy() / np.sum(df_slice.to_numpy()),
                )
                # plt.plot(df_slice.index.get_level_values("x"), df_slice.to_numpy())
                x_vals = df_slice.index.get_level_values("x").to_numpy()
                logger.info(f"Found {len(peaks)} peaks: {x_vals[peaks]}")
                edges = [0]
                for id in range(0, len(peaks) - 1):
                    window = np.s_[peaks[id] : peaks[id + 1]]
                    edge_id = np.argwhere(
                        score[window] == np.min(score[window])
                    )[0]
                    edges.append(x_vals[window.start + edge_id][0])
                    # edge_id = np.argwhere(
                    #     df_slice.values == np.min(df_slice.values[window])
                    # )[0]
                    # ads_edges.append(x_vals[edge_id][0])
                # logger.info(f"ads_edges: {ads_edges}")
                # ads_edges.append(x_vals[peak_dict["right_bases"][-1]])
                # logger.info(f"ads_edges: {ads_edges}")

                # logger.info(f"ads_edges: {ads_edges}")
                try:
                    right_base = peak_dict["right_bases"][-1]
                except IndexError:
                    right_base = None
                # logger.info(f'{right_base}')
                #     print(right_base)
                if len(peaks) <= 1:
                    logger.info("l1")
                    right_edge = right_base
                else:
                    logger.info("ln1")
                    final_slice = np.s_[
                        window.stop : window.stop
                        + np.min([edge_id[0], window.stop - window.start])
                    ]
                    if score[right_base] <= np.min(score[final_slice]):
                        logger.info("a")
                        right_edge = right_base
                    else:
                        right_edge = (
                            window.stop
                            + np.argwhere(
                                score[final_slice]
                                == np.min(score[final_slice])
                            )[-1][0]
                        )
                logger.info(f"{right_base}, {right_edge}")
                if right_edge is not None:
                    edges.append(x_vals[right_edge])  #
                edges.append(x_vals[-1])
                # print(ads_edges)
                plt.scatter(
                    x_vals[peaks],
                    df_slice[peaks] / np.sum(df_slice.to_numpy()),
                    color="red",
                )
                edge_dict = {
                    "ads_edges": edges,
                    "cutoff": self.cutoff,
                    "peak": x_vals[peaks],
                }
                # for peak in edge_dict["peak"]:
                #     plt.axvline(peak, 0, 1, color="green")
                # for edge in edge_dict["ads_edges"]:
                #     plt.axvline(edge, 0, 0.5, color="orange")
                # plt.suptitle(atom_type)
                for p in peaks:
                    #         print(peak)
                    #         print((peak, df_slice[peak_i]))
                    plt.annotate(
                        rf"{np.round(x_vals[p], 1):2.1f} \AA",
                        xy=(
                            x_vals[p],
                            df_slice[p] / np.sum(df_slice.to_numpy()),
                        ),
                        textcoords="offset points",
                        verticalalignment="bottom",
                    )
                logger.info(edge_dict)
                for ei, edge in enumerate(edge_dict["edges"]):
                    plt.axvline(
                        edge, 0, 2, color="orange"
                    )  # , label = f'edge {ei}: {edge:2.1f}')
                    # plt.annotate(fr'{edge:2.1f} \AA', xy=(edge, 0.8))
                plt.suptitle(f"Edges: {atom_type}")
                plt.xlabel(r"z-distance (\AA)")
                plt.ylabel(r"density")
                plt.xticks(np.arange(0, 21, 2))
                # plt.ylim(0, 0.5)
                # plt.show()
                # plt.savefig(
                #     Path(self._get_edge_fname(atom_type=atom_type)).with_suffix(".png")
                # )
                # plt.close()
                with open(outname, "wb") as edge_file:
                    pkl.dump(edge_dict, edge_file)
                logger.info(
                    f"Wrote {atom_type} edges to {outname}."
                )  # plt.show()  # p.get_peaks(atom_type=atom_type)  # edges = self._read_edge_file(atom_type=atom_type, skip=False)  # self._edges[atom_type] = edges  # self._peaks[atom_type] = x_vals[peaks]  #

    def _read_edge_file(self, atom_type: str, skip=True, other=None):
        fname = self._get_edge_fname(atom_type, name="edges", other=other)
        if not fname.exists():
            logger.debug(f"No {atom_type} edge file found.")
            os.makedirs(fname.parent, exist_ok=True)
            logger.info(f"{fname.parent}")
            if skip is True:
                logger.info(f"Continuing without {atom_type} edges")
                p = [self._min, self.cutoff.num]
            else:
                # self._get_edges(atom_type=atom_type)
                raise FileNotFoundError(
                    f"No {atom_type} edge file found {fname!r}."
                )

        else:
            with open(fname, "rb") as edges_file:
                logger.info(f"Reading {atom_type} edges {edges_file.name!r}")
                p = list(
                    map(lambda e: e.round(5), pkl.load(edges_file)["edges"])
                )
        logger.debug(f"edges: {p}")
        return p

    #
    # # def _read_peak_file(self, atom_type):
    # #     fname = self._get_edge_fname(atom_type)
    # #     if not fname.exists():
    # #         logger.info("does not exist")
    # #         os.mkdir(fname.parent)
    # #         from ClayAnalysis.peaks import Peaks
    # #         pks = Peaks(self)
    # #         pks.get_peaks(atom_type=atom_type)
    # #     with open(fname, "rb") as edges_file:
    # #         logger.info(f"Reading peaks {edges_file.name}")
    # #         p =  pkl.load(edges_file)["peaks"]
    # #         print(p)
    # #         return p
    #
    # # @property
    # # def peaks(self):
    # #     if len(self.__peaks) == len(self._atoms):
    # #         pass
    # #     else:
    # #         for atom_type in self._atoms:
    # #             # try:
    # #
    # #             self.__edges[atom_type] = self._read_peak_file(atom_type)
    # #             logger.info(f"Reading peaks")
    # #             # except FileNotFoundError:
    # #             #     logger.info(f"Getting new peaks")
    # #             #     self._get_edges(atom_type)
    # #     return self.__peaks

    @property
    def edges(self):
        if len(self._edges) == len(self._atoms):
            pass
        else:
            for atom_type in self._atoms:
                # try:
                self._edges[atom_type] = self._read_edge_file(
                    atom_type
                )  # logger.info(f"Reading peaks")  # except FileNotFoundError:  #     logger.info(f"Getting new edges")  #     self._get_edges(atom_type)
        return self._edges

    def get_bin_df(self):
        idx = self.df.index.names
        bin_df = self.df.copy()
        atom_types = bin_df.index.get_level_values("_atoms").unique().tolist()
        bin_df.reset_index(["x_bins", "x", "_atoms"], drop=False, inplace=True)
        bin_df = bin_df.convert_dtypes(
            infer_objects=True, convert_integer=True, convert_floating=True
        )
        for col in ["x", "x_bins"]:
            bin_df[col] = bin_df[col].astype(float)
        for atom_type in tqdm(
            atom_types, position=0, desc="atom types", leave=False
        ):
            # logger.info(f"{atom_type}")
            try:
                edges = self._edges[atom_type]
            except KeyError:
                # edge_fname = self._get_edge_fname(atom_type)
                edges = self._read_edge_file(
                    atom_type=atom_type, other=self.other
                )  # if edge_fname.is_file():  #     self._edges[atom_type] = self._read_edge_file(atom_type)  # else:  #     raise  #     self._get_edges(atom_type=atom_type)  # edges = self._edges[atom_type]
            # print(edges, bin_df['x_bins'].where(bin_df['_atoms'] == atom_type))
            bin_df["x_bins"].where(
                bin_df["_atoms"] != atom_type,
                pd.cut(bin_df["x"], [*edges]),
                inplace=True,
            )
        bin_df = bin_df.reset_index(drop=False).set_index(idx)

        # bin_df.set_index(
        #     idx,
        #     inplace=True,
        # )
        self.df = bin_df.copy()

    @property
    def bin_df(self):
        if not self.df.index.get_level_values("x_bins").is_interval():
            logger.info("No Interval")
            self.get_bin_df()
        else:
            logger.info("Interval")
        return self.df

        # area_df = self.df.copy()
        # atom_col = edge_df.loc['atoms']
        # edge_df['atoms'].where(edge_df['atoms'] == 'ions')
        # data_slices = edge_df.groupby(['ions', 'atoms', 'x']).sum()
        # data_slices = data_slices.aggregate('sum', axis=1)
        # ion_slices = data_slices.xs('ions', level='atoms')
        # # other_slices =
        #
        # peaks = find_peaks(data_slices.to_numpy(),
        #                    height=height,
        #                    distance=distance,
        #                    width=width,
        #                    wlen=wlen)
        # check_logger.info(f'Found peaks {peaks[0]}')
        #
        # colours = ['blue', 'orange']
        # fig, ax = plt.subplots(len(data_slices.index.unique('atoms')))
        # y = []
        # fig = plt.figure(figsize=(16, 9))
        # for atom_type in data_slices.index.unique('atoms'):
        #     data_slice = data_slices.xs(atom_type, level='atoms')
        #     plt_slice = data_slice
        #     if atom_type == 'ions':
        #         for ion_type in data_slice.index.unique('ions'):
        #             plt_slice = data_slice.xs(ion_type, level='ions')
        #             y.append((plt_slice.reset_index()['x'].to_numpy(), plt_slice.to_numpy()))
        #     else:
        #         y.append((plt_slice.reset_index()['x'].to_numpy(), plt_slice.to_numpy()))
        #
        # for y_data in y:
        #     # y = plt_slice.to_numpy()
        #     # x = plt_slice.reset_index()['x'].to_numpy()#atom_type)
        #     plt.plot(*y_data)
        #     plt.vlines(data_slice.reset_index()['x'].to_numpy()[peaks[0]], -1, 1, color='red')
        #     plt.xlim(0, 7)

    #
    #     group = data.index.droplevel('x')
    #
    #     # new_idx = pd.MultiIndex.from_product(group = data.index.droplevel('x').get_level_values)
    #
    #     ads_edges = np.array(ads_edges, dtype=np.float32)
    #     if ads_edges[0] != min:
    #         np.insert(ads_edges, 0, min)
    #     if ads_edges[-1] < self.cutoff:
    #         ads_edges.append(self.cutoff)
    #     # intervals = pd.IntervalIndex.from_breaks(ads_edges)
    #
    #     data = data.reset_index(drop=False).set_index(group.names)
    #     print(data.index.names)
    #     print(data.columns)
    #     data['bins'] = pd.cut(data['x'], [min, *ads_edges, self.cutoff])
    #     print(data['bins'].head(5))
    #     data.set_index(['bins'], append=True, inplace=True)
    #     data = data.loc[:, self.clays]
    #     grouped = data.groupby([*group.names, 'bins']).sum()
    #
    #
    #
    #     # data.set_index('bins', append=True, inplace=True)
    #     # data = data.reset_index(level='x').set_index('bins', append=True)
    #
    #
    #     # if type(sel_level) == str:
    #     #     sel_level = [sel_level]
    #     # # group = [g for g in group if g not in sel_level]
    #     # # group.append('area_bins')
    #     # x = data.groupby(by=[*group.names, 'bins']).cumsum()
    #
    #     return grouped
    # # def _get_areas(self, sel, sel_level, ads_edges, min = 0.0):
    # #     idsl = pd.IndexSlice
    # #     data = self.df.xs(sel,
    # #                       level=sel_level,
    # #                       drop_level=False).copy()
    # #     group = data.index.droplevel('x')
    # #
    # #     # new_idx = pd.MultiIndex.from_product(group = data.index.droplevel('x').get_level_values)
    # #
    # #     ads_edges = np.array(ads_edges, dtype=np.float32)
    # #     if ads_edges[0] != min:
    # #         np.insert(ads_edges, 0, min)
    # #     if ads_edges[-1] < self.cutoff:
    # #         ads_edges.append(self.cutoff)
    # #     # intervals = pd.IntervalIndex.from_breaks(ads_edges)
    # #
    # #     data = data.reset_index(drop=False).set_index(group.names)
    # #     print(data.index.names)
    # #     print(data.columns)
    # #     data['bins'] = pd.cut(data['x'], [min, *ads_edges, self.cutoff])
    # #     print(data['bins'].head(5))
    # #     data.set_index(['bins'], append=True, inplace=True)
    # #     data = data.loc[:, self.clays]
    # #     grouped = data.groupby([*group.names, 'bins']).sum()
    # #
    # #
    # #
    # #     # data.set_index('bins', append=True, inplace=True)
    # #     # data = data.reset_index(level='x').set_index('bins', append=True)
    # #
    # #
    # #     # if type(sel_level) == str:
    # #     #     sel_level = [sel_level]
    # #     # # group = [g for g in group if g not in sel_level]
    # #     # # group.append('area_bins')
    # #     # x = data.groupby(by=[*group.names, 'bins']).cumsum()
    # #
    # #     return grouped
    #

    # def _get_bin_label(self, x_bin):
    #     if x_bin.right < np.max(self.x):
    #         label = f'${x_bin.left} - {x_bin.right}$ \AA'
    #         # barwidth = x_bin.right - x_bin.left
    #
    #     else:
    #         label = f'$ > {x_bin.left}$ \AA'
    #     return label

    # def plot_bars(self,
    #               bars: Literal['clays', 'aas', 'ions', 'atoms', 'other'],
    #               x: Literal['clays', 'aas', 'ions', 'atoms', 'other'],
    #               y: Literal['clays', 'aas', 'ions', 'atoms', 'other'],
    #               # y: Literal['clays', 'ions', 'aas', 'atoms', 'other'],
    #               # select: Literal['clays', 'ions', 'aas'],
    #               rowlabel: str = 'y',
    #               columnlabel: str = 'x',
    #               figsize=None,
    #               dpi=None,
    #               # diff=False,
    #               xmax=50,
    #               ymax=50,
    #               save=False,
    #               ylim=None,
    #               odir='.',
    #               barwidth=0.75,
    #               xpad=0.25,
    #               cmap='winter'
    #               ):
    #     """Create stacked Histogram adsorption shell populations.
    #
    #     """
    #     aas_classes = [['arg', 'lys', 'his'],
    #                    ['glu', 'gln'],
    #                    ['cys'],
    #                    ['gly'],
    #                    ['pro'],
    #                    ['ala', 'val', 'ile', 'leu', 'met'],
    #                    ['phe', 'tyr', 'trp'],
    #                    ['ser', 'thr', 'asp', 'gln']]
    #     ions_classes = [['Na', 'Ca'],
    #                     ['Ca', 'Mg']]
    #     atoms_classes = [['ions'],
    #                      ['N'],
    #                      ['OT'],
    #                      ['CA']]
    #     clays_classes = [['NAu-1'],
    #                      ['NAu-2']]
    #     cmaps_seq = ['Purples', 'Blues', 'Greens', 'Oranges', 'Reds']
    #     cmaps_single = ['Dark2']
    #     sel_list = ('clays', 'ions', 'aas', 'atoms')
    #     # for color, attr in zip([''], sel_list):
    #     #     cmaps_dict[attr] = {}
    #     # cm.get_cmap()
    #     cmap_dict = {'clays': []}
    #
    #     title_dict = {'clays': 'Clay type',
    #                   'ions': 'Ion type',
    #                   'aas': 'Amino acid',
    #                   'atoms': 'Atom type',
    #                   'other': 'Other atom type'}
    #
    #     sel_list = ['clays', 'ions', 'aas', 'atoms']
    #
    #     # if self.other != None:
    #     #     sel_list.append('other')
    #
    #     separate = [s for s in sel_list if s not in [x, y, bars]]  # (s != x and s != y and s != bars and s != groups)]
    #
    #     idx = pd.Index([s for s in sel_list if (s != x and s != bars and s not in separate)])
    #
    #
    #     sep = pd.Index(separate)
    #
    #     vx = getattr(self, x)
    #     logger.info(f'x = {x}: {vx}')
    #     lx = len(vx)
    #
    #     vy = getattr(self, y)
    #     logger.info(f'y = {y}: {vy}')
    #     ly = len(vy)
    #
    #
    #     vbars = getattr(self, bars)
    #     lbars = len(vbars)
    #     logger.info(f'bars = {bars}: {vbars}')
    #
    #     bar_dict = dict(zip(vbars, np.arange(lbars)))
    #
    #     yid = np.ravel(np.where(np.array(idx) == y))[0]
    #
    #
    #     # label_key = idx.difference(pd.Index([x, y, *separate]), sort=False).values[0]
    #
    #     # label_id = idx.get_loc(key=label_key)
    #     n_plots = len(sep)
    #
    #     x_dict = dict(zip(vx, np.arange(lx)))
    #
    #
    #     sel = self.clays
    #
    #     # get data for plotting
    #     plot_df = self.bin_df[sel].copy()
    #
    #     # move clays category from columns to index
    #     idx_names = ['clays', *plot_df.index.droplevel(['x', '_atoms']).names]
    #     # DataFrame -> Series
    #     plot_df = plot_df.stack()
    #
    #     # get values for atom types (including separate ions)
    #     atoms = plot_df.index.get_level_values('_atoms')
    #
    #     # make new DataFrame from atom_type index level and values
    #     plot_df.index = plot_df.index.droplevel(['x', '_atoms'])
    #     plot_df = pd.DataFrame({'values': plot_df,
    #                             '_atoms': atoms})
    #
    #     # list of matplotlib sequential cmaps
    #     cmaps = [
    #         'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
    #         'hot', 'afmhot', 'gist_heat', 'copper']
    #
    #     # map unique atom types to colour map
    #     atom_types = atoms.unique()
    #     colour_dict = dict(zip(atom_types, cmaps[:len(atom_types)]))
    #     plot_df['_atoms'] = plot_df['_atoms'].transform(lambda x: colour_dict[x])
    #
    #     # reorder index for grouping
    #     plot_df = plot_df.reorder_levels(idx_names)
    #
    #     # group and sum densities within adsorption shell bins
    #     plot_df = plot_df.groupby(
    #         plot_df.index.names).agg(values=pd.NamedAgg('values', 'sum'),
    #                                  colours=pd.NamedAgg('_atoms', 'first')
    #                                  )
    #
    #     # separate colour column from plot_df -> yields 2 Series
    #     colour_df = plot_df['colours']
    #     plot_df = plot_df['values']
    #
    #     # add missing atom probabilities from bulk to the largest bin
    #     # (bin.left, cutoff] -> (bin.left, all bulk])
    #     inner_sum = plot_df.groupby(plot_df.index.droplevel('x_bins').names).sum()
    #     extra = 1 - inner_sum
    #     plot_df.where(np.rint(plot_df.index.get_level_values('x_bins').right
    #                           ) != int(self.cutoff),
    #                   lambda x: x + extra,
    #                   inplace=True
    #                   )
    #
    #     # determine largest shell bin limit
    #     max_edge = list(map(lambda x: np.max(x[:-1]), self.ads_edges.values()))
    #     max_edge = np.max(max_edge)
    #
    #     # normalise colour map from 0 to max_edge
    #     cnorm = mpc.Normalize(vmin=0, vmax=max_edge, clip=False)
    #
    #     # set figure size
    #     if figsize == None:
    #         figsize = tuple([5 * lx if (10 * lx) < xmax else xmax,
    #                          5 * ly if (5 * ly) < ymax else ymax])
    #
    #     # set resultion
    #     if dpi == None:
    #         dpi = 100
    #
    #     # get plotting iter from index
    #     iters = np.array(np.meshgrid(*[getattr(self, idxit) for idxit in idx])
    #                      ).T.reshape(-1, len(idx))
    #
    #     logger.info(f'Printing bar plots for {sep}\nColumns: {vx}\nRows: {vy}')
    #
    #     # set label modifier function
    #     label_mod = lambda l: ', '.join([li.upper() if namei == 'aas'
    #                                      else li for li, namei in l])
    #
    #
    #     try:
    #         # iterator for more than one plot
    #         sep_it = np.array(np.meshgrid(*[getattr(self, idxit) for idxit in sep])
    #                           ).T.reshape(-1, len(sep))
    #     except ValueError:
    #         # only one plot
    #         sep_it = [None]
    #
    #     # iterate over separate plots
    #     for pl in sep_it:
    #         # index map for y values
    #         y_dict: dict = dict(zip(vy, np.arange(ly)))
    #         print(y_dict)
    #
    #         # initialise legends
    #         legends_list: list = [(a, b) for a in range(ly) for b in range(lx)]
    #         legends: dict = dict(zip(legends_list, [[] for a in range(len(legends_list))]))
    #
    #         # generate figure and axes array
    #         fig, ax = plt.subplots(nrows=ly,
    #                                ncols=lx,
    #                                figsize=figsize,
    #                                sharey=True,
    #                                dpi=dpi,
    #                                constrained_layout=True,
    #                                # sharex=True
    #                                )
    #         # only one plot
    #         if pl is None:
    #             logger.info(f'Generating plot')
    #             sepview = plot_df.view()
    #             plsave = ''
    #
    #         # multiple plots
    #         else:
    #             logger.info(f'Generating {pl} plot')
    #             sepview = plot_df.xs((pl),
    #                                  level=separate,
    #                                  axis=0,
    #                                  drop_level=False)
    #             plsave = pl
    #             fig.suptitle((', '.join([title_dict[s].upper() for s in separate]) +
    #                           f': {label_mod(list(tuple(zip(pl, separate))))}'), size=16,
    #                          weight='bold')
    #
    #         # set plot index
    #         pi = 0
    #
    #         #iterate over subplot columns
    #         for col in vx:
    #             logger.info(col)
    #             try:
    #
    #                 view = sepview.xs(col,
    #                                   level=x,
    #                                   axis=0,
    #                                   drop_level=False)
    #
    #                 pi = 1
    #             except ValueError:
    #                 view = sepview
    #                 col = vx
    #                 pi += 1
    #
    #             table_text = []
    #             table_col = []
    #             table_cmap = None
    #             table_rows = []
    #             for it in iters:
    #                 try:
    #
    #
    #                     values = view.xs(tuple(it),
    #                                      level=idx.tolist(),
    #                                      drop_level=False)
    #
    #                     # x_grouplabels = []
    #                     x_labels = []
    #                     x_ticks = []
    #
    #
    #
    #
    #
    #                     for vbar in vbars:
    #                         # bulk_pad = 2
    #                         # bulk_edge = np.rint((max_edge + bulk_pad))
    #                         x_ticks.append(bar_dict[vbar] * (barwidth + xpad))
    #                         x_labels.append(vbar)
    #
    #
    #                         bottom = 0.0
    #
    #
    #                         # x_tick = bar_dict[vbar] * (barwidth + xpad)
    #
    #
    #                         bar_vals = values.xs(vbar,
    #                                              level=bars,
    #                                              drop_level=False)
    #
    #                         cmap = colormaps[colour_df.loc[bar_vals.index].values[0]]
    #                         if table_cmap is None:
    #                             table_cmap = cmap
    #
    #                         if len(self.__peaks) != len(self.df.index.get_level_values('_atoms').unique()):
    #                             self._get_edges()
    #                         try:
    #                             peaks = self.__peaks[bar_vals.index.get_level_values('atoms').unique().tolist()[0]]
    #                         except:
    #                             peaks = self.__peaks[bar_vals.index.get_level_values('ions').unique().tolist()[0]]
    #
    #                         #       bar_vals.values >= 0)
    #                         if np.all(bar_vals.values) >= 0:
    #                             table_text.append([f'${v * 100:3.1f} %$' for v in bar_vals.values])
    #
    #
    #                             x_id, y_id = x_dict[col], y_dict[it[yid]]
    #
    #                             bar_val_view = bar_vals
    #                             bar_val_view.index = bar_val_view.index.get_level_values('x_bins')
    #
    #
    #                             x_tick = x_ticks[-1]
    #                             # x_ticks.append(x_tick)
    #                             # bar_plots = []
    #                             for bar_id, bar_val in enumerate(bar_val_view.items()):
    #
    #
    #                                 x_bin, y_val = bar_val
    #
    #                                 try:
    #                                     peak = peaks[bar_id]
    #                                 except IndexError:
    #                                     peak = x_bin.right
    #                                 colour = cmap(cnorm(peak))
    #                                 if colour not in table_col and cmap == table_cmap:
    #                                     print('colour', colour)
    #                                     table_col.append(colour)
    #                                 if x_bin.right < np.max(self.x):
    #                                     label = f'${x_bin.left} - {x_bin.right}$ \AA'
    #                                     # barwidth = x_bin.right - x_bin.left
    #                                 else:
    #                                     label = f'$ > {x_bin.left}$ \AA'
    #                                 if label not in table_rows and cmap == table_cmap:
    #                                     table_rows.append(label)
    #                                 if y_val >= 0.010:
    #
    #                                         # barwidth = bulk_edge - x_bin.left
    #                                     # try:
    #                                     # x_tick = x_ticks[-1] + barwidth
    #                                     # x_ticks.append(x_tick)
    #                                     # except IndexError:
    #                                     #     x_tick = x_bin.left
    #
    #
    #
    #
    #                                     try:
    #                                         p = ax[y_id, x_id].bar(x_tick,
    #                                                                y_val,
    #                                                                label=label,
    #                                                                bottom=bottom,
    #                                                                width=barwidth,
    #                                                                align='edge',
    #                                                                color=colour
    #                                                                )
    #                                         ax[y_id, x_id].bar_label(p, labels=[label],
    #                                                                  fmt='%s',
    #                                                                  label_type='center')
    #                                     except IndexError:
    #                                         p = ax[y_id].bar(x_tick,
    #                                                          y_val,
    #                                                          label=label,
    #                                                          bottom=bottom,
    #                                                          width=barwidth,
    #                                                          align='edge',
    #                                                          color=colour
    #                                                          )
    #                                         ax[y_id, x_id].bar_label(p, labels=[label],
    #                                                                  fmt='%s',
    #                                                                  label_type='center')
    #                                     # finally:
    #                                     bottom += y_val
    #                                 print(table_text, table_col, table_rows)
    #                                 table = ax[y_id, x_id].table(cellText=table_text,
    #                                                              rowColours=table_col,
    #                                                              rowLabels=table_rows,
    #                                                              # colLables=...,
    #                                                              loc='bottom')
    #                             # x_ticks = x_ticks[:-1]
    #
    #                             # x_ticks.append(bar_dict[vbar] * (barwidth + xpad))
    #
    #
    #
    #                 #                 values = values
    #                 #                 print('try 1 done')
    #                 #
    #                 #                     # for shell in values:
    #                 #                     # view for group and bars
    #                 #                             label = f'${lims.left} - {lims.right}$ \AA'
    #                 #
    #                 #                     try:
    #                 #                         print('try 2')
    #                 #                         print(x_dict[col], y_dict[it[yid]])
    #
    #                 #                     except:
    #                 # #                         raise ValueError
    #                 #                         x_id, y_id = 0, y_dict[it[yid]]
    #                 #                         label = f'${x_bin.left} - {x_bin.right}$ \AA'
    #
    #                 #                     if pi == 1:
    #                 #                         legends[y_id, x_id].append(it[label_id])
    #                 #                 else:
    #                 #                     check_logger.info('NaN values')
    #
    #                 except KeyError:
    #                     logger.info(f'No data for {pl}, {vx}, {it}')
    #
    #         # x_ticks = [np.linspace(n_bar * bulk_edge + xpad,
    #         #                        n_bar * bulk_edge + bulk_edge, int(bulk_edge)) for n_bar in range(lbars)]
    #         # x_ticks = np.ravel(x_ticks)
    #         # x_labels = np.tile(np.arange(0, bulk_edge, 1), lbars)
    #
    #         for i in range(ly):
    #             try:
    #                 ax[i, 0].set_ylabel(f'{label_mod([(vy[i], y)])}\n' + rowlabel)
    #                 ax[i, 0].set_yticks(np.arange(0.0, 1.1, 0.2))
    #                 for j in range(lx):
    #                     ax[i, j].spines[['top', 'right']].set_visible(False)
    #                     ax[i, j].hlines(1.0, -xpad, lbars * (barwidth + xpad) + xpad, linestyle='--')
    #                     # ax[i, j].legend(ncol=2, loc='lower center')#[leg for leg in legends[i, j]], ncol=3)
    #                     #                  if xlim != None:
    #                     ax[i, j].set_xlim((-xpad, lbars * (barwidth + xpad)))
    #                     ax[i, j].set_xticks([], [])
    #                     #                 if ylim != None:
    #                     ax[i, j].set_ylim((0.0, 1.25))
    #
    #                     ax[ly - 1, j].set_xticks(np.array(x_ticks) + 0.5 * barwidth, x_labels)
    #                     ax[ly - 1, j].set_xlabel(bars + f'\n{label_mod([(vx[j], x)])}')
    #             except IndexError:
    #                 ...
    #     #             ax[i].set_ylabel(f'{label_mod([(vy[i], y)])}\n' + rowlabel)
    #     #             ax[i].legend([label_mod([(leg, label_key)]) for leg in legends[i, 0]], ncol=3)
    #     #             ax[ly - 1].set_xlabel(columnlabel + f'\n{label_mod([(vx[0], x)])}')
    #     # #
    #         fig.supxlabel(f'{title_dict[x]}s', size=14)
    #         fig.supylabel(f'{title_dict[y]}s', size=14)
    #     # #     if save != False:
    #     # #         odir = Path(odir)
    #     # #         if not odir.is_dir():
    #     # #             os.makedirs(odir)
    #     # #         if type(save) == str:
    #     # #             fig.savefig(odir / f'{save}.png')
    #     # #         else:
    #     # #             fig.savefig(odir / f'{self.name}_{diffstr}_{x}_{y}_{plsave}_{self.cutoff}_{self.bins}.png')
    #     # #         fig.clear()
    #         fig.show()
    #     return fig

    @cached_property
    def binned_plot_colour_dfs_1d(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info(f"Getting binned plot and colour dfs")
        sel = self.clays

        # get data for plotting
        plot_df = self.bin_df[sel].copy()

        # move clays category from columns to index
        idx_names = ["clays", *plot_df.index.droplevel(["x"]).names]
        # DataFrame -> Series
        plot_df = plot_df.stack()

        # get values for atom types (including separate ions)
        atoms = plot_df.index.get_level_values("_atoms").to_numpy()

        # make new DataFrame from atom_type index level and values
        plot_df.index = plot_df.index.droplevel(["x"])
        # plot_df.index.names = [name.strip('_') for name in plot_df.index.names]

        plot_df = pd.DataFrame({"values": plot_df, "colours": atoms})

        # list of matplotlib sequential cmaps
        cmaps = [
            "spring",
            "summer",
            "autumn",
            "winter",
            "cool",
            "Wistia",
            "hot",
            "afmhot",
            "gist_heat",
            "copper",
        ]

        # map unique atom types to colour map
        atom_types = atoms.unique()
        colour_dict = dict(zip(atom_types, cmaps[: len(atom_types)]))
        plot_df["colours"] = plot_df["colours"].transform(
            lambda x: colour_dict[x]
        )

        # reorder index for grouping
        plot_df = plot_df.reorder_levels(idx_names)

        # group and sum densities within adsorption shell bins
        plot_df = plot_df.groupby(plot_df.index.names).agg(
            values=pd.NamedAgg("values", "sum"),
            colours=pd.NamedAgg("colours", "first"),
        )

        # separate colour column from plot_df -> yields 2 Series
        colour_df = plot_df["colours"]
        plot_df = plot_df["values"]

        # add missing atom probabilities from bulk to the largest bin
        # (bin.left, cutoff] -> (bin.left, all bulk])
        inner_sum = plot_df.groupby(
            plot_df.index.droplevel("x_bins").names
        ).sum()
        extra = 1 - inner_sum
        plot_df.where(
            np.rint(plot_df.index.get_level_values("x_bins").right)
            != int(self.cutoff),
            lambda x: x + extra,
            inplace=True,
        )
        return plot_df, colour_df

    @cached_property
    def binned_df(self):
        return self.binned_plot_colour_dfs_1d[0]

    @cached_property
    def colour_df(self):
        return self.binned_plot_colour_dfs_1d[1]

    @property
    def max_shell_edge(self) -> float:
        # determine largest shell bin limit
        max_edge = list(map(lambda x: np.max(x[:-1]), self.edges.values()))
        max_edge = np.max(max_edge)
        return max_edge

    def plot_columns(self, sepview, col, x, vx, pi):
        # logger.info(col)
        try:
            view = sepview.xs(col, level=x, axis=0, drop_level=False)
            pi = 1
        except ValueError:
            view = sepview
            col = vx
            pi += 1
        return view, col, pi

    def get_bar_peaks(self, atom_type, other=None):
        # if len(self.__peaks) != len(self.df.index.get_level_values("_atoms").unique()):
        peaks = self._read_edge_file(
            atom_type=atom_type, other=other
        )  # ['peaks']
        return peaks
        # print(peaks)
        # logger.info(f"Found peaks {peaks}")
        # try:
        #     print(peaks)
        #     print(bar_vals.index.get_level_values('atoms'))
        #     bar_peaks = peaks[
        #         bar_vals.index.get_level_values("atoms").unique().tolist()#[0]
        #     ]
        #     print(bar_peaks)
        #     bar_peaks=bar_peaks[0]
        # except:
        #     bar_peaks = peaks[
        #         bar_vals.index.get_level_values("ions").unique().tolist()#[0]
        #     ]
        #     print(bar_peaks)
        #     bar_peaks = bar_peaks[0]
        # return bar_peaks

    # def plot_bars(self,
    #               bars: Literal['clays', 'aas', 'ions', 'other'],
    #               x: Literal['clays', 'aas', 'ions', 'other'],
    #               y: Literal['clays', 'aas', 'ions', 'other'],
    #               rowlabel: str = 'y',
    #               columnlabel: str = 'x',
    #               figsize=None,
    #               dpi=None,
    #               # diff=False,
    #               xmax=50,
    #               ymax=50,
    #               save=False,
    #               ylim=None,
    #               odir='.',
    #               barwidth=0.75,
    #               xpad=0.25,
    #               cmap='winter'
    #               ):
    #     """Create stacked Histogram adsorption shell populations.
    #
    #     """
    #     aas_classes = [['arg', 'lys', 'his'],
    #                    ['glu', 'gln'],
    #                    ['cys'],
    #                    ['gly'],
    #                    ['pro'],
    #                    ['ala', 'val', 'ile', 'leu', 'met'],
    #                    ['phe', 'tyr', 'trp'],
    #                    ['ser', 'thr', 'asp', 'gln']]
    #     ions_classes = [['Na', 'Ca'],
    #                     ['Ca', 'Mg']]
    #     atoms_classes = [['ions'],
    #                      ['N'],
    #                      ['OT'],
    #                      ['CA']]
    #     clays_classes = [['NAu-1'],
    #                      ['NAu-2']]
    #     cmaps_seq = ['Purples', 'Blues', 'Greens', 'Oranges', 'Reds']
    #     cmaps_single = ['Dark2']
    #     sel_list = ('clays', 'ions', 'aas', 'atoms')
    #     # for color, attr in zip([''], sel_list):
    #     #     cmaps_dict[attr] = {}
    #     # cm.get_cmap()
    #     cmap_dict = {'clays': []}
    #
    #     title_dict = {'clays': 'Clay type',
    #                   'ions': 'Ion type',
    #                   'aas': 'Amino acid',
    #                   '_atoms': 'Atom type',
    #                   'other': 'Other atom type'}
    #
    #     sel_list = ['clays', 'ions', 'aas', '_atoms']
    #
    #     if self.other != None:
    #         sel_list.append('other')
    #
    #     assert x in sel_list and x != '_atoms'
    #     assert y in sel_list and y != '_atoms'
    #     assert bars in sel_list and bars != '_atoms'
    #
    #     plot_df, colour_df = self._get_binned_plot_df_1d()
    #
    #     cnorm = self._get_cnorm()
    #
    #    # get data for plotting
    #
    #     bins = 'x_bins'
    #     group = '_atoms'
    #
    #     separate = [s for s in plot_df.index.names if s not in [x, y, bars, bins]]  # (s != x and s != y and s != bars and s != groups)]
    #     logger.info(f'Separate plots: {separate}')
    #     idx = pd.Index([s for s in plot_df.index.names if (s != x and s != bars and s not in [*separate, bins])])
    #     logger.info(f'Iteration index: {idx}')
    #
    #     sep = pd.Index(separate)
    #
    #     vx = getattr(self, x)
    #     logger.info(f'x = {x}: {vx}')
    #     lx = len(vx)
    #
    #     vy = getattr(self, y)
    #     logger.info(f'y = {y}: {vy}')
    #     ly = len(vy)
    #
    #
    #     vbars = getattr(self, bars)
    #     lbars = len(vbars)
    #     logger.info(f'bars = {bars}: {vbars}')
    #
    #     bar_dict = dict(zip(vbars, np.arange(lbars)))
    #
    #     yid = np.ravel(np.where(np.array(idx) == y))[0]
    #
    #     sys.exit(2)
    #
    #     # label_key = idx.difference(pd.Index([x, y, *separate]), sort=False).values[0]
    #
    #     # label_id = idx.get_loc(key=label_key)
    #     n_plots = len(sep)
    #
    #     x_dict = dict(zip(vx, np.arange(lx)))
    #
    #
    #     sel = self.clays
    #
    #     # set figure size
    #     if figsize == None:
    #         figsize = self.get_figsize(lx=lx,
    #                                    ly=ly,
    #                                    xmax=xmax,
    #                                    ymax=ymax)
    #
    #     # set resultion
    #     if dpi == None:
    #         dpi = 100
    #
    #     # get plotting iter from index
    #     iters = self._get_idx_iter(idx=idx)
    #
    #     logger.info(f'Printing bar plots for {sep}\nColumns: {vx}\nRows: {vy}')
    #
    #     # set label modifier function
    #     label_mod = self.modify_plot_labels
    #
    #     try:
    #         # iterator for more than one plot
    #         sep_it = self._get_idx_iter(idx=sep)
    #     except ValueError:
    #         # only one plot
    #         sep_it = [None]
    #
    #     # iterate over separate plots
    #     for pl in sep_it:
    #         # index map for y values
    #         y_dict: dict = dict(zip(vy, np.arange(ly)))
    #         print(y_dict)
    #
    #         legends = self.init_legend(ly=ly,
    #                                    lx=lx)
    #         print(legends)
    #
    #         # generate figure and axes array
    #         fig, ax = plt.subplots(nrows=ly,
    #                                ncols=lx,
    #                                figsize=figsize,
    #                                sharey=True,
    #                                dpi=dpi,
    #                                constrained_layout=True,
    #                                # sharex=True
    #                                )
    #         # only one plot
    #         if pl is None:
    #             logger.info(f'Generating plot')
    #             sepview = plot_df.view()
    #             plsave = ''
    #
    #         # multiple plots
    #         else:
    #             logger.info(f'Generating {pl} plot')
    #             print(plot_df.head(20),'\n', separate, pl)
    #             sepview = plot_df.xs((pl),
    #                                  level=separate,
    #                                  drop_level=False)
    #             plsave = pl
    #             print(pl)
    #             print(separate)
    #             fig.suptitle((', '.join([title_dict[s].upper() for s in separate]) +
    #                           f': {label_mod(list(tuple(zip(pl, separate))))}'), size=16,
    #                          weight='bold')
    #
    #         # set plot index
    #         pi = 0
    #
    #         #iterate over subplot columns
    #         for col in vx:
    #             # logger.info(col)
    #             # view, col, pi = self.plot_columns(sepview=sepview,
    #             #                                   col=col,
    #             #                                   x=x,
    #             #                                   vx=vx,
    #             #                                   pi=pi)
    #             logger.info(col)
    #             try:
    #                 view = sepview.xs(col,
    #                                   level=x,
    #                                   axis=0,
    #                                   drop_level=False)
    #                 pi = 1
    #             except ValueError:
    #                 view = sepview
    #                 col = vx
    #                 pi += 1
    #
    #             table_text = []
    #             table_col = []
    #             table_cmap = None
    #             table_rows = []
    #             for it in iters:
    #                 try:
    #                     values = view.xs(tuple(it),
    #                                      level=idx.tolist(),
    #                                      drop_level=False)
    #
    #                     x_labels = []
    #                     x_ticks = []
    #                     for vbar in vbars:
    #                         print(vbar)
    #                         x_ticks.append(bar_dict[vbar] * (barwidth + xpad))
    #                         x_labels.append(vbar)
    #                         bottom = 0.0
    #                         bar_vals = values.xs(vbar,
    #                                              level=bars,
    #                                              drop_level=False)
    #
    #                         cmap = colormaps[colour_df.loc[bar_vals.index].values[0]]
    #                         if table_cmap is None:
    #                             table_cmap = cmap
    #
    #                         peaks = self.get_bar_peaks(bar_vals=bar_vals)
    #
    #                         if np.all(bar_vals.values) >= 0:
    #                             table_text.append([f'${v * 100:3.1f} %$' for v in bar_vals.values])
    #                             x_id, y_id = x_dict[col], y_dict[it[yid]]
    #
    #                             bar_val_view = bar_vals
    #                             bar_val_view.index = bar_val_view.index.get_level_values('x_bins')
    #
    #                             x_tick = x_ticks[-1]
    #
    #                             for bar_id, bar_val in enumerate(bar_val_view.items()):
    #
    #                                 x_bin, y_val = bar_val
    #
    #                                 try:
    #                                     peak = peaks[bar_id]
    #                                 except IndexError:
    #                                     peak = x_bin.right
    #                                 colour = cmap(cnorm(peak))
    #                                 if colour not in table_col and cmap == table_cmap:
    #                                     print('colour', colour)
    #                                     table_col.append(colour)
    #
    #                                 label = self._get_bin_label(x_bin)
    #
    #                                 # if x_bin.right < np.max(self.x):
    #                                 #     label = f'${x_bin.left} - {x_bin.right}$ \AA'
    #                                 # else:
    #                                 #     label = f'$ > {x_bin.left}$ \AA'
    #                                 if label not in table_rows and cmap == table_cmap:
    #                                     table_rows.append(label)
    #                                 if y_val >= 0.010:
    #
    #                                         # barwidth = bulk_edge - x_bin.left
    #                                     # try:
    #                                     # x_tick = x_ticks[-1] + barwidth
    #                                     # x_ticks.append(x_tick)
    #                                     # except IndexError:
    #                                     #     x_tick = x_bin.left
    #
    #
    #
    #
    #                                     try:
    #                                         p = ax[y_id, x_id].bar(x_tick,
    #                                                                y_val,
    #                                                                label=label,
    #                                                                bottom=bottom,
    #                                                                width=barwidth,
    #                                                                align='edge',
    #                                                                color=colour
    #                                                                )
    #                                         ax[y_id, x_id].bar_label(p, labels=[label],
    #                                                                  fmt='%s',
    #                                                                  label_type='center')
    #                                     except IndexError:
    #                                         p = ax[y_id].bar(x_tick,
    #                                                          y_val,
    #                                                          label=label,
    #                                                          bottom=bottom,
    #                                                          width=barwidth,
    #                                                          align='edge',
    #                                                          color=colour
    #                                                          )
    #                                         ax[y_id, x_id].bar_label(p, labels=[label],
    #                                                                  fmt='%s',
    #                                                                  label_type='center')
    #                                     # finally:
    #                                     bottom += y_val
    #                                 print(table_text, table_col, table_rows)
    #                                 # table = ax[y_id, x_id].table(cellText=table_text,
    #                                 #                              rowColours=table_col,
    #                                 #                              rowLabels=table_rows,
    #                                 #                              # colLables=...,
    #                                 #                              loc='bottom')
    #                             # x_ticks = x_ticks[:-1]
    #
    #                             # x_ticks.append(bar_dict[vbar] * (barwidth + xpad))
    #
    #
    #
    #                 #                 values = values
    #                 #                 print('try 1 done')
    #                 #
    #                 #                     # for shell in values:
    #                 #                     # view for group and bars
    #                 #                             label = f'${lims.left} - {lims.right}$ \AA'
    #                 #
    #                 #                     try:
    #                 #                         print('try 2')
    #                 #                         print(x_dict[col], y_dict[it[yid]])
    #
    #                 #                     except:
    #                 # #                         raise ValueError
    #                 #                         x_id, y_id = 0, y_dict[it[yid]]
    #                 #                         label = f'${x_bin.left} - {x_bin.right}$ \AA'
    #
    #                 #                     if pi == 1:
    #                 #                         legends[y_id, x_id].append(it[label_id])
    #                 #                 else:
    #                 #                     check_logger.info('NaN values')
    #
    #                 except KeyError:
    #                     logger.info(f'No data for {pl}, {vx}, {it}')
    #
    #         # x_ticks = [np.linspace(n_bar * bulk_edge + xpad,
    #         #                        n_bar * bulk_edge + bulk_edge, int(bulk_edge)) for n_bar in range(lbars)]
    #         # x_ticks = np.ravel(x_ticks)
    #         # x_labels = np.tile(np.arange(0, bulk_edge, 1), lbars)
    #
    #         for i in range(ly):
    #             try:
    #                 ax[i, 0].set_ylabel(f'{label_mod([(vy[i], y)])}\n' + rowlabel)
    #                 ax[i, 0].set_yticks(np.arange(0.0, 1.1, 0.2))
    #                 for j in range(lx):
    #                     ax[i, j].spines[['top', 'right']].set_visible(False)
    #                     ax[i, j].hlines(1.0, -xpad, lbars * (barwidth + xpad) + xpad, linestyle='--')
    #                     # ax[i, j].legend(ncol=2, loc='lower center')#[leg for leg in legends[i, j]], ncol=3)
    #                     #                  if xlim != None:
    #                     ax[i, j].set_xlim((-xpad, lbars * (barwidth + xpad)))
    #                     ax[i, j].set_xticks([], [])
    #                     #                 if ylim != None:
    #                     ax[i, j].set_ylim((0.0, 1.25))
    #
    #                     ax[ly - 1, j].set_xticks(np.array(x_ticks) + 0.5 * barwidth, x_labels)
    #                     ax[ly - 1, j].set_xlabel(bars + f'\n{label_mod([(vx[j], x)])}')
    #             except IndexError:
    #                 ...
    #     #             ax[i].set_ylabel(f'{label_mod([(vy[i], y)])}\n' + rowlabel)
    #     #             ax[i].legend([label_mod([(leg, label_key)]) for leg in legends[i, 0]], ncol=3)
    #     #             ax[ly - 1].set_xlabel(columnlabel + f'\n{label_mod([(vx[0], x)])}')
    #     # #
    #         fig.supxlabel(f'{title_dict[x]}s', size=14)
    #         fig.supylabel(f'{title_dict[y]}s', size=14)
    #     # #     if save != False:
    #     # #         odir = Path(odir)
    #     # #         if not odir.is_dir():
    #     # #             os.makedirs(odir)
    #     # #         if type(save) == str:
    #     # #             fig.savefig(odir / f'{save}.png')
    #     # #         else:
    #     # #             fig.savefig(odir / f'{self.name}_{diffstr}_{x}_{y}_{plsave}_{self.cutoff}_{self.bins}.png')
    #     # #         fig.clear()
    #         fig.show()
    #     return fig

    def plot_bars_shifted(
        self,
        bars: Literal["clays", "aas", "ions", "atoms", "other"],
        x: Literal["clays", "aas", "ions", "atoms", "other"],
        y: Literal["clays", "aas", "ions", "atoms", "other"],
        # y: Literal['clays', 'ions', 'aas', 'atoms', 'other'],
        # select: Literal['clays', 'ions', 'aas'],
        rowlabel: str = "y",
        columnlabel: str = "x",
        figsize=None,
        dpi=None,
        # diff=False,
        xmax=50,
        ymax=50,
        save=False,
        ylim=None,
        odir=".",
        # barwidth = 0.75,
        xpad=0.25,
        cmap="winter",
    ):
        """Create stacked Histogram adsorption shell populations."""
        aas_classes = [
            ["arg", "lys", "his"],
            ["glu", "gln"],
            ["cys"],
            ["gly"],
            ["pro"],
            ["ala", "val", "ile", "leu", "met"],
            ["phe", "tyr", "trp"],
            ["ser", "thr", "asp", "gln"],
        ]
        ions_classes = [["Na", "Ca"], ["Ca", "Mg"]]
        atoms_classes = [["ions"], ["N"], ["OT"], ["CA"]]
        clays_classes = [["NAu-1"], ["NAu-2"]]
        cmaps_seq = ["Purples", "Blues", "Greens", "Oranges", "Reds"]
        cmaps_single = ["Dark2"]
        sel_list = ("clays", "ions", "aas", "atoms")
        # for color, attr in zip([''], sel_list):
        #     cmaps_dict[attr] = {}
        # cm.get_cmap()
        cmap_dict = {"clays": []}

        title_dict = {
            "clays": "Clay type",
            "ions": "Ion type",
            "aas": "Amino acid",
            "atoms": "Atom type",
            "other": "Other atom type",
        }

        sel_list = ["clays", "ions", "aas", "atoms"]

        # if self.other != None:
        #     sel_list.append('other')
        cmap = colormaps[cmap]
        separate = [
            s for s in sel_list if s not in [x, y, bars]
        ]  # (s != x and s != y and s != bars and s != groups)]
        # print(separate)
        idx = pd.Index(
            [
                s
                for s in sel_list
                if (s != x and s != bars and s not in separate)
            ]
        )
        # print(idx)

        sep = pd.Index(separate)

        vx = getattr(self, x)
        logger.info(f"x = {x}: {vx}")
        lx = len(vx)

        vy = getattr(self, y)
        logger.info(f"y = {y}: {vy}")
        ly = len(vy)
        # print(ly)

        vbars = getattr(self, bars)
        lbars = len(vbars)
        logger.info(f"bars = {bars}: {vbars}")

        bar_dict = dict(zip(vbars, np.arange(lbars)))

        yid = np.ravel(np.where(np.array(idx) == y))[0]
        # print(yid)

        # label_key = idx.difference(pd.Index([x, y, *separate]), sort=False).values[0]

        # label_id = idx.get_loc(key=label_key)
        n_plots = len(sep)

        x_dict = dict(zip(vx, np.arange(lx)))
        # print(x_dict)

        sel = self.clays

        plot_df = self.bin_df[sel].copy()
        idx_names = ["clays", *plot_df.index.droplevel(["x", "_atoms"]).names]
        # print("idx", idx_names)
        plot_df = plot_df.stack()
        # atoms = plot_df.index.get_level_values('_atoms')
        plot_df.index = plot_df.index.droplevel(["x", "_atoms"])
        # plot_df = pd.DataFrame({'values': plot_df,
        #                         '_atoms': atoms})
        # idx_names.remove('_atoms')
        plot_df = plot_df.reorder_levels(idx_names)
        #
        plot_df.name = "values"
        # print(plot_df.head(3))
        plot_df = plot_df.groupby(plot_df.index.names).sum()
        # print(plot_df.head(3))
        inner_sum = plot_df.groupby(
            plot_df.index.droplevel("x_bins").names
        ).sum()
        extra = 1 - inner_sum
        plot_df.where(
            np.rint(plot_df.index.get_level_values("x_bins").right)
            != int(self.cutoff),
            lambda x: x + extra,
            inplace=True,
        )
        # print(type(self.ads_edges))
        max_edge = list(map(lambda x: np.max(x[:-1]), self.edges.values()))
        max_edge = np.max(max_edge)
        # print("max edge", max_edge)
        # max_edge = np.ravel(np.array(*self.ads_edges.values()))
        # print(max_edge)
        cnorm = mpc.Normalize(vmin=0, vmax=max_edge, clip=False)

        if figsize == None:
            figsize = tuple(
                [
                    5 * lx if (10 * lx) < xmax else xmax,
                    5 * ly if (5 * ly) < ymax else ymax,
                ]
            )
        #
        if dpi == None:
            dpi = 100
        #
        iters = np.array(
            np.meshgrid(*[getattr(self, idxit) for idxit in idx])
        ).T.reshape(-1, len(idx))
        #
        logger.info(f"Printing bar plots for {sep}\nColumns: {vx}\nRows: {vy}")
        #
        label_mod = lambda l: ", ".join(
            [li.upper() if namei == "aas" else li for li, namei in l]
        )
        #
        try:
            sep_it = np.array(
                np.meshgrid(*[getattr(self, idxit) for idxit in sep])
            ).T.reshape(-1, len(sep))
        except ValueError:
            sep_it = [None]
        # check_logger.info(vx, vy, lx, ly)
        #
        for pl in sep_it:
            # print(pl)
            # try:
            #     fig.clear()
            # except:
            #     pass
            y_dict = dict(zip(vy, np.arange(ly)))
            # print(y_dict)
            #     if separate == 'atoms' and pl != '':
            #         ...
            #
            legends_list = [(a, b) for a in range(ly) for b in range(lx)]
            #
            legends = dict(
                zip(legends_list, [[] for a in range(len(legends_list))])
            )
            #
            # if type(pl) in [list, tuple, np.ndarray]:
            #     #     viewlist = []
            #     #     for p in pl:
            #     #         viewlist.append(plot_df.xs((p), level=separate, axis=0))
            #     #
            #     #     sepview = pd.concat(viewlist)
            #     #     plsave = 'ions'
            #     #
            #     # else:
            fig, ax = plt.subplots(
                nrows=ly,
                ncols=lx,
                figsize=figsize,
                sharey=True,
                dpi=dpi,
                constrained_layout=True,
                # sharex=True
            )
            if pl is None:
                sepview = plot_df.view()
                plsave = ""
            else:
                sepview = plot_df.xs(
                    (pl), level=separate, axis=0, drop_level=False
                )
                plsave = pl
                #
                #

                #
                fig.suptitle(
                    (
                        ", ".join([title_dict[s].upper() for s in separate])
                        + f": {label_mod(list(tuple(zip(pl, separate))))}"
                    ),
                    size=16,
                    weight="bold",
                )
            pi = 0
            for col in vx:
                try:
                    view = sepview.xs(col, level=x, axis=0, drop_level=False)
                    pi = 1
                except ValueError:
                    view = sepview
                    col = vx
                    pi += 1
                # print("column", col)
                for it in iters:
                    try:
                        values = view.xs(
                            tuple(it), level=idx.tolist(), drop_level=False
                        )
                        x_grouplabels = []
                        x_labels = []
                        x_ticks = []
                        for vbar in vbars:
                            bulk_pad = 2
                            bulk_edge = np.rint((max_edge + bulk_pad))
                            x_ticks.append(bar_dict[vbar] * bulk_edge + xpad)
                            # print(values)
                            bottom = 0.0
                            x_grouplabels.append(vbar)
                            # x_tick = bar_dict[vbar] * (barwidth + xpad)
                            bar_vals = values.xs(
                                vbar, level=bars, drop_level=False
                            )

                            if len(self.__peaks) != len(
                                self.df.index.get_level_values(
                                    "_atoms"
                                ).unique()
                            ):
                                self._get_edges()
                            try:
                                peaks = self.__peaks[
                                    bar_vals.index.get_level_values("atoms")
                                    .unique()
                                    .tolist()[0]
                                ]
                            except:
                                peaks = self.__peaks[
                                    bar_vals.index.get_level_values("ions")
                                    .unique()
                                    .tolist()[0]
                                ]
                            #       bar_vals.values >= 0)
                            if np.all(bar_vals.values) >= 0:
                                # print("All > 0")
                                x_id, y_id = x_dict[col], y_dict[it[yid]]

                                bar_val_view = bar_vals
                                bar_val_view.index = (
                                    bar_val_view.index.get_level_values(
                                        "x_bins"
                                    )
                                )
                                # x_ticks.append(x_tick + 0.5 * barwidth)
                                # bar_plots = []
                                for bar_id, bar_val in enumerate(
                                    bar_val_view.items()
                                ):
                                    x_bin, y_val = bar_val

                                    try:
                                        peak = peaks[bar_id]
                                    except IndexError:
                                        peak = x_bin.right
                                    colour = cmap(cnorm(peak))

                                    label = self._get_bin_label(x_bin)
                                    if x_bin.right < np.max(self.x):
                                        label = f"${x_bin.left:3.1f} - {x_bin.right:3.1f}$ \AA"
                                        barwidth = x_bin.right - x_bin.left

                                    else:
                                        label = f"$ > {x_bin.left:3.1f}$ \AA"
                                        # print(bar_val_view.index[-1].left)
                                        barwidth = bulk_edge - x_bin.left
                                    # try:
                                    x_tick = x_ticks[-1] + barwidth
                                    x_ticks.append(x_tick)
                                    # except IndexError:
                                    #     x_tick = x_bin.left
                                    x_labels.append(x_bin.left)
                                    # print(x_ticks, x_tick, x_bin)
                                    # print(peaks, bar_id, "label", "peak", label, peak)
                                    # print(label)
                                    try:
                                        p = ax[y_id, x_id].bar(
                                            x_tick,
                                            y_val,
                                            label=label,
                                            left=bottom,
                                            height=-barwidth,
                                            align="edge",
                                            color=colour,
                                        )
                                        ax[y_id, x_id].bar_label(
                                            p,
                                            labels=[label],
                                            fmt="%s",
                                            label_type="center",
                                        )
                                    except IndexError:
                                        p = ax[y_id].bar(
                                            x_tick,
                                            y_val,
                                            label=label,
                                            left=bottom,
                                            height=-barwidth,
                                            align="edge",
                                            color=colour,
                                        )
                                        ax[y_id, x_id].bar_label(
                                            p,
                                            labels=[label],
                                            fmt="%s",
                                            label_type="center",
                                        )
                                    # finally:
                                    bottom += y_val
                                x_ticks = x_ticks[:-1]
                    #                 values = values
                    #                 print('try 1 done')
                    #
                    #                     # for shell in values:
                    #                     # view for group and bars
                    #                             label = f'${lims.left} - {lims.right}$ \AA'
                    #
                    #                     try:
                    #                         print('try 2')
                    #                         print(x_dict[col], y_dict[it[yid]])

                    #                     except:
                    # #                         raise ValueError
                    #                         x_id, y_id = 0, y_dict[it[yid]]
                    #                         label = f'${x_bin.left} - {x_bin.right}$ \AA'

                    #                     if pi == 1:
                    #                         legends[y_id, x_id].append(it[label_id])
                    #                 else:
                    #                     check_logger.info('NaN values')

                    except KeyError:
                        logger.info(f"No data for {pl}, {vx}, {it}")

            x_ticks = [
                np.linspace(
                    n_bar * bulk_edge + xpad,
                    n_bar * bulk_edge + bulk_edge,
                    int(bulk_edge),
                )
                for n_bar in range(lbars)
            ]
            x_ticks = np.ravel(x_ticks)
            x_labels = np.tile(np.arange(0, bulk_edge, 1), lbars)
            # print(x_ticks, x_labels)
            for i in range(ly):
                try:
                    ax[i, 0].set_ylabel(
                        f"{label_mod([(vy[i], y)])}\n" + rowlabel
                    )
                    ax[i, 0].set_yticks(np.arange(0.0, 1.1, 0.2))
                    for j in range(lx):
                        ax[i, j].spines[["top", "right"]].set_visible(False)
                        ax[i, j].hlines(1.0, -xpad, lbars, linestyle="--")
                        # ax[i, j].legend(ncol=2, loc='lower center')#[leg for leg in legends[i, j]], ncol=3)
                        #                  if xlim != None:
                        # ax[i, j].set_xlim((-xpad, lbars))
                        ax[i, j].set_xticks([], [])
                        #                 if ylim != None:
                        ax[i, j].set_ylim((0.0, 1.25))
                        # print(x_ticks, x_labels)
                        ax[ly - 1, j].set_xticks(x_ticks, x_labels)
                #                 ax[ly - 1, j].set_xlabel(columnlabel + f'\n{label_mod([(vx[j], x)])}')
                except IndexError:
                    ...
        #             ax[i].set_ylabel(f'{label_mod([(vy[i], y)])}\n' + rowlabel)
        #             ax[i].legend([label_mod([(leg, label_key)]) for leg in legends[i, 0]], ncol=3)
        #             ax[ly - 1].set_xlabel(columnlabel + f'\n{label_mod([(vx[0], x)])}')
        # #
        # #     fig.supxlabel(f'{title_dict[x]}s', size=14)
        # #     fig.supylabel(f'{title_dict[y]}s', size=14)
        # #     if save != False:
        # #         odir = Path(odir)
        # #         if not odir.is_dir():
        # #             os.makedirs(odir)
        # #         if type(save) == str:
        # #             fig.savefig(odir / f'{save}.png')
        # #         else:
        # #             fig.savefig(odir / f'{self.name}_{diffstr}_{x}_{y}_{plsave}_{self.cutoff}_{self.bins}.png')
        # #         fig.clear()

    def plot_hbars(
        self,
        bars: Literal["clays", "aas", "ions", "atoms", "other"],
        x: Literal["clays", "aas", "ions", "atoms", "other"],
        y: Literal["clays", "aas", "ions", "atoms", "other"],
        # y: Literal['clays', 'ions', 'aas', 'atoms', 'other'],
        # select: Literal['clays', 'ions', 'aas'],
        rowlabel: str = "y",
        columnlabel: str = "x",
        figsize=None,
        dpi=None,
        # diff=False,
        xmax=50,
        ymax=50,
        save=False,
        ylim=None,
        odir=".",
        # barwidth = 0.75,
        xpad=0.25,
        cmap="winter",
    ):
        """Create stacked Histogram adsorption shell populations."""
        aas_classes = [
            ["arg", "lys", "his"],
            ["glu", "gln"],
            ["cys"],
            ["gly"],
            ["pro"],
            ["ala", "val", "ile", "leu", "met"],
            ["phe", "tyr", "trp"],
            ["ser", "thr", "asp", "gln"],
        ]
        ions_classes = [["Na", "Ca"], ["Ca", "Mg"]]
        atoms_classes = [["ions"], ["N"], ["OT"], ["CA"]]
        clays_classes = [["NAu-1"], ["NAu-2"]]
        cmaps_seq = ["Purples", "Blues", "Greens", "Oranges", "Reds"]
        cmaps_single = ["Dark2"]
        sel_list = ("clays", "ions", "aas", "atoms")
        # for color, attr in zip([''], sel_list):
        #     cmaps_dict[attr] = {}
        # cm.get_cmap()
        cmap_dict = {"clays": []}

        title_dict = {
            "clays": "Clay type",
            "ions": "Ion type",
            "aas": "Amino acid",
            "atoms": "Atom type",
            "other": "Other atom type",
        }

        sel_list = ["clays", "ions", "aas", "atoms"]

        # if self.other != None:
        #     sel_list.append('other')
        cmap = colormaps[cmap]
        separate = [
            s for s in sel_list if s not in [x, y, bars]
        ]  # (s != x and s != y and s != bars and s != groups)]
        # print(separate)
        idx = pd.Index(
            [
                s
                for s in sel_list
                if (s != x and s != bars and s not in separate)
            ]
        )
        # print(idx)

        sep = pd.Index(separate)

        vx = getattr(self, x)
        logger.info(f"x = {x}: {vx}")
        lx = len(vx)

        vy = getattr(self, y)
        logger.info(f"y = {y}: {vy}")
        ly = len(vy)
        # print(ly)

        vbars = getattr(self, bars)
        lbars = len(vbars)
        logger.info(f"bars = {bars}: {vbars}")

        bar_dict = dict(zip(vbars, np.arange(lbars)))

        yid = np.ravel(np.where(np.array(idx) == y))[0]
        # print(yid)

        # label_key = idx.difference(pd.Index([x, y, *separate]), sort=False).values[0]

        # label_id = idx.get_loc(key=label_key)
        n_plots = len(sep)

        x_dict = dict(zip(vx, np.arange(lx)))
        # print(x_dict)

        sel = self.clays

        plot_df = self.bin_df[sel].copy()
        idx_names = ["clays", *plot_df.index.droplevel(["x", "_atoms"]).names]
        # print("idx", idx_names)
        plot_df = plot_df.stack()
        # atoms = plot_df.index.get_level_values('_atoms')
        plot_df.index = plot_df.index.droplevel(["x", "_atoms"])
        # plot_df = pd.DataFrame({'values': plot_df,
        #                         '_atoms': atoms})
        # idx_names.remove('_atoms')
        plot_df = plot_df.reorder_levels(idx_names)
        #
        plot_df.name = "values"
        # print(plot_df.head(3))
        plot_df = plot_df.groupby(plot_df.index.names).sum()
        # print(plot_df.head(3))
        inner_sum = plot_df.groupby(
            plot_df.index.droplevel("x_bins").names
        ).sum()
        extra = 1 - inner_sum
        plot_df.where(
            np.rint(plot_df.index.get_level_values("x_bins").right)
            != int(self.cutoff),
            lambda x: x + extra,
            inplace=True,
        )
        # print(type(self.ads_edges))
        max_edge = list(map(lambda x: np.max(x[:-1]), self.edges.values()))
        max_edge = np.max(max_edge)
        # print("max edge", max_edge)
        # max_edge = np.ravel(np.array(*self.ads_edges.values()))
        # print(max_edge)
        cnorm = mpc.Normalize(vmin=0, vmax=max_edge, clip=False)

        if figsize == None:
            figsize = tuple(
                [
                    5 * lx if (10 * lx) < xmax else xmax,
                    5 * ly if (5 * ly) < ymax else ymax,
                ]
            )
        #
        if dpi == None:
            dpi = 100
        #
        iters = np.array(
            np.meshgrid(*[getattr(self, idxit) for idxit in idx])
        ).T.reshape(-1, len(idx))
        #
        logger.info(f"Printing bar plots for {sep}\nColumns: {vx}\nRows: {vy}")
        #
        label_mod = lambda l: ", ".join(
            [li.upper() if namei == "aas" else li for li, namei in l]
        )
        #
        try:
            sep_it = np.array(
                np.meshgrid(*[getattr(self, idxit) for idxit in sep])
            ).T.reshape(-1, len(sep))
        except ValueError:
            sep_it = [None]
        # check_logger.info(vx, vy, lx, ly)
        #
        for pl in sep_it:
            # print(pl)
            # try:
            #     fig.clear()
            # except:
            #     pass
            y_dict = dict(zip(vy, np.arange(ly)))
            # print(y_dict)
            #     if separate == 'atoms' and pl != '':
            #         ...
            #
            legends_list = [(a, b) for a in range(ly) for b in range(lx)]
            #
            legends = dict(
                zip(legends_list, [[] for a in range(len(legends_list))])
            )
            #
            # if type(pl) in [list, tuple, np.ndarray]:
            #     #     viewlist = []
            #     #     for p in pl:
            #     #         viewlist.append(plot_df.xs((p), level=separate, axis=0))
            #     #
            #     #     sepview = pd.concat(viewlist)
            #     #     plsave = 'ions'
            #     #
            #     # else:
            fig, ax = plt.subplots(
                nrows=ly,
                ncols=lx,
                figsize=figsize,
                sharey=True,
                dpi=dpi,
                constrained_layout=True,
                # sharex=True
            )
            if pl is None:
                sepview = plot_df.view()
                plsave = ""
            else:
                sepview = plot_df.xs(
                    (pl), level=separate, axis=0, drop_level=False
                )
                plsave = pl
                #
                #

                #
                fig.suptitle(
                    (
                        ", ".join([title_dict[s].upper() for s in separate])
                        + f": {label_mod(list(tuple(zip(pl, separate))))}"
                    ),
                    size=16,
                    weight="bold",
                )
            pi = 0
            for col in vx:
                try:
                    view = sepview.xs(col, level=x, axis=0, drop_level=False)
                    pi = 1
                except ValueError:
                    view = sepview
                    col = vx
                    pi += 1
                # print("column", col)
                for it in iters:
                    try:
                        values = view.xs(
                            tuple(it), level=idx.tolist(), drop_level=False
                        )
                        x_grouplabels = []
                        x_labels = []
                        x_ticks = []
                        for vbar in vbars:
                            bulk_pad = 2
                            bulk_edge = np.rint((max_edge + bulk_pad))
                            x_ticks.append(bar_dict[vbar] * bulk_edge + xpad)
                            # print(values)
                            bottom = 0.0
                            x_grouplabels.append(vbar)
                            # x_tick = bar_dict[vbar] * (barwidth + xpad)
                            bar_vals = values.xs(
                                vbar, level=bars, drop_level=False
                            )

                            if len(self.__peaks) != len(
                                self.df.index.get_level_values(
                                    "_atoms"
                                ).unique()
                            ):
                                self._get_edges()
                            try:
                                peaks = self.__peaks[
                                    bar_vals.index.get_level_values("atoms")
                                    .unique()
                                    .tolist()[0]
                                ]
                            except:
                                peaks = self.__peaks[
                                    bar_vals.index.get_level_values("ions")
                                    .unique()
                                    .tolist()[0]
                                ]
                            #       bar_vals.values >= 0)
                            if np.all(bar_vals.values) >= 0:
                                # print("All > 0")
                                x_id, y_id = x_dict[col], y_dict[it[yid]]

                                bar_val_view = bar_vals
                                bar_val_view.index = (
                                    bar_val_view.index.get_level_values(
                                        "x_bins"
                                    )
                                )
                                # x_ticks.append(x_tick + 0.5 * barwidth)
                                # bar_plots = []
                                for bar_id, bar_val in enumerate(
                                    bar_val_view.items()
                                ):
                                    x_bin, y_val = bar_val

                                    try:
                                        peak = peaks[bar_id]
                                    except IndexError:
                                        peak = x_bin.right
                                    colour = cmap(cnorm(peak))

                                    if x_bin.right < np.max(self.x):
                                        label = f"${x_bin.left} - {x_bin.right}$ \AA"
                                        barwidth = x_bin.right - x_bin.left

                                    else:
                                        label = f"$ > {x_bin.left}$ \AA"
                                        # print(bar_val_view.index[-1].left)
                                        barwidth = bulk_edge - x_bin.left
                                    # try:
                                    x_tick = x_ticks[-1] + barwidth
                                    x_ticks.append(x_tick)
                                    # except IndexError:
                                    #     x_tick = x_bin.left
                                    x_labels.append(x_bin.left)
                                    # print(x_ticks, x_tick, x_bin)
                                    # print(peaks, bar_id, "label", "peak", label, peak)
                                    # print(label)
                                    try:
                                        p = ax[y_id, x_id].barh(
                                            x_tick,
                                            y_val,
                                            label=label,
                                            left=bottom,
                                            height=-barwidth,
                                            align="edge",
                                            color=colour,
                                        )
                                        ax[y_id, x_id].bar_label(
                                            p,
                                            labels=[label],
                                            fmt="%s",
                                            label_type="center",
                                        )
                                    except IndexError:
                                        p = ax[y_id].barh(
                                            x_tick,
                                            y_val,
                                            label=label,
                                            left=bottom,
                                            height=-barwidth,
                                            align="edge",
                                            color=colour,
                                        )
                                        ax[y_id, x_id].bar_label(
                                            p,
                                            labels=[label],
                                            fmt="%s",
                                            label_type="center",
                                        )
                                    # finally:
                                    bottom += y_val
                                x_ticks = x_ticks[:-1]
                    #                 values = values
                    #                 print('try 1 done')
                    #
                    #                     # for shell in values:
                    #                     # view for group and bars
                    #                             label = f'${lims.left} - {lims.right}$ \AA'
                    #
                    #                     try:
                    #                         print('try 2')
                    #                         print(x_dict[col], y_dict[it[yid]])

                    #                     except:
                    # #                         raise ValueError
                    #                         x_id, y_id = 0, y_dict[it[yid]]
                    #                         label = f'${x_bin.left} - {x_bin.right}$ \AA'

                    #                     if pi == 1:
                    #                         legends[y_id, x_id].append(it[label_id])
                    #                 else:
                    #                     check_logger.info('NaN values')

                    except KeyError:
                        logger.info(f"No data for {pl}, {vx}, {it}")

            x_ticks = [
                np.linspace(
                    n_bar * bulk_edge + xpad,
                    n_bar * bulk_edge + bulk_edge,
                    int(bulk_edge),
                )
                for n_bar in range(lbars)
            ]
            x_ticks = np.ravel(x_ticks)
            x_labels = np.tile(np.arange(0, bulk_edge, 1), lbars)
            # print(x_ticks, x_labels)
            for i in range(ly):
                try:
                    ax[i, 0].set_ylabel(
                        f"{label_mod([(vy[i], y)])}\n" + rowlabel
                    )
                    ax[i, 0].set_xticks(np.arange(0.0, 1.1, 0.2))
                    for j in range(lx):
                        ax[i, j].spines[["top", "right"]].set_visible(False)
                        ax[i, j].hlines(1.0, -xpad, lbars, linestyle="--")
                        # ax[i, j].legend(ncol=2, loc='lower center')#[leg for leg in legends[i, j]], ncol=3)
                        #                  if xlim != None:
                        # ax[i, j].set_xlim((-xpad, lbars))
                        ax[i, j].set_yticks([], [])
                        #                 if ylim != None:
                        ax[i, j].set_ylim((0.0, 1.25))
                        # print(x_ticks, x_labels)
                        ax[ly - 1, j].set_yticks(x_ticks, x_labels)
                #                 ax[ly - 1, j].set_xlabel(columnlabel + f'\n{label_mod([(vx[j], x)])}')
                except IndexError:
                    ...
        #             ax[i].set_ylabel(f'{label_mod([(vy[i], y)])}\n' + rowlabel)
        #             ax[i].legend([label_mod([(leg, label_key)]) for leg in legends[i, 0]], ncol=3)
        #             ax[ly - 1].set_xlabel(columnlabel + f'\n{label_mod([(vx[0], x)])}')
        # #
        # #     fig.supxlabel(f'{title_dict[x]}s', size=14)
        # #     fig.supylabel(f'{title_dict[y]}s', size=14)
        # #     if save != False:
        # #         odir = Path(odir)
        # #         if not odir.is_dir():
        # #             os.makedirs(odir)
        # #         if type(save) == str:
        # #             fig.savefig(odir / f'{save}.png')
        # #         else:
        # #             fig.savefig(odir / f'{self.name}_{diffstr}_{x}_{y}_{plsave}_{self.cutoff}_{self.bins}.png')
        # #         fig.clear()


class Data2D:
    """
    Class for histogram analysis data processing and plotting.
    Reads files in `indir` that match the naming pattern
    "`namestem`*_`cutoff`_`bins`.dat"
    (or "`namestem`*_`cutoff`_`bins`_`analysis`.dat" if `analysis` != `None`.
    The data is stored in a :class:`pandas.DataFrame`
    :param indir: Data directory
    :type indir: Union[str, Path]
    :param cutoff: Maximum value in the histogram bins
    :type cutoff: Union[int, float]
    :param bins: Histogram bin size
    :type bins: float
    :param ions: List of ion types in solvent
    :type ions: List[Literal['Na', 'K', 'Ca', 'Mg']], optional
    :param atoms: List of atom types in selection, defaults to `None`
    :type atoms: List[Literal['ions', 'OT', 'N', 'CA', 'OW']], optional
    :param other: Optional list of atom types in a second selection,
     defaults to `None`
    :type other: List[Literal['ions', 'OT', 'N', 'CA', 'OW']], optional
    :param clays: List of clay types,
     defaults to `None`
    :type clays: List[Literal['NAu-1', 'NAu-2']], optional
    :param aas: List of amino acid types in lower case 3-letter code,
     defaults to `None`
    :type aas: Optional[List[Literal['ala', 'arg', 'asn', 'asp',
                                   'ctl', 'cys', 'gln', 'glu',
                                   'gly', 'his', 'ile', 'leu',
                                   'lys', 'met', 'phe', 'pro',
                                   'ser', 'thr', 'trp', 'tyr',
                                   'val']]]
    :param load: Load,  defaults to False
    :type load: Union[str, Literal[False], Path], optional
    :param odir: Output directory, defaults to `None`
    :type odir: str, optional
    :param nameparts: number of `_`-separated partes in `namestem`
    :type nameparts: int, defaults to 1
    :param namestem: leading string in naming pattern, optional
    :type namestem: str, defaults to ''
    :param analysis: trailing string in naming pattern, optional
    defaults to `None`
    :type analysis: str, optional
    :param df: :class: `pandas.DataFrame`
    """

    aas = [
        "ala",
        "arg",
        "asn",
        "asp",
        "ctl",
        "cys",
        "gln",
        "glu",
        "gly",
        "his",
        "ile",
        "leu",
        "lys",
        "met",
        "phe",
        "pro",
        "ser",
        "thr",
        "trp",
        "tyr",
        "val",
    ]

    ions = ["Na", "K", "Ca", "Mg"]
    atoms = ["ions", "OT", "N", "CA"]
    clays = ["NAu-1", "NAu-2"]  # , 'L31']
    analyses = ["zvel", "surfzvel", "xyvel", "totvel"]

    @redirect_tqdm
    def __init__(
        self,
        indir: Union[str, Path],
        zdir: Union[str, Path],
        cutoff: Union[int, float],
        bins: float,
        ions: List[Literal["Na", "K", "Ca", "Mg"]] = None,
        atoms: List[Literal["ions", "OT", "N", "CA", "OW"]] = None,
        other: List[Literal["ions", "OT", "N", "CA", "OW"]] = None,
        clays: List[Literal["NAu-1", "NAu-2"]] = None,
        aas: List[
            Literal[
                "ala",
                "arg",
                "asn",
                "asp",
                "ctl",
                "cys",
                "gln",
                "glu",
                "gly",
                "his",
                "ile",
                "leu",
                "lys",
                "met",
                "phe",
                "pro",
                "ser",
                "thr",
                "trp",
                "tyr",
                "val",
            ]
        ] = None,
        load: Union[str, Literal[False], Path] = False,
        odir: Optional[str] = None,
        nameparts: int = 1,
        namestem: str = "veldist",
        zstem: str = "zdist",
        analyses: Optional[List[str]] = None,
        zname: Optional[str] = "zdens",
        vel_bins: float = 0.05,
        velrange: Tuple[float, float] = (0.0, 1.0),
    ):
        """Constructor method"""
        logger.info(f"Initialising {self.__class__.__name__}")
        self.filelist: dict = {}
        self.bins: Bins = Bins(bins)
        self.cutoff: float = Cutoff(cutoff)
        x = None

        self.zname = zname

        if type(indir) != Path:
            indir = Path(indir)

        if type(zdir) != Path:
            zdir = Path(zdir)

        self._indir = indir
        self._zdir = zdir

        # if self.analysis is None:
        #     logger.info(
        #         rf"Getting {namestem}*_"
        #         rf"{self.cutoff}_"
        #         rf"{self.bins}.p from {str(indir.resolve())!r}"
        #     )
        #     self.filelist: List[Path] = sorted(
        #         list(
        #             indir.glob(rf"{namestem}*_" rf"{self.cutoff}_" rf"{self.bins}.dat")
        #         )
        #     )
        # else:
        #     logger.info(
        #         rf"Getting {namestem}*_"
        #         rf"{self.cutoff}_"
        #         rf"{self.bins}_"
        #         rf"{analysis}.dat from {str(indir.resolve())!r}"
        #     )
        #     self.filelist: List[Path] = sorted(
        #         list(
        #             indir.glob(
        #                 rf"{namestem}*_"
        #                 rf"{self.cutoff}_"
        #                 rf"{self.bins}_"
        #                 rf"{self.analysis}.dat"
        #             )
        #         )
        #     )
        # logger.info(f"Found {len(self.filelist)} files.")

        if load != False:
            load = Path(load).resolve()
            print(load)
            self.df: pd.DataFrame = pkl.load(load)
            logger.info(f"Using data from {load!r}")
        else:
            if ions == None:
                ions = self.__class__.ions
                logger.info(
                    f"ions not specified, using default {self.__class__.ions}"
                )
            else:
                logger.info(f"Using custom {ions} for ions")
            if atoms == None:
                atoms = self.__class__.atoms
                logger.info(
                    f"atoms not specified, using default {self.__class__.atoms}"
                )
            else:
                logger.info(f"Using custom {atoms} for atoms")
            if aas == None:
                aas = self.__class__.aas
                logger.info(
                    f"aas not specified, using default {self.__class__.aas}"
                )
            else:
                logger.info(f"Using custom {aas} for aas")
            if clays == None:
                clays = self.__class__.clays
                logger.info(
                    f"clays not specified, using default {self.__class__.clays}"
                )
            else:
                logger.info(f"Using custom {clays} for clays")
            if analyses == None:
                analyses = self.__class__.analyses
                logger.info(
                    f"clays not specified, using default {self.__class__.analyses}"
                )
            else:
                logger.info(f"Using custom {analyses} for analyses")
            logger.info(
                rf"Getting {zstem}*_"
                rf"{self.cutoff}_"
                rf"{self.bins}_"
                rf"{self.zname}.p from {str(zdir.resolve())!r}"
            )
            self.filelist[self.zname]: List[Path] = sorted(
                list(
                    zdir.glob(
                        rf"{zstem}*_"
                        rf"{self.cutoff}_"
                        rf"{self.bins}_"
                        rf"{self.zname}.p"
                    )
                )
            )
            logger.info(f"Found {len(self.filelist[self.zname])} files.")

            for analysis in analyses:
                logger.info(
                    rf"Getting {namestem}*_"
                    rf"{self.cutoff}_"
                    rf"{self.bins}_"
                    rf"{analysis}.p from {str(indir.resolve())!r}"
                )
                self.filelist[analysis]: List[Path] = sorted(
                    list(
                        indir.glob(
                            rf"{namestem}*_"
                            rf"{self.cutoff}_"
                            rf"{self.bins}_"
                            rf"{analysis}.p"
                        )
                    )
                )

                logger.info(f"Found {len(self.filelist[analysis])} files.")

            cols = pd.Index([*analyses], name="analyses")

            if other is not None:
                if other is True:
                    other = atoms
                    other.append("OW")
                idx = pd.MultiIndex.from_product(
                    [clays, ions, aas, atoms, other],
                    names=["clays", "ions", "aas", "atoms", "other"],
                )
                self.other: List[str] = other
                logger.info(f"Setting second atom selection to {self.other}")
            else:
                idx = pd.MultiIndex.from_product(
                    [clays, ions, aas, atoms],
                    names=["clays", "ions", "aas", "atoms"],
                )
                self.other: None = None

            self.df: pd.DataFrame = pd.DataFrame(index=idx, columns=cols)
            self.zf = pd.Series(
                index=self.df.index, name=self.zname, dtype="object"
            )
            if other is not None:
                self.zf.reset_index("other", drop=True, inplace=True)
            # logger.info(f'Getting {self.__class__.__name__} data')
            self._get_data(nameparts, analyses)
            # logger.info('Finished getting data')

        setattr(self, f"{self.df}.columns.name", list(self.df.columns))
        self.df.reset_index(level=["ions", "atoms"], inplace=True)
        self.df["_atoms"] = self.df["atoms"].where(
            self.df["atoms"] != "ions", self.df["ions"], axis=0
        )
        self.df.set_index(["ions", "atoms"], inplace=True, append=True)
        atoms_col = self.df["_atoms"].copy()
        atoms_col = atoms_col.reset_index().set_index(self.zf.index.names)
        atoms_col[self.zf.name] = np.nan
        atoms_col[self.zf.name].update(self.zf)
        self.zf = atoms_col
        for df in [self.zf, self.df]:
            df.set_index(["_atoms"], inplace=True, append=True)
            # df.sort_index(inplace=True)
            df.dropna(inplace=True, how="all", axis=0)
        self.df.index = self.df.index.reorder_levels([*idx.names, "_atoms"])
        # self.zf.sort_index(inplace=True, sort_remaining=True)
        # self.df.sort_index(inplace=True, sort_remaining=True)
        _atoms = self.df.index.get_level_values("_atoms").tolist()
        #
        # print(self.zf, "\n", self.zf.index.names)
        # print(self.df, "\n", self.df.index.names)

        for iid, i in enumerate(self.df.index.names):
            value: List[Union[str, float]] = (
                self.df.index._get_level_values(level=iid).unique().tolist()
            )
            logger.info(f"Setting {i} to {value}")
            setattr(self, i, value)

        if odir != None:
            self.odir: Path = Path(odir)
        else:
            self.odir: Path = Path(".").cwd()

        logger.info(f"Output directory set to {str(self.odir.resolve())!r}\n")
        self.__bin_df = pd.DataFrame(columns=self.df.columns)

        self.__edges = {}
        self.__peaks = {}
        self.__z_bins = None
        self.__other_bins = {}

    def get_fname(self, fname):
        return Path(fname).parent / f"{Path(fname).stem}.tar.xz"

    def save(self, savename):
        savename = self.get_fname(savename)
        logger.info(f"Writing dataframe to {savename}")
        pd.to_pickle(self.df, savename)

    @property
    def z_bins(self):
        if self.__z_bins is None:
            with open(self.filelist[self.zname][0], "rb") as file:
                data = pkl.load(file)
                self.__z_bins = HistData(edges=data.edges)
        return self.__z_bins

    @property
    def other_bins(self):
        for analysis in self.analyses:
            if analysis not in self.__other_bins.keys():
                with open(self.filelist[analysis][0], "rb") as file:
                    data = pkl.load(file)
                    self.__other_bins[analysis] = HistData(edges=data.edges)
        return self.__other_bins

    def _get_data(self, nameparts, analyses):
        idsl = pd.IndexSlice
        for f in self.filelist[self.zname]:
            namesplit = f.stem.split("_")
            namesplit.pop(-1)
            namesplit = namesplit[nameparts:]
            try:
                clay, ion, aa, pH, atom, cutoff, bins = namesplit
                assert cutoff == self.cutoff
                assert bins == self.bins
                try:
                    self.zf.loc[idsl[clay, ion, aa, atom]] = f
                except KeyError:
                    print("KeyError")
            except IndexError:
                logger.info(f"Encountered IndexError while getting data")
            except ValueError:
                print(f, namesplit)
                logger.info(f"Encountered ValueError while getting data")

        logger.info(f"Getting {self.__class__.__name__} data")
        for analysis in tqdm(
            analyses, leave=False, position=0, desc="analyses"
        ):
            logger.info(f"\tanalysis: {analysis}")
            for f in tqdm(
                self.filelist[analysis], leave=False, position=1, desc="files"
            ):
                namesplit = f.stem.split("_")
                namesplit.pop(-1)
                namesplit = namesplit[nameparts:]
                if self.other is not None:
                    other = namesplit.pop(5)
                    if other in self.ions:
                        other = "ions"
                try:
                    clay, ion, aa, pH, atom, cutoff, bins = namesplit
                    assert cutoff == self.cutoff
                    assert bins == self.bins
                    try:
                        zdata = self.zf.loc[idsl[clay, ion, aa, atom]]
                        timeseries = Timeseries(f)
                        timeseries.zdata = zdata
                        if self.other is None:
                            self.df.loc[
                                idsl[clay, ion, aa, atom], analysis
                            ] = timeseries
                        else:
                            self.df.loc[
                                idsl[clay, ion, aa, atom, other], analysis
                            ] = timeseries
                    except ValueError:
                        print("KeyError")
                except IndexError:
                    logger.info(f"Encountered IndexError while getting data")
                except ValueError:
                    print(f, namesplit)
                    logger.info(f"Encountered ValueError while getting data")
        logger.info("Finished getting data")
        logger.info(f'{self.df.dropna(how="all", axis=0)}')
        logger.debug(self.zf[self.zf.isna()])
        logger.debug(self.df[self.df.isna()])

    def __repr__(self):
        return self.df.dropna(how="all", axis=0).__repr__()

    def _get_edge_fname(
        self,
        atom_type: str,
        name: Union[Literal["pe"], Literal["edge"]] = "pe",
    ):
        # fname = Path.cwd() / f"edge_data/edges_{atom_type}_{self.cutoff}_{self.bins}.p"
        fname = (
            Path(__file__).parent
            / f"pe_data/{atom_type}_{name}_data_{self.cutoff}_{self.bins}.p"
        )
        logger.info(f"Peak/edge Filename: {fname}")
        return fname

    def _read_edge_file(self, atom_type: str, skip=True):
        fname = self._get_edge_fname(atom_type, name="ads_edges")
        if not fname.exists():
            logger.info("No edge file found.")
            os.makedirs(fname.parent, exist_ok=True)
            logger.info(f"{fname.parent}")
            if skip is True:
                logger.info(f"Continuing without ads_edges")
                p = [0, self.cutoff]
            else:
                # self._get_edges(atom_type=atom_type)
                raise FileNotFoundError(f"No edge file found {fname}.")

        else:
            with open(fname, "rb") as edges_file:
                logger.info(f"Reading ads_edges {edges_file.name}")
                p = pkl.load(edges_file)["edges"]
        logger.info(f"ads_edges:{p}")
        return p

    # def get_bin_df(self):
    #     idx = self.df.index.names
    #     bin_df = self.df.copy()
    #     atom_types = bin_df.index.get_level_values("_atoms").unique().tolist()
    #     bin_df.reset_index(["x_bins", "x", "_atoms"], drop=False, inplace=True)
    #     for atom_type in atom_types:
    #         # logger.info(f"{atom_type}")
    #         try:
    #             ads_edges = self.__edges[atom_type]
    #         except KeyError:
    #             # edge_fname = self._get_edge_fname(atom_type)
    #             ads_edges = self._read_edge_file(atom_type=atom_type)
    #             # if edge_fname.is_file():
    #             #     self.__edges[atom_type] = self._read_edge_file(atom_type)
    #             # else:
    #             #     raise
    #             #     self._get_edges(atom_type=atom_type)
    #             # ads_edges = self.__edges[atom_type]
    #         # print(ads_edges, bin_df['x_bins'].where(bin_df['_atoms'] == atom_type))
    #         bin_df["x_bins"].where(
    #             bin_df["_atoms"] != atom_type,
    #             pd.cut(bin_df["x"], [*ads_edges]),
    #             inplace=True,
    #         )
    #     bin_df.reset_index(drop=False, inplace=True)
    #
    #     bin_df.set_index(
    #         idx,
    #         inplace=True,
    #     )
    #     self.df = bin_df.copy()
    #
    # @property
    # def bin_df(self):
    #     if not self.df.index.get_level_values("x_bins").is_interval():
    #         logger.info("No Interval")
    #         self.get_bin_df()
    #     else:
    #         logger.info("Interval")
    #     return self.df

    # with open(self.zdist, "rb") as zdist_file:
    #     zdist_data = pkl.load(zdist_file)
    #     self.zdist_timeseries = zdist_data.timeseries
    #     self.zdist_edges = zdist_data.ads_edges
    #     self.zdist_bins = zdist_data.bins
    # timeseries_arrays = {}
    # hist_arrays = {}
    # for analysis in self.analyses:
    #     for f in self.filelist[analysis]:
    #         namesplit = f.stem.split("_")
    #         # if self.analysis is not None:
    #         namesplit.pop(-1)
    #         # else:
    #         #     self.analysis = "zdist"
    #         name = namesplit[:nameparts]
    #         namesplit = namesplit[nameparts:]
    #         # if self.other != None:
    #         #     other = namesplit[4]
    #         #     namesplit.pop(4)
    #         #     if other in self.ions:
    #         #         other = "ions"
    #         #     try:
    #         clay, ion, aa, pH, atom, cutoff, bins = namesplit
    #         assert cutoff == self.cutoff
    #         assert bins == self.bins
    #         with open(f, "rb") as file:
    #             array = pkl.load(f)
    #         timeseries_arrays[analysis] = (array.ads_edges, array.timeseries)
    #         hist_arrays[analysis] = np.histogramdd(
    #             [
    #                 np.ravel(self.zdist_timeseries),
    #                 np.ravel(timeseries_arrays[analysis]),
    #             ],
    #             bins=[self.zdist_edges, self.vel_edges],
    #         )
    # analysis_cols = [a for a in self.analyses if self.analyses != "totvel"]
    # for analysis_col in analysis_cols:
    #     hist_arrays[f"{analysis_col}/totvel"] = np.histogramdd(
    #         [
    #             np.ravel(self.zdist_timeseries),
    #             np.divide(
    #                 np.ravel(timeseries_arrays[analysis_col]),
    #                 np.ravel(timeseries_arrays["totvel"]),
    #                 where=np.ravel(timeseries_arrays["totvel"]) != 0,
    #             ),
    #         ],
    #         bins=[self.zdist_edges, self.vel_edges],
    #     )

    # array = pd.read_csv(f, delimiter="\s+", comment="#").to_numpy()
    # try:
    #     self.df.loc[idsl[ion, aa, atom, :], clay] = array[:, 2]
    # except ValueError:
    #     logger.info('Using second atom species')
    #     self.df.loc[idsl[ion, aa, atom, other, :], clay] = array[:, 2]
    # except KeyError:
    #     logger.info('Using 1st column')
    #     self.df.loc[idsl[ion, aa, atom, :], clay] = array[:, 1]
    #     except IndexError:
    #         logger.info(f"Encountered IndexError while getting data")
    #     except ValueError:
    #         logger.info(f"Encountered ValueError while getting data")
    # self.name = "_".join(name)

    # def __repr__(self):
    #     return self.df[self.clays].dropna().__repr__()


class RawData:
    """
    Class for raw simulation data checking and setup.
    Reads files in `rootdir` that follow a directory structure
    "`clay_type/ion_type/aa_type/`".
    The data is stored in a :class:`pandas.DataFrame`
    :param root_dir: Data directory
    :type root_dir: Union[str, Path]
    :param alt_dir: Alternative data directory, defaults to `None`
    :type alt_dir: Union[str, Path], optional
    :param ions: List of ion types in solvent
    :type ions: List[Literal['Na', 'K', 'Ca', 'Mg']], optional
    :param atoms: List of atom types in selection, defaults to `None`
    :type atoms: List[Literal['ions', 'OT', 'N', 'CA', 'OW']], optional
    :param other: Optional list of atom types in a second selection,
     defaults to `None`
    :type other: List[Literal['ions', 'OT', 'N', 'CA', 'OW']], optional
    :param clays: List of clay types,
     defaults to `None`
    :type clays: List[Literal['NAu-1', 'NAu-2']], optional
    :param aas: List of amino acid types in lower case 3-letter code,
     defaults to `None`
    :type aas: Optional[List[Literal['ala', 'arg', 'asn', 'asp',
                                   'ctl', 'cys', 'gln', 'glu',
                                   'gly', 'his', 'ile', 'leu',
                                   'lys', 'met', 'phe', 'pro',
                                   'ser', 'thr', 'trp', 'tyr',
                                   'val']]]
    :param load: Load,  defaults to False
    :type load: Union[str, Literal[False], Path], optional
    :param odir: Output directory, defaults to `None`
    :type odir: str, optional
    :param nameparts: number of `_`-separated partes in `namestem`
    :type nameparts: int, defaults to 1
    :param namestem: leading string in naming pattern, optional
    :type namestem: str, defaults to ''
    :param analysis: trailing string in naming pattern, optional
    defaults to `None`
    :type analysis: str, optional
    :param df: :class: `pandas.DataFrame`
    """

    aas = [
        "ala",
        "arg",
        "asn",
        "asp",
        "ctl",
        "cys",
        "gln",
        "glu",
        "gly",
        "his",
        "ile",
        "leu",
        "lys",
        "met",
        "phe",
        "pro",
        "ser",
        "thr",
        "trp",
        "tyr",
        "val",
    ]

    ions = ["Na", "K", "Ca", "Mg"]
    clays = ["NAu-1", "NAu-2"]
    new_dirs = ["neutral", "setup"]
    idx_names = ["root", "clays", "ions", "aas"]

    def __init__(
        self,
        root_dir: Union[str, Path],
        alt_root: Optional[Union[str, Path]] = None,
        ions: List[Literal["Na", "K", "Ca", "Mg"]] = None,
        clays: List[Literal["NAu-1", "NAu-2"]] = None,
        aas: List[
            Literal[
                "ala",
                "arg",
                "asn",
                "asp",
                "ctl",
                "cys",
                "gln",
                "glu",
                "gly",
                "his",
                "ile",
                "leu",
                "lys",
                "met",
                "phe",
                "pro",
                "ser",
                "thr",
                "trp",
                "tyr",
                "val",
            ]
        ] = None,
        load: Union[str, Literal[False], Path] = False,
        odir: Optional[str] = None,
        new_dirs: List[str] = None,
    ):
        self.filelist: list = []
        if new_dirs is None:
            self.new_dirs = self.__class__.new_dirs
        else:
            self.new_dirs = [Path(dir).name for dir in new_dirs]

        if type(root_dir) != Path:
            root_dir = Path(root_dir)
            if not root_dir.is_dir():
                logger.error(f"No directory found for {root_dir!r}")

        self.root = root_dir.resolve()
        self.root_idx = [self.root]

        if alt_root is not None:
            indir = root_dir / alt_root
            if not indir.is_dir():
                indir = Path(alt_root).resolve()
                if not indir.is_dir():
                    logger.error(
                        f"No alternative directory found for {indir!r}"
                    )
            if indir.is_dir():
                self.alt = indir
                self.root_idx.append(self.alt)
                logger.info(f"Alternative root {self.alt.resolve()} specified")
            else:
                self.alt = None
        # print(self.root_idx)

        # self.root = SimDir(indir.resolve())
        # else:
        #     self._alt_dir = None

        #     self.filelist: List[Path] = sorted(
        #         list(
        #             root_dir.glob(rf"{namestem}*_" rf"{self.cutoff}_" rf"{self.bins}.dat")
        #         )
        #     )
        #
        # logger.info(f"Found {len(self.filelist)} files.")

        if load is not False:
            load = Path(load.resolve())
            self.df: pd.DataFrame = pkl.load(load)
            logger.info(f"Using data from {load!r}")
        else:
            if ions is None:
                ions = self.__class__.ions
                logger.info(
                    f"ions not specified, using default {self.__class__.ions}"
                )
            else:
                logger.info(f"Using custom {ions} for ions")
            if aas is None:
                aas = self.__class__.aas
                logger.info(
                    f"aas not specified, using default {self.__class__.aas}"
                )
            else:
                logger.info(f"Using custom {aas} for aas")
            if clays is None:
                clays = self.__class__.clays
                logger.info(
                    f"clays not specified, using default {self.__class__.clays}"
                )
            else:
                logger.info(f"Using custom {clays} for clays")

            self.clays = clays = self.modify_list_str(clays, suffix="-fe")
            self.ions = ions = self.modify_list_str(ions)
            self.aas = aas = self.modify_list_str(aas, suffix="_7")

            cols = pd.Index(["orig", *self.new_dirs], name="paths")
            # cols = pd.Index(self.new_dirs, name="paths")

            idx = pd.MultiIndex.from_product(
                [self.root_idx, clays, ions, aas],
                names=self.__class__.idx_names,
            )
            # df = pd.DataFrame(index=idx, columns=cols, dtype=object)
            # for col in cols:
            #     df[col] = np.NaN
            # df.columns.name = 'paths'
            # self.df = df
            self.df: pd.DataFrame = pd.DataFrame(
                index=idx, columns=cols, dtype=object
            )
            # self.idx_iter = self._get_idx_iter(self.df.index)

            for iid, i in enumerate(self.df.index.names):
                value: List[Union[str, float]] = (
                    self.df.index._get_level_values(level=iid)
                    .unique()
                    .tolist()
                )
                logger.info(f"Setting {i} to {value}")
                setattr(self, i, value)

            # logger.info(f'Getting data for {self.__class__.__name__}')
            self._get_data()
            # logger.info('Finished getting data')

        # self.df.dropna(inplace=True, how="all", axis=0)

        setattr(self, self.df.columns.name, list(self.df.columns))

        for iid, i in enumerate(self.df.index.names):
            value: List[Union[str, float]] = (
                self.df.index._get_level_values(level=iid).unique().tolist()
            )
            logger.info(f"Setting {i} to {value}")
            setattr(self, i, value)

        # print(self.root)

        if odir is not None:
            self.odir: Path = Path(odir)
        else:
            self.odir: Path = Path(".").cwd()
        print(self.odir)

        logger.info(f"Output directory set to {str(self.odir.resolve())!r}\n")

    def _get_data(self):
        idsl = pd.IndexSlice
        for dir_tree in self.idx_iter:
            root, clay, ion, aa = dir_tree
            p_str = np.apply_along_axis(lambda x: "/".join(x), 0, dir_tree[1:])
            path = root / p_str
            print(root, p_str)
            if path.is_dir():
                self.df.loc[idsl[root, clay, ion, aa], "orig"] = True
                for new_dir in self.new_dirs:
                    if (path / new_dir).is_dir():
                        self.df.loc[idsl[root, clay, ion, aa], new_dir] = True

    # def write_json(self, outpath=None):
    #     json_file = open_outfile(outpath=outpath, suffix='json', default='rawdata_paths')
    #     df = self.df.copy()
    #     df = df.stack(dropna=True)
    #     get_pd_idx_iter(self.df[self.df].index.unique())

    @staticmethod
    def regex_join(match_list, suffix="", prefix="") -> str:
        if not isinstance(match_list[0], str) or suffix != "" or prefix != "":
            match_list = list(
                map(lambda x: f"{str(x.rstrip(suffix))}", match_list)
            )
            match_list = list(
                map(lambda x: f"{str(x.lstrip(prefix))}", match_list)
            )
            match_list = list(
                map(lambda x: f"{prefix}{str(x)}{suffix}", match_list)
            )
        return "|".join(match_list)

    @staticmethod
    def modify_list_str(match_list, suffix="", prefix="") -> List[str]:
        # print(match_list)
        if not isinstance(match_list[0], str) or suffix != "" or prefix != "":
            match_list = list(
                map(lambda x: f"{str(x.rstrip(suffix))}", match_list)
            )
            match_list = list(
                map(lambda x: f"{str(x.lstrip(prefix))}", match_list)
            )
            match_list = list(
                map(lambda x: f"{prefix}{str(x)}{suffix}", match_list)
            )
        return match_list

    def save(self, savename=None, overwrite=True):
        if savename is None:
            savename = self.odir / f"{self.__class__.__name__}.p"
        if not savename.is_file() or overwrite is True:
            with open(savename, "wb") as outfile:
                pkl.dump(self.df, outfile)

    def update_attrs(self):
        for iid, i in enumerate(self.df.index.names):
            value: List[Union[str, float]] = (
                self.df.index._get_level_values(level=iid).unique().tolist()
            )
            logger.info(f"Setting {i} to {value}")
            setattr(self, i, value)

    # @cached_property
    @property
    def idx_iter(self):
        idx_values = [getattr(self, idxit) for idxit in self.df.index.names]
        idx_product = np.array(
            np.meshgrid(*[idx_value for idx_value in idx_values])
        ).T.reshape(-1, len(idx_values))
        # idx_product = np.apply_along_axis(lambda x: '/'.join(x), 1, idx_product)
        return idx_product


class ArrayData2D:
    aas = np.array(
        [
            "ala",
            "arg",
            "asn",
            "asp",
            "ctl",
            "cys",
            "gln",
            "glu",
            "gly",
            "his",
            "ile",
            "leu",
            "lys",
            "met",
            "phe",
            "pro",
            "ser",
            "thr",
            "trp",
            "tyr",
            "val",
        ]
    )
    ions = ["Na", "K", "Ca", "Mg"]
    atoms = ["ions", "OT", "N", "CA"]
    clays = ["NAu-1", "NAu-2"]

    def __init__(
        self,
        indir: Union[str, Path],
        namestem,
        cutoff,
        bins,
        ions=None,
        atoms=None,
        other=None,
        aas=None,
        load=False,
        odir=None,
        nameparts: int = 1,
    ):
        self.filelist = []
        self.bins = Bins(bins)
        self.cutoff = Cutoff(cutoff)

        if type(indir) != Path:
            indir = Path(indir)

        # self.filelist = sorted(list(indir.glob(rf'*_{self.cutoff}_{self.bins}.dat')))
        #

        if load != False:
            load = self.get_fname(load)
            self.df = pd.read_pickle(load)
        else:
            if ions == None:
                ions = self.__class__.ions
            if atoms == None:
                atoms = self.__class__.atoms
            if aas == None:
                aas = self.__class__.aas
            self.filelist = sorted(
                list(indir.glob(rf"{namestem}*_{self.cutoff}_{self.bins}.p"))
            )

            with open(self.filelist[0], "rb") as f:
                data = pkl.load(f)
                array, self.ybins, self.xbins = data.values()

            cols = pd.Index(["NAu-1", "NAu-2"], name="clays")

            if other != None:
                # print("other")
                other = atoms
                other.append("OW")
                idx = pd.MultiIndex.from_product(
                    [ions, aas, atoms, other],
                    names=["ions", "aas", "atoms", "other"],
                )
                self.other = other
            # print(self.other)
            else:
                idx = pd.MultiIndex.from_product(
                    [ions, aas, atoms], names=["ions", "aas", "atoms"]
                )
                self.other = None

            logger.info(f"Getting DataFrame")
            self.df = pd.DataFrame(index=idx, columns=cols)

            logger.info(f"Setting bins")
            self.xbin_step = Bins(
                np.round(np.abs(np.subtract.reduce(self.xbins[:2])), 2)
            )
            self.ybin_step = Bins(
                np.round(np.abs(np.subtract.reduce(self.ybins[:2])), 2)
            )
            logger.info(f"Setting cutoff")
            self.xcutoff = Cutoff(np.rint(self.xbins[-1]))
            self.ycutoff = Cutoff(np.rint(self.ybins[-1]))

            self._get_data(nameparts)
            # self.df.dropna(inplace=True, how='all', axis=0)

        setattr(self, self.df.columns.name, list(self.df.columns))

        for iid, i in enumerate(self.df.index.names):
            value = (
                self.df.index._get_level_values(level=iid).unique().tolist()
            )
            setattr(self, i, value)

        if odir != None:
            self.odir = Path(odir)
        else:
            self.odir = Path(".").cwd()

    def __repr__(self):
        return self.df.__repr__()

    def get_fname(self, fname):
        return Path(fname).with_suffix("tar.xz")

    def save(self, savename):
        savename = self.get_fname(savename)
        logger.info(f"Writing dataframe to {savename}")
        pd.to_pickle(savename)

    def _get_data(self, nameparts: int):
        idsl = pd.IndexSlice
        for f in self.filelist:
            namesplit = f.stem.split("_")
            name = namesplit[:nameparts]
            namesplit = namesplit[nameparts:]
            if self.other != None:
                other = namesplit[4]
                namesplit.pop(4)
                if other in self.ions:
                    other = "ions"
            try:
                clay, ion, aa, pH, atom, cutoff, bins = namesplit

                # clay, ion, aa, pH, cutoff, bins = f.stem.split(sep='_')
                # atom = 'ions'
                # assert cutoff == self.cutoff
                # assert bins == self.xbin_step

                with open(f, "rb") as file:
                    data = pkl.load(file)
                    if type(data) == dict:
                        data = list(data.values())
                    data = np.squeeze(data)
                    assert (
                        type(data) == np.ndarray
                    ), f"Expected array type, found {data.__class__.__name__!r}"
                try:
                    self.df.loc[idsl[ion, aa, atom], clay] = data
                except ValueError:
                    self.df.loc[idsl[ion, aa, atom, other], clay] = data
                except IndexError:
                    pass
                except ValueError:
                    pass
            except:
                pass
        self.name = "_".join(name)

    def plot(
        self,
        x: Literal["clays", "aas", "ions", "atoms", "other"],
        y: Literal["clays", "ions", "aas", "atoms", "other"],
        rowlabel: str = "y",
        columnlabel: str = "x",
        figsize=None,
        dpi=None,
        diff=False,
        xmax=50,
        ymax=50,
        save=False,
        xlim=None,
        ylim=None,
        cmap="magma",
        odir=".",
        plot_table=False,
    ):
        aas_classes = [
            ["arg", "lys", "his"],
            ["glu", "gln"],
            ["cys"],
            ["pro"],
            ["gly"],
            ["pro"],
            ["ala", "val", "ile", "leu", "met"],
            ["phe", "tyr", "trp"],
            ["ser", "thr", "asp", "gln"],
        ]
        ions_classes = [["Na", "Ca"], ["Ca", "Mg"]]
        atoms_classes = [["ions"], ["N"], ["OT"], ["CA"]]
        clays_classes = [["NAu-1"], ["NAu-2"]]
        cmaps_seq = ["Purples", "Blues", "Greens", "Oranges", "Reds"]
        cmaps_single = ["Dark2"]
        sel_list = ("clays", "ions", "aas", "atoms")
        # for color, attr in zip([''], sel_list):
        #     cmaps_dict[attr] = {}
        # cm.get_cmap()
        cmap_dict = {"clays": []}

        title_dict = {
            "clays": "Clay type",
            "ions": "Ion type",
            "aas": "Amino acid",
            "atoms": "Atom type",
            "other": "Other atom type",
        }

        sel_list = ["clays", "ions", "aas", "atoms"]
        if self.other != None:
            sel_list.append("other")

        separate = [s for s in sel_list if (s != x and s != y)]
        idx = pd.Index([s for s in sel_list if (s != x and s not in separate)])

        sep = pd.Index(separate)

        vx = getattr(self, x)

        if diff == True:
            vx = "/".join(vx)
            lx = 1
        else:
            lx = len(vx)

        vy = getattr(self, y)

        ly = len(vy)

        yid = np.ravel(np.where(np.array(idx) == y))[0]

        # label_key = idx.difference(pd.Index([x, y, sep]), sort=False).values[0]
        # label_id = idx.get_loc(key=label_key)
        # label_classes = locals()[f'{label_key}_classes']
        # cmap_dict = {}
        # single_id = 0
        # seq_id = 0
        # for category in label_classes:
        #     if len(category) == 1:
        #         cmap = matplotlib.cycler('color', cm.Dark2.colors)
        #         single_id += 1
        #     else:
        #         cmap = getattr(cm, cmaps_seq[seq_id])(np.linspace(0, 1, len(category)))
        #
        #         cmap = matplotlib.cycler('color', cmap)
        #         # cmap = cmap(np.linspace(0, 1, len(category))).colors
        #             #                                viridis(np.linspace(0,1,N)))
        #             # cm.get_cmap(cmaps_seq[seq_id], len(category))
        #         seq_id += 1
        #     for item_id, item in enumerate(category):
        #         cmap_dict[item] = cmap.__getitem__(item_id)
        #

        x_dict = dict(zip(vx, np.arange(lx)))

        if diff == True:
            diffstr = "diff"
            sel = "diff"
            self._get_densdiff()
        else:
            sel = self.clays
            diffstr = ""

        plot_df = self.df[sel].copy()
        plot_df.reset_index().set_index([*idx])

        if figsize == None:
            figsize = tuple(
                [
                    5 * lx if (10 * lx) < xmax else xmax,
                    5 * ly if (5 * ly) < ymax else ymax,
                ]
            )

        if dpi == None:
            dpi = 100

        iters = np.array(
            np.meshgrid(*[getattr(self, idxit) for idxit in idx])
        ).T.reshape(-1, len(idx))

        logger.info(f"Printing plots for {sep}\nColumns: {vx}\nRows: {vy}")

        label_mod = lambda l: ", ".join(
            [li.upper() if namei == "aas" else li for li, namei in l]
        )

        sep_it = np.array(
            np.meshgrid(*[getattr(self, idxit) for idxit in sep])
        ).T.reshape(-1, len(sep))

        for pl in sep_it:
            # try:
            #     fig.clear()
            # except:
            #     pass
            y_dict = dict(zip(vy, np.arange(ly)))
            # if separate == 'atoms' and pl != '':
            #     ...

            legends_list = [(a, b) for a in range(ly) for b in range(lx)]

            legends = dict(
                zip(legends_list, [[] for a in range(len(legends_list))])
            )

            # if type(pl) in [list, tuple, np.ndarray]:
            #     viewlist = []
            #     for p in pl:
            #         viewlist.append(plot_df.xs((p), level=separate, axis=0))
            #
            #     sepview = pd.concat(viewlist)
            #     plsave = 'ions'
            #
            # else:
            sepview = plot_df.xs(tuple(pl), level=sep.tolist(), axis=0)
            plsave = "_".join(pl)

            fig, ax = plt.subplots(
                nrows=ly,
                ncols=lx,
                figsize=figsize,
                sharey=True,
                dpi=dpi,
                constrained_layout=True,
            )

            fig.suptitle(
                (
                    ", ".join([title_dict[s].upper() for s in separate])
                    + f": {label_mod(list(tuple(zip(pl, separate))))}"
                ),
                size=16,
                weight="bold",
            )
            pi = 0
            for col in vx:
                try:
                    view = sepview.xs(col, axis=1)
                    pi = 1
                except ValueError:
                    view = sepview
                    col = vx
                    pi += 1
                for it in iters:
                    try:
                        values = view.xs(it[0])
                        # values = view.xs(tuple(it), level=idx.tolist()[0])#.reset_index(drop=False)
                        if type(values) == list:
                            values = np.squeeze(values)
                        if type(values) == np.ndarray:
                            values_array = values
                            data, xbins, ybins = values_array
                            if np.all(np.ravel(data)) >= 0:
                                levels = np.linspace(
                                    np.min(data), np.max(data), 50
                                )

                                try:
                                    x_id, y_id = x_dict[col], y_dict[it[yid]]
                                    ax[y_id, x_id].contourf(
                                        xbins, ybins, data, cmap=cmap
                                    )
                                except:
                                    x_id, y_id = 0, y_dict[it[yid]]
                                    ax[y_id].contourf(
                                        xbins, ybins, data, cmap=cmap
                                    )
                                if pi == 1:
                                    legends[y_id, x_id].append(
                                        it
                                    )  # [label_id])
                        else:
                            logger.info(f"Found {type(values)}: NaN values")
                    except KeyError:
                        logger.info(f"No data for {pl}, {vx}, {it}")

            for i in range(ly):
                try:
                    ax[i, 0].set_ylabel(
                        f"{label_mod([(vy[i], y)])}\n" + rowlabel
                    )
                    for j in range(lx):
                        # ax[i, j].legend([label_mod(leg, label_key) for leg in legends[i, j]], ncol=3)
                        if xlim != None:
                            ax[i, j].set_xlim((0.0, float(xlim)))
                        if ylim != None:
                            ax[i, j].set_ylim((0.0, float(ylim)))
                        ax[ly - 1, j].set_xlabel(
                            columnlabel + f"\n{label_mod([(vx[j], x)])}"
                        )
                except:
                    ax[i].set_ylabel(f"{label_mod([(vy[i], y)])}\n" + rowlabel)
                    # ax[i].legend([label_mod(leg, label_key) for leg in legends[i, 0]], ncol=3)
                    ax[ly - 1].set_xlabel(
                        columnlabel + f"\n{label_mod([(vx, x)])}"
                    )

            fig.supxlabel(f"{title_dict[x]}s", size=14)
            fig.supylabel(f"{title_dict[y]}s", size=14)
            if save != False:
                odir = Path(odir)
                print(odir.absolute())
                if not odir.is_dir():
                    os.makedirs(odir)
                if type(save) == str:
                    fig.savefig(str(odir) / f"{save}.png")
                else:
                    fig.savefig(
                        str(odir)
                        / f"{self.name}_{diffstr}_{x}_{y}_{plsave}_{self.cutoff}_{self.bins}.png"
                    )
            else:
                fig.show
                fig.clear()

    def make_df(self):
        new_df = self.df.copy()
        transformation = (
            lambda x: (make_1d(x[0]), x[1:]) if type(x) != float else x
        )
        new_df = new_df.applymap(transformation)
        return new_df


class Timeseries:
    def __init__(
        self, filename: Union[str, Path], axis=None, abs: bool = True
    ):
        self.__filename = None
        self.filename = filename
        self.axis = axis
        self.abs = abs
        self.__zfile = None
        self.has_zdata = False

    def __check_attr_set(self, attr):
        if getattr(self, attr) is None:
            logger.info(f'{attr.strip("_")} not set.')
            return None
        else:
            return True

    def get_bin_hist(self, hist, h_edges, edges):
        hist = hist.T
        z_edges, v_edges = (
            HistData(edges=h_edges[0]),
            HistData(edges=h_edges[1]),
        )
        new_hist = np.ma.array(hist, fill_value=np.NaN, mask=np.isnan(hist))
        mask = new_hist.mask.copy()
        combined_hist = []
        for a_idx, a_min in enumerate(edges[:-1]):
            a_max = edges[a_idx + 1]
            i_mask = np.logical_and(
                z_edges.bins > a_min, z_edges.bins <= a_max
            )
            i_mask = np.broadcast_to(i_mask[np.newaxis, :], new_hist.shape)
            new_hist.mask = i_mask | mask
            mean = np.apply_along_axis(
                lambda a: np.ma.multiply(a, v_edges.bins), arr=new_hist, axis=0
            )
            mean = np.ma.sum(mean)
            mean = np.ma.divide(mean, np.sum(new_hist), where=mean != np.nan)
            combined_hist.append(mean)
        return np.array(combined_hist), z_edges.bins

    @property
    def timeseries(self):
        with open(self.filename, "rb") as file:
            data = pkl.load(file)
        timeseries = np.ravel(data.timeseries).astype(np.float64)
        if self.abs is True:
            timeseries = np.abs(timeseries)
        return timeseries

    @property
    def filename(self):
        if self.__filename is not None:
            return self.__filename

    @filename.setter
    def filename(self, filename: Union[str, Path]):
        filename = Path(filename)
        if filename.is_file():
            self.__filename = filename.absolute()

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return self.__class__.__name__

    @property
    def zdata(self):
        if self.__zfile is not None:
            with open(self.__zfile, "rb") as file:
                data = pkl.load(file)
            zdata = np.ravel(data.timeseries).astype(np.float64)
            return zdata
        else:
            self.has_zdata = False
            logger.info(f"No z-density file specified.")

    @zdata.setter
    def zdata(self, zfile: Union[str, Path]):
        zfile = Path(zfile)
        if zfile.is_file():
            self.__zfile = zfile.absolute()
        self.has_zdata = True

    @property
    def z_bins(self):
        with open(self.__zfile, "rb") as file:
            data = pkl.load(file)
        return data.bins

    @property
    def bins(self):
        with open(self.filename, "rb") as file:
            data = pkl.load(file)
        return data.bins

    @property
    def hist2d(self):
        if self.__histogram is not None and self.__h_edges is not None:
            return self.__histogram, self.__h_edges
        else:
            raise AttributeError(
                "No histogram data found! " "Use make_hist2d to make histogram"
            )

    def make_hist2d(self, other=None, bins=None, z_bins=None):
        if self.__zfile is not None and self.__filename is not None:
            if other is not None:
                timeseries = np.divide(
                    self.timeseries,
                    other.timeseries,
                    where=other.timeseries != 0,
                )
                assert bins is not None
            else:
                timeseries = self.timeseries
            if bins is None:
                bins = self.bins
            if z_bins is None:
                z_bins = self.z_bins
            histogram, edges = np.histogramdd(
                [self.zdata, timeseries], [z_bins, bins], density=False
            )
            return histogram, edges


class HistData:
    __slots__ = ["min", "max", "nbins", "edges", "bins", "stepsize"]

    def __init__(
        self,
        min=None,
        max=None,
        nbins=None,
        edges=None,
        bins=None,
        stepsize=None,
    ):
        if bins is not None:
            self.bins = bins.copy()
            for attr in self.__class__.__slots__:
                if attr != "bins":
                    logger.info(f"{attr}")
                    assert eval(f"{attr} is None")
            logger.info(f"{self.bins}")
            self.stepsize = np.round(self.bins[1] - self.bins[0], 5)
            self.nbins = len(bins)
            self.edges = np.linspace(
                self.bins[0] - 1 / 2 * self.stepsize,
                self.bins[-1] + 1 / 2 * self.stepsize,
                self.nbins + 1,
            )
            self.min, self.max = np.min(self.edges), np.max(self.edges)
        elif edges is not None:
            for attr in self.__class__.__slots__:
                self.edges = edges.copy()
                if attr != "edges":
                    assert eval(f"{attr} is None")
            self.min, self.max = np.min(self.edges), np.max(self.edges)
            self.nbins = len(self.edges) - 1
            self.stepsize = self.edges[1] - self.edges[0]
            self.get_bins_from_edges()
        elif min is not None and max is not None and nbins is not None:
            assert stepsize is None
            self.min, self.max, self.nbins = float(min), float(max), int(nbins)
            self.edges = np.linspace(self.min, self.max, self.nbins + 1)
            self.stepsize = self.edges[1] - self.edges[0]
            self.get_bins_from_edges()

        elif min is not None and max is not None and stepsize is not None:
            self.min, self.max, self.stepsize = (
                float(min),
                float(max),
                float(stepsize),
            )
            self.edges = np.arange(
                self.min, self.max + self.stepsize, self.stepsize
            )
            self.get_bins_from_edges()
            self.nbins = len(self.bins)

        else:
            raise ValueError(
                f"No initialisation with selected arguments is implemented."
            )
        self.stepsize = Bins(self.stepsize)
        self.min, self.max = Cutoff(self.min), Cutoff(self.max)

    def get_bins_from_edges(self):
        try:
            self.bins = np.linspace(
                self.min + 0.5 * self.stepsize,
                self.max - 0.5 * self.stepsize,
                self.nbins,
            )
        except AttributeError:
            self.bins = np.arange(
                self.min + 0.5 * self.stepsize, self.max, self.stepsize
            )

    def __str__(self):
        return f"HistData([{self.min}:{self.max}:{self.stepsize}])"


class AtomTypeData(Data):
    """Class for analysing atom type data."""

    @redirect_tqdm
    def __init__(
        self,
        indir: Union[str, Path],
        cutoff: Union[int, float],
        bins: float,
        ions: List[Literal["Na", "K", "Ca", "Mg"]] = None,
        atoms: List[Literal["ions", "OT", "N", "CA", "OW"]] = None,
        other: List[Literal["ions", "OT", "N", "CA", "OW"]] = None,
        clays: List[Literal["NAu-1", "NAu-2"]] = None,
        aas: List[
            Literal[
                "ala",
                "arg",
                "asn",
                "asp",
                "ctl",
                "cys",
                "gln",
                "glu",
                "gly",
                "his",
                "ile",
                "leu",
                "lys",
                "met",
                "phe",
                "pro",
                "ser",
                "thr",
                "trp",
                "tyr",
                "val",
            ]
        ] = None,
        load: Union[str, Literal[False], Path] = False,
        odir: Optional[str] = None,
        nameparts: int = 1,
        namestem: str = "",
        analysis: Optional[str] = None,
        atomname=True,
    ):
        """Constructor method"""
        logger.info(f"Initialising {self.__class__.__name__}")
        self.filelist: list = []
        self.bins: Bins = Bins(bins)
        self.cutoff: float = Cutoff(cutoff)
        self.analysis: Union[str, None] = analysis

        if type(indir) != Path:
            indir = Path(indir)

        self._indir = indir

        if self.analysis is None:
            logger.info(
                rf"Getting {namestem}*_"
                rf"{self.cutoff}_"
                rf"{self.bins}.dat from {str(indir.resolve())!r}"
            )
            self.filelist: List[Path] = sorted(
                list(
                    indir.glob(
                        rf"{namestem}*_" rf"{self.cutoff}_" rf"{self.bins}.dat"
                    )
                )
            )
        else:
            logger.info(
                rf"Getting {namestem}*_"
                rf"{self.cutoff}_"
                rf"{self.bins}_"
                rf"{analysis}.dat from {str(indir.resolve())!r}"
            )
            self.filelist: List[Path] = sorted(
                list(
                    indir.glob(
                        rf"{namestem}*_"
                        rf"{self.cutoff}_"
                        rf"{self.bins}_"
                        rf"{self.analysis}.dat"
                    )
                )
            )
        logger.info(f"Found {len(self.filelist)} files.")

        if load != False:
            load = Path(load.resolve())
            self.df: pd.DataFrame = pkl.load(load)
            logger.info(f"Using data from {load!r}")
        else:
            if ions is None:
                ions = self.__class__.ions
                logger.info(
                    f"ions not specified, using default {self.__class__.ions}"
                )
            else:
                logger.info(f"Using custom {ions} for ions")
            if atoms is None:
                atoms = self.__class__.atoms
                logger.info(
                    f"atoms not specified, using default {self.__class__.atoms}"
                )
            else:
                logger.info(f"Using custom {atoms} for atoms")
            if atomname is False:
                assert len(atoms) == 1, "Expected one atom category"
                self.atomnames = atoms[0]
            else:
                self.atomnames = None
            if aas is None:
                aas = self.__class__.aas
                logger.info(
                    f"aas not specified, using default {self.__class__.aas}"
                )
            else:
                logger.info(f"Using custom {aas} for aas")
            if clays is None:
                clays = self.__class__.clays
                logger.info(
                    f"clays not specified, using default {self.__class__.clays}"
                )
            else:
                logger.info(f"Using custom {clays} for clays")

            # f = self.filelist[0]
            # print(f)

            # cols = pd.Index(["NAu-1", "NAu-2"], name="clays")

            if other is not None:
                if other is True:
                    other = atoms
                    other.append("OW")
                # idx = pd.MultiIndex.from_product(
                #     [ions, aas, atoms, other],
                #     names=["ions", "aas", "atoms", "other"],
                # )
                self.other: List[str] = other
                logger.info(f"Setting second atom selection to {self.other}")
            else:
                # idx = pd.MultiIndex.from_product(
                #     [ions, aas, atoms, x], names=["ions", "aas", "atoms"]
                # )
                self.other: None = None
            # self.df: pd.DataFrame = pd.DataFrame(index=idx, columns=cols)

            self._get_data(nameparts)

        self.df.dropna(inplace=True, how="all", axis=0)
        self.df.dropna(inplace=True, how="all", axis=1)

        # self.df.reset_index(level=["ions", "atoms"], inplace=True)
        # self.df["_atoms"] = self.df["atoms"].where(
        #     self.df["atoms"] != "ions", self.df["ions"], axis=0
        # )
        # self.df.set_index(["ions", "atoms", "_atoms"], inplace=True, append=True)
        # self.df.index = self.df.index.reorder_levels([*idx.names, "_atoms"])
        # self._atoms = self.df.index.get_level_values("_atoms").tolist()
        # self.df["x_bins"] = np.NaN
        # self.df.set_index(["x_bins"], inplace=True, append=True)

        for iid, i in enumerate(self.df.index.names):
            value: List[Union[str, float]] = (
                self.df.index._get_level_values(level=iid).unique().tolist()
            )
            logger.info(f"Setting {i} to {value}")
            setattr(self, i, value)

        if odir is not None:
            self.odir: Path = Path(odir)
        else:
            self.odir: Path = Path(".").cwd()

        logger.info(f"Output directory set to {str(self.odir.resolve())!r}\n")
        self._bin_df = pd.DataFrame(columns=self.df.columns)

        self._edges = {}
        self._peaks = {}

    @staticmethod
    def get_labels(f: Union[Path, str]) -> np.ndarray:
        """Get labels from file"""
        with open(f, "r") as file:
            labels = file.read()
        labels = re.search(
            r"^.*atypes_flat.*?\[([A-Za-z0-9\"'#\n\s]+)\]\n",
            labels,
            flags=re.MULTILINE | re.DOTALL,
        )
        labels = labels.group(1)
        labels = re.sub(r"['\"\s]*\n#[\s'\"]*", " ", labels)
        labels = labels.split(" ")
        labels = list(map(lambda x: x.strip("'").strip('"'), labels))
        return np.array(labels)

    @redirect_tqdm
    def _get_data(self, nameparts: int) -> None:
        """Get data from files
        :param nameparts: number of `_`-separated partes in `namestem`
        :type nameparts: int
        """
        idsl = pd.IndexSlice
        cols = pd.Index(["NAu-1", "NAu-2"], name="clays")
        for f_id, f in enumerate(self.filelist):
            namesplit = f.stem.split("_")
            if self.analysis is not None:
                namesplit.pop(-1)
            else:
                self.analysis = "zdist"
            name = namesplit[:nameparts]
            namesplit = namesplit[nameparts:]
            if self.other is not None:
                # other = namesplit[5]
                other = namesplit.pop(5)
                if other in self.ions:
                    other = "ions"
            try:
                if self.atomnames is None:
                    clay, ion, aa, pH, atom, cutoff, bins = namesplit
                else:
                    clay, ion, aa, pH, cutoff, bins = namesplit
                    atom = self.atomnames
                assert cutoff == self.cutoff
                assert bins == self.bins
                array = pd.read_csv(
                    f, delimiter="\s+", header=None, comment="#"
                ).to_numpy()
                labels = self.get_labels(f)

                if self.other is not None:
                    if other is True:
                        other = atom
                        other.append("OW")
                        idx = pd.MultiIndex.from_product(
                            [[clay], [ion], [aa], [atom], [other], labels],
                            names=[
                                "clays",
                                "ions",
                                "aas",
                                "atoms",
                                "other",
                                "x",
                            ],
                        )
                        logger.info(
                            f"Setting second atom selection to {self.other}"
                        )
                else:
                    idx = pd.MultiIndex.from_product(
                        [[clay], [ion], [aa], [atom], labels],
                        names=["clays", "ions", "aas", "atoms", "x"],
                    )
                df: pd.DataFrame = pd.DataFrame(index=idx, columns=["values"])
                try:
                    df.loc[idsl[clay, ion, aa, atom, :], "values"] = array[
                        :, 2
                    ]
                except ValueError:
                    df.loc[
                        idsl[clay, ion, aa, atom, other, :], "values"
                    ] = array[:, 2]
                except IndexError:
                    try:
                        df.loc[idsl[clay, ion, aa, atom, :]] = array[:, 1]
                    except ValueError:
                        df.loc[idsl[clay, ion, aa, atom, other, :]] = array[
                            :, 1
                        ]
                except KeyError:
                    pass
                if f_id != 0:
                    self.df = pd.concat([self.df, df.copy()])
                    del df
                else:
                    self.df = df.copy()
                    del df
            except IndexError:
                logger.info(f"Encountered IndexError while getting data")
            except ValueError:
                logger.info(f"Encountered ValueError while getting data")

        self.name = "_".join(name)


class AtomTypeData2D(Data2D, Data):
    @redirect_tqdm
    def __init__(
        self,
        indir: Union[str, Path],
        cutoff: Union[int, float],
        bins: float,
        ions: List[Literal["Na", "K", "Ca", "Mg"]] = None,
        atoms: List[Literal["ions", "OT", "N", "CA", "OW"]] = None,
        other: List[Literal["ions", "OT", "N", "CA", "OW"]] = None,
        clays: List[Literal["NAu-1", "NAu-2"]] = None,
        aas: List[
            Literal[
                "ala",
                "arg",
                "asn",
                "asp",
                "ctl",
                "cys",
                "gln",
                "glu",
                "gly",
                "his",
                "ile",
                "leu",
                "lys",
                "met",
                "phe",
                "pro",
                "ser",
                "thr",
                "trp",
                "tyr",
                "val",
            ]
        ] = None,
        load: Union[str, Literal[False], Path] = False,
        odir: Optional[str] = None,
        nameparts: int = 1,
        namestem: str = "",
        analysis: str = None,
        atomnames: str = None,
        save=None,
        overwrite=False,
        new_bins=None,
        new_cutoff=None,
        group_all=False,
        split=True,
    ):
        """Constructor method"""
        logger.info(f"Initialising {self.__class__.__name__}")
        super().__init__(
            indir=indir,
            zdir=indir,
            cutoff=50,
            bins=0.02,
            atoms=atoms,
            clays=clays,
            ions=ions,
            aas=aas,
            namestem=namestem,
            nameparts=nameparts,
            other=other,
            analyses=[atomnames, analysis],
            zstem=namestem,
            zname=atomnames,
            atomname=False,
        )
        self.atomnames = atomnames
        self.analysis = analysis
        self.df.index = self.df.index.droplevel(["atoms", "_atoms"])
        loadfname = Path(__file__).parent / f"../run_data/{self.analysis}_df.p"
        self.odir = Dir(odir, check=False)
        if not self.odir.is_dir():
            os.makedirs(self.odir)
            logger.info(f"Initialising output directory: {self.odir!r}")
        if split is True:
            if save is False:
                pass
                # if load is True:
                #     loadfile = loadfname
                # else:
                #     loadfile = Path(load)
                # assert loadfile.is_file(), f'No file named {loadfile.name!r} exists.'
                # with open(loadfile, 'rb') as file:
                #     df = pkl.load(file)
                #     assert type(df) == pd.DataFrame, f'Loaded object is {type(df)}, expected pandas.DataFrame'
                #     self.df = df
            else:
                self.split_dfs(
                    overwrite=overwrite, cutoff=new_cutoff, bins=new_bins
                )
        # if save is True:
        #     with open(loadfname, 'wb') as file:
        #         pkl.dump(self.df, file)
        # self.df.index = self.df.index.droplevel('atom_ids')
        # self._process_atom_type_idx(self.df.index)
        # self._init_xbin_idx()
        # self.df
        # self.atoms = self.df.index.get_level_values('atoms')
        # self._edges = {}
        # self._peaks = {}

    @staticmethod
    def __get_dat_info(
        datfile,
        info: Union[Tuple[str], List[str]],
        types: Union[Tuple[str], List[str]],
    ):
        with open(datfile, "r") as file:
            filestr = file.read()
        match_list = []
        for item, itype in zip(info, types):
            match = re.search(
                rf".*{item}.*?({itype}*)\n",
                filestr,
                flags=re.MULTILINE | re.DOTALL,
            )
            match_list.append(match.group(1))
        if len(match_list) != 1:
            match_list = tuple(match_list)
        else:
            match_list = match_list[0]
        return match_list
        # n_frames = int(n_frames.group(1))
        # n_atoms = re.search(r"^.*sel_n_atoms.*?(\d+)\n", filestr, flags=re.MULTILINE | re.DOTALL)
        # n_atoms = int(n_atoms.group(1))

    def yield_first_analysis_group_filename(self, group_key):
        for group, df in self.df.groupby(group_key, group_keys=True):
            analysis = df[self.analysis].head(1).values[0]
            datfile = analysis.filename.with_suffix(".dat")
            yield group, datfile

    def get_atom_id_groups(self, labels, ids):
        at_group_dict = {}
        vec_group = np.vectorize(
            get_atom_type_group, excluded=["group_all_atoms"]
        )  # , signature='(m)->(m)', otypes=[str])
        labels = np.array(labels)
        groups = vec_group(labels, group_all_atoms=True)
        at_types, groups = np.unique(groups, return_inverse=True)
        for group in np.unique(groups):
            at_group_dict[at_types[group]] = ids[groups == group]
        return at_group_dict

    @redirect_tqdm
    def split_dfs(self, bins=None, cutoff=None, overwrite=False):
        if cutoff is None:
            cutoff = self.cutoff
        else:
            cutoff = Cutoff(cutoff)
        if bins is None:
            bins = self.bins
        else:
            bins = Bins(bins)
        idsl = pd.IndexSlice
        df_list = []
        binfile = self.df[self.analysis].iloc[0].filename.with_suffix(".dat")
        n_frames = int(
            self.__get_dat_info(binfile, ["n_frames"], types=["\d"])
        )
        # self.hist_bins = pd.read_csv(binfile, delimiter="\s+", comment="#", header=None).to_numpy()[:, 0]
        mol_n_atoms = {}
        for clay, clay_group_file in self.yield_first_analysis_group_filename(
            "clays"
        ):
            mol_n_atoms[clay] = int(
                self.__get_dat_info(clay_group_file, ["sel_n_atoms"], ["\d"])
            )
        mol_atom_labels = {}
        atom_group_labels = {}
        for group, aa_group_file in self.yield_first_analysis_group_filename(
            "aas"
        ):
            labels = AtomTypeData.get_labels(aa_group_file)
            ids = np.arange(start=0, stop=len(labels), step=1)
            mol_atom_labels[group] = (labels, ids)
            atom_group_labels[group] = self.get_atom_id_groups(
                labels=labels, ids=ids
            )

        # idx_arr = self.df.index.to_frame.to_numpy()
        # idx_arr = np.array([tuple(zip(*idx_arr))], dtype=dtypes)
        # dtypes = list(zip([*self.df.index.names, 'atom_ids', 'atoms', 'x'], ['S10', 'S10', 'S10', 'u4', 'S10', 'f8']))
        self.df.sort_index(level="aas", sort_remaining=True, inplace=True)
        # prev_aa, prev_clay = 'X', 'Y'
        for idx, val in tqdm(
            self.df.iterrows(),
            total=self.df.shape[0],
            leave=False,
            position=0,
            desc="rows",
        ):
            clay, ion, aa = idx
            atoms = val[self.atomnames]
            analysis = val[self.analysis]
            # if aa != prev_aa:
            labels, label_ids = mol_atom_labels[aa]
            group_labels = atom_group_labels[aa]
            # new_idx = np.broadcast_to(np.array(idx)[:, np.newaxis], (len(idx), len(label_ids)))
            # new_idx = np.vstack([new_idx, np.atleast_2d(label_ids), np.atleast_2d(labels)])
            # new_idx_repeated = np.apply_along_axis(self.__repeat_idx, arr=new_idx, axis=1)
            # bins_repeated = np.tile(A=self.hist_bins.T, reps=len(new_idx.T))
            # new_idx = np.column_stack([new_idx_repeated.T, bins_repeated])
            # # new_idx = list(tuple(zip(*new_idx.T)))
            # # new_idx = np.array(arr, dtype=dtypes)
            # new_idx = pd.MultiIndex.from_arrays(new_idx.T, names=[*self.df.index.names, 'atom_ids', 'atoms', 'x'],
            #                                 )
            # new_df = pd.DataFrame(index=new_idx, columns=[self.analysis])
            n_atoms = mol_n_atoms[clay]
            for (
                atom_type,
                group_ids,
            ) in (
                group_labels.items()
            ):  # , leave=False, position=1, desc='atypes'):
                savename = (
                    self.odir
                    / f"{self.name}_{clay}_{ion}_{aa}_{self.pH}_{atom_type}_{cutoff}_{bins}"
                )
                outname = Path(f"{savename}_{self.analysis}.dat")
                if overwrite is True or not outname.is_file():
                    logger.info(f"Getting {outname.stem} histogram")
                    new_ts = analysis.timeseries[
                        np.isin(atoms.timeseries, group_ids)
                    ]
                    data = AnalysisData(
                        name=self.analysis,
                        cutoff=cutoff.num,
                        bin_step=bins.num,
                    )
                    data.timeseries = new_ts
                    data.get_hist_data(n_frames=n_frames)
                    data.get_norm(n_atoms=n_atoms)
                    data.get_df()
                    data.save(
                        savename,
                        n_frames=n_frames,
                        sel_n_atoms=n_atoms,
                        atypes_flat=labels,
                        acodes_flat=label_ids,
                        label=atom_type,
                        group_atypes=labels[np.isin(label_ids, group_ids)],
                        group_ids=group_ids,
                    )
            # prev_aa = aa
            # prev_clay = clay
            # hist, edges = np.histogram(new_ts, bins=[*self.hist_bins, self.cutoff.num])
            # hist = np.divide(hist, n_frames * n_atoms)
            # new_df.loc[idsl[clay, ion, aa, str(atom_id), atom_type, :], self.analysis] = hist
            # new_df = new_df.groupby(new_df.index.names).sum()
            # df_list.append(new_df.copy())
            # del new_df
            # if combined_df is None:
            #     combined_df = new_df.copy()
            #     del new_df
            # else:
            #     combined_df = pd.concat([combined_df.copy(), new_df.copy()])
            #     del new_df
        # combined_df = pd.concat(df_list)
        # self.df = combined_df.copy()
        self.ignore_density_sum = True

    # @redirect_tqdm
    # def split_dfs(self):
    #     idsl = pd.IndexSlice
    #     df_list = []
    #     datfile = self.df[self.analysis].iloc[0].filename.with_suffix('.dat')
    #     with open(datfile, 'r') as file:
    #         filestr = file.read()
    #         n_frames = int(re.search(r"^.*n_frames.*?(\d+)\n", filestr, flags=re.MULTILINE | re.DOTALL).group(1))
    #         n_atoms = int(re.search(r"^.*sel_n_atoms.*?(\d+)\n", filestr, flags=re.MULTILINE | re.DOTALL).group(1))
    #         hist_bins = pd.read_csv(datfile, delimiter="\s+", comment="#", header=None).to_numpy()[:, 0]
    #     for idx, val in tqdm(self.df.iterrows(), total=self.df.shape[0], leave=False, position=0, desc='rows'):
    #         clay, ion, aa = idx
    #         atoms = val[self.atomnames]
    #         analysis = val[self.analysis]
    #         datfile = analysis.filename.with_suffix('.dat')
    #         labels = AtomTypeData.get_labels(datfile)
    #         del datfile
    #         atom_ids = np.arange(len(labels))
    #         new_idx_vals = np.hstack([np.array(idx), np.arange(len(labels)), labels])
    #         new_idx_repeated = np.apply_along_axis(self.__repeat_idx, arr=new_idx_vals, axis=1)
    #         bins_repeated = np.tile(A=hist_bins.T, reps=len(new_idx_vals))
    #         new_idx = np.column_stack([new_idx_repeated.T, bins_repeated])
    #         new_idx = pd.MultiIndex.from_arrays(new_idx.T, names=[*self.df.index.names, 'atom_ids', 'atoms', 'x'])
    #         new_df = pd.DataFrame(index=new_idx, columns=[self.analysis])
    #         atom_timeseries = atoms.timeseries[:, np.newaxis] == atom_ids[np.newaxis, :]
    #         atom_hist = np.apply_along_axis(lambda x: np.histogram(analysis.timeseries[x], hist_bins=hist_bins)[0],
    #                                         arr=atom_timeseries, axis=0)
    #         atom_hist = atom_hist / (n_frames + n_atoms)
    #         new_df.loc[idsl[clay, ion, aa, :, :, :], self.analysis] = atom_hist.reshape(-1, len(hist_bins)).tolist()
    #         new_df = new_df.groupby(new_df.index.names).sum()
    #         df_list.append(new_df.copy())
    #         del new_df
    #     #     if combined_df is None:
    #     #         combined_df = new_df.copy()
    #     #     else:
    #     #         combined_df = pd.concat([combined_df.copy(), new_df.copy()])
    #     #     del new_df
    #     combined_df = pd.concat(df_list)
    #     self.df = combined_df.copy()

    # plt.plot(self.hist_bins, hist)
    # plt.show()
    # plt.legend()

    def __repeat_idx(self, arr, *args, **kwargs):
        return np.repeat(a=arr, repeats=len(self.hist_bins))

        # combined_df = pd.DataFrame({self.atomnames: self.zf,
        #                              self.analysis: self.df[self.analysis]}, index = self.df.index)
        # self.filelist: list = []
        # self.bins: Bins = Bins(bins)
        # self.cutoff: float = Cutoff(cutoff)
        # self.analysis: str = analysis
        # self.atomnames: str = atomnames
        # assert self.analysis is not None
        # assert self.atomnames is not None
        # if type(indir) != Path:
        #     indir = Path(indir)
        #
        # self.filelist_dict = {}
        #
        # self._indir = indir
        # for filetype in [self.analysis, self.atomnames]:
        #     logger.info(
        #         rf"Getting {namestem}*_"
        #         rf"{self.cutoff}_"
        #         rf"{self.bins}_"
        #         rf"{analysis}.dat from {str(indir.resolve())!r}"
        #     )
        #     self.filelist: List[Path] = sorted(
        #         list(
        #             indir.glob(
        #                 rf"{namestem}*_"
        #                 rf"{self.cutoff}_"
        #                 rf"{self.bins}_"
        #                 rf"{self.analysis}.dat"
        #             )
        #         )
        #     )
        #     logger.info(f"Found {len(self.filelist)} {filetype} data files.")
        #
        # if load is not False:
        #     load = Path(load.resolve())
        #     self.df: pd.DataFrame = pkl.load(load)
        #     logger.info(f"Using data from {load!r}")
        # else:
        #     if ions is None:
        #         ions = self.__class__.ions
        #         logger.info(f"ions not specified, using default {self.__class__.ions}")
        #     else:
        #         logger.info(f"Using custom {ions} for ions")
        #     if atoms is None:
        #         atoms = self.__class__.atoms
        #         logger.info(
        #             f"atoms not specified, using default {self.__class__.atoms}"
        #         )
        #     else:
        #         logger.info(f"Using custom {atoms} for atoms")
        #     if atomname is False:
        #         assert len(atoms) == 1, 'Expected one atom category'
        #         self.atomnames = atoms[0]
        #     else:
        #         self.atomnames = None
        #     if aas is None:
        #         aas = self.__class__.aas
        #         logger.info(f"aas not specified, using default {self.__class__.aas}")
        #     else:
        #         logger.info(f"Using custom {aas} for aas")
        #     if clays is None:
        #         clays = self.__class__.clays
        #         logger.info(
        #             f"clays not specified, using default {self.__class__.clays}"
        #         )
        #     else:
        #         logger.info(f"Using custom {clays} for clays")
        #
        #     if other is not None:
        #         if other is True:
        #             other = atoms
        #             other.append("OW")
        #
        #             self.other: List[str] = other
        #             logger.info(f"Setting second atom selection to {self.other}")
        #         else:
        #             # idx = pd.MultiIndex.from_product(
        #             #     [ions, aas, atoms, x], names=["ions", "aas", "atoms"]
        #             # )
        #             self.other: None = None
        #         # self.df: pd.DataFrame = pd.DataFrame(index=idx, columns=cols)
        #
        #     self._get_data(nameparts)
        #
        # self.df.dropna(inplace=True, how="all", axis=0)

    # @redirect_tqdm
    # def get_data(self, nameparts):
    #     idsl = pd.IndexSlice
    #     for f in self.filelist_dict[self.atomnames]:
    #         namesplit = f.stem.split("_")
    #         namesplit.pop(-1)
    #         name = namesplit[:nameparts]
    #         namesplit = namesplit[nameparts:]
    #         if self.other is not None:
    #             # other = namesplit[5]
    #             other = namesplit.pop(5)
    #             if other in self.ions:
    #                 other = "ions"
    #         try:
    #             if self.atomnames is None:
    #                 clay, ion, aa, pH, atom, cutoff, bins = namesplit
    #             else:
    #                 clay, ion, aa, pH, cutoff, bins = namesplit
    #                 atom = self.atomnames
    #             assert cutoff == self.cutoff
    #             assert bins == self.bins
    #             with open(f, 'rb') as file:
    #                 arr = pkl.load(f)
    #                 # arr = np.ravel()


def read_edge_file(
    fname: Union[str, PathType],
    cutoff: Union[int, str, float, Cutoff] = None,
    skip=False,
    edge_type: Optional[str] = "",
):
    fname = File(fname, check=False)
    if edge_type != "":
        edge_type = f"{edge_type}-"
    if not fname.exists():
        logger.info("No edge file found.")
        # os.makedirs(fname.parent, exist_ok=True)
        # logger.info(f"{fname.parent}")
        if skip is True:
            logger.info(f"Continuing without {edge_type}edges")
            if cutoff is None:
                logger.finfo(
                    f"Cutoff required when skipping {edge_type}edges!\nAborting."
                )
                sys.exit(1)
            p = [0, float(cutoff)]
        else:
            raise FileNotFoundError(f"No edge file found {fname.name!r}.")
    else:
        with open(fname, "rb") as edges_file:
            logger.finfo(f"Reading {edge_type}edges {fname.name!r}")
            p = pkl.load(edges_file)["edges"]
        logger.finfo(
            ", ".join(list(map(lambda e: f"{e:.2f}", p))),
            kwd_str=f"{edge_type}edges: ",
            indent="\t",
        )
    return p


def get_edge_fname(
    atom_type: str,
    cutoff: Union[int, str, float],
    bins: Union[int, str, float],
    other: Optional[str] = None,
    path: Union[str, PathType] = PE_DATA,
    name: Union[Literal["pe"], Literal["edge"]] = "pe",
    ls_regex: bool = False,
):
    if other is not None:
        other = f"{other}_"
    else:
        other = ""
    if cutoff is None:
        cutoff = "*"
    else:
        cutoff = Cutoff(cutoff)
    bins = Bins(bins)
    if ls_regex is True:
        cutoff = "{" + f"{cutoff}" + "..99}"
        bins = "{01.." + f"{bins}" + "}"
        regex = " regex"
    else:
        regex = ""
    # fname = Path.cwd() / f"edge_data/edges_{atom_type}_{self.cutoff}_{self.bins}.p"
    fname = (
        Dir(path) / f"{atom_type}_{other}{name}_data_{cutoff}_{bins}.p"
    ).resolve()
    logger.finfo(
        f"Peak/edge filename{regex}: {fname.name!r}",
        initial_linebreak=True,
        indent="\t",
    )
    return Path(fname)
