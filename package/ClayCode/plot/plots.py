#!/usr/bin/env python3
from __future__ import annotations

import copy
import itertools
import logging
import re
import sys
from argparse import ArgumentParser
from functools import wraps
from pathlib import Path

import matplotlib.patches
import scipy.optimize
from ClayCode.analysis.analysisbase import AnalysisData
from ClayCode.analysis.utils import gauss_peak, heavyside_func, n_gauss_peaks
from ClayCode.core.classes import Dir
from numpy.typing import NDArray
from scipy.signal import find_peaks

# import dask as da
# import dask.dataframe

# from dask.diagnostics import ProgressBar


logger = logging.getLogger(Path(__file__).stem)
logger.setLevel(logging.DEBUG)

from scipy.interpolate import make_smoothing_spline
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

tqdm.pandas(desc="pandas")
import logging
import os
import pickle as pkl
from abc import ABC, abstractmethod
from collections import UserString
from functools import cached_property, partialmethod
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import matplotlib.colors as mpc
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ClayCode.analysis.dataclasses import Data
from ClayCode.analysis.utils import _check_methods, redirect_tqdm
from ClayCode.plots.dataclasses import Bins, Cutoff
from matplotlib import colormaps
from matplotlib.table import table

# from dask.diagnostics import ProgressBar

__all__ = [
    "Plot",
    "Axis",
    "LinePlot",
    "SDevLinePlot",
    "HistPlot",
    "HistPlot2D",
    "Data",
]
# from blume.table import table

logger = logging.getLogger(Path(__file__).stem)

logger.setLevel(logging.DEBUG)


# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
# logger.addHandler(ch)
# __all__ = ["Histogram", "Data", "BarPlot"]

# __name__ = 'plots'

# from collections.abc library


# pbar = ProgressBar()
# pbar.register() # global registration


# ------------------------------------------------------------------------------

# class Histogram(Plot):
#     __slots__ = ('data', 'x', 'y', 'n_bins', 'bin_size', 'xmin', 'xmax', 'ymin', 'ymax',
#                  'labels', 'xlabels', 'ylabels', 'xformat', 'yformat', 'figsize', 'dpi',
#                  )
#
#     _figargs = {'figsize', 'dpi', 'facecolor', 'edgecolor', 'linewidth', 'frameon',
#                 'subplotpars'}
#
#     def _process_data(self):
#         if not hasattr(self, 'figsize'):
#             figsize = None
#         if not hasattr(self, 'xmin'):
#             self.xmin = 0
#         if self.data.ndim == 1:
#             self.y = self.data
#             self.x = np.linspace(self.xmin, self.xmax, len(self.y), endpoint=True)
#         elif self.data.ndim == 2:
#             self.x = self.data[0]
#             self.y = self.data[1]
#
#     def plot(self):
#         print(self.x, self.y)
#         figargs = {}
#         for arg in self._figargs:
#             if hasattr(self, arg):
#                 figargs[arg] = getattr(self, arg)
#         fig, ax = plt.subplots(**figargs)
#         ax.plot(self.x, self.y)


# def make_1d(arr: NDArray):
#     arr_list = []
#     idxs = np.arange(arr.ndim)
#     for idx in idxs:
#         dimarr = np.add.reduce(arr, axis=[idxs == idx])
#         arr_list.append(dimarr)
#     return arr_list


# class Plot(ABC):
#     # __slots__ = ('data')
#     sel_list = None
#     title_dict = None
#     cmaps = None
#     _init_args = {}
#     _plot_args = {}
#
#     # def __init_subclass__(cls, data, **kwargs):
#     #     cls.data = data
#
#     def __repr__(self):
#         return f'{self.__class__.__name__!r} ({len(self.data.dropna(how="all"))} entries)'
#
#     @classmethod
#     def __subclasshook__(cls, subclass):
#         if cls is Plot:
#             return _check_methods(
#                 subclass, "plot", "save", "_process_data", "plot_df"
#             )
#
#     @redirect_tqdm
#     def __init__(
#         self,
#         data: Data,
#         # lx: int,
#         # ly: int,
#         # xmax: int,
#         # ymax: int,
#         **kwargs,
#     ):
#         logger.info(f"{self.__class__.__name__}")
#         # assert (
#         #     type(data).__name__ == DataD.__name__
#         # ), f"expected {Data.__name__}, found {type(data).__name__}"
#         self.data = data
#         init_args = {}
#         logger.info(f"{kwargs.keys()}")
#         for key, val in kwargs.items():
#             try:
#                 setattr(self, key, val)
#             except AttributeError:
#                 logger.error(
#                     f"{key} cannot be set for {self.__class__.__name__}"
#                 )
#             if key in self.__class__._init_args:
#                 init_args[key] = val
#         self._process_data(**init_args)
#         logger.info(f"{self.__class__.__name__}")
#
#     @abstractmethod
#     def _process_data(self, **kwargs):
#         pass
#
#     @abstractmethod
#     def plot(self, **kwargs):
#         pass
#
#     @abstractmethod
#     def save(self, **kwargs):
#         pass
#
#     def get_figsize(self, xmax: int, ymax: int):
#         logger.info(f"Setting figsize to x > {xmax}," f" y > {ymax}")
#         return tuple(
#             [
#                 6 * self.x.l if (6 * self.x.l) < xmax else xmax,
#                 7 * self.y.l if (7 * self.y.l) < ymax else ymax,
#             ]
#         )
#
#     def _get_idx_iter(self, idx):
#         # logger.info(f'idx: {idx}')
#         idx_iter = np.array(
#             np.meshgrid(
#                 *[
#                     self.plot_df.index.get_level_values(idxit).unique()
#                     for idxit in idx
#                 ]
#             )
#         ).T.reshape(-1, len(idx))
#         # logger.info(f'{idx_iter}')
#         return idx_iter
#
#     @staticmethod
#     def modify_plot_labels(l: Tuple[str, str]):
#         """
#
#         :param l: tuple of label category and value
#         :type l: Tuple[str, str]
#         :return: label value
#         :rtype: str
#         """
#         # logging.info(f'{l}, {len(l)}')
#         return ", ".join(
#             [li.upper() if namei == "aas" else li for li, namei in l]
#         )
#
#     def init_legend(self, lx, ly):
#         # initialise legends
#         legends_list: list = [(a, b) for a in range(ly) for b in range(lx)]
#         self.legends: dict = dict(
#             zip(legends_list, [[] for a in range(len(legends_list))])
#         )
#
#     def _get_attr(self, attr: Literal["clays", "aas", "atoms", "ions"]):
#         return pd.unique(self.plot_df.index.get_level_values(attr))
#
#     get_clays = partialmethod(_get_attr, attr="clays")
#     get_aas = partialmethod(_get_attr, attr="aas")
#     get_ions = partialmethod(_get_attr, attr="ions")
#     get_atoms = partialmethod(_get_attr, attr="atoms")
#     get_other = partialmethod(_get_attr, attr="other")
#
#     @property
#     @abstractmethod
#     def plot_df(self):
#         ...
#
#     def _init_plot(self, x, y, select):
#         sel_list = ["clays", "aas", "atoms", "ions"]
#
#         if self.select == "ions":
#             sel_list.pop(-1)
#
#         if self.data.other != None:
#             sel_list.append("other")
#
#         assert x in sel_list, f"{x} not in sel list"
#         assert y in sel_list, f"{y} not in sel list"
#         assert select in sel_list, f"{select} not in sel list"
#
#     # @abstractmethod
#     # def plot_ions(self, *args, **kwargs):
#     #     ...
#     #
#     # @abstractmethod
#     # def plot_other(self, *args, **kwargs):
#     #     ...


class Plot(ABC):
    # __slots__ = ('data')
    sel_list = None
    title_dict = None
    cmaps = None
    _init_args = {}
    _plot_args = {}

    # def __init_subclass__(cls, data, **kwargs):
    #     cls.data = data

    def __repr__(self):
        return f'{self.__class__.__name__!r} ({len(self.data.dropna(how="all"))} entries)'

    @classmethod
    def __subclasshook__(cls, subclass):
        if cls is Plot:
            return _check_methods(
                subclass, "plot", "save", "_process_data", "plot_df"
            )

    def plot_args_decorator(select: str):
        def plot_decorator(plot_f):
            def wrapper(self, **kwargs):
                self.select = select
                plot_f(self, **kwargs)
                self.select = None

            return wrapper

        return plot_decorator

    @redirect_tqdm
    def __init__(
        self,
        data: Data,  # lx: int,
        # ly: int,
        # xmax: int,
        # ymax: int,
        **kwargs,
    ):
        logger.info(f"{self.__class__.__name__}")
        # assert (
        #     type(data).__name__ == DataD.__name__
        # ), f"expected {Data.__name__}, found {type(data).__name__}"
        self.data = data
        init_args = {}
        logger.info(f"{kwargs.keys()}")
        for key, val in kwargs.items():
            try:
                setattr(self, key, val)
            except AttributeError:
                logger.error(
                    f"{key} cannot be set for {self.__class__.__name__}"
                )
            if key in self.__class__._init_args:
                init_args[key] = val
        self._process_data(**init_args)
        logger.info(f"{self.__class__.__name__}")

    @abstractmethod
    def _process_data(self, **kwargs):
        pass

    @abstractmethod
    def plot(self, **kwargs):
        pass

    @abstractmethod
    def save(self, **kwargs):
        pass

    def get_figsize(self, xmax: int, ymax: int):
        logger.info(f"Setting figsize to x > {xmax}," f" y > {ymax}")
        return tuple(
            [
                6 * self.x.l if (6 * self.x.l) < xmax else xmax,
                7 * self.y.l if (7 * self.y.l) < ymax else ymax,
            ]
        )

    def _get_idx_iter(self, idx):
        # logger.info(f'idx: {idx}')
        idx_iter = np.array(
            np.meshgrid(
                *[
                    self.plot_df.index.get_level_values(idxit).unique()
                    for idxit in idx
                ]
            )
        ).T.reshape(-1, len(idx))
        # logger.info(f'{idx_iter}')
        return idx_iter

    @staticmethod
    def modify_plot_labels(l: Tuple[str, str]):
        """

        :param l: tuple of label category and value
        :type l: Tuple[str, str]
        :return: label value
        :rtype: str
        """
        # logging.info(f'{l}, {len(l)}')
        new_str = ", ".join(
            [li.upper() if namei == "aas" else li for li, namei in l]
        )
        new_str = re.sub("_", " ", new_str)
        return new_str

    def init_legend(self, lx, ly):
        # initialise legends
        legends_list: list = [(a, b) for a in range(ly) for b in range(lx)]
        self.legends: dict = dict(
            zip(legends_list, [[] for a in range(len(legends_list))])
        )
        self.handles = dict(
            zip(legends_list, [[] for a in range(len(legends_list))])
        )

    def _get_attr(self, attr: Literal["clays", "aas", "atoms", "ions"]):
        return pd.unique(self.plot_df.index.get_level_values(attr))

    get_clays = partialmethod(_get_attr, attr="clays")
    get_aas = partialmethod(_get_attr, attr="aas")
    get_ions = partialmethod(_get_attr, attr="ions")
    get_atoms = partialmethod(_get_attr, attr="atoms")
    get_other = partialmethod(_get_attr, attr="other")

    @property
    @abstractmethod
    def plot_df(self):
        ...

    def _init_plot(self, x, y, select):
        sel_list = ["clays", "aas", "atoms", "ions"]

        if self.select == "ions":
            sel_list.pop(-1)

        if self.data.other is not None:
            sel_list.append("other")

        assert x in sel_list, f"{x} not in sel list"
        assert y in sel_list, f"{y} not in sel list"
        assert select in sel_list, f"{select} not in sel list"

    # @abstractmethod
    # def plot_ions(self, *args, **kwargs):
    #     ...
    #
    # @abstractmethod
    # def plot_other(self, *args, **kwargs):
    #     ...

    def _get_colour_dict(self):
        from seaborn import saturate, set_hls_values

        colours = [
            "#8dd3c7",  # cyan (C)
            "#ffffb3",  # yellow (S)
            "#bebada",  # blue (N)
            "#fb8072",  # red (O)
            "#80b1d3",  # light blue (Mg)
            "#fdb462",  # orange (K)
            "#b3de69",  # light green (Ca)
            "#fccde5",  # pink (Na)
            # "#d9d9d9",  # grey (other)
        ]
        colours = [set_hls_values(saturate(cval), l=0.4) for cval in colours]
        colour_keys = ["C", "S", "N", "O", "Mg", "K", "Ca", "Na"]
        colour_dict = dict(zip(colour_keys, colours))
        return colour_dict

    def get_atom_colour_codes_from_names(self, atom_name):
        # atom_name = AtomHistPlot.get_atom_type_group(atom_name, group_all_atoms=True)
        if atom_name not in self.data.ions:
            atom_name = atom_name[0]
        return atom_name


class Axis(UserString):
    def __init__(self, plot_instance, name):
        # if name == 'atoms':
        #     name = '_atoms'
        self.v = plot_instance._get_attr(name)
        self.data = name
        self.l = len(self.v)
        self.dict = dict(zip(self.v, np.arange(self.l)))
        logger.info(f"{self.data}: {self.v}")


# class LinePlot(Plot):
#     sel_list = ["clays", "ions", "aas", "_atoms"]
#     title_dict = {
#         "clays": "Clay type",
#         "ions": "Ion type",
#         "aas": "Amino acid",
#         "atoms": "Atom type",
#         "other": "Other atom type",
#     }
#     _init_args = {"x", "y", "lines"}
#     _plot_args = {
#         "rowlabel": "y",
#         "columnlabel": "x",
#         "dpi": None,
#         "figsize": None,
#     }
#
#     def __init__(self, data, **kwargs):
#         super().__init__(data, **kwargs)
#         self.bins = "x"
#         self.line = None
#
#     def plot_args_decorator(select: str):
#         def plot_decorator(plot_f):
#             def wrapper(self, **kwargs):
#                 self.select = select
#                 plot_f(self, **kwargs)
#                 self.select = None
#
#             return wrapper
#
#         return plot_decorator
#
#     @property
#     def plot_df(self):
#         try:
#             return getattr(self, f"_{self.select}_bin_df")  # _{self.line}")
#         except AttributeError:
#             logging.error(f"Wrong usage of property, select not assigned yet.")
#
#     @property
#     def colour_df(self):
#         try:
#             return getattr(self, f"_{self.select}_colour_df")  # _{self.line}")
#         except AttributeError:
#             logging.error(f"Wrong usage of property, select not assigned yet.")
#
#     def _get_binned_plot_colour_dfs_1d(
#         self,
#     ) -> Tuple[pd.DataFrame, pd.DataFrame]:
#         logger.info(f"Getting plot and colour dfs")
#         from seaborn import saturate, set_hls_values
#
#         # atom type colour cycle
#         colours = [
#             "#8dd3c7",
#             "#ffffb3",
#             "#bebada",
#             "#fb8072",
#             "#80b1d3",
#             "#fdb462",
#             "#b3de69",
#             "#fccde5",
#         ]
#         colours = [set_hls_values(saturate(cval), l=0.4) for cval in colours]
#         # colours = color_palette('dark').as_hex()
#         sel = self.data.clays
#
#         # get data for plotting
#         plot_df = self.data.bin_df[sel].copy()
#
#         # move clays category from columns to index
#         idx_names = ["clays", *plot_df.index.droplevel(["x_bins"]).names]
#         # DataFrame -> Series
#         plot_df = plot_df.stack()
#
#         # get values for atom types (including separate ions)
#         atoms = plot_df.index.get_level_values("_atoms")
#
#         # make new DataFrame from atom_type index level and values
#         plot_df.index = plot_df.index.droplevel(["x_bins"])
#         # plot_df.index.names = [name.strip('_') for name in plot_df.index.names]
#
#         plot_df = pd.DataFrame({"values": plot_df, "colours": atoms})
#
#         # self._plot_df = {}
#         # self._colour_df = {}
#
#         atom_types = atoms.unique()
#         colour_dict = dict(zip(atom_types, colours[: len(atom_types)]))
#         plot_df["colours"] = plot_df["colours"].transform(
#             lambda x: colour_dict[x]
#         )
#
#         # reorder index for grouping
#         plot_df = plot_df.reorder_levels(idx_names)
#
#         # # group and sum densities within adsorption shell bins
#         # plot_df = plot_df.groupby(plot_df.index.names).agg(
#         #     values=pd.NamedAgg("values", "sum"), colours=pd.NamedAgg("colours", "first")
#         # )
#
#         # separate colour column from plot_df -> yields 2 Series
#         colour_df = plot_df["colours"]
#         plot_df = plot_df["values"]
#
#         # add missing atom probabilities from bulk to the largest bin
#         # (bin.left, cutoff] -> (bin.left, all bulk])
#         # inner_sum = plot_df.groupby(plot_df.index.droplevel("x_bins").names).sum()
#         # extra = 1 - inner_sum
#         # plot_df.where(
#         #     np.rint(plot_df.index.get_level_values("x_bins").right)
#         #     != int(self.data.cutoff),
#         #     lambda x: x + extra,
#         #     inplace=True,
#         # )
#         # setattr(self, "_plot_df", plot_df.copy())
#         self._plot_df, self._colour_df = plot_df.copy(), colour_df.copy()
#         logger.info(f"{self._plot_df}\n{self._colour_df}")
#         # print("sel list", self.sel_list)
#
#         # for line in self.sel_list:
#         #
#         #     line_plot_df = plot_df.copy()
#         #     # get values for atom types (including separate ions)
#         #     if line == "_atoms":
#         #         line = "atoms"
#         #     lines = line_plot_df.index.get_level_values(line)
#         #
#         #     # make new DataFrame from atom_type index level and values
#         #     line_plot_df.index = line_plot_df.index.droplevel(["x_bins"])
#         #
#         #     # plot_df.index.names = [name.strip('_') for name in plot_df.index.names]
#         #
#         #     line_plot_df = pd.DataFrame({"values": line_plot_df, "colours": lines})
#         #
#         #     # map unique atom types to colour map
#         #     line_types = lines.unique()
#         #     colour_dict = dict(zip(line_types, colours[: len(line_types)]))
#         #     print(colour_dict)
#         #     line_plot_df["colours"] = line_plot_df["colours"].transform(
#         #         lambda x: colour_dict[x]
#         #     )
#         #
#         #     # reorder index for grouping
#         #     line_plot_df = line_plot_df.reorder_levels(idx_names)
#         #
#         #     # group and sum densities within adsorption shell bins
#         #     colour_df = line_plot_df.groupby(line_plot_df.index.names).agg(
#         #         colours=pd.NamedAgg("colours", "first")
#         #     )
#         #
#         #     # separate colour column from plot_df -> yields 2 Series
#         #     colour_df = colour_df["colours"]
#         #     # print(colour_df)
#         #     line_plot_df = line_plot_df["values"]
#         #
#         #     # add missing atom probabilities from bulk to the largest bin
#         #     # (bin.left, cutoff] -> (bin.left, all bulk])
#         #     # inner_sum = plot_df.groupby(plot_df.index.droplevel("x_bins").names).sum()
#         #     # extra = 1 - inner_sum
#         #     # plot_df.where(
#         #     #     np.rint(plot_df.index.get_level_values("x_bins").right)
#         #     #     != int(self.data.cutoff),
#         #     #     lambda x: x + extra,
#         #     #     inplace=True,
#         #     # )
#         #     # setattr(self, f"_plot_df_{line}", line_plot_df.copy())
#         #     self._plot_df[line], self._colour_df[line] = (
#         #         line_plot_df.copy(),
#         #         colour_df.copy(),
#         #     )
#
#     def _split_plot_dfs(self):
#         logger.info(f"Splitting plot and colour dfs")
#         bin_df = self._plot_df.copy()
#         plot_df = self._colour_df.copy()
#         dfs = {"bin": bin_df, "colour": plot_df}
#         for key, df in dfs.items():
#             not_ions = [a for a in self.data.atoms if a != "ions"]
#             ion_view = df.xs("ions", level="atoms")
#             ion_view.index = ion_view.index.droplevel(["ions"])
#             ion_view.index.names = [
#                 idstr.strip("_") for idstr in ion_view.index.names
#             ]
#
#             other_view = df.loc[pd.IndexSlice[:, :, :, not_ions, :]]
#             other_view.index = other_view.index.droplevel("_atoms")
#             setattr(self, f"_ions_{key}_df", ion_view.copy())
#             setattr(self, f"_other_{key}_df", other_view.copy())
#         # dfs = {}
#         # # print(self._plot_df.keys())
#         # for line in self.sel_list:
#         #     if line == "_atoms":
#         #         line = "atoms"
#         #     bin_df = self._plot_df[line].copy()
#         #     plot_df = self._colour_df[line].copy()
#         #     dfs[line] = {"bin": bin_df.copy(), "colour": plot_df.copy()}
#         #     for key, df in dfs[line].items():
#         #         not_ions = [a for a in self.data.atoms if a != "ions"]
#         #         ion_view = df.xs("ions", level="atoms")
#         #         ion_view.index = ion_view.index.droplevel(["ions"])
#         #         ion_view.index.names = [
#         #             idstr.strip("_") for idstr in ion_view.index.names
#         #         ]
#         #
#         #         other_view = df.loc[pd.IndexSlice[:, :, :, not_ions, :]]
#         #         other_view.index = other_view.index.droplevel("_atoms")
#         #         print(f"_ions_{key}_df_{line}")
#         #         setattr(self, f"_ions_{key}_df_{line}", ion_view.copy())
#         #         setattr(self, f"_other_{key}_df_{line}", other_view.copy())
#
#     def _process_data(self):
#         self._get_binned_plot_colour_dfs_1d()
#         self._split_plot_dfs()
#
#     def get_suptitle(self, pl, separate):
#         self.fig.suptitle(
#             (
#                 ", ".join([self.title_dict[s].upper() for s in separate])
#                 + f": {self.label_mod(list(tuple(zip(pl, separate))))}"
#             ),
#             size=16,
#             weight="bold",
#         )
#
#     def plot(
#         self,
#         lines: Literal["clays", "ions", "aas", "atoms", "other"],
#         x: Literal["clays", "aas", "ions", "atoms", "other"],
#         y: Literal["clays", "ions", "aas", "atoms", "other"],
#         rowlabel: str = "y",
#         columnlabel: str = "x",
#         figsize=None,
#         dpi=None,
#         diff=False,
#         xmax=50,
#         ymax=50,
#         plsave=False,
#         xlim=None,
#         ylim=None,
#         odir=".",
#         plot_table=None,
#         edges=False,
#     ):
#         # line style cycler
#         linestyle_list = [
#             (0, ()),
#             (0, (1, 2)),
#             (0, (5, 2)),
#             (0, (5, 2, 1, 2)),
#             (0, (5, 2, 1, 2, 1, 2)),
#             (0, (1, 4)),
#             (0, (3, 4, 1, 4)),
#             (0, (3, 4, 1, 4, 1, 4)),
#             (0, (7, 6)),
#             (0, (7, 6, 3, 6)),
#             (0, (7, 6, 3, 6, 1, 2)),
#         ]
#
#         self._init_plot(x, y, lines)
#
#         plot_df = self.plot_df.copy()
#         colour_df = self.colour_df.copy()
#
#         # logger.info(f'{plot_df}')
#
#         avail_attrs = plot_df.index.names
#         logger.info(f"Available attributes: {avail_attrs}")
#
#         logger.info("Setting x:")
#         self.x = Axis(self, x)
#         logger.info("Setting y:")
#         self.y = Axis(self, y)
#         logger.info("Setting lines:")
#         self.line = Axis(self, lines)
#
#         separate = [
#             s
#             for s in avail_attrs
#             if s not in [self.x, self.y, self.line, self.bins]
#         ]
#
#         if len(separate) != 0:
#             logger.info(f"Separate plots: {separate}")
#
#         idx = pd.Index(
#             [
#                 s
#                 for s in avail_attrs
#                 if (
#                     s != self.x
#                     and s != self.line
#                     and s not in [*separate, self.bins]
#                 )
#             ]
#         )
#
#         logger.info(f"Iteration index: {idx.values}")
#
#         sep = pd.Index(separate)
#
#         yid = np.ravel(np.where(np.array(idx) == self.y))[0]
#
#         if figsize == None:
#             logger.info(f"{xmax}, {ymax} figsize")
#             figsize = self.get_figsize(xmax=xmax, ymax=ymax)
#
#         # set resultion
#         if dpi == None:
#             dpi = 100
#
#         # get plotting iter from index
#         iters = self._get_idx_iter(idx=idx)
#
#         logger.info(
#             f"Printing bar plots for {self.line}\nColumns: {self.x.v}\nRows: {self.y.v}"
#         )
#
#         # set label modifier function
#         self.label_mod = self.modify_plot_labels
#
#         try:
#             # iterator for more than one plot
#             sep_it = self._get_idx_iter(idx=sep)
#         except ValueError:
#             # only one plot
#             sep_it = [None]
#         # logger.info(f'{sep_it}\n{plot_df}')
#
#         # iterate over separate plots
#         for pl in sep_it:
#             y_max = 0
#             if pl is None:
#                 pl_str = ""
#             else:
#                 pl_str = f"_{pl[0]}"
#             self.name = f"{self.data.name}_{x}_{y}_{lines}{pl_str}_{self.data.cutoff}_{self.data.bins}"
#             logger.info(f"plot {self.name}")
#
#             # index map for y values
#             # y_dict: dict = dict(zip(vy, np.arange(ly)))
#
#             self.init_legend(ly=self.y.l, lx=self.x.l)
#
#             self.fig, self.ax = plt.subplots(
#                 nrows=self.y.l,
#                 ncols=self.x.l,
#                 figsize=figsize,
#                 sharey=True,
#                 sharex=True,
#                 dpi=dpi,
#                 constrained_layout=True,
#             )
#
#             if pl is None:
#                 logger.info(f"Generating plot")
#                 sepview = plot_df.view()
#                 save_str = ""
#
#             # multiple plots
#             else:
#                 logger.info(f"Generating {pl} plot:")
#                 # print(plot_df.head(20), "\n", separate, pl)
#                 sepview = plot_df.xs((pl), level=separate, drop_level=False)
#                 save_str = pl
#                 self.get_suptitle(pl, separate)
#
#             # set plot index
#             pi = 0
#
#             # iterate over subplot columns
#             for col in self.x.v:
#                 # logger.info(col)
#                 try:
#                     # print(sepview.dropna().index)
#                     view = sepview.xs(
#                         col, level=self.x, axis=0, drop_level=False
#                     )
#                     pi = 1
#                 except ValueError:
#                     view = sepview
#                     col = self.x.v
#                     pi += 1
#                 for it in iters:
#                     try:
#                         values = view.xs(
#                             tuple(it), level=idx.tolist(), drop_level=False
#                         )
#                         # print(values)
#                         # x_labels = []
#
#                         # logging.info(f"it: {it}")  # , {self.line.v}, {values}")
#                         x_id, y_id = self.x.dict[col], self.y.dict[it[yid]]
#                         # if ads_edges is True:
#                         #     atom_type = values.index.get_level_values("atoms").unique()[
#                         #         0
#                         #     ]
#                         #     edge_list = self.data.ads_edges[atom_type]
#                         #     p = self.ax[y_id, x_id].vlines(
#                         #         edge_list, 0, ylim, color="gray", linestyles=(0, (1, 4))
#                         #     )
#
#                         for line_num, vline in enumerate(self.line.v):
#                             # logging.info(f"{vline}")
#                             # set linestyle
#                             if self.line != "atoms":
#                                 print(line_num)
#                                 linestyle = linestyle_list[
#                                     line_num % len(linestyle_list)
#                                 ]
#                             else:
#                                 linestyle = "solid"
#
#                             line_vals = values.xs(
#                                 vline, level=self.line, drop_level=False
#                             )
#
#                             colour = colour_df.loc[line_vals.index].values[0]
#                             bin_list = line_vals.index.get_level_values("x")
#                             line_vals.index = bin_list
#                             # logger.info(f'{vline}, {self.line}')
#                             label = self.label_mod([(vline, self.line)])
#                             if pi == 1:
#                                 self.legends[(y_id, x_id)].append(label)
#                             if not line_vals.hasnans:
#                                 if np.max(line_vals) > y_max:
#                                     y_max = np.max(line_vals)
#                                 try:
#                                     summed = np.sum(line_vals.values)
#                                     assert (
#                                         np.round(summed) == 1.00
#                                     ), "Densities do not sum to 1.00"
#                                     p = self.ax[y_id, x_id].plot(
#                                         bin_list,
#                                         line_vals.values,
#                                         label=label,
#                                         color=colour,
#                                         linestyle=linestyle,
#                                     )
#                                     # self.ax[y_id, x_id].bar_label(p, labels=[label],
#                                     #                          fmt='%s',
#                                     #                          label_type='center')
#                                 except IndexError:
#                                     p = self.ax[x_id].plot(
#                                         bin_list,
#                                         line_vals.values,
#                                         label=label,
#                                         color=colour,
#                                         linestyle=linestyle,
#                                     )
#
#                     except KeyError:
#                         logger.info(f"No data for {pl}, {self.x}, {it}")
#
#             for i in range(self.y.l):
#                 try:
#                     self.ax[i, 0].set_ylabel(
#                         f"{self.label_mod([(self.y.v[i], self.y)])}\n"
#                         + rowlabel
#                     )
#                     for j in range(self.x.l):
#                         label = [
#                             self.label_mod([(leg, self.line)])
#                             for leg in self.legends[i, j]
#                         ]
#                         if len(label) % 3 == 0:
#                             ncol = 3
#                         else:
#                             ncol = 2
#                         self.ax[i, j].legend(label, ncol=ncol)
#                         if xlim != None:
#                             self.ax[i, j].set_xlim((0.0, float(xlim)))
#                         if ylim != None:
#                             self.ax[i, j].set_ylim((0.0, float(ylim)))
#                         else:
#                             self.ax[i, j].set_ylim((0.0, float(y_max)))
#                         self.ax[self.y.l - 1, j].set_xlabel(
#                             columnlabel
#                             + f"\n{self.label_mod([(self.x.v[j], self.x)])}"
#                         )
#
#                         self.ax[i, j].spines[["top", "right"]].set_visible(
#                             False
#                         )
#                     self.fig.supxlabel(f"{self.title_dict[x]}s", size=14)
#                     self.fig.supylabel(f"{self.title_dict[y]}s", size=14)
#                 except IndexError:
#                     self.ax[i].set_ylabel(
#                         f"{self.label_mod([(self.y.v[i], y)])}\n" + rowlabel
#                     )
#                     # self.ax[i].legend([self.label_mod([(leg, self.label_key)]) for leg in self.legends[i, 0]], ncol=3)
#                     self.ax[self.y.l - 1].set_xlabel(
#                         columnlabel
#                         + f"\n{self.label_mod([(self.x.v[0], self.x)])}"
#                     )
#             if edges is True:
#                 for col in self.x.v:
#                     # logger.info(col)
#                     try:
#                         view = sepview.xs(
#                             col, level=self.x, axis=0, drop_level=False
#                         )
#                         pi = 1
#                     except ValueError:
#                         view = sepview
#                         col = self.x.v
#                         pi += 1
#                     for it in iters:
#                         try:
#                             values = view.xs(
#                                 tuple(it), level=idx.tolist(), drop_level=False
#                             )
#                             # print(values)
#                             # x_labels = []
#
#                             logging.info(
#                                 f"it: {it}"
#                             )  # , {self.line.v}, {values}")
#                             x_id, y_id = self.x.dict[col], self.y.dict[it[yid]]
#
#                             atom_type = values.index.get_level_values(
#                                 "atoms"
#                             ).unique()[0]
#                             edge_list = self.data.edges[atom_type]
#                             p = self.ax[y_id, x_id].vlines(
#                                 edge_list,
#                                 0,
#                                 np.max(line_vals),
#                                 color="gray",
#                                 linestyles=(0, (1, 4)),
#                             )
#                         except KeyError:
#                             logger.info(f"No data for {pl}, {self.x}, {it}")
#
#             # self.fig.supxlabel(f"{self.title_dict[x]}s", size=14)
#             # self.fig.supylabel(f"{self.title_dict[y]}s", size=14)
#
#             if plsave != False:
#                 logger.info("Saving plot")
#                 if type(plsave) == str:
#                     outname = f"{plsave}.png"
#                 else:
#                     outname = f"{self.name}.png"
#                 odir = Path(odir).absolute()
#                 logger.info(f"output to {odir.absolute()}")
#                 if not odir.is_dir():
#                     os.makedirs(odir)
#                 self.fig.savefig(str(odir / outname))
#             else:
#                 plt.show()
#             self.fig.clear()
#
#         def _get_bin_label(self, x_bin, bin_list):
#             if x_bin.right < np.max(bin_list.right):
#                 label = f"${x_bin.left} - {x_bin.right}$ \AA"
#                 # barwidth = x_bin.right - x_bin.left
#
#             else:
#                 label = f"$ > {x_bin.left}$ \AA"
#             return label
#
#     # @property
#     # def plot_df(self):
#     #     try:
#     #         return getattr(self, f"_{self.select}_bin_df")
#     #     except AttributeError:
#     #         logging.error(f"Wrong usage of property, line not assigned yet.")
#     #
#     # @property
#     # def colour_df(self):
#     #     try:
#     #         return getattr(self, f"_{self.select}_colour_df")
#     #     except AttributeError:
#     #         logging.error(f"Wrong usage of property, line not assigned yet.")
#
#     def _get_attr(self, attr: Literal["clays", "aas", "atoms", "ions"]):
#         return pd.unique(self.plot_df.index.get_level_values(attr))
#
#     get_clays = partialmethod(_get_attr, attr="clays")
#     get_aas = partialmethod(_get_attr, attr="aas")
#     get_ions = partialmethod(_get_attr, attr="ions")
#     get_atoms = partialmethod(_get_attr, attr="atoms")
#     get_other = partialmethod(_get_attr, attr="other")
#
#     def save(self, **kwargs):
#         ...
#
#     @plot_args_decorator(select="ions")
#     def plot_ions(
#         self,
#         lines: Literal["clays", "ions", "aas", "atoms", "other"],
#         x: Literal["clays", "aas", "ions", "atoms", "other"],
#         y: Literal["clays", "ions", "aas", "atoms", "other"],
#         rowlabel: str = "y",
#         columnlabel: str = "x",
#         figsize=None,
#         dpi=None,
#         diff=False,
#         xmax=50,
#         ymax=50,
#         plsave=False,
#         xlim=None,
#         ylim=None,
#         odir=".",
#         plot_table=None,
#         edges=True,
#     ):
#         self.plot(
#             lines,
#             x,
#             y,
#             rowlabel,
#             columnlabel,
#             figsize,
#             dpi,
#             diff,
#             xmax,
#             ymax,
#             plsave,
#             xlim,
#             ylim,
#             odir,
#             plot_table,
#             edges,
#         )
#
#     @plot_args_decorator(select="other")
#     def plot_other(
#         self,
#         lines: Literal["clays", "ions", "aas", "atoms", "other"],
#         x: Literal["clays", "aas", "ions", "atoms", "other"],
#         y: Literal["clays", "ions", "aas", "atoms", "other"],
#         rowlabel: str = "y",
#         columnlabel: str = "x",
#         figsize=None,
#         dpi=None,
#         diff=False,
#         xmax=50,
#         ymax=50,
#         plsave=False,
#         xlim=None,
#         ylim=None,
#         odir=".",
#         plot_table=None,
#         edges=True,
#     ):
#         self.plot(
#             lines,
#             x,
#             y,
#             rowlabel,
#             columnlabel,
#             figsize,
#             dpi,
#             diff,
#             xmax,
#             ymax,
#             plsave,
#             xlim,
#             ylim,
#             odir,
#             plot_table,
#             edges,
#         )


class LinePlot(Plot):
    sel_list = ["clays", "ions", "aas", "_atoms"]
    title_dict = {
        "clays": "Clay type",
        "ions": "Ion type",
        "aas": "Amino acid",
        "atoms": "Atom type",
        "other": "Other atom type",
    }
    _init_args = {"x", "y", "lines"}
    _plot_args = {
        "rowlabel": "y",
        "columnlabel": "x",
        "dpi": None,
        "figsize": None,
    }
    _default_colour = "#7f7f7f"

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.bins = "x"
        self.line = None

    def plot_args_decorator(select: str):
        def plot_decorator(plot_f):
            def wrapper(self, **kwargs):
                self.select = select
                plot_f(self, **kwargs)
                self.select = None

            return wrapper

        return plot_decorator

    @property
    def plot_df(self):
        try:
            return getattr(self, f"_{self.select}_bin_df")  # _{self.line}")
        except AttributeError:
            logging.error(f"Wrong usage of property, select not assigned yet.")

    @property
    def colour_df(self):
        try:
            return getattr(self, f"_{self.select}_colour_df")  # _{self.line}")
        except AttributeError:
            logging.error(f"Wrong usage of property, select not assigned yet.")

    def _get_binned_plot_colour_dfs_1d(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info(f"Getting plot and colour dfs")
        # from seaborn import set_hls_values, saturate
        #
        # # atom type colour cycle
        # colours = [
        #     "#8dd3c7",  # cyan (C)
        #     "#ffffb3",  # yellow (S)
        #     "#bebada",  # blue (N)
        #     "#fb8072",  # red (O)
        #     "#80b1d3",  # light blue (Mg)
        #     "#fdb462",  # orange (K)
        #     "#b3de69",  # light green (Ca)
        #     "#fccde5",  # pink (Na)
        # ]
        # colours = [set_hls_values(saturate(cval), l=0.4) for cval in colours]
        # colour_keys = ["C", "S", "N", "O", "Mg", "K", "Ca", "Na"]
        colour_dict = (
            self._get_colour_dict()
        )  # dict(zip(colour_keys, colours))
        # colours = color_palette('dark').as_hex()
        sel = self.data.clays

        # get data for plotting
        try:
            # clays still in columns
            plot_df = self.data.bin_df[sel].copy()

            # move clays category from columns to index
            idx_names = ["clays", *plot_df.index.droplevel(["x_bins"]).names]
            # DataFrame -> Series
            plot_df = plot_df.stack()
        except KeyError:
            plot_df = self.data.bin_df.copy()
            idx_names = plot_df.index.droplevel(["x_bins"]).names

        # get values for atom types (including separate ions)
        atoms: np.array = plot_df.index.get_level_values("_atoms").to_numpy()
        # atom_type_groups = np.unique(atoms)
        transform_atom_names_vec = np.vectorize(
            self.get_atom_colour_codes_from_names, otypes=[str]
        )
        # transform_dict = dict(zip(atom_type_groups, transform_atom_names_vec(atom_type_groups)))
        atoms = transform_atom_names_vec(atoms)
        # atom_type_groups = transform_atom_names_vec(atom_type_groups)

        # make new DataFrame from atom_type index level and values
        plot_df.index = plot_df.index.droplevel(["x_bins"])
        # plot_df.index.names = [name.strip('_') for name in plot_df.index.names]

        try:
            plot_df = pd.DataFrame({"values": plot_df, "colours": atoms})
        except ValueError:
            #     if self.select == 'ions':
            plot_df["colours"] = atoms
            plot_df["values"] = plot_df[
                self.data.analysis
            ]  # else:  #     plot_df['colours'] = atoms.transform(lambda x: x[0])

        # self._plot_df = {}
        # self._colour_df = {}

        # color_cycle = itertools.cycle(colours)
        # colour_dict = {}
        # for atom_type in np.unique(atoms):
        #     if atom_type not in colour_dict.keys():
        #         colour_dict[atom_type] = next(color_cycle)
        # colour_dict = dict(zip(atom_type_groups, colours[:len(atom_type_groups)]))
        try:
            plot_df["colours"] = plot_df["colours"].apply(
                lambda x: colour_dict[x]
            )
        except KeyError:
            plot_df["colours"] = self._default_colour

        # reorder index for grouping
        plot_df = plot_df.reorder_levels(idx_names)

        # # group and sum densities within adsorption shell bins
        # plot_df = plot_df.groupby(plot_df.index.names).agg(
        #     values=pd.NamedAgg("values", "sum"), colours=pd.NamedAgg("colours", "first")
        # )

        # separate colour column from plot_df -> yields 2 Series
        colour_df = plot_df["colours"]
        plot_df = plot_df["values"]

        # add missing atom probabilities from bulk to the largest bin
        # (bin.left, cutoff] -> (bin.left, all bulk])
        # inner_sum = plot_df.groupby(plot_df.index.droplevel("x_bins").names).sum()
        # extra = 1 - inner_sum
        # plot_df.where(
        #     np.rint(plot_df.index.get_level_values("x_bins").right)
        #     != int(self.data.cutoff),
        #     lambda x: x + extra,
        #     inplace=True,
        # )
        # setattr(self, "_plot_df", plot_df.copy())
        self._plot_df, self._colour_df = plot_df.copy(), colour_df.copy()
        logger.info(
            f"{self._plot_df}\n{self._colour_df}"
        )  # print("sel list", self.sel_list)

        # for line in self.sel_list:  #  #     line_plot_df = plot_df.copy()  #     # get values for atom types (including separate ions)  #     if line == "_atoms":  #         line = "atoms"  #     lines = line_plot_df.index.get_level_values(line)  #  #     # make new DataFrame from atom_type index level and values  #     line_plot_df.index = line_plot_df.index.droplevel(["x_bins"])  #  #     # plot_df.index.names = [name.strip('_') for name in plot_df.index.names]  #  #     line_plot_df = pd.DataFrame({"values": line_plot_df, "colours": lines})  #  #     # map unique atom types to colour map  #     line_types = lines.unique()  #     colour_dict = dict(zip(line_types, colours[: len(line_types)]))  #     print(colour_dict)  #     line_plot_df["colours"] = line_plot_df["colours"].transform(  #         lambda x: colour_dict[x]  #     )  #  #     # reorder index for grouping  #     line_plot_df = line_plot_df.reorder_levels(idx_names)  #  #     # group and sum densities within adsorption shell bins  #     colour_df = line_plot_df.groupby(line_plot_df.index.names).agg(  #         colours=pd.NamedAgg("colours", "first")  #     )  #  #     # separate colour column from plot_df -> yields 2 Series  #     colour_df = colour_df["colours"]  #     # print(colour_df)  #     line_plot_df = line_plot_df["values"]  #  #     # add missing atom probabilities from bulk to the largest bin  #     # (bin.left, cutoff] -> (bin.left, all bulk])  #     # inner_sum = plot_df.groupby(plot_df.index.droplevel("x_bins").names).sum()  #     # extra = 1 - inner_sum  #     # plot_df.where(  #     #     np.rint(plot_df.index.get_level_values("x_bins").right)  #     #     != int(self.data.cutoff),  #     #     lambda x: x + extra,  #     #     inplace=True,  #     # )  #     # setattr(self, f"_plot_df_{line}", line_plot_df.copy())  #     self._plot_df[line], self._colour_df[line] = (  #         line_plot_df.copy(),  #         colour_df.copy(),  #     )

    @redirect_tqdm
    def _split_plot_dfs(self):
        logger.info(f"Splitting plot and colour dfs")
        bin_df = self._plot_df.copy()
        plot_df = self._colour_df.copy()
        dfs = {"bin": bin_df, "colour": plot_df}
        for key, df in dfs.items():
            if "ions" in self.data.atoms:
                not_ions = [a for a in self.data.atoms if a != "ions"]
                ion_view = df.xs("ions", level="atoms")
                ion_view.index = ion_view.index.droplevel(["ions"])
                ion_view.index.names = [
                    idstr.strip("_") for idstr in ion_view.index.names
                ]
                setattr(self, f"_ions_{key}_df", ion_view.copy())
            else:
                not_ions = np.unique(self.data.atoms).tolist()

            other_view = df.loc[pd.IndexSlice[:, :, :, not_ions, :]]
            other_view.index = other_view.index.droplevel("_atoms")

            setattr(
                self, f"_other_{key}_df", other_view.copy()
            )  # dfs = {}  # # print(self._plot_df.keys())  # for line in self.sel_list:  #     if line == "_atoms":  #         line = "atoms"  #     bin_df = self._plot_df[line].copy()  #     plot_df = self._colour_df[line].copy()  #     dfs[line] = {"bin": bin_df.copy(), "colour": plot_df.copy()}  #     for key, df in dfs[line].items():  #         not_ions = [a for a in self.data.atoms if a != "ions"]  #         ion_view = df.xs("ions", level="atoms")  #         ion_view.index = ion_view.index.droplevel(["ions"])  #         ion_view.index.names = [  #             idstr.strip("_") for idstr in ion_view.index.names  #         ]  #  #         other_view = df.loc[pd.IndexSlice[:, :, :, not_ions, :]]  #         other_view.index = other_view.index.droplevel("_atoms")  #         print(f"_ions_{key}_df_{line}")  #         setattr(self, f"_ions_{key}_df_{line}", ion_view.copy())  #         setattr(self, f"_other_{key}_df_{line}", other_view.copy())

    def _process_data(self):
        self._get_binned_plot_colour_dfs_1d()
        self._split_plot_dfs()

    def get_suptitle(self, pl, separate):
        self.fig.suptitle(
            (
                ", ".join([self.title_dict[s].upper() for s in separate])
                + f": {self.label_mod(list(tuple(zip(pl, separate))))}"
            ),
            size=16,
            weight="bold",
        )

    @redirect_tqdm
    def plot(
        self,
        lines: Literal["clays", "ions", "aas", "atoms", "other"],
        x: Literal["clays", "aas", "ions", "atoms", "other"],
        y: Literal["clays", "ions", "aas", "atoms", "other"],
        rowlabel: str = "y",
        columnlabel: str = "x",
        figsize=None,
        dpi=None,
        diff=False,
        xmax=50,
        ymax=50,
        plsave=False,
        xlim=None,
        ylim=None,
        odir=".",
        plot_table=None,
        edges=False,
        sparse=None,
        contract_binsize=None,
        antialiased=True,
        smooth_line=0.01,
        colours=None,
    ):
        # line style cycler
        linestyle_list = [
            (0, ()),
            (0, (1, 2)),
            (0, (5, 2)),
            (0, (5, 2, 1, 2)),
            (0, (5, 2, 1, 2, 1, 2)),
            (0, (1, 4)),
            (0, (3, 4, 1, 4)),
            (0, (3, 4, 1, 4, 1, 4)),
            (0, (7, 6)),
            (0, (7, 6, 3, 6)),
            (0, (7, 6, 3, 6, 1, 2)),
        ]
        if sparse is True:
            pass
        self._init_plot(x, y, lines)

        plot_df = self.plot_df.copy()
        plot_df = plot_df.groupby(plot_df.index.names).sum()
        colour_df = self.colour_df.copy()
        colour_df = colour_df.groupby(colour_df.index.names).sum()
        # linestyle_dict = {}
        line_cycle = itertools.cycle(linestyle_list)
        # line_idx = colour_df.index.droplevel('x')
        line_grouped = (
            colour_df.groupby("atoms").first().duplicated(False).to_dict()
        )
        for at_type, duplicate in line_grouped.items():
            if duplicate is True:
                line_grouped[at_type] = next(line_cycle)
            else:
                line_cycle = itertools.cycle(linestyle_list)
                line_grouped[at_type] = linestyle_list[0]
        del at_type, duplicate, line_cycle

        # logger.info(f'{plot_df}')

        avail_attrs = plot_df.index.names
        logger.info(f"Available attributes: {avail_attrs}")

        logger.info("Setting x:")
        self.x = Axis(self, x)
        logger.info("Setting y:")
        self.y = Axis(self, y)
        logger.info("Setting lines:")
        self.line = Axis(self, lines)

        separate = [
            s
            for s in avail_attrs
            if s not in [self.x, self.y, self.line, self.bins]
        ]

        if len(separate) != 0:
            logger.info(f"Separate plots: {separate}")

        idx = pd.Index(
            [
                s
                for s in avail_attrs
                if (
                    s != self.x
                    and s != self.line
                    and s not in [*separate, self.bins]
                )
            ]
        )

        if colours is not None:
            from seaborn import saturate, set_hls_values

            colours = [
                set_hls_values(saturate(cval), l=0.4) for cval in colours
            ]
            colour_dict = dict(zip(self.line.v, colours))

        logger.info(f"Iteration index: {idx.values}")

        sep = pd.Index(separate)

        yid = np.ravel(np.where(np.array(idx) == self.y))[0]

        if figsize is None:
            logger.info(f"{xmax}, {ymax} figsize")
            figsize = self.get_figsize(xmax=xmax, ymax=ymax)

        # set resultion
        if dpi is None:
            dpi = 100

        # get plotting iter from index
        iters = self._get_idx_iter(idx=idx)

        logger.info(
            f"Printing line plots for {self.line}\nColumns: {self.x.v}\nRows: {self.y.v}"
        )

        # set label modifier function
        self.label_mod = self.modify_plot_labels

        try:
            # iterator for more than one plot
            sep_it = self._get_idx_iter(idx=sep)
        except ValueError:
            # only one plot
            sep_it = [None]
        # logger.info(f'{sep_it}\n{plot_df}')

        # iterate over separate plots
        for pl in sep_it:
            y_max = 0
            if pl is None:
                pl_str = ""
            else:
                pl_str = f"_{pl[0]}"
            self.name = (
                f"{self.data.name}_{self.data.analysis}_{self.select}"
                f"_{x}_{y}_{lines}{pl_str}_{self.data.cutoff}_{self.data.bins}"
            )
            if xlim is not None:
                self.name += f"_{xlim:.0f}"
            logger.info(f"plot {self.name}")

            # index map for y values
            # y_dict: dict = dict(zip(vy, np.arange(ly)))

            self.init_legend(ly=self.y.l, lx=self.x.l)

            self.fig, self.ax = plt.subplots(
                nrows=self.y.l,
                ncols=self.x.l,
                figsize=tuple(figsize),
                sharey=True,
                sharex=True,
                dpi=dpi,
                constrained_layout=True,
            )

            if pl is None:
                logger.info(f"Generating plot")
                sepview = plot_df.view()
                save_str = ""

            # multiple plots
            else:
                logger.info(f"Generating {pl} plot:")
                # print(plot_df.head(20), "\n", separate, pl)
                sepview = plot_df.xs((pl), level=separate, drop_level=False)
                save_str = pl  # self.get_suptitle(pl, separate)

            # set plot index
            pi = 0

            # iterate over subplot columns
            for col in self.x.v:
                # logger.info(col)
                try:
                    # print(sepview.dropna().index)
                    view = sepview.xs(
                        col, level=self.x, axis=0, drop_level=False
                    )
                    pi = 1
                except ValueError:
                    view = sepview
                    col = self.x.v
                    pi += 1
                except KeyError:
                    continue
                for it in iters:
                    try:
                        values = view.xs(
                            tuple(it), level=idx.tolist(), drop_level=False
                        )
                        v = values.index.get_level_values("atoms").unique()
                        logger.info(v)
                        # print(values)
                        # x_labels = []

                        # logging.info(f"it: {it}")  # , {self.line.v}, {values}")
                        x_id, y_id = self.x.dict[col], self.y.dict[it[yid]]
                        # if edges is True:
                        #     atom_type = values.index.get_level_values("atoms").unique()[
                        #         0
                        #     ]
                        #     edge_list = self.data.edges[atom_type]
                        #     p = self.ax[y_id, x_id].vlines(
                        #         edge_list, 0, ylim, color="gray", linestyles=(0, (1, 4))
                        #     )
                        line_cycle = itertools.cycle(linestyle_list)
                        for line_num, vline in enumerate(self.line.v):
                            logging.info(
                                f"line {vline} ({line_num + 1}/{self.line.l})"
                            )
                            # set linestyle
                            if self.line != "atoms":
                                # line_cycle = itertools.cycle(linestyle_list)
                                # linestyle_dict = {}
                                # for atom_type in np.unique(self.data.atoms):
                                #     linestyle_dict[atom_type] = next(line_cycle)
                                linestyle = next(
                                    line_cycle
                                )  # linestyle_list[line_num % len(linestyle_list)]
                            else:
                                linestyle = line_grouped[vline]

                            try:
                                line_vals: pd.Series = values.xs(
                                    vline, level=self.line, drop_level=False
                                )
                                if contract_binsize is not None:
                                    line_vals = (
                                        line_vals.rolling(
                                            contract_binsize,
                                            closed="right",
                                            center=True,
                                            step=contract_binsize,
                                        )
                                        .sum()
                                        .fillna(0)
                                    )
                            except KeyError:
                                logger.info(f"No data for {vline}")
                                continue
                            if colours is None:
                                colour = colour_df.loc[line_vals.index].values[
                                    0
                                ]
                            else:
                                colour = colour_dict[vline]
                            # if sparse is True:
                            #     line_vals = line_vals.rolling(10, min_periods=1, step=10).sum()
                            bin_list = line_vals.index.get_level_values("x")
                            if (
                                smooth_line is not None
                                and smooth_line is not False
                            ):
                                if isinstance(smooth_line, float):
                                    if self.data.bins.num < smooth_line:
                                        break
                                elif isinstance(smooth_line, int):
                                    smooth_line = (
                                        self.data.bins.num / smooth_line
                                    )
                                else:
                                    smooth_line = self.data.bins.num / 10
                                dense_bin_list = np.arange(
                                    bin_list[0], bin_list[-1], smooth_line
                                )
                            line_vals.index = bin_list
                            # logger.info(f'{vline}, {self.line}')
                            # print(line_vals.hasnans, np.max(line_vals.values), vline, line_num, y_id, x_id)
                            summed = np.round(np.sum(line_vals.values), 4)
                            if (
                                not line_vals.hasnans and summed > 0.0
                            ):  # and vline != 'C':
                                if np.max(line_vals) > y_max:
                                    y_max = np.max(line_vals)
                                    logger.info(
                                        f"adjusting max value: {y_max}"
                                    )
                                try:
                                    label = f"{self.label_mod([(vline, self.line)]):<2}"
                                    if hasattr(
                                        self.data, "ignore_density_sum"
                                    ):
                                        logger.debug(
                                            f"Ignoring line value sum check sum={summed:.2f}"
                                        )
                                        if summed != 1.0:
                                            logger.info(
                                                f"labelling sum {summed}"
                                            )
                                            label_add = (
                                                rf" ({summed * 100:>.0f} \%)"
                                            )
                                            label = f"{label:<3}{label_add:>7}"
                                    else:
                                        assert (
                                            np.round(summed) == 1.00
                                        ), f"Densities ({np.round(summed)}) do not sum to 1.00"
                                    if pi == 1:
                                        self.legends[(y_id, x_id)].append(
                                            label
                                        )
                                    # if sparse is True:
                                    #     line_vals_mask = line_vals.values != 0
                                    # spline = splrep(bin_list,#[line_vals_mask],
                                    #                               line_vals.values)#[line_vals_mask])
                                    # line_vals = splev(bin_list, spline)
                                    # else:
                                    #     line_vals = line_vals.values
                                    # print(x_id)
                                    if (
                                        smooth_line is not None
                                        and smooth_line is not False
                                    ):
                                        bin_list_plot = dense_bin_list
                                        line_vals_plot = make_smoothing_spline(
                                            bin_list, line_vals.values
                                        )(bin_list_plot)
                                    else:
                                        bin_list_plot = bin_list
                                        line_vals_plot = line_vals.values
                                    try:
                                        # f, ax = plt.subplots()
                                        # ax.plot(bin_list, line_vals)
                                        # f.suptitle(vline)
                                        # f.show()
                                        logger.info(
                                            f"{len(bin_list)}, {len(line_vals)}"
                                        )
                                        # p = self.ax[y_id, x_id].plot([1,1], [0,1], label='a')
                                        try:
                                            p = self.ax[y_id, x_id].plot(
                                                bin_list_plot,
                                                line_vals_plot,
                                                label=label,
                                                color=colour,
                                                linestyle=linestyle,
                                                antialiased=antialiased,
                                            )

                                        except IndexError:
                                            xy_id = np.max([x_id, y_id])
                                            p = self.ax[xy_id].plot(
                                                bin_list_plot,
                                                line_vals_plot,
                                                label=label,
                                                color=colour,
                                                linestyle=linestyle,
                                                antialiased=antialiased,
                                            )
                                        except TypeError:
                                            p = self.ax.plot(
                                                bin_list_plot,
                                                line_vals_plot,
                                                label=label,
                                                color=colour,
                                                linestyle=linestyle,
                                                antialiased=antialiased,
                                            )  # self.fig.show()  # self.fig.show()  # self.fig.savefig('a.png')  # logger.info('wrote file')  # logger.info(f'{pl} {col} {vline} {it[yid]}')  # handles[(y_id, x_id)] = p
                                    except Exception as e:
                                        # del p
                                        print(e, y_id, x_id, line_num, vline)

                                    # self.ax[y_id, x_id].bar_label(p, labels=[label],  #                          fmt='%s',  #                          label_type='center')
                                except IndexError:
                                    p = self.ax[x_id].plot(
                                        bin_list,
                                        line_vals.values,
                                        label=label,
                                        color=colour,
                                        linestyle=linestyle,
                                        antialiased=antialiased,
                                    )
                            else:
                                logger.info(
                                    f"nan or 0: {vline}"
                                )  # self.ax[y_id, x_id].legend()
                    # except FileNotFoundError:
                    except KeyError:
                        logger.info(
                            f"No data for {pl}, {self.x}, {it} {it[yid]}"
                        )
            i = 0
            j = 0
            for i in range(self.y.l):
                y_ax_label = f"{self.label_mod([(self.y.v[i], self.y)])}\n"
                if self.y.l > 1:
                    if i == 0:
                        self.fig.supylabel(f"{self.title_dict[y]}s", size=14)
                else:
                    if i == 0 and not re.match(
                        y_ax_label, "\nctl", flags=re.IGNORECASE
                    ):
                        self.fig.supylabel(
                            f"{self.title_dict[y]}: {y_ax_label}", size=14
                        )
                    self.name = re.sub(
                        rf"(.*)_{y}_(.*)",
                        r"\1_" + y_ax_label.strip("\n") + r"_\2",
                        self.name,
                    )
                    y_ax_label = (
                        ""  # f"{self.label_mod([(self.y.v[i], self.y)])}\n"
                    )
                try:
                    self.ax[i, 0].set_ylabel(y_ax_label + rowlabel)
                except IndexError:
                    self.ax[i].set_ylabel(y_ax_label + rowlabel)
                except TypeError:
                    self.ax.set_ylabel(y_ax_label + rowlabel)
                for j in range(self.x.l):
                    x_ax_label = f"\n{self.label_mod([(self.x.v[j], self.x)])}"
                    if self.x.l > 1:
                        if j == 0:
                            self.fig.supxlabel(
                                f"{self.title_dict[x]}s", size=14
                            )
                    else:
                        if j == 0 and not re.match(
                            x_ax_label, "\nctl", flags=re.IGNORECASE
                        ):
                            self.fig.supxlabel(
                                f"{self.title_dict[x]}: {x_ax_label}", size=14
                            )

                        self.name = re.sub(
                            rf"(.*)_{x}_(.*)",
                            r"\1_" + x_ax_label.strip("\n") + r"_\2",
                            self.name,
                        )
                        x_ax_label = ""  # f"\n{self.label_mod([(self.x.v[j], self.x)])}"
                    label = [
                        self.label_mod([(leg, self.line)])
                        for leg in self.legends[i, j]
                    ]
                    if len(label) % 3 == 0:
                        ncol = 3
                    else:
                        ncol = 2
                    try:
                        self.ax[i, j].legend(
                            label,
                            ncol=ncol,
                            title=self.title_dict[lines],
                            frameon=False,
                            loc="upper right",
                        )
                        if xlim is not None:
                            self.ax[i, j].set_xlim((0.0, float(xlim)))
                        if ylim is not None:
                            self.ax[i, j].set_ylim((0.0, float(ylim)))
                        else:
                            self.ax[i, j].set_ylim((0.0, float(y_max)))
                        self.ax[self.y.l - 1, j].set_xlabel(
                            columnlabel + x_ax_label
                        )
                        self.ax[i, j].spines[["top", "right"]].set_visible(
                            False
                        )
                    except IndexError:
                        try:
                            max_id = np.max(i, j)
                        except IndexError:
                            max_id = np.max(i, j)[0]
                        self.ax[max_id].legend(
                            label,
                            ncol=ncol,
                            title=self.title_dict[lines],
                            frameon=False,
                            loc="upper right",
                        )
                        if xlim is not None:
                            self.ax[j].set_xlim((0.0, float(xlim)))
                        if ylim is not None:
                            self.ax[i].set_ylim((0.0, float(ylim)))
                        else:
                            self.ax[i].set_ylim(
                                (0.0, np.round(float(y_max + 0.05 * y_max), 2))
                            )
                        self.ax[self.y.l - 1].set_xlabel(
                            columnlabel + x_ax_label
                        )
                        self.ax[j].spines[["top", "right"]].set_visible(False)
                        self.ax[i].spines[["top", "right"]].set_visible(False)
                    except TypeError:
                        self.ax.legend(
                            label,
                            ncol=ncol,
                            title=self.title_dict[lines],
                            frameon=False,
                            loc="upper right",
                        )
                        if xlim is not None:
                            self.ax.set_xlim((0.0, float(xlim)))
                        if ylim is not None:
                            self.ax.set_ylim((0.0, float(ylim)))
                        else:
                            self.ax.set_ylim(
                                (0.0, np.round(float(y_max + 0.05 * y_max), 2))
                            )
                        self.ax.set_xlabel(columnlabel + x_ax_label)
                        self.ax.spines[["top", "right"]].set_visible(False)
                    if j == 0:
                        try:
                            y_ticks = self.ax[i, 0].get_yticks()
                            x_ticks = self.ax[i, j].get_xticks()
                        except IndexError:
                            y_ticks = self.ax[i].get_yticks()
                            x_ticks = self.ax[j].get_xticks()
                        except TypeError:
                            y_ticks = self.ax.get_yticks()
                            x_ticks = self.ax.get_xticks()
                        finally:
                            y_int = np.rint(len(y_ticks) / 5).astype(int)
                            y_ticks = y_ticks[::y_int]
                            x_int = np.rint(len(x_ticks) / 5).astype(int)
                            x_ticks = x_ticks[::x_int]
                            try:
                                self.ax[i, j].set_xticks(x_ticks, x_ticks)
                            except IndexError:
                                self.ax[j].set_xticks(
                                    x_ticks, x_ticks.astype(int)
                                )
                            except TypeError:
                                self.ax.set_xticks(
                                    x_ticks, x_ticks.astype(int)
                                )
                            if i == 0:
                                try:
                                    self.ax[i, 0].set_yticks(y_ticks, y_ticks)
                                except IndexError:
                                    self.ax[i].set_yticks(y_ticks, y_ticks)
                                except TypeError:
                                    self.ax.set_yticks(
                                        y_ticks, y_ticks
                                    )  # self.fig.supxlabel(f"{self.title_dict[x]}s", size=14)  # self.fig.supylabel(f"{self.title_dict[y]}s", size=14)

                # except IndexError:  #     self.ax[i].set_ylabel(  #         f"{self.label_mod([(self.y.v[i], y)])}\n" + rowlabel  #     )  #     # self.ax[i].legend([self.label_mod([(leg, self.label_key)]) for leg in self.legends[i, 0]], ncol=3)  #     self.ax[self.y.l - 1].set_xlabel(  #         columnlabel + f"\n{self.label_mod([(self.x.v[0], self.x)])}"  #     )
            if edges is True:
                if self.line != "atoms":
                    max_y = plot_df.max()
                    for col in self.x.v:
                        # logger.info(col)
                        try:
                            view = sepview.xs(
                                col, level=self.x, axis=0, drop_level=False
                            )
                            pi = 1
                        except ValueError:
                            view = sepview
                            col = self.x.v
                            pi += 1
                        for it in iters:
                            try:
                                values = view.xs(
                                    tuple(it),
                                    level=idx.tolist(),
                                    drop_level=False,
                                )
                                # print(values)
                                # x_labels = []

                                logging.info(
                                    f"it: {it}"
                                )  # , {self.line.v}, {values}")
                                x_id, y_id = (
                                    self.x.dict[col],
                                    self.y.dict[it[yid]],
                                )

                                atom_type = values.index.get_level_values(
                                    "atoms"
                                ).unique()[0]
                                edge_list = self.data.edges[atom_type]
                                try:
                                    p = self.ax[y_id, x_id].vlines(
                                        edge_list,
                                        0,  # np.max(line_vals),
                                        max_y,
                                        color="gray",
                                        linestyles=(0, (1, 4)),
                                    )
                                except IndexError:
                                    xy_id = np.max([x_id, y_id])
                                    p = self.ax[xy_id].vlines(
                                        edge_list,
                                        0,  # np.max(line_vals),
                                        max_y,
                                        color="gray",
                                        linestyles=(0, (1, 4)),
                                    )
                            except TypeError:
                                xy_id = np.max([x_id, y_id])
                                p = self.ax.vlines(
                                    edge_list,
                                    0,  # np.max(line_vals),
                                    max_y,
                                    color="gray",
                                    linestyles=(0, (1, 4)),
                                )
                            except KeyError:
                                logger.info(
                                    f"No data for {pl}, {self.x}, {it}"
                                )

            # self.fig.supxlabel(f"{self.title_dict[x]}s", size=14)
            # self.fig.supylabel(f"{self.title_dict[y]}s", size=14)

            if plsave != False:
                logger.info("Saving plot")
                if type(plsave) == str:
                    outname = f"{plsave}.png"
                else:
                    outname = f"{self.name}.png"
                odir = Path(odir).absolute()
                logger.info(f"output to {odir.absolute()}")
                if not odir.is_dir():
                    os.makedirs(odir)
                logger.info(odir / outname)
                self.fig.savefig(str(odir / outname))
            else:
                if type(plsave) == str:
                    outname = f"{plsave}.p"
                else:
                    outname = f"{self.name}.p"
                odir = Path(odir).absolute()
                logger.info(f"output to {odir.absolute()}")
                if not odir.is_dir():
                    os.makedirs(odir)
                logger.info(odir / outname)
                with open(odir / outname, "wb") as pklfile:
                    pkl.dump(
                        {"fig": self.fig, "ax": self.ax, "data": self.data.df},
                        pklfile,
                    )
                self.fig.show()
            self.fig.clear()

        def _get_bin_label(self, x_bin, bin_list):
            if x_bin.right < np.max(bin_list.right):
                label = f"${x_bin.left} - {x_bin.right}$ \AA"  # barwidth = x_bin.right - x_bin.left

            else:
                label = f"$ > {x_bin.left}$ \AA"
            return label

    # @property
    # def plot_df(self):
    #     try:
    #         return getattr(self, f"_{self.select}_bin_df")
    #     except AttributeError:
    #         logging.error(f"Wrong usage of property, line not assigned yet.")
    #
    # @property
    # def colour_df(self):
    #     try:
    #         return getattr(self, f"_{self.select}_colour_df")
    #     except AttributeError:
    #         logging.error(f"Wrong usage of property, line not assigned yet.")

    def _get_attr(self, attr: Literal["clays", "aas", "atoms", "ions"]):
        return pd.unique(self.plot_df.index.get_level_values(attr))

    get_clays = partialmethod(_get_attr, attr="clays")
    get_aas = partialmethod(_get_attr, attr="aas")
    get_ions = partialmethod(_get_attr, attr="ions")
    get_atoms = partialmethod(_get_attr, attr="atoms")
    get_other = partialmethod(_get_attr, attr="other")

    def save(self, **kwargs):
        ...

    @plot_args_decorator(select="ions")
    def plot_ions(
        self,
        lines: Literal["clays", "ions", "aas", "atoms", "other"],
        x: Literal["clays", "aas", "ions", "atoms", "other"],
        y: Literal["clays", "ions", "aas", "atoms", "other"],
        rowlabel: str = "y",
        columnlabel: str = "x",
        figsize=None,
        dpi=None,
        diff=False,
        xmax=50,
        ymax=50,
        plsave=False,
        xlim=None,
        ylim=None,
        odir=".",
        plot_table=None,
        edges=True,
        sparse=None,
        antialiased=True,
        smooth_line=0.01,
        colours=None,
    ):
        self.plot(
            lines=lines,
            x=x,
            y=y,
            rowlabel=rowlabel,
            columnlabel=columnlabel,
            figsize=figsize,
            dpi=dpi,
            diff=diff,
            xmax=xmax,
            ymax=ymax,
            plsave=plsave,
            xlim=xlim,
            ylim=ylim,
            odir=odir,
            plot_table=plot_table,
            edges=edges,
            sparse=sparse,
            antialiased=antialiased,
            smooth_line=smooth_line,
            colours=colours,
        )

    @plot_args_decorator(select="other")
    def plot_other(
        self,
        lines: Literal["clays", "ions", "aas", "atoms", "other"],
        x: Literal["clays", "aas", "ions", "atoms", "other"],
        y: Literal["clays", "ions", "aas", "atoms", "other"],
        rowlabel: str = "y",
        columnlabel: str = "x",
        figsize=None,
        dpi=None,
        diff=False,
        xmax=50,
        ymax=50,
        plsave=False,
        xlim=None,
        ylim=None,
        odir=".",
        plot_table=None,
        edges=True,
        sparse=None,
        contract_binsize=None,
        antialiased=True,
        smooth_line=0.01,
        colours=None,
    ):
        self.plot(
            lines=lines,
            x=x,
            y=y,
            rowlabel=rowlabel,
            columnlabel=columnlabel,
            figsize=figsize,
            dpi=dpi,
            diff=diff,
            xmax=xmax,
            ymax=ymax,
            plsave=plsave,
            xlim=xlim,
            ylim=ylim,
            odir=odir,
            plot_table=plot_table,
            edges=edges,
            sparse=sparse,
            contract_binsize=contract_binsize,
            antialiased=antialiased,
            smooth_line=smooth_line,
            colours=colours,
        )


# class SDevLinePlot(LinePlot):
#     def __init__(
#         self, data: List[Data] = None, indirs=None, select="other", **kwargs
#     ):
#         if data is None and indirs is not None:
#             data = [Data(indir=indir, **kwargs) for indir in indirs]
#             for d in data:
#                 d.df.sort_index(inplace=True)
#                 d.select = select
#         elif data is not None and indirs is None:
#             pass
#         else:
#             raise ValueError(f"Specify either data or indirs")
#         new_data = copy.deepcopy(data[0])
#         new_data.var_df = new_data.bin_df.copy()
#         new_data.var_df = np.NaN
#         if len(data) > 1:
#             for d in data:
#                 d.__class__.__name__ == "Data", f"Expected Data, found {d.__class__.__name__}"
#             props = {}
#             for prop_name in [
#                 "cutoff",
#                 "bins",
#                 "analysis",
#                 "_arr_col",
#                 "atomnames",
#                 "select",
#             ]:
#                 prop = np.unique(
#                     [
#                         d.__getattribute__(prop_name)
#                         for d in data
#                         if d.__getattribute__(prop_name) is not None
#                     ]
#                 )
#                 if not prop:
#                     props[prop_name] = prop
#                 else:
#                     assert (
#                         len(prop) == 1
#                     ), "Incompatible Data objects, cutoff doesn't match"
#                     props[prop_name] = prop[0]
#             for d in data[1:]:
#                 # assert d.df.index.identical(new_data.df.index), 'Non-matching indices'
#                 assert d.bin_df.index.identical(
#                     new_data.bin_df.index
#                 ), "Non-matching indices"
#                 new_data.df += d.bin_df
#             new_data.df = new_data.bin_df / len(data)
#             new_data.var_df = (new_data.bin_df - new_data.bin_df) ** 2
#             for d in data[1:]:
#                 new_data.var_df += (d.bin_df - new_data.bin_df) ** 2
#             new_data.var_df = new_data.var_df.divide(len(data))
#             new_data.std_df = np.sqrt(new_data.var_df)
#         # self.data = new_data
#         # for attr in ['clays', 'ions', 'aas', 'atoms']:
#         #     self.__setattr__(attr, new_data.__getattribute__(attr))
#         super().__init__(new_data)
#
#     def _get_binned_plot_colour_dfs_1d(
#         self,
#     ) -> Tuple[pd.DataFrame, pd.DataFrame]:
#         logger.info(f"Getting plot and colour dfs")
#         # from seaborn import set_hls_values, saturate
#         #
#         # # atom type colour cycle
#         # colours = [
#         #     "#8dd3c7",  # cyan (C)
#         #     "#ffffb3",  # yellow (S)
#         #     "#bebada",  # blue (N)
#         #     "#fb8072",  # red (O)
#         #     "#80b1d3",  # light blue (Mg)
#         #     "#fdb462",  # orange (K)
#         #     "#b3de69",  # light green (Ca)
#         #     "#fccde5",  # pink (Na)
#         # ]
#         # colours = [set_hls_values(saturate(cval), l=0.4) for cval in colours]
#         # colour_keys = ["C", "S", "N", "O", "Mg", "K", "Ca", "Na"]
#         colour_dict = (
#             self._get_colour_dict()
#         )  # dict(zip(colour_keys, colours))
#
#         sel = self.data.clays
#         for df_name in ["bin", "std"]:
#             # get data for plotting
#             try:
#                 # clays still in columns
#                 plot_df = self.data.__getattribute__(f"{df_name}_df")[
#                     sel
#                 ].copy()
#
#                 # move clays category from columns to index
#                 idx_names = [
#                     "clays",
#                     *plot_df.index.droplevel(["x_bins"]).names,
#                 ]
#                 # DataFrame -> Series
#                 plot_df = plot_df.stack()
#             except KeyError:
#                 plot_df = self.data.bin_df.copy()
#                 idx_names = plot_df.index.droplevel(["x_bins"]).names
#
#             # get values for atom types (including separate ions)
#             atoms: np.array = plot_df.index.get_level_values(
#                 "_atoms"
#             ).to_numpy()
#
#             transform_atom_names_vec = np.vectorize(
#                 self.get_atom_colour_codes_from_names, otypes=[str]
#             )
#
#             atoms = transform_atom_names_vec(atoms)
#
#             # make new DataFrame from atom_type index level and values
#             plot_df.index = plot_df.index.droplevel(["x_bins"])
#
#             try:
#                 plot_df = pd.DataFrame({"values": plot_df, "colours": atoms})
#             except ValueError:
#                 plot_df["colours"] = atoms
#                 plot_df["values"] = plot_df[self.data.analysis]
#
#             try:
#                 plot_df["colours"] = plot_df["colours"].apply(
#                     lambda x: colour_dict[x]
#                 )
#             except KeyError:
#                 plot_df["colours"] = self._default_colour
#
#             # reorder index for grouping
#             plot_df = plot_df.reorder_levels(idx_names)
#             colour_df = plot_df["colours"]
#             plot_df = plot_df["values"]
#             setattr(self, f"_plot_{df_name}_df", plot_df.copy())
#             setattr(
#                 self, f"_colour_{df_name}_df", colour_df.copy()
#             )  # self._plot_df, self._colour_df = plot_df.copy(), colour_df.copy()  # logger.info(f"{self._plot_df}\n{self._colour_df}")
#
#     @redirect_tqdm
#     def _split_plot_dfs(self):
#         logger.info(f"Splitting plot and colour dfs")
#         for df_name in ["bin", "std"]:
#             bin_df = self.__getattribute__(f"_plot_{df_name}_df").copy()
#             plot_df = self.__getattribute__(f"_colour_{df_name}_df").copy()
#             dfs = {"bin": bin_df, "colour": plot_df}
#             for key, df in dfs.items():
#                 if "ions" in self.data.atoms:
#                     not_ions = [a for a in self.data.atoms if a != "ions"]
#                     ion_view = df.xs("ions", level="atoms")
#                     ion_view.index = ion_view.index.droplevel(["ions"])
#                     ion_view.index.names = [
#                         idstr.strip("_") for idstr in ion_view.index.names
#                     ]
#                     setattr(self, f"_ions_{key}_{df_name}_df", ion_view.copy())
#                 else:
#                     not_ions = np.unique(self.data.atoms).tolist()
#
#                 other_view = df.loc[pd.IndexSlice[:, :, :, not_ions, :]]
#                 other_view.index = other_view.index.droplevel("_atoms")
#
#                 setattr(self, f"_other_{key}_{df_name}_df", other_view.copy())
#
#     @property
#     def plot_df(self):
#         try:
#             return getattr(
#                 self, f"_{self.select}_bin_bin_df"
#             )  # _{self.line}")
#         except AttributeError:
#             logging.error(f"Wrong usage of property, select not assigned yet.")
#
#     @property
#     def colour_df(self):
#         try:
#             return getattr(
#                 self, f"_{self.select}_colour_bin_df"
#             )  # _{self.line}")
#         except AttributeError:
#             logging.error(f"Wrong usage of property, select not assigned yet.")
#
#     @property
#     def plot_sdev_df(self):
#         try:
#             return getattr(
#                 self, f"_{self.select}_bin_std_df"
#             )  # _{self.line}")
#         except AttributeError:
#             logging.error(f"Wrong usage of property, select not assigned yet.")
#
#     @property
#     def colour_sdev_df(self):
#         try:
#             return getattr(
#                 self, f"_{self.select}_colour_std_df"
#             )  # _{self.line}")
#         except AttributeError:
#             logging.error(f"Wrong usage of property, select not assigned yet.")
#
#     def compare(
#         self, first, second, category, df=None, sdev=None, select="other"
#     ):
#         if not self.select:
#             sel_reset = True
#             self.select = select
#         else:
#             sel_reset = False
#         if df is not None:
#             pass
#         else:
#             df = self.plot_df.copy()
#             sdev = self.plot_sdev_df.copy()
#         avail_attrs = df.index.names
#         group_attrs = [a for a in avail_attrs if a != category]
#         df_slice_1 = df.xs(first, level=category)
#         df_slice_2 = df.xs(second, level=category)
#         df_diff = df_slice_1 - df_slice_2
#         sdev_diff = None
#         if sdev is not None:
#             sdev_slice_1 = sdev.xs(first, level=category)
#             sdev_slice_2 = sdev.xs(second, level=category)
#             sdev_diff = (sdev_slice_1**2 - sdev_slice_2**2).apply(np.sqrt)
#         if sel_reset:
#             self.select = None
#         return df_diff, sdev_diff
#
#     @redirect_tqdm
#     def plot(
#         self,
#         lines: Literal["clays", "ions", "aas", "atoms", "other"],
#         x: Literal["clays", "aas", "ions", "atoms", "other"],
#         y: Literal["clays", "ions", "aas", "atoms", "other"],
#         rowlabel: str = "y",
#         columnlabel: str = "x",
#         figsize=None,
#         dpi=None,
#         diff=False,
#         xmax=50,
#         ymax=50,
#         plsave=False,
#         xlim=None,
#         ylim=None,
#         odir=".",
#         plot_table=None,
#         edges=False,
#         sparse=None,
#         contract_binsize=None,
#         antialiased=True,
#     ):
#         # line style cycler
#         linestyle_list = [
#             (0, ()),
#             (0, (1, 2)),
#             (0, (5, 2)),
#             (0, (5, 2, 1, 2)),
#             (0, (5, 2, 1, 2, 1, 2)),
#             (0, (1, 4)),
#             (0, (3, 4, 1, 4)),
#             (0, (3, 4, 1, 4, 1, 4)),
#             (0, (7, 6)),
#             (0, (7, 6, 3, 6)),
#             (0, (7, 6, 3, 6, 1, 2)),
#         ]
#         if sparse is True:
#             pass
#         self._init_plot(x, y, lines)
#
#         plot_df = self.plot_df.copy()
#         plot_df = plot_df.groupby(plot_df.index.names).sum()
#         plot_sdev_df = self.plot_sdev_df.copy()
#         plot_sdev_df = plot_sdev_df.groupby(plot_sdev_df.index.names).sum()
#         colour_df = self.colour_df.copy()
#         colour_df = colour_df.groupby(colour_df.index.names).sum()
#         colour_sdev_df = self.colour_sdev_df.copy()
#         colour_sdev_df = colour_sdev_df.groupby(
#             colour_sdev_df.index.names
#         ).sum()
#         # linestyle_dict = {}
#         line_cycle = itertools.cycle(linestyle_list)
#         # line_idx = colour_df.index.droplevel('x')
#         line_grouped = (
#             colour_df.groupby("atoms").first().duplicated(False).to_dict()
#         )
#         for at_type, duplicate in line_grouped.items():
#             if duplicate is True:
#                 line_grouped[at_type] = next(line_cycle)
#             else:
#                 line_cycle = itertools.cycle(linestyle_list)
#                 line_grouped[at_type] = linestyle_list[0]
#         del at_type, duplicate, line_cycle
#
#         # logger.info(f'{plot_df}')
#
#         avail_attrs = plot_df.index.names
#         logger.info(f"Available attributes: {avail_attrs}")
#
#         logger.info("Setting x:")
#         self.x = Axis(self, x)
#         logger.info("Setting y:")
#         self.y = Axis(self, y)
#         logger.info("Setting lines:")
#         self.line = Axis(self, lines)
#
#         separate = [
#             s
#             for s in avail_attrs
#             if s not in [self.x, self.y, self.line, self.bins]
#         ]
#
#         if len(separate) != 0:
#             logger.info(f"Separate plots: {separate}")
#
#         idx = pd.Index(
#             [
#                 s
#                 for s in avail_attrs
#                 if (
#                     s != self.x
#                     and s != self.line
#                     and s not in [*separate, self.bins]
#                 )
#             ]
#         )
#
#         logger.info(f"Iteration index: {idx.values}")
#
#         sep = pd.Index(separate)
#
#         yid = np.ravel(np.where(np.array(idx) == self.y))[0]
#
#         if figsize is None:
#             logger.info(f"{xmax}, {ymax} figsize")
#             figsize = self.get_figsize(xmax=xmax, ymax=ymax)
#
#         # set resultion
#         if dpi is None:
#             dpi = 100
#
#         # get plotting iter from index
#         iters = self._get_idx_iter(idx=idx)
#
#         logger.info(
#             f"Printing line plots for {self.line}\nColumns: {self.x.v}\nRows: {self.y.v}"
#         )
#
#         # set label modifier function
#         self.label_mod = self.modify_plot_labels
#
#         try:
#             # iterator for more than one plot
#             sep_it = self._get_idx_iter(idx=sep)
#         except ValueError:
#             # only one plot
#             sep_it = [None]
#         # logger.info(f'{sep_it}\n{plot_df}')
#
#         # iterate over separate plots
#         for pl in sep_it:
#             y_max = 0
#             if pl is None:
#                 pl_str = ""
#             else:
#                 pl_str = f"_{pl[0]}"
#             self.name = (
#                 f"{self.data.name}_{self.data.analysis}_{self.select}"
#                 f"_{x}_{y}_{lines}{pl_str}_{self.data.cutoff}_{self.data.bins}"
#             )
#             if xlim is not None:
#                 self.name += f"_{xlim:.0f}"
#             logger.info(f"plot {self.name}")
#
#             # index map for y values
#             # y_dict: dict = dict(zip(vy, np.arange(ly)))
#
#             self.init_legend(ly=self.y.l, lx=self.x.l)
#
#             self.fig, self.ax = plt.subplots(
#                 nrows=self.y.l,
#                 ncols=self.x.l,
#                 figsize=tuple(figsize),
#                 sharey=True,
#                 sharex=True,
#                 dpi=dpi,
#                 constrained_layout=True,
#             )
#
#             if pl is None:
#                 logger.info(f"Generating plot")
#                 sepview = plot_df.view()
#                 save_str = ""
#
#             # multiple plots
#             else:
#                 logger.info(f"Generating {pl} plot:")
#                 # print(plot_df.head(20), "\n", separate, pl)
#                 sepview = plot_df.xs((pl), level=separate, drop_level=False)
#                 sepview_sdev = plot_sdev_df.xs(
#                     (pl), level=separate, drop_level=False
#                 )
#                 save_str = pl  # self.get_suptitle(pl, separate)
#
#             # set plot index
#             pi = 0
#
#             # iterate over subplot columns
#             for col in self.x.v:
#                 # logger.info(col)
#                 try:
#                     # print(sepview.dropna().index)
#                     view = sepview.xs(
#                         col, level=self.x, axis=0, drop_level=False
#                     )
#                     view_sdev = sepview_sdev.xs(
#                         col, level=self.x, axis=0, drop_level=False
#                     )
#                     pi = 1
#                 except ValueError:
#                     view = sepview
#                     view_sdev = sepview_sdev
#                     col = self.x.v
#                     pi += 1
#                 except KeyError:
#                     continue
#                 for it in iters:
#                     try:
#                         values = view.xs(
#                             tuple(it), level=idx.tolist(), drop_level=False
#                         )
#                         values_sdev = view_sdev.xs(
#                             tuple(it), level=idx.tolist(), drop_level=False
#                         )
#                         v = values.index.get_level_values("atoms").unique()
#                         v_sdev = values_sdev.index.get_level_values(
#                             "atoms"
#                         ).unique()
#                         logger.info(v)
#                         # print(values)
#                         # x_labels = []
#
#                         # logging.info(f"it: {it}")  # , {self.line.v}, {values}")
#                         x_id, y_id = self.x.dict[col], self.y.dict[it[yid]]
#                         # if edges is True:
#                         #     atom_type = values.index.get_level_values("atoms").unique()[
#                         #         0
#                         #     ]
#                         #     edge_list = self.data.edges[atom_type]
#                         #     p = self.ax[y_id, x_id].vlines(
#                         #         edge_list, 0, ylim, color="gray", linestyles=(0, (1, 4))
#                         #     )
#                         line_cycle = itertools.cycle(linestyle_list)
#                         for line_num, vline in enumerate(self.line.v):
#                             logging.info(
#                                 f"line {vline} ({line_num + 1}/{self.line.l})"
#                             )
#                             # set linestyle
#                             if self.line != "atoms":
#                                 # line_cycle = itertools.cycle(linestyle_list)
#                                 # linestyle_dict = {}
#                                 # for atom_type in np.unique(self.data.atoms):
#                                 #     linestyle_dict[atom_type] = next(line_cycle)
#                                 linestyle = next(
#                                     line_cycle
#                                 )  # linestyle_list[line_num % len(linestyle_list)]
#                             else:
#                                 linestyle = line_grouped[vline]
#
#                             try:
#                                 line_vals: pd.Series = values.xs(
#                                     vline, level=self.line, drop_level=False
#                                 )
#                                 line_vals_sdev: pd.Series = values_sdev.xs(
#                                     vline, level=self.line, drop_level=False
#                                 )
#                                 if contract_binsize is not None:
#                                     line_vals = (
#                                         line_vals.rolling(
#                                             contract_binsize,
#                                             closed="right",
#                                             center=True,
#                                             step=contract_binsize,
#                                         )
#                                         .sum()
#                                         .fillna(0)
#                                     )
#                                     line_vals_sdev = (
#                                         line_vals.rolling(
#                                             contract_binsize,
#                                             closed="right",
#                                             center=True,
#                                             step=contract_binsize,
#                                         )
#                                         .sum()
#                                         .fillna(0)
#                                     )
#                             except KeyError:
#                                 logger.info(f"No data for {vline}")
#                                 continue
#
#                             colour = colour_df.loc[line_vals.index].values[0]
#                             # if sparse is True:
#                             #     line_vals = line_vals.rolling(10, min_periods=1, step=10).sum()
#                             bin_list = line_vals.index.get_level_values("x")
#                             line_vals.index = bin_list
#                             line_vals_sdev.index = bin_list
#                             # logger.info(f'{vline}, {self.line}')
#                             # print(line_vals.hasnans, np.max(line_vals.values), vline, line_num, y_id, x_id)
#                             summed = np.round(np.sum(line_vals.values), 4)
#                             if (
#                                 not line_vals.hasnans and summed > 0.0
#                             ):  # and vline != 'C':
#                                 if np.max(line_vals) > y_max:
#                                     y_max = (
#                                         np.max(line_vals)
#                                         + line_vals_sdev.values[
#                                             np.argmax(line_vals)
#                                         ]
#                                     )
#                                     logger.info(
#                                         f"adjusting max value: {y_max}"
#                                     )
#                                 try:
#                                     label = f"{self.label_mod([(vline, self.line)]):<2}"
#                                     if hasattr(
#                                         self.data, "ignore_density_sum"
#                                     ):
#                                         logger.debug(
#                                             f"Ignoring line value sum check sum={summed:.2f}"
#                                         )
#                                         if summed != 1.0:
#                                             logger.info(
#                                                 f"labelling sum {summed}"
#                                             )  # label_add = rf" ({summed*100:>.0f} \%)"  # label = f"{label:<3}{label_add:>7}"
#                                     else:
#                                         assert (
#                                             np.round(summed) == 1.00
#                                         ), f"Densities ({np.round(summed)}) do not sum to 1.00"
#                                     if pi == 1:
#                                         self.legends[(y_id, x_id)].append(
#                                             label
#                                         )
#                                     # if sparse is True:
#                                     #     line_vals_mask = line_vals.values != 0
#                                     # spline = splrep(bin_list,#[line_vals_mask],
#                                     #                               line_vals.values)#[line_vals_mask])
#                                     # line_vals = splev(bin_list, spline)
#                                     # else:
#                                     #     line_vals = line_vals.values
#                                     # print(x_id)
#                                     try:
#                                         # f, ax = plt.subplots()
#                                         # ax.plot(bin_list, line_vals)
#                                         # f.suptitle(vline)
#                                         # f.show()
#                                         logger.info(
#                                             f"{len(bin_list)}, {len(line_vals)}, {len(line_vals_sdev)}"
#                                         )
#                                         # p = self.ax[y_id, x_id].plot([1,1], [0,1], label='a')
#                                         try:
#                                             p1 = self.ax[
#                                                 y_id, x_id
#                                             ].fill_between(
#                                                 bin_list,
#                                                 (
#                                                     line_vals - line_vals_sdev
#                                                 ).astype(float),
#                                                 (
#                                                     line_vals + line_vals_sdev
#                                                 ).astype(float),
#                                                 alpha=0.2,
#                                                 color=colour,
#                                             )
#                                             p = self.ax[y_id, x_id].plot(
#                                                 bin_list,
#                                                 line_vals,
#                                                 label=label,
#                                                 color=colour,
#                                                 linestyle=linestyle,
#                                                 antialiased=antialiased,
#                                             )
#
#                                         except IndexError:
#                                             xy_id = np.max([x_id, y_id])
#                                             p1 = self.ax[xy_id].fill_between(
#                                                 bin_list,
#                                                 (
#                                                     line_vals - line_vals_sdev
#                                                 ).astype(float),
#                                                 (
#                                                     line_vals + line_vals_sdev
#                                                 ).astype(float),
#                                                 alpha=0.2,
#                                                 color=colour,
#                                             )
#                                             p = self.ax[xy_id].plot(
#                                                 bin_list,
#                                                 line_vals,
#                                                 label=label,
#                                                 color=colour,
#                                                 linestyle=linestyle,
#                                                 antialiased=antialiased,
#                                             )
#
#                                         except TypeError:
#                                             p1 = self.ax.fill_between(
#                                                 bin_list,
#                                                 (
#                                                     line_vals - line_vals_sdev
#                                                 ).astype(float),
#                                                 (
#                                                     line_vals + line_vals_sdev
#                                                 ).astype(float),
#                                                 alpha=0.2,
#                                                 color=colour,
#                                             )
#                                             p = self.ax.plot(
#                                                 bin_list,
#                                                 line_vals,
#                                                 label=label,
#                                                 color=colour,
#                                                 linestyle=linestyle,
#                                             )
#
#                                         # self.fig.show()  # self.fig.show()  # self.fig.savefig('a.png')  # logger.info('wrote file')  # logger.info(f'{pl} {col} {vline} {it[yid]}')
#                                     except Exception as e:
#                                         # del p
#                                         print(e, y_id, x_id, line_num, vline)
#
#                                     # self.ax[y_id, x_id].bar_label(p, labels=[label],  #                          fmt='%s',  #                          label_type='center')
#                                 except IndexError:
#                                     p1 = self.ax[x_id].fill_between(
#                                         bin_list,
#                                         (line_vals + line_vals_sdev).astype(
#                                             float
#                                         ),
#                                         (line_vals - line_vals_sdev).astype(
#                                             float
#                                         ),
#                                         alpha=0.2,
#                                         label=label,
#                                         color=colour,
#                                     )
#                                     p = self.ax[x_id].plot(
#                                         bin_list,
#                                         line_vals.values,
#                                         label=label,
#                                         color=colour,
#                                         linestyle=linestyle,
#                                         antialiased=antialiased,
#                                     )
#                                 finally:
#                                     self.handles[(y_id, x_id)].append(p)
#                             else:
#                                 logger.info(f"nan or 0: {vline}")
#
#                         # self.ax[y_id, x_id].legend()
#                     # except FileNotFoundError:
#                     except KeyError:
#                         logger.info(
#                             f"No data for {pl}, {self.x}, {self.y}, {self.line}, {it} {it[yid]}"
#                         )
#             i = 0
#             j = 0
#             for i in range(self.y.l):
#                 y_ax_label = f"{self.label_mod([(self.y.v[i], self.y)])}\n"
#                 if self.y.l > 1:
#                     if i == 0:
#                         pass  # self.fig.supylabel(f"{self.title_dict[y]}s", size=14)
#                 else:
#                     if i == 0:
#                         pass  # self.fig.supylabel(f"{self.title_dict[y]}: {y_ax_label}", size=14)
#                     y_ax_label = (
#                         ""  # f"{self.label_mod([(self.y.v[i], self.y)])}\n"
#                     )
#                 try:
#                     self.ax[i, 0].set_ylabel(rowlabel)  # y_ax_label +
#                 except IndexError:
#                     self.ax[i].set_ylabel(rowlabel)  # y_ax_label +
#                 except TypeError:
#                     self.ax.set_ylabel(rowlabel)  # y_ax_label +
#                 for j in range(self.x.l):
#                     x_ax_label = f"\n{self.label_mod([(self.x.v[j], self.x)])}"
#                     if self.x.l > 1:
#                         if j == 0:
#                             pass  # self.fig.supxlabel(f"{self.title_dict[x]}s", size=14)
#                     else:
#                         if j == 0:
#                             pass
#                         # self.fig.supxlabel(f"{self.title_dict[x]}: {x_ax_label}", size=14)
#                         x_ax_label = ""  # f"\n{self.label_mod([(self.x.v[j], self.x)])}"
#                     label = [
#                         self.label_mod([(leg, self.line)])
#                         for leg in self.legends[i, j]
#                     ]
#                     handle = self.handles[i, j]
#                     if len(label) % 3 == 0:
#                         ncol = 3
#                     else:
#                         ncol = 2
#                     try:
#                         if j == self.x.l - 1 and i == 0:
#                             self.ax[i, j].legend(  # handles=handle,
#                                 # label,
#                                 ncol=ncol,
#                                 title=lines,
#                                 frameon=False,
#                                 loc="upper right",
#                             )
#                         if xlim is not None:
#                             self.ax[i, j].set_xlim((0.0, float(xlim)))
#                         if ylim is not None:
#                             self.ax[i, j].set_ylim((0.0, float(ylim)))
#                         else:
#                             self.ax[i, j].set_ylim((0.0, float(y_max)))
#                         self.ax[self.y.l - 1, j].set_xlabel(
#                             columnlabel
#                         )  # + x_ax_label
#                         self.ax[i, j].spines[["top", "right"]].set_visible(
#                             False
#                         )
#                     except IndexError:
#                         try:
#                             max_id = np.max(i, j)
#                         except IndexError:
#                             max_id = np.max(i, j)[0]
#                         self.ax[max_id].legend(
#                             label,
#                             ncol=ncol,
#                             title=lines,
#                             frameon=False,
#                             loc="upper right",
#                         )
#                         if xlim is not None:
#                             self.ax[j].set_xlim((0.0, float(xlim)))
#                         if ylim is not None:
#                             self.ax[i].set_ylim((0.0, float(ylim)))
#                         else:
#                             self.ax[i].set_ylim(
#                                 (0.0, np.round(float(y_max + 0.05 * y_max), 2))
#                             )
#                         self.ax[self.y.l - 1].set_xlabel(
#                             columnlabel
#                         )  # + x_ax_label
#                         self.ax[j].spines[["top", "right"]].set_visible(False)
#                         self.ax[i].spines[["top", "right"]].set_visible(False)
#                     except TypeError:
#                         self.ax.legend(
#                             label,
#                             ncol=ncol,
#                             title=lines,
#                             frameon=False,
#                             loc="upper right",
#                         )
#                         if xlim is not None:
#                             self.ax.set_xlim((0.0, float(xlim)))
#                         if ylim is not None:
#                             self.ax.set_ylim((0.0, float(ylim)))
#                         else:
#                             self.ax.set_ylim(
#                                 (0.0, np.round(float(y_max + 0.05 * y_max), 2))
#                             )
#                         self.ax.set_xlabel(columnlabel)  # + x_ax_label
#                         self.ax.spines[["top", "right"]].set_visible(False)
#                     if j == 0:
#                         try:
#                             y_ticks = self.ax[i, 0].get_yticks()
#                             x_ticks = self.ax[i, j].get_xticks()
#                         except IndexError:
#                             y_ticks = self.ax[i].get_yticks()
#                             x_ticks = self.ax[j].get_xticks()
#                         except TypeError:
#                             y_ticks = self.ax.get_yticks()
#                             x_ticks = self.ax.get_xticks()
#                         finally:
#                             y_int = np.rint(len(y_ticks) / 5).astype(int)
#                             y_ticks = y_ticks[::y_int]
#                             x_int = np.rint(len(x_ticks) / 5).astype(int)
#                             x_ticks = x_ticks[::x_int]
#                             try:
#                                 self.ax[i, j].set_xticks(
#                                     x_ticks.astype(int), x_ticks.astype(int)
#                                 )
#                             except IndexError:
#                                 self.ax[j].set_xticks(
#                                     x_ticks, x_ticks.astype(int)
#                                 )
#                             except TypeError:
#                                 self.ax.set_xticks(
#                                     x_ticks, x_ticks.astype(int)
#                                 )
#                             if i == 0:
#                                 try:
#                                     self.ax[i, 0].set_yticks(
#                                         y_ticks, [f"{t:.3f}" for t in y_ticks]
#                                     )
#                                 except IndexError:
#                                     self.ax[i].set_yticks(y_ticks, y_ticks)
#                                 except TypeError:
#                                     self.ax.set_yticks(
#                                         y_ticks, y_ticks
#                                     )  # self.fig.supxlabel(f"{self.title_dict[x]}s", size=14)  # self.fig.supylabel(f"{self.title_dict[y]}s", size=14)
#
#                 # except IndexError:  #     self.ax[i].set_ylabel(  #         f"{self.label_mod([(self.y.v[i], y)])}\n" + rowlabel  #     )  #     # self.ax[i].legend([self.label_mod([(leg, self.label_key)]) for leg in self.legends[i, 0]], ncol=3)  #     self.ax[self.y.l - 1].set_xlabel(  #         columnlabel + f"\n{self.label_mod([(self.x.v[0], self.x)])}"  #     )
#             if edges is True:
#                 if self.line != "atoms":
#                     max_y = plot_df.max()
#                     for col in self.x.v:
#                         # logger.info(col)
#                         try:
#                             view = sepview.xs(
#                                 col, level=self.x, axis=0, drop_level=False
#                             )
#                             view_sdev = sepview_sdev.xs(
#                                 col, level=self.x, axis=0, drop_level=False
#                             )
#                             pi = 1
#                         except ValueError:
#                             view = sepview
#                             view_sdev = sepview_sdev
#                             col = self.x.v
#                             pi += 1
#                         for it in iters:
#                             try:
#                                 values = view.xs(
#                                     tuple(it),
#                                     level=idx.tolist(),
#                                     drop_level=False,
#                                 )
#                                 # print(values)
#                                 # x_labels = []
#
#                                 logging.info(
#                                     f"it: {it}"
#                                 )  # , {self.line.v}, {values}")
#                                 x_id, y_id = (
#                                     self.x.dict[col],
#                                     self.y.dict[it[yid]],
#                                 )
#
#                                 atom_type = values.index.get_level_values(
#                                     "atoms"
#                                 ).unique()[0]
#                                 edge_list = self.data.edges[atom_type]
#                                 try:
#                                     p = self.ax[y_id, x_id].vlines(
#                                         edge_list,
#                                         0,  # np.max(line_vals),
#                                         max_y,
#                                         color="gray",
#                                         linestyles=(0, (1, 4)),
#                                     )
#                                 except IndexError:
#                                     xy_id = np.max([x_id, y_id])
#                                     p = self.ax[xy_id].vlines(
#                                         edge_list,
#                                         0,  # np.max(line_vals),
#                                         max_y,
#                                         color="gray",
#                                         linestyles=(0, (1, 4)),
#                                     )
#                             except TypeError:
#                                 xy_id = np.max([x_id, y_id])
#                                 p = self.ax.vlines(
#                                     edge_list,
#                                     0,  # np.max(line_vals),
#                                     max_y,
#                                     color="gray",
#                                     linestyles=(0, (1, 4)),
#                                 )
#                             except KeyError:
#                                 logger.info(
#                                     f"No data for {pl}, {self.x}, {it}"
#                                 )
#
#             # self.fig.supxlabel(f"{self.title_dict[x]}s", size=14)
#             # self.fig.supylabel(f"{self.title_dict[y]}s", size=14)
#
#             if plsave != False:
#                 logger.info("Saving plot")
#                 if type(plsave) == str:
#                     outname = f"{plsave}.png"
#                 else:
#                     outname = f"{self.name}.png"
#                 odir = Path(odir).absolute()
#                 logger.info(f"output to {odir.absolute()}")
#                 if not odir.is_dir():
#                     os.makedirs(odir)
#                 logger.info(odir / outname)
#                 self.fig.savefig(str(odir / outname))
#             else:
#                 if type(plsave) == str:
#                     outname = f"{plsave}.p"
#                 else:
#                     outname = f"{self.name}.p"
#                 odir = Path(odir).absolute()
#                 logger.info(f"output to {odir.absolute()}")
#                 if not odir.is_dir():
#                     os.makedirs(odir)
#                 logger.info(odir / outname)
#                 with open(odir / outname, "wb") as pklfile:
#                     pkl.dump(
#                         {"fig": self.fig, "ax": self.ax, "data": self.data.df},
#                         pklfile,
#                     )
#                 self.fig.show()
#             self.fig.clear()
#
#         def _get_bin_label(self, x_bin, bin_list):
#             if x_bin.right < np.max(bin_list.right):
#                 label = f"${x_bin.left} - {x_bin.right}$ \AA"  # barwidth = x_bin.right - x_bin.left
#
#             else:
#                 label = f"$ > {x_bin.left}$ \AA"
#             return label


# https://colorbrewer2.org/?type=qualitative&scheme=Set3&n=5


# class HistPlot(Plot):
#     # __slots__ = ()
#     sel_list = ["clays", "ions", "aas", "_atoms"]
#     title_dict = {
#         "clays": "Clay type",
#         "ions": "Ion type",
#         "aas": "Amino acid",
#         "atoms": "Atom type",
#         "other": "Other atom type",
#     }
#     cmaps = [
#         "spring",
#         "summer",
#         "autumn",
#         "winter",
#         "cool",
#         "Wistia",
#         "hot",
#         "afmhot",
#         "gist_heat",
#         "copper",
#     ]
#     _init_args = {"x", "y", "bars"}
#     _plot_args = {
#         "rowlabel": "y",
#         "columnlabel": "x",
#         "dpi": None,
#         "figsize": None,
#     }
#
#     def __init__(self, data, **kwargs):
#         super().__init__(data, **kwargs)
#         self.bins = "x_bins"
#         self.group = "atoms"
#         self.bars = None
#
#     def plot_args_decorator(select: str):
#         def plot_decorator(plot_f):
#             def wrapper(self, **kwargs):
#                 self.select = select
#                 plot_f(self, **kwargs)
#                 self.select = None
#
#             return wrapper
#
#         return plot_decorator
#
#     def _get_binned_plot_colour_dfs_1d(
#         self,
#     ) -> Tuple[pd.DataFrame, pd.DataFrame]:
#         logger.info(f"Getting binned plot and colour dfs")
#
#         # list of matplotlib sequential cmaps
#         cmaps = [  # 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
#             "spring",
#             "summer",
#             "autumn",
#             "winter",
#             "cool",
#             "Wistia",
#             "hot",
#             "afmhot",
#             "gist_heat",
#             "copper",
#         ]
#
#         sel = self.data.clays
#
#         # get data for plotting
#         plot_df = self.data.bin_df[sel].copy()
#
#         # move clays category from columns to index
#         idx_names = ["clays", *plot_df.index.droplevel(["x"]).names]
#         # DataFrame -> Series
#         plot_df = plot_df.stack()
#
#         # get values for atom types (including separate ions)
#         atoms = plot_df.index.get_level_values("_atoms")
#
#         # make new DataFrame from atom_type index level and values
#         plot_df.index = plot_df.index.droplevel(["x"])
#         # plot_df.index.names = [name.strip('_') for name in plot_df.index.names]
#
#         plot_df = pd.DataFrame({"values": plot_df, "colours": atoms})
#
#         # map unique atom types to colour map
#         atom_types = atoms.unique()
#         colour_dict = dict(zip(atom_types, cmaps[: len(atom_types)]))
#         plot_df["colours"] = plot_df["colours"].transform(
#             lambda x: colour_dict[x]
#         )
#
#         # reorder index for grouping
#         plot_df = plot_df.reorder_levels(idx_names)
#
#         # group and sum densities within adsorption shell bins
#         plot_df = plot_df.groupby(plot_df.index.names).agg(
#             values=pd.NamedAgg("values", "sum"),
#             colours=pd.NamedAgg("colours", "first"),
#         )
#
#         # separate colour column from plot_df -> yields 2 Series
#         colour_df = plot_df["colours"]
#         plot_df = plot_df["values"]
#
#         # add missing atom probabilities from bulk to the largest bin
#         # (bin.left, cutoff] -> (bin.left, all bulk])
#         inner_sum = plot_df.groupby(
#             plot_df.index.droplevel("x_bins").names
#         ).sum()
#         extra = 1 - inner_sum
#         plot_df.where(
#             np.rint(plot_df.index.get_level_values("x_bins").right)
#             != int(self.data.cutoff),
#             lambda x: x + extra,
#             inplace=True,
#         )
#         # setattr(self, "_plot_df", plot_df.copy())
#         self._plot_df, self._colour_df = plot_df.copy(), colour_df.copy()
#
#     def _split_plot_dfs(self):
#         logger.info(f"Splitting binned plot and colour dfs")
#         bin_df = self._plot_df.copy()
#         plot_df = self._colour_df.copy()
#         dfs = {"bin": bin_df, "colour": plot_df}
#         for key, df in dfs.items():
#             not_ions = [a for a in self.data.atoms if a != "ions"]
#             ion_view = df.xs("ions", level="atoms")
#             ion_view.index = ion_view.index.droplevel(["ions"])
#             ion_view.index.names = [
#                 idstr.strip("_") for idstr in ion_view.index.names
#             ]
#             # logger.info(f'\n{not_ions}, {df.index.names}')
#             other_view = df.loc[pd.IndexSlice[:, :, :, :, not_ions, :]]
#             other_view.index = other_view.index.droplevel("_atoms")
#             # logger.info(f'{other_view.index.get_level_values("atoms").unique()}, {other_view.index.names}')
#             setattr(self, f"_ions_{key}_df", ion_view.copy())
#             setattr(self, f"_other_{key}_df", other_view.copy())
#
#     def _process_data(self, **kwargs):
#         self._get_binned_plot_colour_dfs_1d()
#         self._split_plot_dfs()
#
#     @property
#     def plot_df(self):
#         try:
#             plot_df = getattr(self, f"_{self.select}_bin_df")
#             # logger.info(f'{plot_df.index.get_level_values("atoms").unique()}')
#             return plot_df
#         except AttributeError:
#             logging.error(f"Wrong usage of property, select not assigned yet.")
#
#     @property
#     def colour_df(self):
#         try:
#             return getattr(self, f"_{self.select}_colour_df")
#         except AttributeError:
#             logging.error(f"Wrong usage of property, select not assigned yet.")
#
#     def save(self, **kwargs):
#         ...
#
#     @cached_property
#     def cnorm(self) -> mpc.Normalize:
#         # normalise colour map from 0 to max_edge
#         return mpc.Normalize(vmin=0, vmax=self.data.max_shell_edge, clip=False)
#
#     def _get_attr(self, attr: Literal["clays", "aas", "atoms", "ions"]):
#         return pd.unique(self.plot_df.index.get_level_values(attr))
#
#     get_clays = partialmethod(_get_attr, attr="clays")
#     get_aas = partialmethod(_get_attr, attr="aas")
#     get_ions = partialmethod(_get_attr, attr="ions")
#     get_atoms = partialmethod(_get_attr, attr="atoms")
#     get_other = partialmethod(_get_attr, attr="other")
#
#     @plot_args_decorator(select="ions")
#     def plot_ions(
#         self,
#         bars: Literal["clays", "aas", "ions", "other"],
#         x: Literal["clays", "aas", "ions", "other"],
#         y: Literal["clays", "aas", "ions", "other"],
#         rowlabel: str = "y",
#         columnlabel: str = "x",
#         figsize=None,
#         dpi=None,
#         xmax=50,
#         ymax=50,
#         plsave=False,
#         ylim=None,
#         odir=".",
#         barwidth=0.75,
#         xpad=0.25,
#         cmap="winter",
#         plot_table=False,
#     ):
#         self.plot(
#             bars,
#             x,
#             y,
#             rowlabel,
#             columnlabel,
#             figsize,
#             dpi,
#             xmax,
#             ymax,
#             plsave,
#             ylim,
#             odir,
#             barwidth,
#             xpad,
#             cmap,
#             plot_table=plot_table,
#         )
#
#     @plot_args_decorator(select="other")
#     def plot_other(
#         self,
#         bars: Literal["clays", "aas", "ions", "other"],
#         x: Literal["clays", "aas", "ions", "other"],
#         y: Literal["clays", "aas", "ions", "other"],
#         rowlabel: str = "y",
#         columnlabel: str = "x",
#         figsize=None,
#         dpi=None,
#         xmax=50,
#         ymax=50,
#         plsave=False,
#         ylim=None,
#         odir=".",
#         barwidth=0.75,
#         xpad=0.25,
#         cmap="winter",
#         plot_table=None,
#     ):
#         self.plot(
#             bars,
#             x,
#             y,
#             rowlabel,
#             columnlabel,
#             figsize,
#             dpi,
#             xmax,
#             ymax,
#             plsave,
#             ylim,
#             odir,
#             barwidth,
#             xpad,
#             cmap,
#         )
#
#     def get_suptitle(self, pl, separate):
#         self.fig.suptitle(
#             (
#                 ", ".join([self.title_dict[s].upper() for s in separate])
#                 + f": {self.label_mod(list(tuple(zip(pl, separate))))}"
#             ),
#             size=16,
#             weight="bold",
#         )
#
#     def plot(
#         self,
#         bars: Literal["clays", "aas", "ions", "other"],
#         x: Literal["clays", "aas", "ions", "other"],
#         y: Literal["clays", "aas", "ions", "other"],
#         rowlabel: str = "y",
#         columnlabel: str = "x",
#         figsize=None,
#         dpi=None,  # diff=False,
#         xmax=50,
#         ymax=50,
#         plsave=False,
#         ylim=None,
#         odir=".",
#         barwidth=0.75,
#         xpad=0.25,
#         cmap="winter",
#         tab_fontsize=12,
#         plot_table=False,
#     ):
#         """Create stacked Histogram adsorption shell populations."""
#
#         logger.info(f"plot table {plot_table}")
#
#         self._init_plot(x, y, bars)
#
#         plot_df = self.plot_df.copy()
#         colour_df = self.colour_df.copy()
#
#         avail_attrs = plot_df.index.names
#         logger.info(f"Available attributes: {avail_attrs}")
#
#         logger.info("Setting x:")
#         self.x = Axis(self, x)
#         logger.info("Setting y:")
#         self.y = Axis(self, y)
#         logger.info("Setting bars:")
#         self.bars = Axis(self, bars)
#
#         print(avail_attrs)
#         separate = [
#             s
#             for s in avail_attrs
#             if s not in [self.x, self.y, self.bars, self.bins]
#         ]
#
#         if len(separate) != 0:
#             logger.info(f"Separate plots: {separate}")
#
#         idx = pd.Index(
#             [
#                 s
#                 for s in avail_attrs
#                 if (
#                     s != self.x
#                     and s != self.bars
#                     and s not in [*separate, self.bins]
#                 )
#             ]
#         )
#
#         logger.info(f"Iteration index: {idx.values}")
#
#         sep = pd.Index(separate)
#         # logger.info(f'{sep}')
#
#         yid = np.ravel(np.where(np.array(idx) == self.y))[0]
#
#         if figsize == None:
#             if plot_table is True:
#                 figsize = self.get_figsize(xmax=xmax, ymax=ymax * 2)
#             else:
#                 figsize = self.get_figsize(xmax=xmax, ymax=ymax)
#
#         # set resultion
#         if dpi == None:
#             dpi = 100
#
#         # get plotting iter from index
#         iters = self._get_idx_iter(idx=idx)
#
#         logger.info(
#             f"Printing bar plots for {self.bars}\nColumns: {self.x.v}\nRows: {self.y.v}"
#         )
#
#         # set label modifier function
#         self.label_mod = self.modify_plot_labels
#
#         try:
#             # iterator for more than one plot
#             sep_it = self._get_idx_iter(idx=sep)  # logger.info(f'{sep_it}')
#         except ValueError:
#             # only one plot
#             sep_it = [None]
#
#         # iterate over separate plots
#         for pl in sep_it:
#             if pl is None:
#                 pl_str = ""
#             else:
#                 pl_str = f"_{pl[0]}"
#             self.name = f"{self.data.name}_{x}_{y}_{bars}{pl_str}_{self.data.cutoff}_{self.data.bins}"
#             logger.info(f"plot {self.name}")
#             # index map for y values
#             # y_dict: dict = dict(zip(vy, np.arange(ly)))
#
#             self.init_legend(ly=self.y.l, lx=self.x.l)
#
#             # if plot_table is True:
#             #     plt_nrows = self.y.l * 2
#             #     logger.info('Has table')
#             # else:
#             #     plt_nrows = self.y.l
#             # print('rows ', plt_nrows)
#
#             # generate figure and axes array
#             if plot_table is True:
#                 self.fig = plt.figure(figsize=figsize, dpi=dpi)
#             else:
#                 self.fig = plt.figure(
#                     figsize=figsize, dpi=dpi, constrained_layout=True
#                 )
#
#             # xmul = 2
#
#             self.gs = gridspec.GridSpec(
#                 nrows=self.y.l,
#                 ncols=self.x.l,
#                 figure=self.fig,  # width_ratios=[2,1]
#                 # sharey=True,
#                 # sharex=True
#             )
#
#             self.ax = {}
#
#             if plot_table is True:
#                 self.gs = gridspec.GridSpec(
#                     nrows=self.y.l,
#                     ncols=self.x.l,
#                     figure=self.fig,
#                     wspace=2 / self.x.l,
#                     hspace=0.5,  # width_ratios=
#                     # sharey=True,
#                     # sharex=True
#                 )
#                 self.subax = {}
#                 # self.tabax = {}
#                 for i in range(self.y.l):
#                     for j in range(self.x.l):
#                         #     if i == 0 and j == 0:
#                         #         self.subax[i, j] = gridspec.GridSpecFromSubplotSpec(
#                         #     2, 1, subplot_spec=self.gs[i, j]
#                         # )
#                         #     else:
#                         self.subax[i, j] = self.gs[i, j].subgridspec(
#                             10, self.x.l, wspace=1.5, hspace=1.5
#                         )
#                         self.ax[i, j] = self.fig.add_subplot(
#                             self.subax[i, j][:8, :]
#                         )
#
#                         # if i == (self.y.l - 1) or j == 0:
#
#                         # else:  #     self.ax[i, j] = self.fig.add_subplot(self.subax[i, j][:8, :])#,  #                                          # sharex=self.ax[0, i],  #                                          # sharey=self.ax[j, (self.y.l - 1)])  # if i != (self.y.l - 1):  #     self.ax[i, j].tick_params(labelbottom=False)  # if j != 0:  #     self.ax[i, j].tick_params(labelleft=False)
#
#                         # self.tabax[i, j] = self.fig.add_subplot(self.subax[i, j][1])  # self.tabax[i, j].set
#
#             else:
#                 for i in range(self.y.l):
#                     for j in range(self.x.l):
#                         self.ax[i, j] = self.fig.add_subplot(self.gs[i, j])
#
#             # only one plot
#             if pl is None:
#                 logger.info(f"Generating plot")
#                 sepview = plot_df.view()
#                 plsave = ""
#
#             # multiple plots
#             else:
#                 logger.info(f"Generating {pl} plot:")
#                 # print(plot_df.head(20), "\n", separate, pl)
#                 sepview = plot_df.xs((pl), level=separate, drop_level=False)
#                 plsave = pl
#                 self.get_suptitle(pl, separate)
#
#             # set plot index
#             pi = 0
#
#             # iterate over subplot columns
#             for col in self.x.v:
#                 logger.info(f"Column {col}")
#                 try:
#                     view = sepview.xs(
#                         col, level=self.x, axis=0, drop_level=False
#                     )
#                     pi = 1
#                 except ValueError:
#                     view = sepview
#                     col = self.x.v
#                     pi += 1
#
#                 for it in iters:
#                     try:
#                         values = view.xs(
#                             tuple(it), level=idx.tolist(), drop_level=False
#                         )
#
#                         x_labels = []
#                         x_ticks = []
#                         for bar_num, vbar in enumerate(self.bars.v):
#                             tab_colours = []
#                             tab_rows = []
#                             if plot_table is True and bar_num == 0:
#                                 # tab_colours = []
#                                 # tab_rows = []
#                                 try:
#                                     tab_text = values.values.reshape(
#                                         (self.bars.l, -1)
#                                     ).T
#                                 except ValueError:
#                                     tab_text = values.values
#                                 # tab_text = tab_text.astype(str)
#                                 tab_format = lambda x: rf"${x * 100:3.1f} \%$"
#                                 tab_format_vec = np.vectorize(tab_format)
#                                 tab_text = tab_format_vec(tab_text)
#                                 tab_cols = list(
#                                     self.bars.v[
#                                         : np.atleast_2d(tab_text).shape[0]
#                                     ]
#                                 )
#
#                             x_ticks.append(
#                                 self.bars.dict[vbar] * (barwidth + xpad)
#                             )
#                             x_labels.append(vbar)
#                             bottom = 0.0
#                             bar_vals = values.xs(
#                                 vbar, level=self.bars, drop_level=False
#                             )
#                             # tab_text.append(bar_vals.values)
#
#                             cmap = colormaps[
#                                 colour_df.loc[bar_vals.index].values[0]
#                             ]
#
#                             peaks = self.data.get_bar_peaks(
#                                 atom_type=col, other=None
#                             )
#                             if np.all(bar_vals.values) >= 0:
#                                 # table_text.append([f'${v * 100:3.1f} %$' for v in bar_vals.values])
#                                 x_id, y_id = (
#                                     self.x.dict[col],
#                                     self.y.dict[it[yid]],
#                                 )
#                                 # if plot_table is True:
#                                 #     # logger.info(y_id)
#                                 #     logger.info("Has table")
#                                 #     y_id *= 2
#                                 #     logger.info(y_id)
#                                 # print(y_id)
#
#                                 bar_val_view = bar_vals
#                                 bin_list = bar_val_view.index.get_level_values(
#                                     "x_bins"
#                                 )
#                                 bar_val_view.index = bin_list
#
#                                 x_tick = x_ticks[-1]
#
#                                 for bar_id, bar_val in enumerate(
#                                     bar_val_view.items()
#                                 ):
#                                     x_bin, y_val = bar_val
#
#                                     try:
#                                         peak = peaks[bar_id]
#                                     except IndexError:
#                                         peak = x_bin.right
#                                     colour = cmap(self.cnorm(peak))
#                                     # if colour not in table_col and cmap == table_cmap:
#                                     #     print('colour', colour)
#                                     #     table_col.append(colour)
#
#                                     label = self._get_bin_label(
#                                         x_bin, bin_list
#                                     )
#
#                                     if bar_num == 0:
#                                         tab_colours.append(list(colour))
#                                         tab_rows.append(label)
#                                         print(tab_colours)
#
#                                     # if x_bin.right < np.max(self.x):
#                                     #     label = f'${x_bin.left} - {x_bin.right}$ \AA'
#                                     # else:
#                                     #     label = f'$ > {x_bin.left}$ \AA'
#                                     # if label not in table_rows and cmap == table_cmap:
#                                     #     table_rows.append(label)
#                                     if y_val >= 0.010:
#                                         # barwidth = bulk_edge - x_bin.left
#                                         # try:
#                                         # x_tick = x_ticks[-1] + barwidth
#                                         # x_ticks.append(x_tick)
#                                         # except IndexError:
#                                         #     x_tick = x_bin.left
#                                         try:
#                                             p = self.ax[y_id, x_id].bar(
#                                                 x_tick,
#                                                 y_val,
#                                                 label=label,
#                                                 bottom=bottom,
#                                                 width=barwidth,
#                                                 align="edge",
#                                                 color=colour,
#                                             )  # self.ax[y_id, x_id].bar_label(p, labels=[label],  #                          fmt='%s',  #                          label_type='center')
#                                         except IndexError:
#                                             p = self.ax[y_id].bar(
#                                                 x_tick,
#                                                 y_val,
#                                                 label=label,
#                                                 bottom=bottom,
#                                                 width=barwidth,
#                                                 align="edge",
#                                                 color=colour,
#                                             )  # self.ax[y_id, x_id].bar_label(p, labels=[label],  #                          fmt='%s',  #                          label_type='center')
#                                         # finally:
#                                         bottom += y_val
#                                 # try:
#                                 #     self.fig.subplots_adjust(left=0.2, bottom=0.2)
#                                 # except IndexError:
#                                 #     self.ax[y_id].subplots_adjust(left=0.2, bottom=0.2)
#                                 if plot_table is True:
#                                     # y_id += 1
#                                     # print(tab_colours)
#                                     logger.info(f"Has table, {y_id}")
#                                     has_table = False
#                                     try:
#                                         print("rows", tab_rows)
#                                         tab_colours = np.array(tab_colours)
#                                         print("colours", tab_colours)
#                                         # self.tabax[y_id, x_id].clear()
#                                         the_table: table = table(
#                                             self.ax[y_id, x_id],
#                                             cellText=tab_text,
#                                             rowLabels=tab_rows,
#                                             rowColours=tab_colours,
#                                             colLabels=tab_cols,
#                                             loc="bottom",
#                                             edges="horizontal",
#                                         )
#                                         has_table = True
#                                     except IndexError:
#                                         try:
#                                             # self.tabax[y_id].clear()
#                                             the_table: table = table(
#                                                 self.ax[y_id],
#                                                 cellText=tab_text,
#                                                 rowLabels=tab_rows,
#                                                 rowColours=tab_colours,
#                                                 colLabels=tab_cols,
#                                                 loc="bottom",
#                                                 edges="horizontal",
#                                             )
#                                             has_table = True
#                                         except ValueError:
#                                             logger.info("could not plot table")
#                                     except ValueError:
#                                         logger.info("could not plot table")
#                                     if has_table is True:
#                                         logger.info("Plotting table")
#                                         # renderer = self.fig.canvas.get_renderer()
#                                         the_table.auto_set_font_size(False)
#                                         the_table.set_fontsize(12)
#                                         h = the_table.get_celld()[
#                                             (0, 0)
#                                         ].get_height()
#                                         w = the_table.get_celld()[
#                                             (0, 0)
#                                         ].get_width()
#                                         table_cells = the_table.properties()[
#                                             "children"
#                                         ]
#                                         for cell in table_cells:
#                                             max_width = w
#                                             if cell.get_width() > max_width:
#                                                 max_width = cell.get_width()
#                                             # print(cell.get_y())
#                                             # cell.set_y(5)
#                                             # print(cell.get_y())
#                                             cell.set_width(barwidth + xpad)
#                                             cell.set_height(0.13)
#                                             # cell.set_fontsize(12)
#                                             cell.set_text_props(
#                                                 verticalalignment="center"
#                                             )
#                                         for cell in table_cells:
#                                             cell.set_width(
#                                                 max_width
#                                             )  # the_table.set_fontsize(tab_fontsize)  # has_table = True  # the_table.scale(1, 1.2)  # the_table.stale = True  # the_table.set_clip_on(False)  # the_table._update_positions(renderer)  # bbox = the_table.get_window_extent(renderer)  # the_table.set(clip_bbox=bbox)  # self.ax[y_id, x_id].stale = True  # renderer = self.fig.canvas.get_renderer()  #  #  # # text_height = the_table._approx_text_height()  #  # # the_table.auto_set_font_size(False)  # # new_font_size = the_table.FONTSIZE  # # the_table.FONTSIZE = tab_fontsize  # the_table.set_fontsize(tab_fontsize)  #  # cell = the_table[0, 0]  # # cell_height = cell.get_text()  # win_ext = the_table.get_window_extent(renderer=renderer)  # text_height = cell.get_text_bounds(renderer=renderer)[-1]  # tab_font = the_table.FONTSIZE  # text_height = cell.get_text_bounds()  # # the_table._do_  # tab_font_new = tab_fontsize  # # sys.exit(1)  # # text_height = cells.get_text_bounds()  #  # font_ratio = tab_fontsize / the_table.FONTSIZE  #  # # # the_table.set_fontsize(tab_font_new)  # # # tab_dict = the_table.get_celld()  # #  # # # for col_id in range(0, len(tab_cols)):  # # #     tab_dict[(0, col_id)].set_height(text_height + 0.1)  # # #     for row_id in range(1, len(tab_rows)):  # # #         tab_dict[(row_id, col_id)].set_height(text_height + 0.1)  # the_table.scale(1, font_ratio + 0.3)  # the_table._stale = True  # the_table[...].set_text_props(linespacing=1.5,  #                          verticalalignment='bottom')  # the_table.scale(1, 1.2)  # the_table.stale = True  # the_table.set_clip_on(False)  # the_table._update_positions(renderer)
#
#                                     else:
#                                         logger.info(
#                                             "No table"
#                                         )  # else:  # print(y_val)
#                     # print("table false")
#                     # sys.exit(1)
#                     # tab_dict = the_table.get_celld()
#                     # table = ax[y_id, x_id].table(cellText=table_text,
#                     #                              rowColours=table_col,
#                     #                              rowLabels=table_rows,
#                     #                              # colLables=...,
#                     #                              loc='bottom')
#                     # x_ticks = x_ticks[:-1]
#                     # x_ticks.append(bar_dict[vbar] * (barwidth + xpad))
#                     #                 values = values
#                     #                 print('try 1 done')
#                     #
#                     #                     # for shell in values:
#                     #                     # view for group and bars
#                     #                             label = f'${lims.left} - {lims.right}$ \AA'
#                     #
#                     #                     try:
#                     #                         print('try 2')
#                     #                         print(x_dict[col], y_dict[it[yid]])
#
#                     #                     except:
#                     # #                         raise ValueError
#                     #                         x_id, y_id = 0, y_dict[it[yid]]
#                     #                         label = f'${x_bin.left} - {x_bin.right}$ \AA'
#
#                     #                     if pi == 1:
#                     #                         legends[y_id, x_id].append(it[label_id])
#                     #                 else:
#                     #                     check_logger.info('NaN values')
#
#                     except KeyError:
#                         logger.info(f"No data for {pl}, {self.x}, {it}")
#
#             # x_ticks = [np.linspace(n_bar * bulk_edge + xpad,
#             #                        n_bar * bulk_edge + bulk_edge, int(bulk_edge)) for n_bar in range(lbars)]
#             # x_ticks = np.ravel(x_ticks)
#             # x_labels = np.tile(np.arange(0, bulk_edge, 1), lbars)
#             for i in range(self.y.l):
#                 # if plot_table is True:
#                 #     ax_multi = 2
#                 # else:
#                 #     ax_multi = 1
#                 # ax_i = i * ax_multi
#                 # print(f"Axis index: {ax_i}, multi: {ax_multi}")
#
#                 try:
#                     # self.ax[]
#                     # self.ax[i, 0].set_ylabel(
#                     #     f"{self.label_mod([(self.y.v[i], self.y)])}\n" + rowlabel
#                     # )
#                     for j in range(self.x.l):
#                         # if plot_table is True:
#                         #     self.ax[i, j].subplots_adjust(bottom=0.2)
#                         #     f"{self.label_mod([(self.y.v[i], self.y)])}\n" + rowlabel
#                         # )
#                         if j == 0:
#                             self.ax[i, 0].set_ylabel(
#                                 f"{self.label_mod([(self.y.v[i], self.y)])}\n"
#                                 + rowlabel
#                             )
#                             self.ax[i, j].set_yticks(np.arange(0.0, 1.1, 0.2))
#                         else:
#                             self.ax[i, j].set_yticks(
#                                 np.arange(0.0, 1.1, 0.2), label=""
#                             )
#                             self.ax[i, j].set_yticklabels(
#                                 []
#                             )  # np.arange(0.0, 1.1, 0.2))
#                         self.ax[i, j].spines[["top", "right"]].set_visible(
#                             False
#                         )
#                         self.ax[i, j].hlines(
#                             1.0,
#                             -xpad,
#                             self.bars.l * (barwidth + xpad) + xpad,
#                             linestyle="--",
#                         )
#                         # self.ax[i, j].legend(ncol=2, loc='lower center')#[leg for leg in legends[i, j]], ncol=3)
#                         #                  if xlim != None:
#                         self.ax[i, j].set_xlim(
#                             (-xpad, self.bars.l * (barwidth + xpad))
#                         )
#                         self.ax[i, j].set_xticks([], [])
#                         #                 if ylim != None:
#                         self.ax[i, j].set_ylim((0.0, 1.2))
#                         self.ax[self.y.l - 1, j].set_xticks(
#                             np.array(x_ticks) + 0.5 * barwidth, x_labels
#                         )
#                         self.ax[self.y.l - 1, j].set_xlabel(
#                             bars
#                             + f"\n{self.label_mod([(self.x.v[j], self.x)])}"
#                         )  # self.ax[i, j].set_yticklabels(np.arange(0.0, 1.1, 0.2))
#                 except IndexError:
#                     self.ax[i].set_ylabel(
#                         f"{self.label_mod([(self.y.v[i], y)])}\n" + rowlabel
#                     )
#                     # self.ax[i].legend([self.label_mod([(leg, self.label_key)]) for leg in self.legends[i, 0]], ncol=3)
#                     self.ax[self.y.l - 1].set_xlabel(
#                         columnlabel
#                         + f"\n{self.label_mod([(self.x.v[0], self.x)])}"
#                     )
#             # # #
#             self.fig.supxlabel(f"{self.title_dict[x]}s", size=14)
#             self.fig.supylabel(f"{self.title_dict[y]}s", size=14)
#             if plsave != False:
#                 logger.info("Saving plot")
#                 if type(plsave) == str and plsave != "":
#                     outname = f"{plsave}.png"
#                 else:
#                     outname = f"{self.name}.png"
#                 odir = Path(odir).absolute()
#                 logger.info(f"output to {odir.absolute()}")
#                 if not odir.is_dir():
#                     os.makedirs(odir)
#                 logger.info(odir)
#                 logger.info(outname)
#                 self.fig.savefig(str(odir / outname))
#             else:
#                 plt.show()
#             self.fig.clear()
#
#     def _get_bin_label(self, x_bin, bin_list):
#         if x_bin.right < np.max(bin_list.right):
#             label = f"${x_bin.left:>3.1f} - {x_bin.right:3.1f}$ \AA"  # barwidth = x_bin.right - x_bin.left
#
#         else:
#             label = f"$ > {x_bin.left:>3.1f}$ \AA"
#         return label


# class HistPlot2D(Plot):
#     # __slots__ = ()
#     sel_list = ["clays", "ions", "aas", "_atoms"]
#     title_dict = {
#         "clays": "Clay type",
#         "ions": "Ion type",
#         "aas": "Amino acid",
#         "atoms": "Atom type",
#         "other": "Other atom type",
#     }
#     cmaps = [
#         "spring",
#         "summer",
#         "autumn",
#         "winter",
#         "cool",
#         "Wistia",
#         "hot",
#         "afmhot",
#         "gist_heat",
#         "copper",
#     ]
#     _init_args = {"x", "y", "bars"}
#     _plot_args = {
#         "rowlabel": "y",
#         "columnlabel": "x",
#         "dpi": None,
#         "figsize": None,
#     }
#
#     def __init__(self, data, sdir=None, **kwargs):
#         logger.info(f"Initialising {self.__class__.__name__}")
#         super().__init__(data, **kwargs)
#         self.bins = "x_bins"
#         self.group = "atoms"
#         self.bars = None
#
#     def plot_args_decorator(select: str):
#         def plot_decorator(plot_f):
#             def wrapper(self, **kwargs):
#                 self.select = select
#                 plot_f(self, **kwargs)
#                 self.select = None
#
#             return wrapper
#
#         return plot_decorator
#
#     @plot_args_decorator(select="ions")
#     def plot_ions(
#         self,
#         bars: Literal["clays", "aas", "ions", "other"],
#         x: Literal["clays", "aas", "ions", "other"],
#         y: Literal["clays", "aas", "ions", "other"],
#         col_sel,
#         edges: Tuple[float, float] = None,
#         nbins: int = None,
#         rowlabel: str = "y",
#         columnlabel: str = "x",
#         figsize=None,
#         dpi=None,
#         xmax=50,
#         ymax=50,
#         plsave=False,
#         ylim=None,
#         odir=".",
#         barwidth=0.75,
#         xpad=0.25,
#         cmap="winter",
#         plot_table=False,
#         get_new_data=False,
#     ):
#         self.plot(
#             bars,
#             x,
#             y,
#             col_sel,
#             edges,
#             nbins,
#             rowlabel,
#             columnlabel,
#             figsize,
#             dpi,
#             xmax,
#             ymax,
#             plsave,
#             ylim,
#             odir,
#             barwidth,
#             xpad,
#             cmap,
#             plot_table=plot_table,
#             get_new_data=get_new_data,
#         )
#
#     @plot_args_decorator(select="other")
#     def plot_other(
#         self,
#         bars: Literal["clays", "aas", "ions", "other"],
#         x: Literal["clays", "aas", "ions", "other"],
#         y: Literal["clays", "aas", "ions", "other"],
#         col_sel: str,
#         edges: Tuple[float, float] = None,
#         nbins: int = None,
#         rowlabel: str = "y",
#         columnlabel: str = "x",
#         figsize=None,
#         dpi=None,
#         xmax=50,
#         ymax=50,
#         plsave=False,
#         ylim=None,
#         odir=".",
#         barwidth=0.75,
#         xpad=0.25,
#         cmap="winter",
#         plot_table=None,
#         get_new_data=False,
#     ):
#         self.plot(
#             bars,
#             x,
#             y,
#             col_sel,
#             edges,
#             nbins,
#             rowlabel,
#             columnlabel,
#             figsize,
#             dpi,
#             xmax,
#             ymax,
#             plsave,
#             ylim,
#             odir,
#             barwidth,
#             xpad,
#             cmap,
#             get_new_data=get_new_data,
#         )
#
#     def get_suptitle(self, pl, separate):
#         logger.info(self.title_dict)
#         self.fig.suptitle(
#             (
#                 ", ".join([self.title_dict[s].upper() for s in separate])
#                 + f": {self.label_mod(list(tuple(zip(pl, separate))))}"
#             ),
#             size=16,
#             weight="bold",
#         )
#
#     @redirect_tqdm
#     def plot(
#         self,
#         bars: Literal["clays", "aas", "ions", "other"],
#         x: Literal["clays", "aas", "ions", "other"],
#         y: Literal["clays", "aas", "ions", "other"],
#         col_sel: Union[str, List[str]],
#         edges: Tuple[float, float] = None,
#         nbins: int = None,
#         rowlabel: str = "y",
#         columnlabel: str = "x",
#         figsize=None,
#         dpi=None,  # diff=False,
#         xmax=50,
#         ymax=50,
#         plsave=False,
#         ylim=None,
#         odir=".",
#         barwidth=0.75,
#         xpad=0.25,
#         cmap="winter",
#         tab_fontsize=12,
#         plot_table=False,
#         get_new_data=True,
#     ):
#         """Create stacked Histogram adsorption shell populations."""
#
#         logger.info(f"plot table {plot_table}")
#
#         self._init_plot(x, y, bars)
#         self.col_sel = col_sel
#         self._process_plot_data(
#             get_new_data=get_new_data,
#             col_sel=self.col_sel,
#             v_edges=edges,
#             v_nbins=nbins,
#         )
#         plot_df = self.plot_df.copy()
#         # plot_df.sort_index(inplace=True, sort_remaining=True)
#         colour_df = self.colour_df.copy()
#         # colour_df.sort_index(inplace=True, sort_remaining=True)
#
#         avail_attrs = plot_df.index.names
#         logger.info(f"Available attributes: {avail_attrs}")
#
#         logger.info("Setting x:")
#         self.x = Axis(self, x)
#         logger.info("Setting y:")
#         self.y = Axis(self, y)
#         logger.info("Setting bars:")
#         self.bars = Axis(self, bars)
#
#         separate = [
#             s
#             for s in avail_attrs
#             if s not in [self.x, self.y, self.bars, self.bins]
#         ]
#
#         if len(separate) != 0:
#             logger.info(f"Separate plots: {separate}")
#
#         idx = pd.Index(
#             [
#                 s
#                 for s in avail_attrs
#                 if (
#                     s != self.x
#                     and s != self.bars
#                     and s not in [*separate, self.bins]
#                 )
#             ]
#         )
#
#         logger.info(f"Iteration index: {idx.values}")
#
#         sep = pd.Index(separate)
#         logger.info(f"{sep}")
#
#         yid = np.ravel(np.where(np.array(idx) == self.y))[0]
#
#         if figsize is None:
#             if plot_table is True:
#                 figsize = self.get_figsize(xmax=xmax, ymax=ymax * 2)
#             else:
#                 figsize = self.get_figsize(xmax=xmax, ymax=ymax)
#
#         # set resultion
#         if dpi == None:
#             dpi = 100
#
#         # get plotting iter from index
#         iters = self._get_idx_iter(idx=idx, plot_df=plot_df)
#         plot_df.sort_index(
#             level=[*sep, *idx.values], sort_remaining=True, inplace=True
#         )
#         # colour_df.sort_index(level=)
#         logger.info(
#             f"Printing bar plots for {self.bars}\nColumns: {self.x.v}\nRows: {self.y.v}"
#         )
#
#         # set label modifier function
#         self.label_mod = self.modify_plot_labels
#
#         try:
#             # iterator for more than one plot
#             sep_it = self._get_idx_iter(idx=sep, plot_df=plot_df)
#             logger.info(f"{sep_it}")
#         except ValueError:
#             # only one plot
#             sep_it = [None]
#             logger.info(f"{sep_it}")
#
#         # iterate over separate plots
#         for pl in sep_it:
#             if pl is None:
#                 pl_str = ""
#             else:
#                 pl_str = f"_{pl[0]}"
#             self.savename = f"{self.name}_{x}_{y}_{bars}{pl_str}_{self.data.cutoff}_{self.data.bins}"
#             logger.info(f"plot {self.savename}")
#
#             self.init_legend(ly=self.y.l, lx=self.x.l)
#
#             # generate figure and axes array
#             self.fig = plt.figure(
#                 figsize=figsize, dpi=dpi, constrained_layout=True
#             )
#
#             self.gs = gridspec.GridSpec(
#                 nrows=self.y.l,
#                 ncols=self.x.l,
#                 figure=self.fig,  # width_ratios=[2,1]
#                 # sharey=True,
#                 # sharex=True
#             )
#
#             self.ax = {}
#
#             for i in range(self.y.l):
#                 for j in range(self.x.l):
#                     self.ax[i, j] = self.fig.add_subplot(self.gs[i, j])
#
#             # only one plot
#             if pl is None:
#                 logger.info(f"Generating plot")
#                 sepview = plot_df
#                 plsave = ""
#
#             else:
#                 logger.info(f"Generating {pl} plot:")
#                 # print(plot_df.head(20), "\n", separate, pl)
#                 sepview = plot_df.xs((pl), level=separate, drop_level=False)
#                 plsave = pl
#                 self.get_suptitle(pl, separate)
#
#             # set plot index
#             pi = 0
#
#             # iterate over subplot columns
#             for col in self.x.v:
#                 logger.info(f"Column {col}")
#
#                 try:
#                     view = sepview.xs((col), level=self.x, drop_level=False)
#                     pi = 1
#                 except ValueError:
#                     view = sepview
#                     col = self.x.v
#                     pi += 1
#
#                 for it in iters:
#                     try:
#                         values = view.xs(
#                             tuple(it), level=idx.tolist(), drop_level=False
#                         )
#                         # print(values.columns)
#                         x_labels = []
#                         x_ticks = []
#                         # print(self.bars.v)
#                         for bar_num, vbar in enumerate(self.bars.v):
#                             logger.info(f"Bar {bar_num}: {vbar}")
#                             tab_colours = []
#                             tab_rows = []
#                             if plot_table is True and bar_num == 0:
#                                 try:
#                                     tab_text = values.values.reshape(
#                                         (self.bars.l, -1)
#                                     ).T
#                                 except ValueError:
#                                     tab_text = values.values
#                                 # tab_text = tab_text.astype(str)
#                                 tab_format = lambda x: rf"${x * 100:3.1f} \%$"
#                                 tab_format_vec = np.vectorize(tab_format)
#                                 tab_text = tab_format_vec(tab_text)
#                                 tab_cols = list(
#                                     self.bars.v[
#                                         : np.atleast_2d(tab_text).shape[0]
#                                     ]
#                                 )
#                             print(self.bars.dict[vbar])
#                             print(barwidth, xpad)
#                             x_ticks.append(
#                                 self.bars.dict[vbar] * (barwidth + xpad)
#                             )
#                             x_labels.append(vbar)
#                             bottom = 0.0
#                             bar_vals = values.xs(
#                                 vbar, level=self.bars, drop_level=False
#                             )
#                             try:
#                                 col_1, col_2 = self.name.split("_")
#                                 hist = bar_vals[col_1].make_hist2d(
#                                     other=bar_vals[col_2]
#                                 )
#                             except:
#                                 bins = self.v_axis.bins
#                                 z_bins = self.z_axis.bins
#                                 bar_vals = np.ravel(bar_vals.values)[0]
#                                 print(bar_vals.__dict__)
#                                 print(bar_vals.timeseries)
#                                 print(bar_vals.zdata)
#                                 hist = bar_vals.make_hist2d(
#                                     bins=self.v_axis.bins,
#                                     z_bins=self.z_axis.bins,
#                                 )
#
#                                 # timeseries_z = bar_vals["timeseries_z"]  # bins_z = bar_vals["timeseries_z"]["edges_z"]  # timeseries = bar_vals["timeseries_v"]
#
#                             # tab_text.append(bar_vals.values)
#                             #
#                             cmap = colormaps[
#                                 colour_df.loc[bar_vals.index].values[0]
#                             ]
#                             #
#                             peaks = self.data.get_bar_peaks(
#                                 bar_vals=bar_vals, atom_type=col
#                             )
#                     #
#                     #     if np.all(bar_vals.values) >= 0:
#                     #         # table_text.append([f'${v * 100:3.1f} %$' for v in bar_vals.values])
#                     #         x_id, y_id = self.x.dict[col], self.y.dict[it[yid]]
#                     #         # if plot_table is True:
#                     #         #     # logger.info(y_id)
#                     #         #     logger.info("Has table")
#                     #         #     y_id *= 2
#                     #         #     logger.info(y_id)
#                     #         # print(y_id)
#                     #
#                     #         bar_val_view = bar_vals
#                     #         bin_list = bar_val_view.index.get_level_values("x_bins")
#                     #         bar_val_view.index = bin_list
#                     #
#                     #         x_tick = x_ticks[-1]
#                     #
#                     #         for bar_id, bar_val in enumerate(bar_val_view.items()):
#                     #
#                     #             x_bin, y_val = bar_val
#                     #
#                     #             try:
#                     #                 peak = peaks[bar_id]
#                     #             except IndexError:
#                     #                 peak = x_bin.right
#                     #             colour = cmap(self.cnorm(peak))
#                     #
#                     #             # if colour not in table_col and cmap == table_cmap:
#                     #             #     print('colour', colour)
#                     #             #     table_col.append(colour)
#                     #
#                     #             label = self._get_bin_label(x_bin, bin_list)
#                     #
#                     #             if bar_num == 0:
#                     #                 tab_colours.append(colour)
#                     #                 tab_rows.append(label)
#                     #             # if x_bin.right < np.max(self.x):
#                     #             #     label = f'${x_bin.left} - {x_bin.right}$ \AA'
#                     #             # else:
#                     #             #     label = f'$ > {x_bin.left}$ \AA'
#                     #             # if label not in table_rows and cmap == table_cmap:
#                     #             #     table_rows.append(label)
#                     #             if y_val >= 0.010:
#                     #                 # barwidth = bulk_edge - x_bin.left
#                     #                 # try:
#                     #                 # x_tick = x_ticks[-1] + barwidth
#                     #                 # x_ticks.append(x_tick)
#                     #                 # except IndexError:
#                     #                 #     x_tick = x_bin.left
#                     #                 try:
#                     #                     p = self.ax[y_id, x_id].bar(
#                     #                         x_tick,
#                     #                         y_val,
#                     #                         label=label,
#                     #                         bottom=bottom,
#                     #                         width=barwidth,
#                     #                         align="edge",
#                     #                         color=colour,
#                     #                     )
#                     #                     # self.ax[y_id, x_id].bar_label(p, labels=[label],
#                     #                     #                          fmt='%s',
#                     #                     #                          label_type='center')
#                     #                 except IndexError:
#                     #                     p = self.ax[y_id].bar(
#                     #                         x_tick,
#                     #                         y_val,
#                     #                         label=label,
#                     #                         bottom=bottom,
#                     #                         width=barwidth,
#                     #                         align="edge",
#                     #                         color=colour,
#                     #                     )
#                     #                     # self.ax[y_id, x_id].bar_label(p, labels=[label],
#                     #                     #                          fmt='%s',
#                     #                     #                          label_type='center')
#                     #                 # finally:
#                     #                 bottom += y_val
#                     except:
#                         print("except")
#
#     def save(self):
#         ...
#
#     def df_apply(self, dfs, function):
#         return [function(df) for df in dfs]
#
#     def _process_data(self, sdir=None, **kwargs):
#         self.sdir = sdir
#
#     @redirect_tqdm
#     def _process_plot_data(
#         self,
#         col_sel: Union[List[str], str],
#         col_abs: Optional[Union[Tuple[bool, bool], bool]] = None,
#         z_edges: Optional[Tuple[float, float]] = None,
#         v_edges: Optional[Tuple[float, float]] = None,
#         z_nbins: Optional[int] = None,
#         v_nbins: Optional[int] = None,
#         get_new_data=False,
#         # df_name: Union[Literal["df"], Literal["zf"]] = "df",
#         **kwargs,
#     ):
#         # data save directory
#         if self.sdir is None:
#             sdir = Path("../data/")
#         else:
#             sdir = Path(self.sdir)
#             if not sdir.is_dir():
#                 os.makedirs(sdir)
#         # list of matplotlib sequential cmaps
#         cmaps = [  # 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
#             "spring",
#             "summer",
#             "autumn",
#             "winter",
#             "cool",
#             "Wistia",
#             "hot",
#             "afmhot",
#             "gist_heat",
#             "copper",
#         ]
#         # df_name='df'
#         # for df_name in ['df', 'zf']:
#         # logger.info(f'df name {df_name}')
#         # plot_df = getattr(self.data, df_name).copy()
#         logger.info(f"Processing {self.__class__.__name__} data")
#         plot_df = self.data.df
#         # logger.info(f"{df_name}, {plot_df.index.names}")
#         idx_names = plot_df.index.names
#         atoms = plot_df.index.get_level_values("_atoms")
#         plot_df["colours"] = atoms
#         atom_types = atoms.unique()
#         colour_dict = dict(zip(atom_types, cmaps[: len(atom_types)]))
#         plot_df["colours"] = plot_df["colours"].transform(
#             lambda x: colour_dict[x]
#         )
#         # reorder index for grouping
#         colour_df = plot_df.pop("colours")
#         self._colour_df = colour_df
#         plot_df = plot_df.reorder_levels(idx_names)
#
#         # check column selection for values df
#         # if df_name == 'df':
#         # only one column
#         if type(col_sel) == list and len(col_sel) == 1:
#             col_sel = col_sel[0]
#         if type(col_sel) == str:
#             assert (
#                 col_sel in plot_df.columns
#             ), f"Selected column {col_sel} does not exist"
#             self.name = col_sel
#             self.other_col = None  # plot_col = [plot_df[col_sel]]
#         # 2 columns -> ratio between columns (col1 / col2)
#         else:
#             assert (
#                 type(col_sel) == list and len(col_sel) == 2
#             ), f"Found wrong length ({len(col_sel)}) for plotting df columns"
#             assert np.all(
#                 [c in plot_df.columns for c in col_sel]
#             ), f"One of {col_sel} is not in plotting df columns"
#             self.name = "_".join(col_sel)
#             self.other_col = col_sel[
#                 -1
#             ]  # plot_col = [plot_df[col_sel[0]], plot_df[col_sel[1]]]
#         col_len = len(self.name.split("_"))
#         # check if absolute values of columns should be used
#         # default is absolute
#         if col_abs is None:
#             self.col_abs = tuple([True for col in self.name.split("_")])
#         elif type(col_abs) == bool:
#             self.col_abs = [col_abs]
#             assert (
#                 len(col_abs) == col_len
#             ), f"Length for absolute value specifications ({len(col_abs)}) does not match length of columns ({col_len})"
#
#             # if df_name == 'zf':  #     # _plot_zf = plot_df.copy()  #     _plot_zf = pd.Series(index=plot_df.index, name='timeseries', dtype='object')  #     print(plot_df)  #     _plot_zf = plot_df[self.data.zname]  #     print(_plot_zf)  #     # _plot_zf.index = _plot_zf.index.to_flat_index()  # _plot_zf = dask.dataframe.from_pandas(_plot_zf, npartitions=5)
#         logger.info("Setting z-density bins")
#         if z_edges is None and z_nbins is None:
#             self.z_axis = HistData(bins=self.data.z_bins.bins)
#         else:
#             self.z_axis = HistData(
#                 min=z_edges[0], max=z_edges[1], nbins=z_nbins
#             )
#         logger.info(f"{self.z_axis}")
#         # save = (
#         #         sdir
#         #         / f"{df_name}_{self.data.cutoff}_{self.data.bins}.tar.xz"
#         # )
#         # if not save.is_file() or (save.is_file and get_new_data is True):
#         #     for idx, row in plot_df[self.data.zname].items():
#         #         fname = row
#         #         with open(fname, "rb") as f:
#         #             data = pkl.load(f)
#         # _plot_zf.assign(timeseries=lambda x: [np.ravel(pkl.load(open(x[self.zname], 'rb')).timeseries)])
#         #         _plot_zf.loc[idx] = _plot_zf.loc[idx].apply(lambda x: Timeseries(filename=x))
#         # logger.info(f"Saving {save!r}")
#         # _plot_zf.to_pickle(save)
#         # else:
#         #     logger.info(f"Reading {save!r}")
#         #     _plot_zf = pd.read_pickle(save)
#         # plot_df = _plot_zf.copy()
#         # del _plot_zf
#         # else:
#         # _plot_df = plot_df.copy()
#         # _plot_df.index = _plot_df.index.to_flat_index()
#         # _plot_df = dask.dataframe.from_pandas(_plot_df,
#         #     index=plot_df.index.to_flat_index(),
#         # _plot_df = pd.DataFrame(
#         #     index=plot_df.index,
#         #     columns=[
#         #         f"{x}_{y}"
#         #         for x in ["timeseries"]
#         #         for y in self.name.split("_")
#         #     ]
#         # )
#         #     npartitions=10)
#         logger.info(f"Setting histogram {self.name} bins")
#         if v_edges is not None and v_nbins is not None:
#             self.v_axis = HistData(
#                 min=v_edges[0], max=v_edges[1], nbins=v_nbins
#             )
#         elif col_len == 1:
#             with open(self.data.filelist[self.name][0], "rb") as file:
#                 data = pkl.load(file)
#                 self.v_axis = HistData(edges=data.edges)
#         elif col_len == 2 and v_nbins is not None:
#             self.v_axis = HistData(min=0, max=1, nbins=v_nbins)
#         else:
#             self.v_axis = HistData(
#                 min=0,
#                 max=1,
#                 nbins=HistData(
#                     bins=self.data.other_bins[self.col_sel[0]].bins
#                 ).nbins,
#             )  # raise ValueError(f"No bins specified")
#         logger.info(f"{self.v_axis}")
#         # for col_name in name.split('_'):
#         # save = (
#         #         sdir
#         #         / f"{col_name}_{self.data.cutoff}_{self.data.bins}.tar.xz"
#         # )
#         # if not save.is_file() or (save.is_file() and get_new_data is True):
#         # col_ts = f"timeseries_{col_name}"
#         # col = plot_df[col_name]
#         # _plot_df[col_ts] = _plot_df[col_name].map(lambda x: pkl.load(open(Path(x), 'rb')),
#         #                                           meta=pd.Series(index=_plot_df[col_name].index, dtype='object')).compute()
#         # _plot_df[col_ts] = _plot_df[col_ts].map(lambda x: [da.array.ravel(pkl.load(open(Path(x), 'rb')).timeseries)],
#         #                                           meta=pd.Series(index=_plot_df[col_name].index, dtype='object')).compute()
#         # _plot_df.assign(col_ts=lambda x: [np.ravel(pkl.load(open(x[col_name], 'rb')).timeseries)])
#         # col = col.apply(lambda x: Timeseries(filename=x))
#         # for idx, row in col.items():
#         #     fname = row
#         #     with open(fname, "rb") as f:
#         #         data = pkl.load(f)
#         # slice = _plot_df.xs(idx, drop_level=False)#[
#         # pd.IndexSlice[
#         #     f"timeseries_{col.name}"
#         # ]]
#         # ]_plot_df.loc[
#         #     idx,
#         #     pd.IndexSlice[
#         #         f"timeseries_{col.name}"
#         #     ],
#         # ]
#         # z_idx = slice.index.intersection(self.data.zf.index)
#         # print(z_idx, self.data.zf.index.names, slice.index.names)
#         # sel_id = self.data.zf.loc(z_idx)
#         # print(sel_id)
#         # _plot_df.loc[
#         #             idx,
#         #             pd.IndexSlice[
#         #                 f"timeseries_{col.name}"
#         #             ],
#         #         ].map(lambda x: setattr(x, 'zdata', self.zf.loc[z_idx])) #
#         # = [np.ravel(data.timeseries)]
#         # logger.info(f"Saving {save.name!r}")
#         # _plot_df.to_pickle(save)
#         # else:
#         #  logger.info(f"Reading {save.name!r}")
#         # _plot_df = pd.read_pickle(save)
#
#         # if col_len != 1:
#         #     # arr1 = dask.array.from_array(_plot_df.iloc[:, 0].values)
#         #     # arr2 = dask.array.from_array(_plot_df.iloc[:, 1].values)
#         #     arr1 = _plot_df.iloc[:, 0].values
#         #     arr2 = _plot_df.iloc[:, 1].values
#         #     arr = np.divide(arr1, arr2, where=arr2 != 0)
#         #     _plot_df["timeseries"] = arr
#         # else:
#         #     _plot_df["timeseries"] = _plot_df[f"timeseries_{name}"].values
#         #
#         # plot_df = _plot_df.copy()
#         # del _plot_df
#
#         plot_df = plot_df.loc[..., self.col_sel].copy()
#         # setattr(self, f"_plot_df", plot_df)
#
#         # plot_df = self._plot_df['timeseries'].copy()
#         # plot_df = plot_df.join(self._plot_zf, how='inner', lsuffix='_v', rsuffix='_z')
#         colour_df = self._colour_df
#
#         dfs = {"bin": plot_df, "colour": colour_df}
#
#         # split DataFrames for ions and other atom types
#         for key, df in dfs.items():
#             # get non-ion atom type names
#             not_ions = [a for a in self.data.atoms if a != "ions"]
#             # get only ion DataFrame slice
#             ion_view = df.xs("ions", level="atoms")
#             # remove ions from index
#             ion_view.index = ion_view.index.droplevel(["ions"])
#             # rename _atoms atom type column to replace generic atoms column
#             idx_names = [idstr.strip("_") for idstr in ion_view.index.names]
#             ion_view.index.names = idx_names
#             # get only non-ion DataFrame slice
#             other_view = df.loc[pd.IndexSlice[:, :, :, :, not_ions, :]]
#             # remove unnecessary _atoms column from index
#             other_view.index = other_view.index.droplevel("_atoms")
#             # set dataframes as attributes
#             setattr(self, f"_ions_{key}_df", ion_view.copy())
#             setattr(self, f"_other_{key}_df", other_view.copy())
#
#     # else:
#     #     with open(savename)
#
#     def __repr__(self):
#         return self.plot_df.index.__repr__()
#
#     @property
#     def plot_df(self):
#         select = self.select
#         if not hasattr(self, f"_{select}_bin_df"):
#             self._process_plot_data(col_sel=self.col_sel)
#         logger.info(self.select)
#         return getattr(self, f"_{select}_bin_df")
#
#     @property
#     def colour_df(self):
#         select = self.select
#         if not hasattr(self, f"_{select}_colour_df"):
#             self._process_plot_data(col_sel=self.col_sel)
#         return getattr(self, f"_{select}_colour_df")
#
#     def _get_idx_iter(self, idx, plot_df):
#         logger.info(f"idx: {idx}")
#         idx_iter = np.array(
#             np.meshgrid(
#                 *[
#                     plot_df.index.get_level_values(idxit).unique()
#                     for idxit in idx
#                 ]
#             )
#         ).T.reshape(-1, len(idx))
#         logger.info(f"{idx_iter}")
#         return idx_iter
#
#     def _get_attr(self, attr: Literal["clays", "aas", "atoms", "ions"]):
#         return pd.unique(self.plot_df.index.get_level_values(attr))
#
#     get_clays = partialmethod(_get_attr, attr="clays")
#     get_aas = partialmethod(_get_attr, attr="aas")
#     get_ions = partialmethod(_get_attr, attr="ions")
#     get_atoms = partialmethod(_get_attr, attr="atoms")
#     get_other = partialmethod(_get_attr, attr="other")


# class ModTable(matplotlib.table.Table):
#     def __new__(cls, *args, **kwargs):
#         super().FONTSIZE = 12
#         self = super().__init__(*args, **kwargs)
#         # self.FONTSIZE = 12
#         return self
#


# class HistData:
#     __slots__ = ["min", "max", "nbins", "edges", "bins", "stepsize"]
#
#     def __init__(
#         self,
#         min=None,
#         max=None,
#         nbins=None,
#         edges=None,
#         bins=None,
#         stepsize=None,
#     ):
#         if bins is not None:
#             self.bins = bins.copy()
#             for attr in self.__class__.__slots__:
#                 if attr != "bins":
#                     logger.info(f"{attr}")
#                     assert eval(f"{attr} is None")
#             logger.info(f"{self.bins}")
#             self.stepsize = np.round(self.bins[1] - self.bins[0], 5)
#             self.nbins = len(bins)
#             self.edges = np.linspace(
#                 self.bins[0] - 1 / 2 * self.stepsize,
#                 self.bins[-1] + 1 / 2 * self.stepsize,
#                 self.nbins + 1,
#             )
#             self.min, self.max = np.min(self.edges), np.max(self.edges)
#         elif edges is not None:
#             for attr in self.__class__.__slots__:
#                 self.edges = edges.copy()
#                 if attr != "ads_edges":
#                     assert eval(f"{attr} is None")
#             self.min, self.max = np.min(self.edges), np.max(self.edges)
#             self.nbins = len(self.edges) - 1
#             self.stepsize = self.edges[1] - self.edges[0]
#             self.get_bins_from_edges()
#         elif min is not None and max is not None and nbins is not None:
#             assert stepsize is None
#             self.min, self.max, self.nbins = float(min), float(max), int(nbins)
#             self.edges = np.linspace(self.min, self.max, self.nbins + 1)
#             self.stepsize = self.edges[1] - self.edges[0]
#             self.get_bins_from_edges()
#
#         elif min is not None and max is not None and stepsize is not None:
#             self.min, self.max, self.stepsize = (
#                 float(min),
#                 float(max),
#                 float(stepsize),
#             )
#             self.edges = np.arange(
#                 self.min, self.max + self.stepsize, self.stepsize
#             )
#             self.get_bins_from_edges()
#             self.nbins = len(self.bins)
#
#         else:
#             raise ValueError(
#                 f"No initialisation with selected arguments is implemented."
#             )
#         self.stepsize = Bins(self.stepsize)
#         self.min, self.max = Cutoff(self.min), Cutoff(self.max)
#
#     def get_bins_from_edges(self):
#         self.bins = np.linspace(
#             self.min + 0.5 * self.stepsize,
#             self.max - 0.5 * self.stepsize,
#             self.nbins,
#         )
#
#     def __str__(self):
#         return f"HistData([{self.min}:{self.max}:{self.stepsize}])"


if __name__ == "__main__":
    # d = RawData("../test_data/")
    data = Data(
        "/storage/results/zdist_full",
        cutoff=50,
        bins=0.02,
        odir="/storage/plots/zdist_full",  # _plots",
        namestem="zdist",
        nameparts=1,
        analysis="zdens",  # aas=["ctl"],
    )
    logger.info(f"Start")
    # vdata = Data2D(
    #     indir="/storage/results/rdf_full",
    #     zdir="/storage/results/zdist_full",
    #     cutoff=50,
    #     bins=0.02,
    #     # aas=['gly'],
    #     # clays=['NAu-1'],
    #     # ions=['Na'],
    #     odir="/storage/plots/rdf_full",  # _plots",
    #     namestem="rdf",
    #     other=['OW'],
    #     zstem="zdist",
    #     nameparts=1,
    #     zname="zdens",
    #     # aas=["ctl"],
    #     # load='../data/vdata.pkl'
    #     analyses=['rdens']
    # )
    # vdata.save("../data/rdata.pkl")
    # print(vdata.df)
    # print(vdata.zf)
    # print(data.df)
    # for i in ['Na', 'K', 'Mg', 'Ca', '']:
    # data._get_edges()
    # data.plot(x='clays', y='atoms', select='aas')

    # plot = HistPlot2D(vdata)
    # plot = HistPlot(data)
    plot_l = LinePlot(data)
    # plot_l.select = 'ions'
    # plot.select = 'ions'
    # print(plot_l._plot_df, plot_l._colour_df)#, plot_l.colour_df)
    # plot_l.plot_ions(
    #     y="aas", x="atoms", lines="clays", xlim=10, ylim=0.4
    # )  # , plot_table=True)
    plot_l.plot_ions(
        y="atoms",
        x="clays",
        lines="aas",  # ylim=0.1,
        dpi=200,
        edges=True,
        columnlabel=r"distance from surface (\AA)",
        rowlabel=r"$\rho_z$ ()",
        plsave=True,
        xlim=50,
        odir="/storage/plots/zdist_u/",
    )  # , plot_table=True)
    # plot_l.plot_ions(
    #     y="aas",
    #     x="clays",
    #     lines="atoms",
    #     # ylim=0.1,
    #     dpi=200,
    #     ads_edges=False,
    #     columnlabel=r"distance from surface (\AA)",
    #     rowlabel="position density ()",
    #     plsave=True,
    #     xlim=10,
    #     odir='/storage/plots/zdist_u/'
    # )  # , plot_table=True)
    plot_l.plot_other(
        y="aas",
        x="ions",
        lines="clays",  # ylim=0.02,
        dpi=200,
        edges=False,
        columnlabel=r"distance from surface (\AA)",
        rowlabel=r"$\rho_z$ ()",
        xlim=50,
        plsave=True,
        odir="/storage/plots/zdist_u/",
    )  # plot_l.plot_other(  #     y="aas",  #     x="ions",  #     lines="atoms",  #     # ylim=0.02,  #     dpi=200,  #     xlim=20,  #     ads_edges=False,  #     columnlabel=r"distance from surface (\AA)",  #     rowlabel="position density ()",  #     plsave=True,  #     odir='/storage/plots/zdist_u/'  # )  # plot.plot_ions(  #     y="aas",  #     x="atoms",  #     bars="clays",  #     plot_table=False,  #     plsave=True,  #     odir="/storage/plots/zdist_r/",  #     col_sel=["rdens"],  #     get_new_data=True,  #     ads_edges=[0, 1],  #     nbins=200,  # )  # plot.plot_other(y="aas", x="ions", bars="clays", plot_table=True)  # plot_l.plot_ions(y="aas", x="atoms", bars="clays", plot_table=True,  #                plsave=True, odir="/storage/plots/zdist_h/",)  # fig = data.plot_bars(y='ions', x='clays', bars='aas')


# class AtomHistPlot(Plot):
#     sel_list = ["clays", "ions", "aas", "atoms"]
#     title_dict = {
#         "clays": "Clay type",
#         "ions": "Ion type",
#         "aas": "Amino acid",
#         "atoms": "Atom type",
#         "other": "Other atom type",
#     }
#     cmaps = [
#         "spring",
#         "summer",
#         "autumn",
#         "winter",
#         "cool",
#         "Wistia",
#         "hot",
#         "afmhot",
#         "gist_heat",
#         "copper",
#     ]
#     id_dict = {
#         "A": r"\alpha",
#         "B": r"\beta",
#         "G": r"\gamma",
#         "D": r"\delta",
#         "H": r"\eta",
#         "E": r"\epsilon",
#         "Z": r"\zeta",
#         "R": "side",
#     }
#     _init_args = {"x", "y", "bars", "group_atoms"}
#     _plot_args = {
#         "rowlabel": "y",
#         "columnlabel": "x",
#         "dpi": None,
#         "figsize": None,
#     }
#
#     def __init__(self, data, sdir=None, group_atoms=False, **kwargs):
#         logger.info(f"Initialising {self.__class__.__name__}")
#         super().__init__(data, sdir=sdir, group_atoms=group_atoms, **kwargs)
#         self.bins = "x"
#         self.group = "aas"
#         self.bars = "x"
#
#     def plot_args_decorator(select: str):
#         def plot_decorator(plot_f):
#             def wrapper(self, **kwargs):
#                 self.select = select
#                 plot_f(self, **kwargs)
#                 self.select = None
#
#             return wrapper
#
#         return plot_decorator
#
#     def _split_plot_dfs(self):
#         self._plot_df = ...
#         logger.info(f"Splitting binned plot and colour dfs")
#         bin_df = self._plot_df.copy()
#         plot_df = self._colour_df.copy()
#         dfs = {"bin": bin_df, "colour": plot_df}
#         for key, df in dfs.items():
#             not_ions = [a for a in self.data.atoms if a != "ions"]
#             ion_view = df.xs("ions", level="atoms")
#             ion_view.index = ion_view.index.droplevel(["ions"])
#             ion_view.index.names = [
#                 idstr.strip("_") for idstr in ion_view.index.names
#             ]
#             # logger.info(f'\n{not_ions}, {df.index.names}')
#             other_view = df.loc[pd.IndexSlice[:, :, :, :, not_ions, :]]
#             other_view.index = other_view.index.droplevel("_atoms")
#             # logger.info(f'{other_view.index.get_level_values("atoms").unique()}, {other_view.index.names}')
#             setattr(self, f"_ions_{key}_df", ion_view.copy())
#             setattr(self, f"_other_{key}_df", other_view.copy())
#
#     def get_atom_type(self, atom_name):
#         if atom_name in ["O", "OT", "OXT"]:
#             atom_name = "OT"
#         elif len(atom_name) == 1:
#             pass
#         else:
#             atom_type = atom_name[0]
#             atom_pos_id = atom_name[1]
#             if atom_type != "C" and atom_pos_id == "H":
#                 pass
#             else:
#                 atom_pos_id = self.id_dict[atom_pos_id]
#             if atom_pos_id != "H":
#                 atom_pos_id = "_{" + atom_pos_id
#             else:
#                 atom_pos_id = atom_pos_id + "_{"
#             try:
#                 atom_id = re.search(r"[A-Z]+([0-9])", atom_name).group(1)
#                 atom_pos_id += atom_id
#             except AttributeError:
#                 pass
#             finally:
#                 atom_pos_id += "}"
#             atom_name = rf"{atom_type}{atom_pos_id}"
#         atom_name = r"$\mathrm{" + rf"{atom_name}" + "}$"
#         return atom_name
#
#     # def _get_colour_dict(self, group_atoms=True):
#     #     from seaborn import set_hls_values, saturate
#     #
#     #     # atom type colour cycle
#     #     colours = [
#     #         "#8dd3c7",
#     #         "#ffffb3",
#     #         "#bebada",
#     #         "#fb8072",
#     #         "#80b1d3",
#     #         "#fdb462",
#     #         "#b3de69",
#     #         "#fccde5",
#     #     ]
#     #     colours = [set_hls_values(saturate(cval), l=0.4) for cval in colours]
#     #     plot_df = self.data.df.copy()
#     #     plot_df.reset_index("x", inplace=True)
#     #     plot_df['x'] = plot_df["x"].apply(
#     #         lambda x: self.get_atom_type_group(x, group_all_atoms=group_atoms)
#     #     )
#     #     plot_df.set_index("x", append=True, inplace=True)
#     #     plot_df = plot_df.groupby(plot_df.index.names).sum()
#     #     atoms = plot_df.index.get_level_values("x")
#     #     atoms.transform(lambda x: x[0])
#     #     # plot_df = pd.DataFrame({'a-types': atoms, 'values': plot_df, 'colours': atoms})
#     #     atom_types = atoms.unique()
#     #     colour_dict = dict(zip(atom_types, colours[: len(atom_types)]))
#     #     # plot_df["colours"] = plot_df["colours"].transform(lambda x: colour_dict[x])
#     #     return colour_dict
#
#     def _process_data(self, group_atoms=False, **kwargs):
#         logger.info(f"Getting binned plot and colour dfs")
#         logger.info(f"Getting plot and colour dfs")
#         colour_dict = self._get_colour_dict()
#         colour_dict = {k: [v, "k"] for k, v in colour_dict.items()}
#
#         # atom type colour cycle
#         # colours = [
#         #     "#8dd3c7",
#         #     "#ffffb3",
#         #     "#bebada",
#         #     "#fb8072",
#         #     "#80b1d3",
#         #     "#fdb462",
#         #     "#b3de69",
#         #     "#fccde5",
#         # ]
#         # colours = [set_hls_values(saturate(cval), l=0.4) for cval in colours]
#         plot_df = self.data.df.copy()
#         # plot_df = plot_df.groupby(plot_df.index.names).sum()
#         plot_df.reset_index("x", inplace=True)
#         plot_df["x"] = plot_df["x"].apply(
#             lambda x: self.get_atom_type_group(x, group_all_atoms=group_atoms)
#         )
#         plot_df.set_index("x", append=True, inplace=True)
#         plot_df = plot_df.groupby(plot_df.index.names).sum()
#         self.atomname = plot_df.index.get_level_values("atoms").unique()[0]
#         plot_df.index = plot_df.index.droplevel("atoms")
#         atoms = plot_df.index.get_level_values("x")
#
#         plot_df["colours"] = atoms
#         plot_df["colours"] = plot_df["colours"].transform(lambda x: x[0])
#         # plot_df = pd.DataFrame({'a-types': atoms, 'values': plot_df, 'colours': atoms})
#         # atom_types = plot_df["colours"].unique()
#         # colour_dict = dict(zip(atom_types, colours[: len(atom_types)]))
#         plot_df["colours"] = plot_df["colours"].transform(
#             lambda x: colour_dict[x]
#         )
#
#         # separate colour column from plot_df -> yields 2 Series
#         colour_df = plot_df["colours"]
#         plot_df = plot_df["values"]
#         self._plot_df, self._colour_df = plot_df.copy(), colour_df.copy()
#         logger.info(f"{self._plot_df}\n{self._colour_df}")
#
#     @property
#     def plot_df(self):
#         try:
#             plot_df = getattr(self, f"_plot_df")
#             # logger.info(f'{plot_df.index.get_level_values("atoms").unique()}')
#             return plot_df
#         except AttributeError:
#             logging.error(f"Wrong usage of property, select not assigned yet.")
#
#     @property
#     def colour_df(self):
#         try:
#             return getattr(self, f"_colour_df")
#         except AttributeError:
#             logging.error(f"Wrong usage of property, select not assigned yet.")
#
#     def save(self, **kwargs):
#         ...
#
#     @cached_property
#     def cnorm(self) -> mpc.Normalize:
#         # normalise colour map from 0 to max_edge
#         return mpc.Normalize(vmin=0, vmax=self.data.max_shell_edge, clip=False)
#
#     def _get_attr(self, attr: Literal["clays", "aas", "atoms", "ions"]):
#         return pd.unique(self.plot_df.index.get_level_values(attr))
#
#     get_clays = partialmethod(_get_attr, attr="clays")
#     get_aas = partialmethod(_get_attr, attr="aas")
#     get_ions = partialmethod(_get_attr, attr="ions")
#     # get_atoms = partialmethod(_get_attr, attr="atoms")
#     get_other = partialmethod(_get_attr, attr="other")
#
#     @redirect_tqdm
#     @plot_args_decorator(select="other")
#     def plot_other(
#         self,
#         bars: Literal["clays", "aas", "ions", "other"],
#         x: Literal["clays", "aas", "ions", "other"],
#         y: Literal["clays", "aas", "ions", "other"],
#         rowlabel: str = "y",
#         columnlabel: str = "x",
#         figsize=None,
#         dpi=None,
#         xmax=50,
#         ymax=50,
#         plsave=False,
#         ylim=None,
#         odir=".",
#         barwidth=0.75,
#         xpad=0.25,
#         cmap="winter",  # group_atoms=False,
#         plot_table=None,
#     ):
#         self.plot(
#             bars=bars,
#             x=x,
#             y=y,
#             rowlabel=rowlabel,
#             columnlabel=columnlabel,
#             figsize=figsize,
#             dpi=dpi,
#             xmax=xmax,
#             ymax=ymax,
#             plsave=plsave,
#             ylim=ylim,
#             odir=odir,
#             barwidth=barwidth,
#             xpad=xpad,
#             cmap=cmap,
#             # group_atoms=group_atoms
#         )
#
#     def get_suptitle(self, pl, separate):
#         self.fig.suptitle(
#             (
#                 ", ".join([self.title_dict[s].upper() for s in separate])
#                 + f": {self.label_mod(list(tuple(zip(pl, separate))))}"
#             ),
#             size=16,
#             weight="bold",
#         )
#
#     @redirect_tqdm
#     def plot(
#         self,
#         bars: Literal["clays", "aas", "ions", "other"],
#         x: Literal["clays", "aas", "ions", "other"],
#         y: Literal["clays", "aas", "ions", "other"],
#         rowlabel: str = "y",
#         columnlabel: str = "x",
#         figsize=None,
#         dpi=None,  # diff=False,
#         xmax=50,
#         ymax=50,
#         plsave=False,
#         ylim=None,
#         odir=".",
#         barwidth=0.75,
#         xpad=0.25,
#         cmap="winter",
#         # group_atoms=False,
#         tab_fontsize=12,
#         plot_table=False,
#     ):
#         """Create stacked Histogram adsorption shell populations."""
#
#         logger.info(f"plot table {plot_table}")
#
#         hatch_list = [
#             r"//",
#             r"--",
#             r"\\",
#             r"//",
#             r".",
#             r"///",
#             r"...",
#             "x",
#             r"+",
#             r"..",
#         ]
#
#         self._init_plot(x, y, bars)
#
#         plot_df = self.plot_df.copy()
#         colour_df = self.colour_df.copy()
#
#         avail_attrs = plot_df.index.names
#         logger.info(f"Available attributes: {avail_attrs}")
#
#         logger.info("Setting x:")
#         self.x = Axis(self, x)
#         logger.info("Setting y:")
#         self.y = Axis(self, y)
#         logger.info("Setting bars:")
#         self.bars = Axis(self, bars)
#
#         print(avail_attrs)
#         separate = [
#             s
#             for s in avail_attrs
#             if s not in [self.x, self.y, self.bars, self.bins]
#         ]
#
#         if len(separate) != 0:
#             logger.info(f"Separate plots: {separate}")
#
#         idx = pd.Index(
#             [
#                 s
#                 for s in avail_attrs
#                 if (
#                     s != self.x
#                     and s != self.bars
#                     and s not in [*separate, self.bins]
#                 )
#             ]
#         )
#
#         logger.info(f"Iteration index: {idx.values}")
#
#         sep = pd.Index(separate)
#         # logger.info(f'{sep}')
#
#         yid = np.ravel(np.where(np.array(idx) == self.y))[0]
#
#         if figsize is None:
#             if plot_table is True:
#                 figsize = self.get_figsize(xmax=xmax, ymax=ymax * 2)
#             else:
#                 figsize = self.get_figsize(xmax=xmax, ymax=ymax)
#
#         # set resultion
#         if dpi is None:
#             dpi = 100
#
#         # get plotting iter from index
#         iters = self._get_idx_iter(idx=idx)
#
#         logger.info(
#             f"Printing bar plots for {self.bars}\nColumns: {self.x.v}\nRows: {self.y.v}"
#         )
#
#         # set label modifier function
#         self.label_mod = self.modify_plot_labels
#
#         try:
#             # iterator for more than one plot
#             sep_it = self._get_idx_iter(idx=sep)  # logger.info(f'{sep_it}')
#         except ValueError:
#             # only one plot
#             sep_it = [None]
#
#         # iterate over separate plots
#         for pl in sep_it:
#             if pl is None:
#                 pl_str = ""
#             else:
#                 pl_str = f"_{pl[0]}"
#             self.name = f"{self.data.name}_{self.data.analysis}_{x}_{y}_{bars}{pl_str}_{self.data.cutoff}_{self.data.bins}"
#             if self.group_atoms != False:
#                 self.name += "_grouped"
#             logger.info(f"Plot name: {self.name!r}")
#             # index map for y values
#             # y_dict: dict = dict(zip(vy, np.arange(ly)))
#
#             self.init_legend(ly=self.y.l, lx=self.x.l)
#
#             # if plot_table is True:
#             #     plt_nrows = self.y.l * 2
#             #     logger.info('Has table')
#             # else:
#             #     plt_nrows = self.y.l
#             # print('rows ', plt_nrows)
#
#             # generate figure and axes array
#             self.ax = {}
#
#             if plot_table is True:
#                 self.fig = plt.figure(figsize=figsize, dpi=dpi)
#                 self.gs = gridspec.GridSpec(
#                     nrows=self.y.l,
#                     ncols=self.x.l,
#                     figure=self.fig,
#                     wspace=2 / self.x.l,
#                     hspace=0.5,  # width_ratios=
#                     # sharey=True,
#                     # sharex=True
#                 )
#                 self.subax = {}
#                 # self.tabax = {}
#                 for i in range(self.y.l):
#                     for j in range(self.x.l):
#                         #     if i == 0 and j == 0:
#                         #         self.subax[i, j] = gridspec.GridSpecFromSubplotSpec(
#                         #     2, 1, subplot_spec=self.gs[i, j]
#                         # )
#                         #     else:
#                         self.subax[i, j] = self.gs[i, j].subgridspec(
#                             10, self.x.l, wspace=1.5, hspace=1.5
#                         )
#                         self.ax[i, j] = self.fig.add_subplot(
#                             self.subax[i, j][:8, :]
#                         )
#
#                         # if i == (self.y.l - 1) or j == 0:
#
#                         # else:  #     self.ax[i, j] = self.fig.add_subplot(self.subax[i, j][:8, :])#,  #                                          # sharex=self.ax[0, i],  #                                          # sharey=self.ax[j, (self.y.l - 1)])  # if i != (self.y.l - 1):  #     self.ax[i, j].tick_params(labelbottom=False)  # if j != 0:  #     self.ax[i, j].tick_params(labelleft=False)
#
#                         # self.tabax[i, j] = self.fig.add_subplot(self.subax[i, j][1])  # self.tabax[i, j].set
#
#             else:
#                 self.fig = plt.figure(
#                     figsize=figsize, dpi=dpi, constrained_layout=True
#                 )
#
#                 # xmul = 2
#
#                 self.gs = gridspec.GridSpec(
#                     nrows=self.y.l,
#                     ncols=self.x.l,
#                     figure=self.fig,  # width_ratios=[2,1]
#                     # sharey=True,
#                     # sharex=True
#                 )
#                 for i in range(self.y.l):
#                     for j in range(self.x.l):
#                         self.ax[i, j] = self.fig.add_subplot(self.gs[i, j])
#
#             # only one plot
#             if pl is None:
#                 logger.info(f"Generating plot")
#                 sepview = plot_df.view()
#                 plsave = ""
#
#             # multiple plots
#             else:
#                 logger.info(f"Generating {pl} plot:")
#                 # print(plot_df.head(20), "\n", separate, pl)
#                 sepview = plot_df.xs((pl), level=separate, drop_level=False)
#                 plsave = pl
#                 self.get_suptitle(pl, separate)
#
#             # set plot index
#             pi = 0
#
#             # iterate over subplot columns
#             for col in self.x.v:
#                 logger.info(f"Column {col}")
#                 try:
#                     view = sepview.xs(
#                         col, level=self.x, axis=0, drop_level=False
#                     )
#                     pi = 1
#                 except ValueError:
#                     view = sepview
#                     col = self.x.v
#                     pi += 1
#
#                 for it in iters:
#                     x_id, y_id = self.x.dict[col], self.y.dict[it[yid]]
#                     try:
#                         values = view.xs(
#                             tuple(it), level=idx.tolist(), drop_level=False
#                         )
#
#                         x_labels = []
#                         x_label_ticks = []
#                         for bar_num, vbar in enumerate(self.bars.v):
#                             tab_colours = []
#                             tab_rows = []
#                             if plot_table is True and bar_num == 0:
#                                 # tab_colours = []
#                                 # tab_rows = []
#                                 try:
#                                     tab_text = values.values.reshape(
#                                         (self.bars.l, -1)
#                                     ).T
#                                 except ValueError:
#                                     tab_text = values.values
#                                 # tab_text = tab_text.astype(str)
#                                 tab_format = lambda x: rf"${x * 100:3.1f} \%$"
#                                 tab_format_vec = np.vectorize(tab_format)
#                                 tab_text = tab_format_vec(tab_text)
#                                 tab_cols = list(
#                                     self.bars.v[
#                                         : np.atleast_2d(tab_text).shape[0]
#                                     ]
#                                 )
#
#                             # x_ticks.append(self.bars.dict[vbar] * (barwidth + xpad))
#                             # x_labels.append(vbar)
#                             bottom = 0.0
#                             bar_vals = values.xs(
#                                 vbar, level=self.bars, drop_level=False
#                             )
#                             start = bar_num * (barwidth + xpad)
#                             step = barwidth + xpad
#                             x_ticks = np.linspace(
#                                 start,
#                                 start + (len(bar_vals) * step * self.bars.l),
#                                 len(bar_vals),
#                                 endpoint=True,
#                             ).tolist()
#                             x_label_ticks.append([x_ticks])
#                             x_labels = list(
#                                 map(
#                                     lambda x: self.get_atom_type(x),
#                                     bar_vals.index.get_level_values("x"),
#                                 )
#                             )
#                             bar_hatch = [
#                                 hatch_list[bar_num % len(hatch_list)]
#                                 for val in bar_vals
#                             ]
#                             # tab_text.append(bar_vals.values)
#
#                             cmap = colour_df.loc[bar_vals.index]
#
#                             # peaks = self.data.get_bar_peaks(
#                             #     atom_type=col, other=None
#                             # )
#                             # plot if all values > 0
#                             if np.all(bar_vals.values) >= 0:
#                                 # table_text.append([f'${v * 100:3.1f} %$' for v in bar_vals.values])
#
#                                 # if plot_table is True:
#                                 #     # logger.info(y_id)
#                                 #     logger.info("Has table")
#                                 #     y_id *= 2
#                                 #     logger.info(y_id)
#                                 # print(y_id)
#
#                                 bar_val_view = bar_vals
#                                 bin_list = bar_val_view.index.get_level_values(
#                                     "x"
#                                 )
#                                 bar_val_view.index = bin_list
#
#                                 # x_tick = x_ticks[-1]
#
#                                 # for bar_id, bar_val in enumerate(bar_val_view.items()):
#                                 #
#                                 #     x_bin, y_val = bar_val
#
#                                 # try:
#                                 # peak = peaks[bar_id]
#                                 # except IndexError:
#                                 # peak = x_bin.right
#                                 # colour = cmap[bar_id] #(self.cnorm(peak))
#                                 # if colour not in table_col and cmap == table_cmap:
#                                 #     print('colour', colour)
#                                 #     table_col.append(colour)
#
#                                 # label = self._get_bin_label(x_bin, bin_list)
#
#                                 # if bar_num == 0:
#                                 #     tab_colours.append(list(colour))
#                                 #     tab_rows.append(label)
#                                 #     print(tab_colours)
#
#                                 # if x_bin.right < np.max(self.x):
#                                 #     label = f'${x_bin.left} - {x_bin.right}$ \AA'
#                                 # else:
#                                 #     label = f'$ > {x_bin.left}$ \AA'
#                                 # if label not in table_rows and cmap == table_cmap:
#                                 #     table_rows.append(label)
#                                 # if y_val >= 0.0010:
#                                 # barwidth = bulk_edge - x_bin.left
#                                 # try:
#                                 #     x_tick = x_ticks[-1] + barwidth #+ (barwidth + xpad) * bar_num
#                                 #     x_ticks.append(x_tick)
#                                 # except IndexError:
#                                 #     x_tick = 0
#                                 # plot
#                                 try:
#                                     p = self.ax[y_id, x_id].bar(
#                                         x_ticks,
#                                         bar_val_view,  # label=x_labels,
#                                         bottom=bottom,
#                                         width=barwidth,
#                                         align="center",
#                                         color=cmap,
#                                         hatch=bar_hatch,
#                                     )  # self.ax[y_id, x_id].bar_label(p, labels=[label],  #                          fmt='%s',  #                          label_type='center')
#                                 except IndexError:
#                                     p = self.ax[y_id].bar(
#                                         x_ticks,
#                                         bar_val_view,  # label=x_labels,
#                                         bottom=bottom,
#                                         width=barwidth,
#                                         align="edge",
#                                         color=cmap,
#                                         hatch=bar_hatch,
#                                     )  # self.ax[y_id, x_id].bar_label(p, labels=[label],  #                          fmt='%s',  #                          label_type='center')  # finally:  # bottom += y_val
#                                 # try:
#                                 #     self.fig.subplots_adjust(left=0.2, bottom=0.2)
#                                 # except IndexError:
#                                 #     self.ax[y_id].subplots_adjust(left=0.2, bottom=0.2)
#                                 if plot_table is True:
#                                     # y_id += 1
#                                     # print(tab_colours)
#                                     logger.info(f"Has table, {y_id}")
#                                     has_table = False
#                                     try:
#                                         print("rows", tab_rows)
#                                         tab_colours = np.array(tab_colours)
#                                         print("colours", tab_colours)
#                                         # self.tabax[y_id, x_id].clear()
#                                         the_table: table = table(
#                                             self.ax[y_id, x_id],
#                                             cellText=tab_text,
#                                             rowLabels=tab_rows,
#                                             rowColours=tab_colours,
#                                             colLabels=tab_cols,
#                                             loc="bottom",
#                                             edges="horizontal",
#                                         )
#                                         has_table = True
#                                     except IndexError:
#                                         try:
#                                             # self.tabax[y_id].clear()
#                                             the_table: table = table(
#                                                 self.ax[y_id],
#                                                 cellText=tab_text,
#                                                 rowLabels=tab_rows,
#                                                 rowColours=tab_colours,
#                                                 colLabels=tab_cols,
#                                                 loc="bottom",
#                                                 edges="horizontal",
#                                             )
#                                             has_table = True
#                                         except ValueError:
#                                             logger.info("could not plot table")
#                                     except ValueError:
#                                         logger.info("could not plot table")
#                                     if has_table is True:
#                                         logger.info("Plotting table")
#                                         # renderer = self.fig.canvas.get_renderer()
#                                         the_table.auto_set_font_size(False)
#                                         the_table.set_fontsize(12)
#                                         h = the_table.get_celld()[
#                                             (0, 0)
#                                         ].get_height()
#                                         w = the_table.get_celld()[
#                                             (0, 0)
#                                         ].get_width()
#                                         table_cells = the_table.properties()[
#                                             "children"
#                                         ]
#                                         for cell in table_cells:
#                                             max_width = w
#                                             if cell.get_width() > max_width:
#                                                 max_width = cell.get_width()
#                                             # print(cell.get_y())
#                                             # cell.set_y(5)
#                                             # print(cell.get_y())
#                                             cell.set_width(barwidth + xpad)
#                                             cell.set_height(0.13)
#                                             # cell.set_fontsize(12)
#                                             cell.set_text_props(
#                                                 verticalalignment="center"
#                                             )
#                                         for cell in table_cells:
#                                             cell.set_width(
#                                                 max_width
#                                             )  # the_table.set_fontsize(tab_fontsize)  # has_table = True  # the_table.scale(1, 1.2)  # the_table.stale = True  # the_table.set_clip_on(False)  # the_table._update_positions(renderer)  # bbox = the_table.get_window_extent(renderer)  # the_table.set(clip_bbox=bbox)  # self.ax[y_id, x_id].stale = True  # renderer = self.fig.canvas.get_renderer()  #  #  # # text_height = the_table._approx_text_height()  #  # # the_table.auto_set_font_size(False)  # # new_font_size = the_table.FONTSIZE  # # the_table.FONTSIZE = tab_fontsize  # the_table.set_fontsize(tab_fontsize)  #  # cell = the_table[0, 0]  # # cell_height = cell.get_text()  # win_ext = the_table.get_window_extent(renderer=renderer)  # text_height = cell.get_text_bounds(renderer=renderer)[-1]  # tab_font = the_table.FONTSIZE  # text_height = cell.get_text_bounds()  # # the_table._do_  # tab_font_new = tab_fontsize  # # sys.exit(1)  # # text_height = cells.get_text_bounds()  #  # font_ratio = tab_fontsize / the_table.FONTSIZE  #  # # # the_table.set_fontsize(tab_font_new)  # # # tab_dict = the_table.get_celld()  # #  # # # for col_id in range(0, len(tab_cols)):  # # #     tab_dict[(0, col_id)].set_height(text_height + 0.1)  # # #     for row_id in range(1, len(tab_rows)):  # # #         tab_dict[(row_id, col_id)].set_height(text_height + 0.1)  # the_table.scale(1, font_ratio + 0.3)  # the_table._stale = True  # the_table[...].set_text_props(linespacing=1.5,  #                          verticalalignment='bottom')  # the_table.scale(1, 1.2)  # the_table.stale = True  # the_table.set_clip_on(False)  # the_table._update_positions(renderer)
#
#                                     else:
#                                         logger.info(
#                                             "No table"
#                                         )  # else:  # print(y_val)
#                         x_label_ticks = np.mean(
#                             np.array(x_label_ticks), axis=0
#                         )
#                         # self.ax[y_id, x_id].legend(ncol=2, loc='lower center')#[leg for leg in legends[i, j]], ncol=3)
#                         # if xlim != None:
#                         # self.ax[i, j].set_xlim((-xpad, self.bars.l * (barwidth + xpad)))
#                         # self.ax[i, j].set_xticks([], [])
#                         #                 if ylim != None:
#                         # self.ax[i, j].set_ylim((0.0, 1.2))
#                         self.ax[y_id, x_id].set_xticks(
#                             np.ravel(x_label_ticks), np.ravel(x_labels)
#                         )
#                     # print("table false")
#                     # sys.exit(1)
#                     # tab_dict = the_table.get_celld()
#                     # table = ax[y_id, x_id].table(cellText=table_text,
#                     #                              rowColours=table_col,
#                     #                              rowLabels=table_rows,
#                     #                              # colLables=...,
#                     #                              loc='bottom')
#                     # x_ticks = x_ticks[:-1]
#                     # x_ticks.append(bar_dict[vbar] * (barwidth + xpad))
#                     #                 values = values
#                     #                 print('try 1 done')
#                     #
#                     #                     # for shell in values:
#                     #                     # view for group and bars
#                     #                             label = f'${lims.left} - {lims.right}$ \AA'
#                     #
#                     #                     try:
#                     #                         print('try 2')
#                     #                         print(x_dict[col], y_dict[it[yid]])
#
#                     #                     except:
#                     # #                         raise ValueError
#                     #                         x_id, y_id = 0, y_dict[it[yid]]
#                     #                         label = f'${x_bin.left} - {x_bin.right}$ \AA'
#
#                     #                     if pi == 1:
#                     #                         legends[y_id, x_id].append(it[label_id])
#                     #                 else:
#                     #                     check_logger.info('NaN values')
#
#                     except KeyError:
#                         logger.info(f"No data for {pl}, {self.x}, {it}")
#
#             # x_ticks = [np.linspace(n_bar * bulk_edge + xpad,
#             #                        n_bar * bulk_edge + bulk_edge, int(bulk_edge)) for n_bar in range(lbars)]
#             # x_ticks = np.ravel(x_ticks)
#             # x_labels = np.tile(np.arange(0, bulk_edge, 1), lbars)
#             for i in range(self.y.l):
#                 # if plot_table is True:
#                 #     ax_multi = 2
#                 # else:
#                 #     ax_multi = 1
#                 # ax_i = i * ax_multi
#                 # print(f"Axis index: {ax_i}, multi: {ax_multi}")
#
#                 try:
#                     # self.ax[]
#                     # self.ax[i, 0].set_ylabel(
#                     #     f"{self.label_mod([(self.y.v[i], self.y)])}\n" + rowlabel
#                     # )
#                     for j in range(self.x.l):
#                         # if plot_table is True:
#                         #     self.ax[i, j].subplots_adjust(bottom=0.2)
#                         #     f"{self.label_mod([(self.y.v[i], self.y)])}\n" + rowlabel
#                         # )
#                         if j == 0:
#                             self.ax[i, 0].set_ylabel(
#                                 f"{self.label_mod([(self.y.v[i], self.y)])}\n"
#                                 + rowlabel
#                             )
#                             self.ax[i, j].set_yticks(np.arange(0.0, 1.1, 0.2))
#                         else:
#                             self.ax[i, j].set_yticks(
#                                 np.arange(0.0, 1.1, 0.2)
#                             )  # , label="")
#                             self.ax[i, j].set_yticklabels(
#                                 []
#                             )  # np.arange(0.0, 1.1, 0.2))
#                         self.ax[i, j].spines[["top", "right"]].set_visible(
#                             False
#                         )
#                         # self.ax[i, j].hlines(
#                         #     1.0,
#                         #     -xpad,
#                         #     self.bars.l * (barwidth + xpad) + xpad,
#                         #     linestyle="--",
#                         # )
#                         # legend = [matplotlib.patches.Patch(color = 'black',
#                         #                                    hatch=hatch_list[xi],
#                         #                                    label=x) for xi, x in enumerate(self.bars.v)]
#                         # self.ax[i, j].legend(ncol = 2, handles = legend)
#                         # self.ax[i, j].legend(ncol=2, loc='lower center')#[leg for leg in legends[i, j]], ncol=3)
#                         #                  if xlim != None:
#                         # self.ax[i, j].set_xlim((-xpad, self.bars.l * (barwidth + xpad)))
#                         # self.ax[i, j].set_xticks([], [])
#                         #                 if ylim != None:
#                         self.ax[i, j].set_ylim((0.0, 1.1))
#                         # self.ax[self.y.l - 1, j].set_xticks(
#                         #     np.array(x_ticks) + 0.5 * barwidth, x_labels
#                         # )
#                         self.ax[self.y.l - 1, j].set_xlabel(
#                             "atom types"
#                             + f"\n{self.label_mod([(self.x.v[j], self.x)])}"
#                         )  # self.ax[self.y.l-1, j].subplots_adjust(bottom=0.1)  # self.ax[i, j].set_yticklabels(np.arange(0.0, 1.1, 0.2))
#                 except IndexError:
#                     self.ax[i].set_ylabel(
#                         f"{self.label_mod([(self.y.v[i], y)])}\n" + rowlabel
#                     )
#                     # self.ax[i].legend([self.label_mod([(leg, self.label_key)]) for leg in self.legends[i, 0]], ncol=3)
#                     self.ax[self.y.l - 1].set_xlabel(
#                         columnlabel
#                         + f"\n{self.label_mod([(self.x.v[0], self.x)])}"
#                     )
#             # # #
#             legend_dict = {
#                 x: matplotlib.patches.Patch(
#                     edgecolor="black",
#                     hatch=hatch_list[xi],
#                     label=x,
#                     fill=False,
#                 )
#                 for xi, x in enumerate(self.bars.v)
#             }
#             self.fig.legend(
#                 ncol=2,
#                 handles=list(legend_dict.values()),
#                 labels=list(legend_dict.keys()),
#                 loc="outside upper center",
#                 title=bars,
#                 frameon=False,
#             )
#             self.fig.supxlabel(f"{self.title_dict[x]}s", size=14)
#             self.fig.supylabel(f"{self.title_dict[y]}s", size=14)
#             if plsave != False:
#                 logger.info("Saving plot")
#                 if type(plsave) == str and plsave != "":
#                     outname = f"{self.atomname}_{plsave}.png"
#                 else:
#                     outname = f"{self.atomname}_{self.name}.png"
#                 odir = Path(odir).absolute()
#                 logger.info(f"output to {odir.absolute()}")
#                 if not odir.is_dir():
#                     os.makedirs(odir)
#                 logger.info(odir)
#                 logger.info(outname)
#                 self.fig.savefig(str(odir / outname))
#             else:
#                 plt.show()
#             self.fig.clear()
#
#     def _get_bin_label(self, x_bin, bin_list):
#         # if x_bin.right < np.max(bin_list.right):
#         #     label = f"${x_bin.left:>3.1f} - {x_bin.right:3.1f}$ \AA"
#         #     # barwidth = x_bin.right - x_bin.left
#         #
#         # else:
#         #     label = f"$ > {x_bin.left:>3.1f}$ \AA"
#         return x_bin


# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
# logger.addHandler(ch)
# __all__ = ["Histogram", "Data", "BarPlot"]

# __name__ = 'plots'

# from collections.abc library


# pbar = ProgressBar()
# pbar.register() # global registration


def redirect_tqdm(f):
    """Decorator to redirect tqdm output to logger."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        with logging_redirect_tqdm():
            result = f(*args, **kwargs)
        return result

    return wrapper


# ------------------------------------------------------------------------------

# class Histogram(Plot):
#     __slots__ = ('data', 'x', 'y', 'n_bins', 'bin_size', 'xmin', 'xmax', 'ymin', 'ymax',
#                  'labels', 'xlabels', 'ylabels', 'xformat', 'yformat', 'figsize', 'dpi',
#                  )
#
#     _figargs = {'figsize', 'dpi', 'facecolor', 'edgecolor', 'linewidth', 'frameon',
#                 'subplotpars'}
#
#     def _process_data(self):
#         if not hasattr(self, 'figsize'):
#             figsize = None
#         if not hasattr(self, 'xmin'):
#             self.xmin = 0
#         if self.data.ndim == 1:
#             self.y = self.data
#             self.x = np.linspace(self.xmin, self.xmax, len(self.y), endpoint=True)
#         elif self.data.ndim == 2:
#             self.x = self.data[0]
#             self.y = self.data[1]
#
#     def plot(self):
#         print(self.x, self.y)
#         figargs = {}
#         for arg in self._figargs:
#             if hasattr(self, arg):
#                 figargs[arg] = getattr(self, arg)
#         fig, ax = plt.subplots(**figargs)
#         ax.plot(self.x, self.y)


# class Cutoff(str):
#     """Class for histogram cutoff processing."""
#
#     def __new__(cls, length):
#         string = f"{int(length):02}"
#         return super().__new__(cls, string)
#
#     def __init__(self, length):
#         self.num = length


# class Bins(str):
#     """Class for histogram bin size processing."""
#
#     def __new__(cls, length):
#         string = f"{float(length):.02f}"[2:]
#         return super().__new__(cls, string)
#
#     def __init__(self, length):
#         self.num = length


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

            f = self.filelist[0]
            # print(f)

            x = pd.read_csv(
                f, delimiter="\s+", comment="#", header=None
            ).to_numpy()
            x = x[:, 0]

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
                mod_at[mod_at != "ions"].apply(lambda x: x[0])
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
                and atom not in self.df.index.get_level_values("ions").unique()
            )
        ]

        setattr(self, self.df.columns.name, list(self.df.columns))
        self._process_atom_type_idx(idx)
        self._init_xbin_idx()

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
                    f, delimiter="\s+", comment="#", header=None
                ).to_numpy()

                try:
                    self.df.loc[idsl[ion, aa, atom, :], clay] = array[
                        :, self._arr_col
                    ]  # print(array[:, self._arr_col].sum(), clay, ion)

                except ValueError:
                    try:
                        self.df.loc[
                            idsl[ion, aa, atom, other, :], clay
                        ] = array[:, self._arr_col]
                    except NameError:
                        pass
                except IndexError:
                    try:
                        self.df.loc[idsl[ion, aa, atom, :], clay] = array[:, 1]
                    except ValueError:
                        self.df.loc[
                            idsl[ion, aa, atom, other, :], clay
                        ] = array[:, 1]
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

    # @redirect_tqdm
    # def plot(
    #     self,
    #     x: Literal["clays", "aas", "ions", "atoms", "other"],
    #     y: Literal["clays", "ions", "aas", "atoms", "other"],
    #     select: Literal["clays", "ions", "aas", "atoms", "other"],
    #     rowlabel: str = "y",
    #     columnlabel: str = "x",
    #     figsize=None,
    #     dpi=None,
    #     diff=False,
    #     xmax=50,
    #     ymax=50,
    #     save=False,
    #     xlim=None,
    #     ylim=None,
    #     odir=".",
    #     plot_table=None,antialiased=True):
    #     aas_classes = [
    #         ["arg", "lys", "his"],
    #         ["glu", "gln"],
    #         ["cys"],
    #         ["gly"],
    #         ["pro"],
    #         ["ala", "val", "ile", "leu", "met"],
    #         ["phe", "tyr", "trp"],
    #         ["ser", "thr", "asp", "gln"],
    #     ]
    #     ions_classes = [["Na", "Ca"], ["Ca", "Mg", "Ba"]]
    #     atoms_classes = [["ions"], ["N"], ["OT"], ["CA"]]
    #     clays_classes = [["NAu-1"], ["NAu-2"]]
    #     cmaps_seq = ["Purples", "Blues", "Greens", "Oranges", "Reds"]
    #     cmaps_single = ["Dark2"]
    #     sel_list = ("clays", "ions", "aas", "atoms")
    #     # for color, attr in zip([''], sel_list):
    #     #     cmaps_dict[attr] = {}
    #     # cm.get_cmap()
    #     cmap_dict = {"clays": []}
    #
    #     title_dict = {
    #         "clays": "Clay type",
    #         "ions": "Ion type",
    #         "aas": "Amino acid",
    #         "atoms": "Atom type",
    #         "other": "Other atom type",
    #     }
    #
    #     sel_list = ["clays", "ions", "aas", "atoms"]
    #
    #     if self.other is not None:
    #         sel_list.append("other")
    #
    #     separate = [s for s in sel_list if (s != x and s != y and s != select)]
    #     idx = pd.Index([s for s in sel_list if (s != x and s not in separate)])
    #
    #     sep = pd.Index(separate)
    #
    #     vx = getattr(self, x)
    #     # print(vx)
    #
    #     if diff == True:
    #         vx = "/".join(vx)
    #         lx = 1
    #     else:
    #         lx = len(vx)
    #
    #     vy = getattr(self, y)
    #
    #     ly = len(vy)
    #     # print(ly)
    #
    #     yid = np.ravel(np.where(np.array(idx) == y))[0]
    #     # print(yid)
    #
    #     label_key = idx.difference(pd.Index([x, y, *separate]), sort=False).values[0]
    #
    #     label_id = idx.get_loc(key=label_key)
    #
    #     # label_classes = locals()[f'{label_key}_classes']
    #     # cmap_dict = {}
    #     # single_id = 0
    #     # seq_id = 0
    #     # for category in label_classes:
    #     #     if len(category) == 1:
    #     #         cmap = matplotlib.cycler('color', cm.Dark2.colors)
    #     #         single_id += 1
    #     #     else:
    #     #         cmap = getattr(cm, cmaps_seq[seq_id])(np.linspace(0, 1, len(category)))
    #     #
    #     #         cmap = matplotlib.cycler('color', cmap)
    #     #         # cmap = cmap(np.linspace(0, 1, len(category))).colors
    #     #             #                                viridis(np.linspace(0,1,N)))
    #     #             # cm.get_cmap(cmaps_seq[seq_id], len(category))
    #     #         seq_id += 1
    #     #     for item_id, item in enumerate(category):
    #     #         cmap_dict[item] = cmap.__getitem__(item_id)
    #     #
    #     n_plots = len(sep)
    #
    #     x_dict = dict(zip(vx, np.arange(lx)))
    #
    #     if diff == True:
    #         diffstr = "diff"
    #         sel = "diff"
    #         self._get_densdiff()
    #     else:
    #         sel = self.clays
    #         diffstr = ""
    #
    #     plot_df = self.df[sel].copy()
    #     plot_df.index = plot_df.index.droplevel(["_atoms", "x_bins"])
    #     plot_df.reset_index().set_index([*idx, "x"])
    #     # print(plot_df.head(5))
    #
    #     if figsize is None:
    #         figsize = tuple(
    #             [
    #                 5 * lx if (10 * lx) < xmax else xmax,
    #                 5 * ly if (5 * ly) < ymax else ymax,
    #             ]
    #         )
    #
    #     if dpi is None:
    #         dpi = 300
    #
    #     iters = np.array(
    #         np.meshgrid(*[getattr(self, idxit) for idxit in idx])
    #     ).T.reshape(-1, len(idx))
    #
    #     logger.info(f"Printing plots for {sep}\nColumns: {vx}\nRows: {vy}")
    #
    #     label_mod = lambda l: ", ".join(
    #         [li.upper() if namei == "aas" else li for li, namei in l]
    #     )
    #
    #     sep_it = np.array(
    #         np.meshgrid(*[getattr(self, idxit) for idxit in sep])
    #     ).T.reshape(-1, len(sep))
    #     # print(vx, vy, lx, ly)
    #
    #     for pl in sep_it:
    #         # print(pl)
    #         # try:
    #         #     fig.clear()
    #         # except:
    #         #     pass
    #         y_dict = dict(zip(vy, np.arange(ly)))
    #         if separate == "atoms" and pl != "":
    #             ...
    #
    #         legends_list = [(a, b) for a in range(ly) for b in range(lx)]
    #
    #         legends = dict(zip(legends_list, [[] for a in range(len(legends_list))]))
    #         handles = legends.copy()
    #         # if type(pl) in [list, tuple, np.ndarray]:
    #         #     viewlist = []
    #         #     for p in pl:
    #         #         viewlist.append(plot_df.xs((p), level=separate, axis=0))
    #         #
    #         #     sepview = pd.concat(viewlist)
    #         #     plsave = 'ions'
    #         #
    #         # else:
    #         sepview = plot_df.xs((pl), level=separate, axis=0)
    #         plsave = pl
    #
    #         fig, ax = plt.subplots(
    #             nrows=ly,
    #             ncols=lx,
    #             figsize=figsize,
    #             sharey=True,
    #             dpi=dpi,
    #             constrained_layout=True,
    #         )
    #
    #         fig.suptitle(
    #             (
    #                 ", ".join([title_dict[s].upper() for s in separate])
    #                 + f": {label_mod(list(tuple(zip(pl, separate))))}"
    #             ),
    #             size=16,
    #             weight="bold",
    #         )
    #         pi = 0
    #         for col in vx:
    #             try:
    #                 view = sepview.xs(col, axis=1)
    #                 pi = 1
    #             except ValueError:
    #                 view = sepview
    #                 col = vx
    #                 pi += 1
    #             for it in iters:
    #                 try:
    #                     values = view.xs(tuple(it), level=idx.tolist()).reset_index(
    #                         drop=False
    #                     )
    #                     values = values.values
    #                     if np.all(values) >= 0:
    #                         try:
    #                             x_id, y_id = x_dict[col], y_dict[it[yid]]
    #                             ax[y_id, x_id].plot(
    #                                 values[:, 0],
    #                                 values[:, 1],
    #                                 label=it[label_id],
    #                                 antialiased=antialiased
    #                             )
    #                         except:
    #                             x_id, y_id = 0, y_dict[it[yid]]
    #                             ax[y_id].plot(
    #                                 values[:, 0],
    #                                 values[:, 1],
    #                                 label=it[label_id],
    #                                 antialiased=antialiased
    #                             )
    #                         if pi == 1:
    #                             legends[y_id, x_id].append(it[label_id])
    #                     else:
    #                         logger.info("NaN values")
    #                 except KeyError:
    #                     logger.info(f"No data for {pl}, {vx}, {it}")
    #         for i in range(ly):
    #             try:
    #                 ax[i, 0].set_ylabel(f"{label_mod([(vy[i], y)])}\n" + rowlabel)
    #                 for j in range(lx):
    #                     ax[i, j].legend(
    #                         [label_mod([(leg, label_key)]) for leg in legends[i, j]],
    #                         ncol=3,
    #                         frameon=False,
    #                     )
    #                     if xlim is not None:
    #                         ax[i, j].set_xlim((0.0, float(xlim)))
    #                     if ylim is not None:
    #                         ax[i, j].set_ylim((0.0, float(ylim)))
    #                     ax[ly - 1, j].set_xlabel(
    #                         columnlabel + f"\n{label_mod([(vx[j], x)])}"
    #                     )
    #             except IndexError:
    #                 ax[i].set_ylabel(f"{label_mod([(vy[i], y)])}\n" + rowlabel)
    #                 ax[i].legend(
    #                     [label_mod([(leg, label_key)]) for leg in legends[i, 0]], ncol=3
    #                 )
    #                 ax[ly - 1].set_xlabel(columnlabel + f"\n{label_mod([(vx[0], x)])}")
    #
    #         fig.supxlabel(f"{title_dict[x]}s", size=14)
    #         fig.supylabel(f"{title_dict[y]}s", size=14)
    #         if save != False:
    #             odir = Path(odir)
    #             if not odir.is_dir():
    #                 os.makedirs(odir)
    #             if type(save) == str:
    #                 fig.savefig(odir / f"{save}.png")
    #             else:
    #                 logger.info(
    #                     f"Saving to {self.name}_{diffstr}_{x}_{y}_{plsave}_{self.cutoff}_{self.bins}.png"
    #                 )
    #                 fig.savefig(
    #                     odir
    #                     / f"{self.name}_{diffstr}_{x}_{y}_{plsave}_{self.cutoff}_{self.bins}.png"
    #                 )
    #         else:
    #             plt.show()
    #         self.fig.clear()

    def _get_edge_fname(
        self,
        atom_type: str,
        other: Optional[str],
        name: Union[Literal["pe"], Literal["edge"]] = "pe",
    ):
        if other is not None:
            other = f"{other}_"
        else:
            other = ""
        # fname = Path.cwd() / f"edge_data/edges_{atom_type}_{self.cutoff}_{self.bins}.p"
        fname = (
            Path.cwd()
            / f"pe_data/{atom_type}_{other}{name}_data_{self.cutoff}_{self.bins}.p"
        )
        logger.info(f"Peak/edge Filename: {fname}")
        return fname

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
        """Identifies edges and maxima of position density peaks.
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
                    score,  # df_slice.to_numpy(),
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
                    edges.append(
                        x_vals[window.start + edge_id][0]
                    )  # edge_id = np.argwhere(  #     df_slice.values == np.min(df_slice.values[window])  # )[0]  # edges.append(x_vals[edge_id][0])
                # logger.info(f"edges: {edges}")
                # edges.append(x_vals[peak_dict["right_bases"][-1]])
                # logger.info(f"edges: {edges}")

                # logger.info(f"edges: {edges}")
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
                # print(edges)
                plt.scatter(
                    x_vals[peaks],
                    df_slice[peaks] / np.sum(df_slice.to_numpy()),
                    color="red",
                )
                edge_dict = {
                    "edges": edges,
                    "cutoff": self.cutoff,
                    "peak": x_vals[peaks],
                }
                # for peak in edge_dict["peak"]:
                #     plt.axvline(peak, 0, 1, color="green")
                # for edge in edge_dict["edges"]:
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
                    )  # , label = f'edge {ei}: {edge:2.1f}')  # plt.annotate(fr'{edge:2.1f} \AA', xy=(edge, 0.8))
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
                p = [0, self.cutoff.num]
            else:
                # self._get_edges(atom_type=atom_type)
                raise FileNotFoundError(
                    f"No {atom_type} edge file found {fname!r}."
                )

        else:
            with open(fname, "rb") as edges_file:
                logger.info(f"Reading {atom_type} edges {edges_file.name!r}")
                p = pkl.load(edges_file)["edges"]
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
    # #     if len(self._peaks) == len(self._atoms):
    # #         pass
    # #     else:
    # #         for atom_type in self._atoms:
    # #             # try:
    # #
    # #             self._edges[atom_type] = self._read_peak_file(atom_type)
    # #             logger.info(f"Reading peaks")
    # #             # except FileNotFoundError:
    # #             #     logger.info(f"Getting new peaks")
    # #             #     self._get_edges(atom_type)
    # #     return self._peaks

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

        # area_df = self.df.copy()  # atom_col = edge_df.loc['atoms']  # edge_df['atoms'].where(edge_df['atoms'] == 'ions')  # data_slices = edge_df.groupby(['ions', 'atoms', 'x']).sum()  # data_slices = data_slices.aggregate('sum', axis=1)  # ion_slices = data_slices.xs('ions', level='atoms')  # # other_slices =  #  # peaks = find_peaks(data_slices.to_numpy(),  #                    height=height,  #                    distance=distance,  #                    width=width,  #                    wlen=wlen)  # check_logger.info(f'Found peaks {peaks[0]}')  #  # colours = ['blue', 'orange']  # fig, ax = plt.subplots(len(data_slices.index.unique('atoms')))  # y = []  # fig = plt.figure(figsize=(16, 9))  # for atom_type in data_slices.index.unique('atoms'):  #     data_slice = data_slices.xs(atom_type, level='atoms')  #     plt_slice = data_slice  #     if atom_type == 'ions':  #         for ion_type in data_slice.index.unique('ions'):  #             plt_slice = data_slice.xs(ion_type, level='ions')  #             y.append((plt_slice.reset_index()['x'].to_numpy(), plt_slice.to_numpy()))  #     else:  #         y.append((plt_slice.reset_index()['x'].to_numpy(), plt_slice.to_numpy()))  #  # for y_data in y:  #     # y = plt_slice.to_numpy()  #     # x = plt_slice.reset_index()['x'].to_numpy()#atom_type)  #     plt.plot(*y_data)  #     plt.vlines(data_slice.reset_index()['x'].to_numpy()[peaks[0]], -1, 1, color='red')  #     plt.xlim(0, 7)

    #
    #     group = data.index.droplevel('x')
    #
    #     # new_idx = pd.MultiIndex.from_product(group = data.index.droplevel('x').get_level_values)
    #
    #     edges = np.array(edges, dtype=np.float32)
    #     if edges[0] != min:
    #         np.insert(edges, 0, min)
    #     if edges[-1] < self.cutoff:
    #         edges.append(self.cutoff)
    #     # intervals = pd.IntervalIndex.from_breaks(edges)
    #
    #     data = data.reset_index(drop=False).set_index(group.names)
    #     print(data.index.names)
    #     print(data.columns)
    #     data['bins'] = pd.cut(data['x'], [min, *edges, self.cutoff])
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
    # # def _get_areas(self, sel, sel_level, edges, min = 0.0):
    # #     idsl = pd.IndexSlice
    # #     data = self.df.xs(sel,
    # #                       level=sel_level,
    # #                       drop_level=False).copy()
    # #     group = data.index.droplevel('x')
    # #
    # #     # new_idx = pd.MultiIndex.from_product(group = data.index.droplevel('x').get_level_values)
    # #
    # #     edges = np.array(edges, dtype=np.float32)
    # #     if edges[0] != min:
    # #         np.insert(edges, 0, min)
    # #     if edges[-1] < self.cutoff:
    # #         edges.append(self.cutoff)
    # #     # intervals = pd.IntervalIndex.from_breaks(edges)
    # #
    # #     data = data.reset_index(drop=False).set_index(group.names)
    # #     print(data.index.names)
    # #     print(data.columns)
    # #     data['bins'] = pd.cut(data['x'], [min, *edges, self.cutoff])
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
    #                     ['Ca', 'Mg', "Ba"]]
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
    #     max_edge = list(map(lambda x: np.max(x[:-1]), self.edges.values()))
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
    #                         if len(self._peaks) != len(self.df.index.get_level_values('_atoms').unique()):
    #                             self._get_edges()
    #                         try:
    #                             peaks = self._peaks[bar_vals.index.get_level_values('atoms').unique().tolist()[0]]
    #                         except:
    #                             peaks = self._peaks[bar_vals.index.get_level_values('ions').unique().tolist()[0]]
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
        atoms = plot_df.index.get_level_values("_atoms")

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
        # if len(self._peaks) != len(self.df.index.get_level_values("_atoms").unique()):
        peaks = self._read_edge_file(
            atom_type=atom_type, other=other
        )  # ['peaks']
        return peaks  # print(peaks)  # logger.info(f"Found peaks {peaks}")  # try:  #     print(peaks)  #     print(bar_vals.index.get_level_values('atoms'))  #     bar_peaks = peaks[  #         bar_vals.index.get_level_values("atoms").unique().tolist()#[0]  #     ]  #     print(bar_peaks)  #     bar_peaks=bar_peaks[0]  # except:  #     bar_peaks = peaks[  #         bar_vals.index.get_level_values("ions").unique().tolist()#[0]  #     ]  #     print(bar_peaks)  #     bar_peaks = bar_peaks[0]  # return bar_peaks

    # def plot_bars(self,  #               bars: Literal['clays', 'aas', 'ions', 'other'],  #               x: Literal['clays', 'aas', 'ions', 'other'],  #               y: Literal['clays', 'aas', 'ions', 'other'],  #               rowlabel: str = 'y',  #               columnlabel: str = 'x',  #               figsize=None,  #               dpi=None,  #               # diff=False,  #               xmax=50,  #               ymax=50,  #               save=False,  #               ylim=None,  #               odir='.',  #               barwidth=0.75,  #               xpad=0.25,  #               cmap='winter'  #               ):  #     """Create stacked Histogram adsorption shell populations.  #  #     """  #     aas_classes = [['arg', 'lys', 'his'],  #                    ['glu', 'gln'],  #                    ['cys'],  #                    ['gly'],  #                    ['pro'],  #                    ['ala', 'val', 'ile', 'leu', 'met'],  #                    ['phe', 'tyr', 'trp'],  #                    ['ser', 'thr', 'asp', 'gln']]  #     ions_classes = [['Na', 'Ca'],  #                     ['Ca', 'Mg', "Ba"]]  #     atoms_classes = [['ions'],  #                      ['N'],  #                      ['OT'],  #                      ['CA']]  #     clays_classes = [['NAu-1'],  #                      ['NAu-2']]  #     cmaps_seq = ['Purples', 'Blues', 'Greens', 'Oranges', 'Reds']  #     cmaps_single = ['Dark2']  #     sel_list = ('clays', 'ions', 'aas', 'atoms')  #     # for color, attr in zip([''], sel_list):  #     #     cmaps_dict[attr] = {}  #     # cm.get_cmap()  #     cmap_dict = {'clays': []}  #  #     title_dict = {'clays': 'Clay type',  #                   'ions': 'Ion type',  #                   'aas': 'Amino acid',  #                   '_atoms': 'Atom type',  #                   'other': 'Other atom type'}  #  #     sel_list = ['clays', 'ions', 'aas', '_atoms']  #  #     if self.other != None:  #         sel_list.append('other')  #  #     assert x in sel_list and x != '_atoms'  #     assert y in sel_list and y != '_atoms'  #     assert bars in sel_list and bars != '_atoms'  #  #     plot_df, colour_df = self._get_binned_plot_df_1d()  #  #     cnorm = self._get_cnorm()  #  #    # get data for plotting  #  #     bins = 'x_bins'  #     group = '_atoms'  #  #     separate = [s for s in plot_df.index.names if s not in [x, y, bars, bins]]  # (s != x and s != y and s != bars and s != groups)]  #     logger.info(f'Separate plots: {separate}')  #     idx = pd.Index([s for s in plot_df.index.names if (s != x and s != bars and s not in [*separate, bins])])  #     logger.info(f'Iteration index: {idx}')  #  #     sep = pd.Index(separate)  #  #     vx = getattr(self, x)  #     logger.info(f'x = {x}: {vx}')  #     lx = len(vx)  #  #     vy = getattr(self, y)  #     logger.info(f'y = {y}: {vy}')  #     ly = len(vy)  #  #  #     vbars = getattr(self, bars)  #     lbars = len(vbars)  #     logger.info(f'bars = {bars}: {vbars}')  #  #     bar_dict = dict(zip(vbars, np.arange(lbars)))  #  #     yid = np.ravel(np.where(np.array(idx) == y))[0]  #  #     sys.exit(2)  #  #     # label_key = idx.difference(pd.Index([x, y, *separate]), sort=False).values[0]  #  #     # label_id = idx.get_loc(key=label_key)  #     n_plots = len(sep)  #  #     x_dict = dict(zip(vx, np.arange(lx)))  #  #  #     sel = self.clays  #  #     # set figure size  #     if figsize == None:  #         figsize = self.get_figsize(lx=lx,  #                                    ly=ly,  #                                    xmax=xmax,  #                                    ymax=ymax)  #  #     # set resultion  #     if dpi == None:  #         dpi = 100  #  #     # get plotting iter from index  #     iters = self._get_idx_iter(idx=idx)  #  #     logger.info(f'Printing bar plots for {sep}\nColumns: {vx}\nRows: {vy}')  #  #     # set label modifier function  #     label_mod = self.modify_plot_labels  #  #     try:  #         # iterator for more than one plot  #         sep_it = self._get_idx_iter(idx=sep)  #     except ValueError:  #         # only one plot  #         sep_it = [None]  #  #     # iterate over separate plots  #     for pl in sep_it:  #         # index map for y values  #         y_dict: dict = dict(zip(vy, np.arange(ly)))  #         print(y_dict)  #  #         legends = self.init_legend(ly=ly,  #                                    lx=lx)  #         print(legends)  #  #         # generate figure and axes array  #         fig, ax = plt.subplots(nrows=ly,  #                                ncols=lx,  #                                figsize=figsize,  #                                sharey=True,  #                                dpi=dpi,  #                                constrained_layout=True,  #                                # sharex=True  #                                )  #         # only one plot  #         if pl is None:  #             logger.info(f'Generating plot')  #             sepview = plot_df.view()  #             plsave = ''  #  #         # multiple plots  #         else:  #             logger.info(f'Generating {pl} plot')  #             print(plot_df.head(20),'\n', separate, pl)  #             sepview = plot_df.xs((pl),  #                                  level=separate,  #                                  drop_level=False)  #             plsave = pl  #             print(pl)  #             print(separate)  #             fig.suptitle((', '.join([title_dict[s].upper() for s in separate]) +  #                           f': {label_mod(list(tuple(zip(pl, separate))))}'), size=16,  #                          weight='bold')  #  #         # set plot index  #         pi = 0  #  #         #iterate over subplot columns  #         for col in vx:  #             # logger.info(col)  #             # view, col, pi = self.plot_columns(sepview=sepview,  #             #                                   col=col,  #             #                                   x=x,  #             #                                   vx=vx,  #             #                                   pi=pi)  #             logger.info(col)  #             try:  #                 view = sepview.xs(col,  #                                   level=x,  #                                   axis=0,  #                                   drop_level=False)  #                 pi = 1  #             except ValueError:  #                 view = sepview  #                 col = vx  #                 pi += 1  #  #             table_text = []  #             table_col = []  #             table_cmap = None  #             table_rows = []  #             for it in iters:  #                 try:  #                     values = view.xs(tuple(it),  #                                      level=idx.tolist(),  #                                      drop_level=False)  #  #                     x_labels = []  #                     x_ticks = []  #                     for vbar in vbars:  #                         print(vbar)  #                         x_ticks.append(bar_dict[vbar] * (barwidth + xpad))  #                         x_labels.append(vbar)  #                         bottom = 0.0  #                         bar_vals = values.xs(vbar,  #                                              level=bars,  #                                              drop_level=False)  #  #                         cmap = colormaps[colour_df.loc[bar_vals.index].values[0]]  #                         if table_cmap is None:  #                             table_cmap = cmap  #  #                         peaks = self.get_bar_peaks(bar_vals=bar_vals)  #  #                         if np.all(bar_vals.values) >= 0:  #                             table_text.append([f'${v * 100:3.1f} %$' for v in bar_vals.values])  #                             x_id, y_id = x_dict[col], y_dict[it[yid]]  #  #                             bar_val_view = bar_vals  #                             bar_val_view.index = bar_val_view.index.get_level_values('x_bins')  #  #                             x_tick = x_ticks[-1]  #  #                             for bar_id, bar_val in enumerate(bar_val_view.items()):  #  #                                 x_bin, y_val = bar_val  #  #                                 try:  #                                     peak = peaks[bar_id]  #                                 except IndexError:  #                                     peak = x_bin.right  #                                 colour = cmap(cnorm(peak))  #                                 if colour not in table_col and cmap == table_cmap:  #                                     print('colour', colour)  #                                     table_col.append(colour)  #  #                                 label = self._get_bin_label(x_bin)  #  #                                 # if x_bin.right < np.max(self.x):  #                                 #     label = f'${x_bin.left} - {x_bin.right}$ \AA'  #                                 # else:  #                                 #     label = f'$ > {x_bin.left}$ \AA'  #                                 if label not in table_rows and cmap == table_cmap:  #                                     table_rows.append(label)  #                                 if y_val >= 0.010:  #  #                                         # barwidth = bulk_edge - x_bin.left  #                                     # try:  #                                     # x_tick = x_ticks[-1] + barwidth  #                                     # x_ticks.append(x_tick)  #                                     # except IndexError:  #                                     #     x_tick = x_bin.left  #  #  #  #  #                                     try:  #                                         p = ax[y_id, x_id].bar(x_tick,  #                                                                y_val,  #                                                                label=label,  #                                                                bottom=bottom,  #                                                                width=barwidth,  #                                                                align='edge',  #                                                                color=colour  #                                                                )  #                                         ax[y_id, x_id].bar_label(p, labels=[label],  #                                                                  fmt='%s',  #                                                                  label_type='center')  #                                     except IndexError:  #                                         p = ax[y_id].bar(x_tick,  #                                                          y_val,  #                                                          label=label,  #                                                          bottom=bottom,  #                                                          width=barwidth,  #                                                          align='edge',  #                                                          color=colour  #                                                          )  #                                         ax[y_id, x_id].bar_label(p, labels=[label],  #                                                                  fmt='%s',  #                                                                  label_type='center')  #                                     # finally:  #                                     bottom += y_val  #                                 print(table_text, table_col, table_rows)  #                                 # table = ax[y_id, x_id].table(cellText=table_text,  #                                 #                              rowColours=table_col,  #                                 #                              rowLabels=table_rows,  #                                 #                              # colLables=...,  #                                 #                              loc='bottom')  #                             # x_ticks = x_ticks[:-1]  #  #                             # x_ticks.append(bar_dict[vbar] * (barwidth + xpad))  #  #  #  #                 #                 values = values  #                 #                 print('try 1 done')  #                 #  #                 #                     # for shell in values:  #                 #                     # view for group and bars  #                 #                             label = f'${lims.left} - {lims.right}$ \AA'  #                 #  #                 #                     try:  #                 #                         print('try 2')  #                 #                         print(x_dict[col], y_dict[it[yid]])  #  #                 #                     except:  #                 # #                         raise ValueError  #                 #                         x_id, y_id = 0, y_dict[it[yid]]  #                 #                         label = f'${x_bin.left} - {x_bin.right}$ \AA'  #  #                 #                     if pi == 1:  #                 #                         legends[y_id, x_id].append(it[label_id])  #                 #                 else:  #                 #                     check_logger.info('NaN values')  #  #                 except KeyError:  #                     logger.info(f'No data for {pl}, {vx}, {it}')  #  #         # x_ticks = [np.linspace(n_bar * bulk_edge + xpad,  #         #                        n_bar * bulk_edge + bulk_edge, int(bulk_edge)) for n_bar in range(lbars)]  #         # x_ticks = np.ravel(x_ticks)  #         # x_labels = np.tile(np.arange(0, bulk_edge, 1), lbars)  #  #         for i in range(ly):  #             try:  #                 ax[i, 0].set_ylabel(f'{label_mod([(vy[i], y)])}\n' + rowlabel)  #                 ax[i, 0].set_yticks(np.arange(0.0, 1.1, 0.2))  #                 for j in range(lx):  #                     ax[i, j].spines[['top', 'right']].set_visible(False)  #                     ax[i, j].hlines(1.0, -xpad, lbars * (barwidth + xpad) + xpad, linestyle='--')  #                     # ax[i, j].legend(ncol=2, loc='lower center')#[leg for leg in legends[i, j]], ncol=3)  #                     #                  if xlim != None:  #                     ax[i, j].set_xlim((-xpad, lbars * (barwidth + xpad)))  #                     ax[i, j].set_xticks([], [])  #                     #                 if ylim != None:  #                     ax[i, j].set_ylim((0.0, 1.25))  #  #                     ax[ly - 1, j].set_xticks(np.array(x_ticks) + 0.5 * barwidth, x_labels)  #                     ax[ly - 1, j].set_xlabel(bars + f'\n{label_mod([(vx[j], x)])}')  #             except IndexError:  #                 ...  #     #             ax[i].set_ylabel(f'{label_mod([(vy[i], y)])}\n' + rowlabel)  #     #             ax[i].legend([label_mod([(leg, label_key)]) for leg in legends[i, 0]], ncol=3)  #     #             ax[ly - 1].set_xlabel(columnlabel + f'\n{label_mod([(vx[0], x)])}')  #     # #  #         fig.supxlabel(f'{title_dict[x]}s', size=14)  #         fig.supylabel(f'{title_dict[y]}s', size=14)  #     # #     if save != False:  #     # #         odir = Path(odir)  #     # #         if not odir.is_dir():  #     # #             os.makedirs(odir)  #     # #         if type(save) == str:  #     # #             fig.savefig(odir / f'{save}.png')  #     # #         else:  #     # #             fig.savefig(odir / f'{self.name}_{diffstr}_{x}_{y}_{plsave}_{self.cutoff}_{self.bins}.png')  #     # #         fig.clear()  #         fig.show()  #     return fig

    # def plot_bars_shifted(  #     self,  #     bars: Literal["clays", "aas", "ions", "atoms", "other"],  #     x: Literal["clays", "aas", "ions", "atoms", "other"],  #     y: Literal["clays", "aas", "ions", "atoms", "other"],  #     # y: Literal['clays', 'ions', 'aas', 'atoms', 'other'],  #     # select: Literal['clays', 'ions', 'aas'],  #     rowlabel: str = "y",  #     columnlabel: str = "x",  #     figsize=None,  #     dpi=None,  #     # diff=False,  #     xmax=50,  #     ymax=50,  #     save=False,  #     ylim=None,  #     odir=".",  #     # barwidth = 0.75,  #     xpad=0.25,  #     cmap="winter",  # ):  #     """Create stacked Histogram adsorption shell populations."""  #     aas_classes = [  #         ["arg", "lys", "his"],  #         ["glu", "gln"],  #         ["cys"],  #         ["gly"],  #         ["pro"],  #         ["ala", "val", "ile", "leu", "met"],  #         ["phe", "tyr", "trp"],  #         ["ser", "thr", "asp", "gln"],  #     ]  #     ions_classes = [["Na", "Ca"], ["Ca", "Mg", "Ba"]]  #     atoms_classes = [["ions"], ["N"], ["OT"], ["CA"]]  #     clays_classes = [["NAu-1"], ["NAu-2"]]  #     cmaps_seq = ["Purples", "Blues", "Greens", "Oranges", "Reds"]  #     cmaps_single = ["Dark2"]  #     sel_list = ("clays", "ions", "aas", "atoms")  #     # for color, attr in zip([''], sel_list):  #     #     cmaps_dict[attr] = {}  #     # cm.get_cmap()  #     cmap_dict = {"clays": []}  #  #     title_dict = {  #         "clays": "Clay type",  #         "ions": "Ion type",  #         "aas": "Amino acid",  #         "atoms": "Atom type",  #         "other": "Other atom type",  #     }  #  #     sel_list = ["clays", "ions", "aas", "atoms"]  #  #     # if self.other != None:  #     #     sel_list.append('other')  #     cmap = colormaps[cmap]  #     separate = [  #         s for s in sel_list if s not in [x, y, bars]  #     ]  # (s != x and s != y and s != bars and s != groups)]  #     # print(separate)  #     idx = pd.Index(  #         [s for s in sel_list if (s != x and s != bars and s not in separate)]  #     )  #     # print(idx)  #  #     sep = pd.Index(separate)  #  #     vx = getattr(self, x)  #     logger.info(f"x = {x}: {vx}")  #     lx = len(vx)  #  #     vy = getattr(self, y)  #     logger.info(f"y = {y}: {vy}")  #     ly = len(vy)  #     # print(ly)  #  #     vbars = getattr(self, bars)  #     lbars = len(vbars)  #     logger.info(f"bars = {bars}: {vbars}")  #  #     bar_dict = dict(zip(vbars, np.arange(lbars)))  #  #     yid = np.ravel(np.where(np.array(idx) == y))[0]  #     # print(yid)  #  #     # label_key = idx.difference(pd.Index([x, y, *separate]), sort=False).values[0]  #  #     # label_id = idx.get_loc(key=label_key)  #     n_plots = len(sep)  #  #     x_dict = dict(zip(vx, np.arange(lx)))  #     # print(x_dict)  #  #     sel = self.clays  #  #     plot_df = self.bin_df[sel].copy()  #     idx_names = ["clays", *plot_df.index.droplevel(["x", "_atoms"]).names]  #     # print("idx", idx_names)  #     plot_df = plot_df.stack()  #     # atoms = plot_df.index.get_level_values('_atoms')  #     plot_df.index = plot_df.index.droplevel(["x", "_atoms"])  #     # plot_df = pd.DataFrame({'values': plot_df,  #     #                         '_atoms': atoms})  #     # idx_names.remove('_atoms')  #     plot_df = plot_df.reorder_levels(idx_names)  #     #  #     plot_df.name = "values"  #     # print(plot_df.head(3))  #     plot_df = plot_df.groupby(plot_df.index.names).sum()  #     # print(plot_df.head(3))  #     inner_sum = plot_df.groupby(plot_df.index.droplevel("x_bins").names).sum()  #     extra = 1 - inner_sum  #     plot_df.where(  #         np.rint(plot_df.index.get_level_values("x_bins").right) != int(self.cutoff),  #         lambda x: x + extra,  #         inplace=True,  #     )  #     # print(type(self.edges))  #     max_edge = list(map(lambda x: np.max(x[:-1]), self.edges.values()))  #     max_edge = np.max(max_edge)  #     # print("max edge", max_edge)  #     # max_edge = np.ravel(np.array(*self.edges.values()))  #     # print(max_edge)  #     cnorm = mpc.Normalize(vmin=0, vmax=max_edge, clip=False)  #  #     if figsize is None:  #         figsize = tuple(  #             [  #                 5 * lx if (10 * lx) < xmax else xmax,  #                 5 * ly if (5 * ly) < ymax else ymax,  #             ]  #         )  #     #  #     if dpi is None:  #         dpi = 100  #     #  #     iters = np.array(  #         np.meshgrid(*[getattr(self, idxit) for idxit in idx])  #     ).T.reshape(-1, len(idx))  #     #  #     logger.info(f"Printing bar plots for {sep}\nColumns: {vx}\nRows: {vy}")  #     #  #     label_mod = lambda l: ", ".join(  #         [li.upper() if namei == "aas" else li for li, namei in l]  #     )  #     #  #     try:  #         sep_it = np.array(  #             np.meshgrid(*[getattr(self, idxit) for idxit in sep])  #         ).T.reshape(-1, len(sep))  #     except ValueError:  #         sep_it = [None]  #     # check_logger.info(vx, vy, lx, ly)  #     #  #     for pl in sep_it:  #         # print(pl)  #         # try:  #         #     fig.clear()  #         # except:  #         #     pass  #         y_dict = dict(zip(vy, np.arange(ly)))  #         # print(y_dict)  #         #     if separate == 'atoms' and pl != '':  #         #         ...  #         #  #         legends_list = [(a, b) for a in range(ly) for b in range(lx)]  #         #  #         legends = dict(zip(legends_list, [[] for a in range(len(legends_list))]))  #         #  #         # if type(pl) in [list, tuple, np.ndarray]:  #         #     #     viewlist = []  #         #     #     for p in pl:  #         #     #         viewlist.append(plot_df.xs((p), level=separate, axis=0))  #         #     #  #         #     #     sepview = pd.concat(viewlist)  #         #     #     plsave = 'ions'  #         #     #  #         #     # else:  #         fig, ax = plt.subplots(  #             nrows=ly,  #             ncols=lx,  #             figsize=figsize,  #             sharey=True,  #             dpi=dpi,  #             constrained_layout=True,  #             # sharex=True  #         )  #         if pl is None:  #             sepview = plot_df.view()  #             plsave = ""  #         else:  #             sepview = plot_df.xs((pl), level=separate, axis=0, drop_level=False)  #             plsave = pl  #             #  #             #  #  #             #  #             fig.suptitle(  #                 (  #                     ", ".join([title_dict[s].upper() for s in separate])  #                     + f": {label_mod(list(tuple(zip(pl, separate))))}"  #                 ),  #                 size=16,  #                 weight="bold",  #             )  #         pi = 0  #         for col in vx:  #             try:  #                 view = sepview.xs(col, level=x, axis=0, drop_level=False)  #                 pi = 1  #             except ValueError:  #                 view = sepview  #                 col = vx  #                 pi += 1  #             # print("column", col)  #             for it in iters:  #                 try:  #                     values = view.xs(  #                         tuple(it), level=idx.tolist(), drop_level=False  #                     )  #                     x_grouplabels = []  #                     x_labels = []  #                     x_ticks = []  #                     for vbar in vbars:  #                         bulk_pad = 2  #                         bulk_edge = np.rint((max_edge + bulk_pad))  #                         x_ticks.append(bar_dict[vbar] * bulk_edge + xpad)  #                         # print(values)  #                         bottom = 0.0  #                         x_grouplabels.append(vbar)  #                         # x_tick = bar_dict[vbar] * (barwidth + xpad)  #                         bar_vals = values.xs(vbar, level=bars, drop_level=False)  #  #                         if len(self._peaks) != len(  #                             self.df.index.get_level_values("_atoms").unique()  #                         ):  #                             self._get_edges()  #                         try:  #                             peaks = self._peaks[  #                                 bar_vals.index.get_level_values("atoms")  #                                 .unique()  #                                 .tolist()[0]  #                             ]  #                         except:  #                             peaks = self._peaks[  #                                 bar_vals.index.get_level_values("ions")  #                                 .unique()  #                                 .tolist()[0]  #                             ]  #                         #       bar_vals.values >= 0)  #                         if np.all(bar_vals.values) >= 0:  #                             # print("All > 0")  #                             x_id, y_id = x_dict[col], y_dict[it[yid]]  #  #                             bar_val_view = bar_vals  #                             bar_val_view.index = (  #                                 bar_val_view.index.get_level_values("x_bins")  #                             )  #                             # x_ticks.append(x_tick + 0.5 * barwidth)  #                             # bar_plots = []  #                             for bar_id, bar_val in enumerate(bar_val_view.items()):  #                                 x_bin, y_val = bar_val  #  #                                 try:  #                                     peak = peaks[bar_id]  #                                 except IndexError:  #                                     peak = x_bin.right  #                                 colour = cmap(cnorm(peak))  #  #                                 label = self._get_bin_label(x_bin)  #                                 if x_bin.right < np.max(self.x):  #                                     label = f"${x_bin.left:3.1f} - {x_bin.right:3.1f}$ \AA"  #                                     barwidth = x_bin.right - x_bin.left  #  #                                 else:  #                                     label = f"$ > {x_bin.left:3.1f}$ \AA"  #                                     # print(bar_val_view.index[-1].left)  #                                     barwidth = bulk_edge - x_bin.left  #                                 # try:  #                                 x_tick = x_ticks[-1] + barwidth  #                                 x_ticks.append(x_tick)  #                                 # except IndexError:  #                                 #     x_tick = x_bin.left  #                                 x_labels.append(x_bin.left)  #                                 # print(x_ticks, x_tick, x_bin)  #                                 # print(peaks, bar_id, "label", "peak", label, peak)  #                                 # print(label)  #                                 try:  #                                     p = ax[y_id, x_id].bar(  #                                         x_tick,  #                                         y_val,  #                                         label=label,  #                                         left=bottom,  #                                         height=-barwidth,  #                                         align="edge",  #                                         color=colour,  #                                     )  #                                     ax[y_id, x_id].bar_label(  #                                         p,  #                                         labels=[label],  #                                         fmt="%s",  #                                         label_type="center",  #                                     )  #                                 except IndexError:  #                                     p = ax[y_id].bar(  #                                         x_tick,  #                                         y_val,  #                                         label=label,  #                                         left=bottom,  #                                         height=-barwidth,  #                                         align="edge",  #                                         color=colour,  #                                     )  #                                     ax[y_id, x_id].bar_label(  #                                         p,  #                                         labels=[label],  #                                         fmt="%s",  #                                         label_type="center",  #                                     )  #                                 # finally:  #                                 bottom += y_val  #                             x_ticks = x_ticks[:-1]  #                 #                 values = values  #                 #                 print('try 1 done')  #                 #  #                 #                     # for shell in values:  #                 #                     # view for group and bars  #                 #                             label = f'${lims.left} - {lims.right}$ \AA'  #                 #  #                 #                     try:  #                 #                         print('try 2')  #                 #                         print(x_dict[col], y_dict[it[yid]])  #  #                 #                     except:  #                 # #                         raise ValueError  #                 #                         x_id, y_id = 0, y_dict[it[yid]]  #                 #                         label = f'${x_bin.left} - {x_bin.right}$ \AA'  #  #                 #                     if pi == 1:  #                 #                         legends[y_id, x_id].append(it[label_id])  #                 #                 else:  #                 #                     check_logger.info('NaN values')  #  #                 except KeyError:  #                     logger.info(f"No data for {pl}, {vx}, {it}")  #  #         x_ticks = [  #             np.linspace(  #                 n_bar * bulk_edge + xpad,  #                 n_bar * bulk_edge + bulk_edge,  #                 int(bulk_edge),  #             )  #             for n_bar in range(lbars)  #         ]  #         x_ticks = np.ravel(x_ticks)  #         x_labels = np.tile(np.arange(0, bulk_edge, 1), lbars)  #         # print(x_ticks, x_labels)  #         for i in range(ly):  #             try:  #                 ax[i, 0].set_ylabel(f"{label_mod([(vy[i], y)])}\n" + rowlabel)  #                 ax[i, 0].set_yticks(np.arange(0.0, 1.1, 0.2))  #                 for j in range(lx):  #                     ax[i, j].spines[["top", "right"]].set_visible(False)  #                     ax[i, j].hlines(1.0, -xpad, lbars, linestyle="--")  #                     # ax[i, j].legend(ncol=2, loc='lower center')#[leg for leg in legends[i, j]], ncol=3)  #                     #                  if xlim != None:  #                     # ax[i, j].set_xlim((-xpad, lbars))  #                     ax[i, j].set_xticks([], [])  #                     #                 if ylim != None:  #                     ax[i, j].set_ylim((0.0, 1.25))  #                     # print(x_ticks, x_labels)  #                     ax[ly - 1, j].set_xticks(x_ticks, x_labels)  #             #                 ax[ly - 1, j].set_xlabel(columnlabel + f'\n{label_mod([(vx[j], x)])}')  #             except IndexError:  #                 ...  #     #             ax[i].set_ylabel(f'{label_mod([(vy[i], y)])}\n' + rowlabel)  #     #             ax[i].legend([label_mod([(leg, label_key)]) for leg in legends[i, 0]], ncol=3)  #     #             ax[ly - 1].set_xlabel(columnlabel + f'\n{label_mod([(vx[0], x)])}')  #     # #  #     # #     fig.supxlabel(f'{title_dict[x]}s', size=14)  #     # #     fig.supylabel(f'{title_dict[y]}s', size=14)  #     # #     if save != False:  #     # #         odir = Path(odir)  #     # #         if not odir.is_dir():  #     # #             os.makedirs(odir)  #     # #         if type(save) == str:  #     # #             fig.savefig(odir / f'{save}.png')  #     # #         else:  #     # #             fig.savefig(odir / f'{self.name}_{diffstr}_{x}_{y}_{plsave}_{self.cutoff}_{self.bins}.png')  #     # #         fig.clear()  #  # def plot_hbars(  #     self,  #     bars: Literal["clays", "aas", "ions", "atoms", "other"],  #     x: Literal["clays", "aas", "ions", "atoms", "other"],  #     y: Literal["clays", "aas", "ions", "atoms", "other"],  #     # y: Literal['clays', 'ions', 'aas', 'atoms', 'other'],  #     # select: Literal['clays', 'ions', 'aas'],  #     rowlabel: str = "y",  #     columnlabel: str = "x",  #     figsize=None,  #     dpi=None,  #     # diff=False,  #     xmax=50,  #     ymax=50,  #     save=False,  #     ylim=None,  #     odir=".",  #     # barwidth = 0.75,  #     xpad=0.25,  #     cmap="winter",  # ):  #     """Create stacked Histogram adsorption shell populations."""  #     aas_classes = [  #         ["arg", "lys", "his"],  #         ["glu", "gln"],  #         ["cys"],  #         ["gly"],  #         ["pro"],  #         ["ala", "val", "ile", "leu", "met"],  #         ["phe", "tyr", "trp"],  #         ["ser", "thr", "asp", "gln"],  #     ]  #     ions_classes = [["Na", "Ca"], ["Ca", "Mg", "Ba"]]  #     atoms_classes = [["ions"], ["N"], ["OT"], ["CA"]]  #     clays_classes = [["NAu-1"], ["NAu-2"]]  #     cmaps_seq = ["Purples", "Blues", "Greens", "Oranges", "Reds"]  #     cmaps_single = ["Dark2"]  #     sel_list = ("clays", "ions", "aas", "atoms")  #     # for color, attr in zip([''], sel_list):  #     #     cmaps_dict[attr] = {}  #     # cm.get_cmap()  #     cmap_dict = {"clays": []}  #  #     title_dict = {  #         "clays": "Clay type",  #         "ions": "Ion type",  #         "aas": "Amino acid",  #         "atoms": "Atom type",  #         "other": "Other atom type",  #     }  #  #     sel_list = ["clays", "ions", "aas", "atoms"]  #  #     # if self.other != None:  #     #     sel_list.append('other')  #     cmap = colormaps[cmap]  #     separate = [  #         s for s in sel_list if s not in [x, y, bars]  #     ]  # (s != x and s != y and s != bars and s != groups)]  #     # print(separate)  #     idx = pd.Index(  #         [s for s in sel_list if (s != x and s != bars and s not in separate)]  #     )  #     # print(idx)  #  #     sep = pd.Index(separate)  #  #     vx = getattr(self, x)  #     logger.info(f"x = {x}: {vx}")  #     lx = len(vx)  #  #     vy = getattr(self, y)  #     logger.info(f"y = {y}: {vy}")  #     ly = len(vy)  #     # print(ly)  #  #     vbars = getattr(self, bars)  #     lbars = len(vbars)  #     logger.info(f"bars = {bars}: {vbars}")  #  #     bar_dict = dict(zip(vbars, np.arange(lbars)))  #  #     yid = np.ravel(np.where(np.array(idx) == y))[0]  #     # print(yid)  #  #     # label_key = idx.difference(pd.Index([x, y, *separate]), sort=False).values[0]  #  #     # label_id = idx.get_loc(key=label_key)  #     n_plots = len(sep)  #  #     x_dict = dict(zip(vx, np.arange(lx)))  #     # print(x_dict)  #  #     sel = self.clays  #  #     plot_df = self.bin_df[sel].copy()  #     idx_names = ["clays", *plot_df.index.droplevel(["x", "_atoms"]).names]  #     # print("idx", idx_names)  #     plot_df = plot_df.stack()  #     # atoms = plot_df.index.get_level_values('_atoms')  #     plot_df.index = plot_df.index.droplevel(["x", "_atoms"])  #     # plot_df = pd.DataFrame({'values': plot_df,  #     #                         '_atoms': atoms})  #     # idx_names.remove('_atoms')  #     plot_df = plot_df.reorder_levels(idx_names)  #     #  #     plot_df.name = "values"  #     # print(plot_df.head(3))  #     plot_df = plot_df.groupby(plot_df.index.names).sum()  #     # print(plot_df.head(3))  #     inner_sum = plot_df.groupby(plot_df.index.droplevel("x_bins").names).sum()  #     extra = 1 - inner_sum  #     plot_df.where(  #         np.rint(plot_df.index.get_level_values("x_bins").right) != int(self.cutoff),  #         lambda x: x + extra,  #         inplace=True,  #     )  #     # print(type(self.edges))  #     max_edge = list(map(lambda x: np.max(x[:-1]), self.edges.values()))  #     max_edge = np.max(max_edge)  #     # print("max edge", max_edge)  #     # max_edge = np.ravel(np.array(*self.edges.values()))  #     # print(max_edge)  #     cnorm = mpc.Normalize(vmin=0, vmax=max_edge, clip=False)  #  #     if figsize is None:  #         figsize = tuple(  #             [  #                 5 * lx if (10 * lx) < xmax else xmax,  #                 5 * ly if (5 * ly) < ymax else ymax,  #             ]  #         )  #     #  #     if dpi is None:  #         dpi = 100  #     #  #     iters = np.array(  #         np.meshgrid(*[getattr(self, idxit) for idxit in idx])  #     ).T.reshape(-1, len(idx))  #     #  #     logger.info(f"Printing bar plots for {sep}\nColumns: {vx}\nRows: {vy}")  #     #  #     label_mod = lambda l: ", ".join(  #         [li.upper() if namei == "aas" else li for li, namei in l]  #     )  #     #  #     try:  #         sep_it = np.array(  #             np.meshgrid(*[getattr(self, idxit) for idxit in sep])  #         ).T.reshape(-1, len(sep))  #     except ValueError:  #         sep_it = [None]  #     # check_logger.info(vx, vy, lx, ly)  #     #  #     for pl in sep_it:  #         # print(pl)  #         # try:  #         #     fig.clear()  #         # except:  #         #     pass  #         y_dict = dict(zip(vy, np.arange(ly)))  #         # print(y_dict)  #         #     if separate == 'atoms' and pl != '':  #         #         ...  #         #  #         legends_list = [(a, b) for a in range(ly) for b in range(lx)]  #         #  #         legends = dict(zip(legends_list, [[] for a in range(len(legends_list))]))  #         #  #         # if type(pl) in [list, tuple, np.ndarray]:  #         #     #     viewlist = []  #         #     #     for p in pl:  #         #     #         viewlist.append(plot_df.xs((p), level=separate, axis=0))  #         #     #  #         #     #     sepview = pd.concat(viewlist)  #         #     #     plsave = 'ions'  #         #     #  #         #     # else:  #         fig, ax = plt.subplots(  #             nrows=ly,  #             ncols=lx,  #             figsize=figsize,  #             sharey=True,  #             dpi=dpi,  #             constrained_layout=True,  #             # sharex=True  #         )  #         if pl is None:  #             sepview = plot_df.view()  #             plsave = ""  #         else:  #             sepview = plot_df.xs((pl), level=separate, axis=0, drop_level=False)  #             plsave = pl  #             #  #             #  #  #             #  #             fig.suptitle(  #                 (  #                     ", ".join([title_dict[s].upper() for s in separate])  #                     + f": {label_mod(list(tuple(zip(pl, separate))))}"  #                 ),  #                 size=16,  #                 weight="bold",  #             )  #         pi = 0  #         for col in vx:  #             try:  #                 view = sepview.xs(col, level=x, axis=0, drop_level=False)  #                 pi = 1  #             except ValueError:  #                 view = sepview  #                 col = vx  #                 pi += 1  #             # print("column", col)  #             for it in iters:  #                 try:  #                     values = view.xs(  #                         tuple(it), level=idx.tolist(), drop_level=False  #                     )  #                     x_grouplabels = []  #                     x_labels = []  #                     x_ticks = []  #                     for vbar in vbars:  #                         bulk_pad = 2  #                         bulk_edge = np.rint((max_edge + bulk_pad))  #                         x_ticks.append(bar_dict[vbar] * bulk_edge + xpad)  #                         # print(values)  #                         bottom = 0.0  #                         x_grouplabels.append(vbar)  #                         # x_tick = bar_dict[vbar] * (barwidth + xpad)  #                         bar_vals = values.xs(vbar, level=bars, drop_level=False)  #  #                         if len(self._peaks) != len(  #                             self.df.index.get_level_values("_atoms").unique()  #                         ):  #                             self._get_edges()  #                         try:  #                             peaks = self._peaks[  #                                 bar_vals.index.get_level_values("atoms")  #                                 .unique()  #                                 .tolist()[0]  #                             ]  #                         except:  #                             peaks = self._peaks[  #                                 bar_vals.index.get_level_values("ions")  #                                 .unique()  #                                 .tolist()[0]  #                             ]  #                         #       bar_vals.values >= 0)  #                         if np.all(bar_vals.values) >= 0:  #                             # print("All > 0")  #                             x_id, y_id = x_dict[col], y_dict[it[yid]]  #  #                             bar_val_view = bar_vals  #                             bar_val_view.index = (  #                                 bar_val_view.index.get_level_values("x_bins")  #                             )  #                             # x_ticks.append(x_tick + 0.5 * barwidth)  #                             # bar_plots = []  #                             for bar_id, bar_val in enumerate(bar_val_view.items()):  #                                 x_bin, y_val = bar_val  #  #                                 try:  #                                     peak = peaks[bar_id]  #                                 except IndexError:  #                                     peak = x_bin.right  #                                 colour = cmap(cnorm(peak))  #  #                                 if x_bin.right < np.max(self.x):  #                                     label = f"${x_bin.left} - {x_bin.right}$ \AA"  #                                     barwidth = x_bin.right - x_bin.left  #  #                                 else:  #                                     label = f"$ > {x_bin.left}$ \AA"  #                                     # print(bar_val_view.index[-1].left)  #                                     barwidth = bulk_edge - x_bin.left  #                                 # try:  #                                 x_tick = x_ticks[-1] + barwidth  #                                 x_ticks.append(x_tick)  #                                 # except IndexError:  #                                 #     x_tick = x_bin.left  #                                 x_labels.append(x_bin.left)  #                                 # print(x_ticks, x_tick, x_bin)  #                                 # print(peaks, bar_id, "label", "peak", label, peak)  #                                 # print(label)  #                                 try:  #                                     p = ax[y_id, x_id].barh(  #                                         x_tick,  #                                         y_val,  #                                         label=label,  #                                         left=bottom,  #                                         height=-barwidth,  #                                         align="edge",  #                                         color=colour,  #                                     )  #                                     ax[y_id, x_id].bar_label(  #                                         p,  #                                         labels=[label],  #                                         fmt="%s",  #                                         label_type="center",  #                                     )  #                                 except IndexError:  #                                     p = ax[y_id].barh(  #                                         x_tick,  #                                         y_val,  #                                         label=label,  #                                         left=bottom,  #                                         height=-barwidth,  #                                         align="edge",  #                                         color=colour,  #                                     )  #                                     ax[y_id, x_id].bar_label(  #                                         p,  #                                         labels=[label],  #                                         fmt="%s",  #                                         label_type="center",  #                                     )  #                                 # finally:  #                                 bottom += y_val  #                             x_ticks = x_ticks[:-1]  #                 #                 values = values  #                 #                 print('try 1 done')  #                 #  #                 #                     # for shell in values:  #                 #                     # view for group and bars  #                 #                             label = f'${lims.left} - {lims.right}$ \AA'  #                 #  #                 #                     try:  #                 #                         print('try 2')  #                 #                         print(x_dict[col], y_dict[it[yid]])  #  #                 #                     except:  #                 # #                         raise ValueError  #                 #                         x_id, y_id = 0, y_dict[it[yid]]  #                 #                         label = f'${x_bin.left} - {x_bin.right}$ \AA'  #  #                 #                     if pi == 1:  #                 #                         legends[y_id, x_id].append(it[label_id])  #                 #                 else:  #                 #                     check_logger.info('NaN values')  #  #                 except KeyError:  #                     logger.info(f"No data for {pl}, {vx}, {it}")  #  #         x_ticks = [  #             np.linspace(  #                 n_bar * bulk_edge + xpad,  #                 n_bar * bulk_edge + bulk_edge,  #                 int(bulk_edge),  #             )  #             for n_bar in range(lbars)  #         ]  #         x_ticks = np.ravel(x_ticks)  #         x_labels = np.tile(np.arange(0, bulk_edge, 1), lbars)  #         # print(x_ticks, x_labels)  #         for i in range(ly):  #             try:  #                 ax[i, 0].set_ylabel(f"{label_mod([(vy[i], y)])}\n" + rowlabel)  #                 ax[i, 0].set_xticks(np.arange(0.0, 1.1, 0.2))  #                 for j in range(lx):  #                     ax[i, j].spines[["top", "right"]].set_visible(False)  #                     ax[i, j].hlines(1.0, -xpad, lbars, linestyle="--")  #                     # ax[i, j].legend(ncol=2, loc='lower center')#[leg for leg in legends[i, j]], ncol=3)  #                     #                  if xlim != None:  #                     # ax[i, j].set_xlim((-xpad, lbars))  #                     ax[i, j].set_yticks([], [])  #                     #                 if ylim != None:  #                     ax[i, j].set_ylim((0.0, 1.25))  #                     # print(x_ticks, x_labels)  #                     ax[ly - 1, j].set_yticks(x_ticks, x_labels)  #             #                 ax[ly - 1, j].set_xlabel(columnlabel + f'\n{label_mod([(vx[j], x)])}')  #             except IndexError:  #                 ...  #     #             ax[i].set_ylabel(f'{label_mod([(vy[i], y)])}\n' + rowlabel)  #     #             ax[i].legend([label_mod([(leg, label_key)]) for leg in legends[i, 0]], ncol=3)  #     #             ax[ly - 1].set_xlabel(columnlabel + f'\n{label_mod([(vx[0], x)])}')  #     # #  #     # #     fig.supxlabel(f'{title_dict[x]}s', size=14)  #     # #     fig.supylabel(f'{title_dict[y]}s', size=14)  #     # #     if save != False:  #     # #         odir = Path(odir)  #     # #         if not odir.is_dir():  #     # #             os.makedirs(odir)  #     # #         if type(save) == str:  #     # #             fig.savefig(odir / f'{save}.png')  #     # #         else:  #     # #             fig.savefig(odir / f'{self.name}_{diffstr}_{x}_{y}_{plsave}_{self.cutoff}_{self.bins}.png')  #     # #         fig.clear()  #


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

    ions = ["Na", "K", "Ca", "Mg", "Ba"]
    atoms = ["ions", "OT", "N", "CA"]
    clays = [
        "NAu-1",
        "NAu-2",
        "SWy-123",
        "SWy-1-simplified",
        "KGa-1",
        "IMt-1",
    ]  # , 'LDH31']  # , 'LDH31']
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
        atomname=True,
        vel_bins: float = 0.05,
        velrange: Tuple[float, float] = (0.0, 1.0),
        atomnames=False,
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
            if zdir is None:
                zdir = indir
            else:
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
            if analyses is None:
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
            self._get_data(
                nameparts, analyses
            )  # logger.info('Finished getting data')

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
            df.dropna(inplace=True, how="all", axis=1)
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

        if odir is not None:
            self.odir: Path = Path(odir)
        else:
            self.odir: Path = Path(".").cwd()

        logger.info(f"Output directory set to {str(self.odir.resolve())!r}\n")
        self._bin_df = pd.DataFrame(columns=self.df.columns)

        self._edges = {}
        self._peaks = {}
        self._z_bins = None
        self._other_bins = {}

    def get_fname(self, fname: Union[str, Path]) -> Path:
        """Get filename for saving plot data
        :param fname: filename
        :type fname: Union[str, Path]
        :return: filename
        :rtype: Path"""
        return Path(fname).parent / f"{Path(fname).stem}.tar.xz"

    def save(self, savename: Union[str, Path]) -> None:
        """Save dataframe to file
        :param savename: filename
        :type savename: Union[str, Path]
        """
        savename = self.get_fname(savename)
        logger.info(f"Writing dataframe to {savename}")
        pd.to_pickle(self.df, savename)

    @property
    def z_bins(self) -> HistData:
        """Get z bins
        :return: z bins
        :rtype: HistData
        """
        if self._z_bins is None:
            with open(self.filelist[self.zname][0], "rb") as file:
                data = pkl.load(file)
                self._z_bins = HistData(edges=data.edges)
        return self._z_bins

    @property
    def other_bins(self):
        for analysis in self.analyses:
            if analysis not in self._other_bins.keys():
                with open(self.filelist[analysis][0], "rb") as file:
                    data = pkl.load(file)
                    self._other_bins[analysis] = HistData(edges=data.edges)
        return self._other_bins

    def _get_data(self, nameparts, analyses):
        idsl = pd.IndexSlice
        for f in self.filelist[self.zname]:
            namesplit = f.stem.split("_")
            namesplit.pop(-1)
            namesplit = namesplit[nameparts:]
            try:
                if self.atomnames is None:
                    clay, ion, aa, pH, atom, cutoff, bins = namesplit
                else:
                    clay, ion, aa, pH, cutoff, bins = namesplit
                    atom = self.atomnames
                assert cutoff == self.cutoff
                assert bins == self.bins
                try:
                    self.zf.loc[idsl[clay, ion, aa, atom]] = f
                except KeyError:
                    self.filelist[self.zname].remove(f)
            except IndexError:
                logger.info(f"Encountered IndexError while getting data")
            except ValueError:
                logger.info(f"Encountered ValueError while getting data")

        logger.info(f"Getting {self.__class__.__name__} data")
        for analysis in tqdm(
            analyses, leave=False, position=0, desc="analyses"
        ):
            logger.info(f"\tanalysis: {analysis}")
            if len(self.filelist[analysis]) == 0:
                logger.info(f"No files found for {analysis}")
                sys.exit(1)
            for f in tqdm(
                self.filelist[analysis], leave=False, position=1, desc="files"
            ):
                namesplit = f.stem.split("_")
                name = namesplit[:nameparts]
                namesplit.pop(-1)
                namesplit = namesplit[nameparts:]
                if self.other is not None:
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
                    try:
                        zdata = self.zf.loc[idsl[clay, ion, aa, atom]]
                        timeseries = Timeseries(f)
                        timeseries.zdata = zdata
                    except KeyError as e:
                        self.filelist[analysis].remove(f)
                    if timeseries.has_zdata:
                        if self.other is None:
                            self.df.loc[
                                idsl[clay, ion, aa, atom], analysis
                            ] = timeseries
                        else:
                            self.df.loc[
                                idsl[clay, ion, aa, atom, other], analysis
                            ] = timeseries
                    else:
                        self.filelist[analysis].remove(
                            f
                        )  # except ValueError as e:  #     print(e)

                except IndexError:
                    logger.info(f"Encountered IndexError while getting data")
                except ValueError:
                    logger.info(f"Encountered ValueError while getting data")
        logger.info("Finished getting data")
        logger.info(f'{self.df.dropna(how="all", axis=0)}')
        logger.info(f'{self.df.dropna(how="all", axis=1)}')
        logger.debug(self.zf[self.zf.isna()])
        logger.debug(self.df[self.df.isna()])
        self.name = "_".join(name)
        self.pH = pH

    def __repr__(self):
        return self.df.dropna(how="all", axis=0).__repr__()

    def _get_edge_fname(
        self,
        atom_type: str,
        other: Optional[str],
        name: Union[Literal["pe"], Literal["edge"]] = "pe",
    ):
        if other is not None:
            other = f"{other}_"
        else:
            other = ""
        # fname = Path.cwd() / f"edge_data/edges_{atom_type}_{self.cutoff}_{self.bins}.p"
        fname = (
            Path.cwd()
            / f"pe_data/{atom_type}_{other}{name}_data_{self.cutoff}_{self.bins}.p"
        )
        logger.info(f"Peak/edge Filename: {fname}")
        return fname

    def _read_edge_file(self, atom_type: str, skip=True, other=None):
        fname = self._get_edge_fname(atom_type, name="edges", other=other)
        if not fname.exists():
            logger.info("No edge file found.")
            os.makedirs(fname.parent, exist_ok=True)
            logger.info(f"{fname.parent}")
            if skip is True:
                logger.info(f"Continuing without edges")
                p = [0, self.cutoff.num]
            else:
                # self._get_edges(atom_type=atom_type)
                raise FileNotFoundError(f"No edge file found {fname}.")

        else:
            with open(fname, "rb") as edges_file:
                logger.info(f"Reading edges {edges_file.name}")
                p = pkl.load(edges_file)["edges"]
        logger.info(f"edges:{p}")
        return p

    # def get_bin_df(self):  #     idx = self.df.index.names  #     bin_df = self.df.copy()  #     atom_types = bin_df.index.get_level_values("_atoms").unique().tolist()  #     bin_df.reset_index(["x_bins", "x", "_atoms"], drop=False, inplace=True)  #     for atom_type in atom_types:  #         # logger.info(f"{atom_type}")  #         try:  #             edges = self._edges[atom_type]  #         except KeyError:  #             # edge_fname = self._get_edge_fname(atom_type)  #             edges = self._read_edge_file(atom_type=atom_type)  #             # if edge_fname.is_file():  #             #     self._edges[atom_type] = self._read_edge_file(atom_type)  #             # else:  #             #     raise  #             #     self._get_edges(atom_type=atom_type)  #             # edges = self._edges[atom_type]  #         # print(edges, bin_df['x_bins'].where(bin_df['_atoms'] == atom_type))  #         bin_df["x_bins"].where(  #             bin_df["_atoms"] != atom_type,  #             pd.cut(bin_df["x"], [*edges]),  #             inplace=True,  #         )  #     bin_df.reset_index(drop=False, inplace=True)  #  #     bin_df.set_index(  #         idx,  #         inplace=True,  #     )  #     self.df = bin_df.copy()  #  # @property  # def bin_df(self):  #     if not self.df.index.get_level_values("x_bins").is_interval():  #         logger.info("No Interval")  #         self.get_bin_df()  #     else:  #         logger.info("Interval")  #     return self.df

    # with open(self.zdist, "rb") as zdist_file:  #     zdist_data = pkl.load(zdist_file)  #     self.zdist_timeseries = zdist_data.timeseries  #     self.zdist_edges = zdist_data.edges  #     self.zdist_bins = zdist_data.bins  # timeseries_arrays = {}  # hist_arrays = {}  # for analysis in self.analyses:  #     for f in self.filelist[analysis]:  #         namesplit = f.stem.split("_")  #         # if self.analysis is not None:  #         namesplit.pop(-1)  #         # else:  #         #     self.analysis = "zdist"  #         name = namesplit[:nameparts]  #         namesplit = namesplit[nameparts:]  #         # if self.other != None:  #         #     other = namesplit[4]  #         #     namesplit.pop(4)  #         #     if other in self.ions:  #         #         other = "ions"  #         #     try:  #         clay, ion, aa, pH, atom, cutoff, bins = namesplit  #         assert cutoff == self.cutoff  #         assert bins == self.bins  #         with open(f, "rb") as file:  #             array = pkl.load(f)  #         timeseries_arrays[analysis] = (array.edges, array.timeseries)  #         hist_arrays[analysis] = np.histogramdd(  #             [  #                 np.ravel(self.zdist_timeseries),  #                 np.ravel(timeseries_arrays[analysis]),  #             ],  #             bins=[self.zdist_edges, self.vel_edges],  #         )  # analysis_cols = [a for a in self.analyses if self.analyses != "totvel"]  # for analysis_col in analysis_cols:  #     hist_arrays[f"{analysis_col}/totvel"] = np.histogramdd(  #         [  #             np.ravel(self.zdist_timeseries),  #             np.divide(  #                 np.ravel(timeseries_arrays[analysis_col]),  #                 np.ravel(timeseries_arrays["totvel"]),  #                 where=np.ravel(timeseries_arrays["totvel"]) != 0,  #             ),  #         ],  #         bins=[self.zdist_edges, self.vel_edges],  #     )

    # array = pd.read_csv(f, delimiter="\s+", comment="#").to_numpy()  # try:  #     self.df.loc[idsl[ion, aa, atom, :], clay] = array[:, 2]  # except ValueError:  #     logger.info('Using second atom species')  #     self.df.loc[idsl[ion, aa, atom, other, :], clay] = array[:, 2]  # except KeyError:  #     logger.info('Using 1st column')  #     self.df.loc[idsl[ion, aa, atom, :], clay] = array[:, 1]  #     except IndexError:  #         logger.info(f"Encountered IndexError while getting data")  #     except ValueError:  #         logger.info(f"Encountered ValueError while getting data")  # self.name = "_".join(name)

    # def __repr__(self):  #     return self.df[self.clays].dropna().__repr__()


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
                pass  # if load is True:  #     loadfile = loadfname  # else:  #     loadfile = Path(load)  # assert loadfile.is_file(), f'No file named {loadfile.name!r} exists.'  # with open(loadfile, 'rb') as file:  #     df = pkl.load(file)  #     assert type(df) == pd.DataFrame, f'Loaded object is {type(df)}, expected pandas.DataFrame'  #     self.df = df
            else:
                self.split_dfs(
                    overwrite=overwrite, cutoff=new_cutoff, bins=new_bins
                )  # if save is True:  #     with open(loadfname, 'wb') as file:  #         pkl.dump(self.df, file)  # self.df.index = self.df.index.droplevel('atom_ids')  # self._process_atom_type_idx(self.df.index)  # self._init_xbin_idx()  # self.df  # self.atoms = self.df.index.get_level_values('atoms')  # self._edges = {}  # self._peaks = {}

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
        return match_list  # n_frames = int(n_frames.group(1))  # n_atoms = re.search(r"^.*sel_n_atoms.*?(\d+)\n", filestr, flags=re.MULTILINE | re.DOTALL)  # n_atoms = int(n_atoms.group(1))

    def yield_first_analysis_group_filename(self, group_key):
        for group, df in self.df.groupby(group_key, group_keys=True):
            analysis = df[self.analysis].head(1).values[0]
            datfile = analysis.filename.with_suffix(".dat")
            yield group, datfile

    def get_atom_id_groups(self, labels, ids):
        at_group_dict = {}
        vec_group = np.vectorize(
            AtomHistPlot.get_atom_type_group, excluded=["group_all_atoms"]
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
                    )  # prev_aa = aa  # prev_clay = clay  # hist, edges = np.histogram(new_ts, bins=[*self.hist_bins, self.cutoff.num])  # hist = np.divide(hist, n_frames * n_atoms)  # new_df.loc[idsl[clay, ion, aa, str(atom_id), atom_type, :], self.analysis] = hist  # new_df = new_df.groupby(new_df.index.names).sum()  # df_list.append(new_df.copy())  # del new_df  # if combined_df is None:  #     combined_df = new_df.copy()  #     del new_df  # else:  #     combined_df = pd.concat([combined_df.copy(), new_df.copy()])  #     del new_df
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

        # combined_df = pd.DataFrame({self.atomnames: self.zf,  #                              self.analysis: self.df[self.analysis]}, index = self.df.index)  # self.filelist: list = []  # self.bins: Bins = Bins(bins)  # self.cutoff: float = Cutoff(cutoff)  # self.analysis: str = analysis  # self.atomnames: str = atomnames  # assert self.analysis is not None  # assert self.atomnames is not None  # if type(indir) != Path:  #     indir = Path(indir)  #  # self.filelist_dict = {}  #  # self._indir = indir  # for filetype in [self.analysis, self.atomnames]:  #     logger.info(  #         rf"Getting {namestem}*_"  #         rf"{self.cutoff}_"  #         rf"{self.bins}_"  #         rf"{analysis}.dat from {str(indir.resolve())!r}"  #     )  #     self.filelist: List[Path] = sorted(  #         list(  #             indir.glob(  #                 rf"{namestem}*_"  #                 rf"{self.cutoff}_"  #                 rf"{self.bins}_"  #                 rf"{self.analysis}.dat"  #             )  #         )  #     )  #     logger.info(f"Found {len(self.filelist)} {filetype} data files.")  #  # if load is not False:  #     load = Path(load.resolve())  #     self.df: pd.DataFrame = pkl.load(load)  #     logger.info(f"Using data from {load!r}")  # else:  #     if ions is None:  #         ions = self.__class__.ions  #         logger.info(f"ions not specified, using default {self.__class__.ions}")  #     else:  #         logger.info(f"Using custom {ions} for ions")  #     if atoms is None:  #         atoms = self.__class__.atoms  #         logger.info(  #             f"atoms not specified, using default {self.__class__.atoms}"  #         )  #     else:  #         logger.info(f"Using custom {atoms} for atoms")  #     if atomname is False:  #         assert len(atoms) == 1, 'Expected one atom category'  #         self.atomnames = atoms[0]  #     else:  #         self.atomnames = None  #     if aas is None:  #         aas = self.__class__.aas  #         logger.info(f"aas not specified, using default {self.__class__.aas}")  #     else:  #         logger.info(f"Using custom {aas} for aas")  #     if clays is None:  #         clays = self.__class__.clays  #         logger.info(  #             f"clays not specified, using default {self.__class__.clays}"  #         )  #     else:  #         logger.info(f"Using custom {clays} for clays")  #  #     if other is not None:  #         if other is True:  #             other = atoms  #             other.append("OW")  #  #             self.other: List[str] = other  #             logger.info(f"Setting second atom selection to {self.other}")  #         else:  #             # idx = pd.MultiIndex.from_product(  #             #     [ions, aas, atoms, x], names=["ions", "aas", "atoms"]  #             # )  #             self.other: None = None  #         # self.df: pd.DataFrame = pd.DataFrame(index=idx, columns=cols)  #  #     self._get_data(nameparts)  #  # self.df.dropna(inplace=True, how="all", axis=0)

    # @redirect_tqdm  # def get_data(self, nameparts):  #     idsl = pd.IndexSlice  #     for f in self.filelist_dict[self.atomnames]:  #         namesplit = f.stem.split("_")  #         namesplit.pop(-1)  #         name = namesplit[:nameparts]  #         namesplit = namesplit[nameparts:]  #         if self.other is not None:  #             # other = namesplit[5]  #             other = namesplit.pop(5)  #             if other in self.ions:  #                 other = "ions"  #         try:  #             if self.atomnames is None:  #                 clay, ion, aa, pH, atom, cutoff, bins = namesplit  #             else:  #                 clay, ion, aa, pH, cutoff, bins = namesplit  #                 atom = self.atomnames  #             assert cutoff == self.cutoff  #             assert bins == self.bins  #             with open(f, 'rb') as file:  #                 arr = pkl.load(f)  #                 # arr = np.ravel()


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

    ions = ["Na", "K", "Ca", "Mg", "Ba"]
    clays = [
        "NAu-1",
        "NAu-2",
        "SWy-123",
        "SWy-1-simplified",
        "KGa-1",
        "IMt-1",
    ]  # , 'LDH31']
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
                self.alt = None  # Is this condition even necessary?
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
            self._get_data()  # logger.info('Finished getting data')

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
    ions = ["Na", "K", "Ca", "Mg", "Ba"]
    atoms = ["ions", "OT", "N", "CA"]
    clays = [
        "NAu-1",
        "NAu-2",
        "SWy-123",
        "SWy-1-simplified",
        "KGa-1",
        "IMt-1",
    ]  # , 'LDH31']

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
            if ions is None:
                ions = self.__class__.ions
            if atoms is None:
                atoms = self.__class__.atoms
            if aas is None:
                aas = self.__class__.aas
            self.filelist = sorted(
                list(indir.glob(rf"{namestem}*_{self.cutoff}_{self.bins}.p"))
            )

            with open(self.filelist[0], "rb") as f:
                data = pkl.load(f)
                array, self.ybins, self.xbins = data.values()

            cols = pd.Index(["NAu-1", "NAu-2"], name="clays")

            if other is not None:
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

            self._get_data(
                nameparts
            )  # self.df.dropna(inplace=True, how='all', axis=0)

        setattr(self, self.df.columns.name, list(self.df.columns))

        for iid, i in enumerate(self.df.index.names):
            value = (
                self.df.index._get_level_values(level=iid).unique().tolist()
            )
            setattr(self, i, value)

        if odir is not None:
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
            if self.other is not None:
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
        ions_classes = [["Na", "Ca"], ["Ca", "Mg", "Ba"]]
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
        if self.other is not None:
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

        if figsize is None:
            figsize = tuple(
                [
                    5 * lx if (10 * lx) < xmax else xmax,
                    5 * ly if (5 * ly) < ymax else ymax,
                ]
            )

        if dpi is None:
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
                        if xlim is not None:
                            ax[i, j].set_xlim((0.0, float(xlim)))
                        if ylim is not None:
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


# def make_1d(arr: NDArray):
#     arr_list = []
#     idxs = np.arange(arr.ndim)
#     for idx in idxs:
#         dimarr = np.add.reduce(arr, axis=[idxs == idx])
#         arr_list.append(dimarr)
#     return arr_list


def make_1d(arr: NDArray, sel=None, return_dims=False):
    idxs = np.arange(arr.ndim)
    arr_list = []
    if sel is None:
        sel = idxs
    else:
        sel = np.ravel(sel)
    for idx in sel:
        dimarr = np.add.reduce(arr, axis=tuple(idxs[idxs != idx]))
        arr_list.append(dimarr)
    if return_dims == True:
        return np.array(*dimarr), sel
    else:
        return np.array(*dimarr)


class Plot(ABC):
    # __slots__ = ('data')
    sel_list = None
    title_dict = None
    cmaps = None
    _init_args = {}
    _plot_args = {}

    # def __init_subclass__(cls, data, **kwargs):
    #     cls.data = data

    def __repr__(self):
        return f'{self.__class__.__name__!r} ({len(self.data.df.dropna(how="all"))} entries)'

    @classmethod
    def __subclasshook__(cls, subclass):
        if cls is Plot:
            return _check_methods(
                subclass, "plot", "save", "_process_data", "plot_df"
            )

    def plot_args_decorator(select: str):
        def plot_decorator(plot_f):
            def wrapper(self, **kwargs):
                self.select = select
                plot_f(self, **kwargs)
                self.select = None

            return wrapper

        return plot_decorator

    @redirect_tqdm
    def __init__(
        self,
        data: Data,  # lx: int,
        # ly: int,
        # xmax: int,
        # ymax: int,
        **kwargs,
    ):
        logger.info(f"{self.__class__.__name__}")
        # assert (
        #     type(data).__name__ == DataD.__name__
        # ), f"expected {Data.__name__}, found {type(data).__name__}"
        self.data = data
        init_args = {}
        logger.info(f"{kwargs.keys()}")
        for key, val in kwargs.items():
            try:
                setattr(self, key, val)
            except AttributeError:
                logger.error(
                    f"{key} cannot be set for {self.__class__.__name__}"
                )
            if key in self.__class__._init_args:
                init_args[key] = val
        self._process_data(**init_args)
        logger.info(f"{self.__class__.__name__}")

    @abstractmethod
    def _process_data(self, **kwargs):
        pass

    @abstractmethod
    def plot(self, **kwargs):
        pass

    @abstractmethod
    def save(self, **kwargs):
        pass

    def get_figsize(self, xmax: int, ymax: int):
        logger.info(f"Setting figsize to x > {xmax}," f" y > {ymax}")
        return tuple(
            [
                6 * self.x.l if (6 * self.x.l) < xmax else xmax,
                7 * self.y.l if (7 * self.y.l) < ymax else ymax,
            ]
        )

    def _get_idx_iter(self, idx):
        # logger.info(f'idx: {idx}')
        idx_iter = np.array(
            np.meshgrid(
                *[
                    self.plot_df.index.get_level_values(idxit).unique()
                    for idxit in idx
                ]
            )
        ).T.reshape(-1, len(idx))
        # logger.info(f'{idx_iter}')
        return idx_iter

    @staticmethod
    def modify_plot_labels(l: Tuple[str, str]):
        """

        :param l: tuple of label category and value
        :type l: Tuple[str, str]
        :return: label value
        :rtype: str
        """
        # logging.info(f'{l}, {len(l)}')
        new_str = ", ".join(
            [li.upper() if namei == "aas" else li for li, namei in l]
        )
        new_str = re.sub("_", " ", new_str)
        return new_str

    def init_legend(self, lx, ly):
        # initialise legends
        legends_list: list = [(a, b) for a in range(ly) for b in range(lx)]
        self.legends: dict = dict(
            zip(legends_list, [[] for a in range(len(legends_list))])
        )
        self.handles = dict(
            zip(legends_list, [[] for a in range(len(legends_list))])
        )

    def _get_attr(self, attr: Literal["clays", "aas", "atoms", "ions"]):
        return pd.unique(self.plot_df.index.get_level_values(attr))

    get_clays = partialmethod(_get_attr, attr="clays")
    get_aas = partialmethod(_get_attr, attr="aas")
    get_ions = partialmethod(_get_attr, attr="ions")
    get_atoms = partialmethod(_get_attr, attr="atoms")
    get_other = partialmethod(_get_attr, attr="other")

    @property
    @abstractmethod
    def plot_df(self):
        ...

    def _init_plot(self, x, y, select):
        sel_list = ["clays", "aas", "atoms", "ions"]

        if self.select == "ions":
            sel_list.pop(-1)

        if self.data.other is not None:
            sel_list.append("other")

        assert x in sel_list, f"{x} not in sel list"
        assert y in sel_list, f"{y} not in sel list"
        assert select in sel_list, f"{select} not in sel list"

    # @abstractmethod
    # def plot_ions(self, *args, **kwargs):
    #     ...
    #
    # @abstractmethod
    # def plot_other(self, *args, **kwargs):
    #     ...

    def _get_colour_dict(self):
        from seaborn import saturate, set_hls_values

        colours = [
            "#8dd3c7",  # cyan (C)
            "#ffffb3",  # yellow (S)
            "#bebada",  # blue (N)
            "#fb8072",  # red (O)
            "#80b1d3",  # light blue (Mg)
            "#fdb462",  # orange (K)
            "#b3de69",  # light green (Ca)
            "#fccde5",  # pink (Na)
            # "#d9d9d9",  # grey (other)
        ]
        colours = [set_hls_values(saturate(cval), l=0.4) for cval in colours]
        colour_keys = ["C", "S", "N", "O", "Mg", "K", "Ca", "Na"]
        colour_dict = dict(zip(colour_keys, colours))
        return colour_dict

    @staticmethod
    def get_atom_type_group(atom_name: str, group_all_atoms):
        if atom_name in ["O", "OT", "OXT"]:
            atom_name = "OT"
        if group_all_atoms is True:
            if (
                len(atom_name) == 1
                or atom_name == "CA"
                or atom_name == "OT"
                or atom_name[1].islower()
            ):
                pass
            elif atom_name[1] == "H" and atom_name[0] != "C":
                atom_name = atom_name[:2]
            else:
                atom_name = f"{atom_name[0]}R"
        return atom_name

    def get_atom_colour_codes_from_names(self, atom_name):
        # atom_name = AtomHistPlot.get_atom_type_group(atom_name, group_all_atoms=True)
        if atom_name not in self.data.ions:
            atom_name = atom_name[0]
        return atom_name


# class Axis(UserString):
#     def __init__(self, plot_instance, name):
#         # if name == 'atoms':
#         #     name = '_atoms'
#         self.v = sorted(plot_instance._get_attr(name))
#         self.data = name
#         self.l = len(self.v)
#         self.dict = dict(zip(self.v, np.arange(self.l)))
#         logger.info(f"{self.data}: {self.v}")


class LinePlot(Plot):
    sel_list = ["clays", "ions", "aas", "_atoms"]
    title_dict = {
        "clays": "Clay type",
        "ions": "Ion type",
        "aas": "Amino acid",
        "atoms": "Atom type",
        "other": "Other atom type",
    }
    _init_args = {"x", "y", "lines"}
    _plot_args = {
        "rowlabel": "y",
        "columnlabel": "x",
        "dpi": None,
        "figsize": None,
    }
    _default_colour = "#7f7f7f"

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.bins = "x"
        self.line = None

    def plot_args_decorator(select: str):
        def plot_decorator(plot_f):
            def wrapper(self, **kwargs):
                self.select = select
                plot_f(self, **kwargs)
                self.select = None

            return wrapper

        return plot_decorator

    @property
    def plot_df(self):
        try:
            return getattr(self, f"_{self.select}_bin_df")  # _{self.line}")
        except AttributeError:
            logging.error(f"Wrong usage of property, select not assigned yet.")

    @property
    def colour_df(self):
        try:
            return getattr(self, f"_{self.select}_colour_df")  # _{self.line}")
        except AttributeError:
            logging.error(f"Wrong usage of property, select not assigned yet.")

    def _get_binned_plot_colour_dfs_1d(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info(f"Getting plot and colour dfs")
        # from seaborn import set_hls_values, saturate
        #
        # # atom type colour cycle
        # colours = [
        #     "#8dd3c7",  # cyan (C)
        #     "#ffffb3",  # yellow (S)
        #     "#bebada",  # blue (N)
        #     "#fb8072",  # red (O)
        #     "#80b1d3",  # light blue (Mg)
        #     "#fdb462",  # orange (K)
        #     "#b3de69",  # light green (Ca)
        #     "#fccde5",  # pink (Na)
        # ]
        # colours = [set_hls_values(saturate(cval), l=0.4) for cval in colours]
        # colour_keys = ["C", "S", "N", "O", "Mg", "K", "Ca", "Na"]
        colour_dict = (
            self._get_colour_dict()
        )  # dict(zip(colour_keys, colours))
        # colours = color_palette('dark').as_hex()
        sel = self.data.clays

        # get data for plotting
        try:
            # clays still in columns
            plot_df = self.data.bin_df[sel].copy()

            # move clays category from columns to index
            idx_names = ["clays", *plot_df.index.droplevel(["x_bins"]).names]
            # DataFrame -> Series
            plot_df = plot_df.stack()
        except KeyError:
            plot_df = self.data.bin_df.copy()
            idx_names = plot_df.index.droplevel(["x_bins"]).names

        # get values for atom types (including separate ions)
        atoms: np.array = plot_df.index.get_level_values("_atoms").to_numpy()
        # atom_type_groups = np.unique(atoms)
        transform_atom_names_vec = np.vectorize(
            self.get_atom_colour_codes_from_names, otypes=[str]
        )
        # transform_dict = dict(zip(atom_type_groups, transform_atom_names_vec(atom_type_groups)))
        atoms = transform_atom_names_vec(atoms)
        # atom_type_groups = transform_atom_names_vec(atom_type_groups)

        # make new DataFrame from atom_type index level and values
        plot_df.index = plot_df.index.droplevel(["x_bins"])
        # plot_df.index.names = [name.strip('_') for name in plot_df.index.names]

        try:
            plot_df = pd.DataFrame({"values": plot_df, "colours": atoms})
        except ValueError:
            #     if self.select == 'ions':
            plot_df["colours"] = atoms
            plot_df["values"] = plot_df[
                self.data.analysis
            ]  # else:  #     plot_df['colours'] = atoms.transform(lambda x: x[0])

        # self._plot_df = {}
        # self._colour_df = {}

        # color_cycle = itertools.cycle(colours)
        # colour_dict = {}
        # for atom_type in np.unique(atoms):
        #     if atom_type not in colour_dict.keys():
        #         colour_dict[atom_type] = next(color_cycle)
        # colour_dict = dict(zip(atom_type_groups, colours[:len(atom_type_groups)]))
        try:
            plot_df["colours"] = plot_df["colours"].apply(
                lambda x: colour_dict[x]
            )
        except KeyError:
            plot_df["colours"] = self._default_colour

        # reorder index for grouping
        plot_df = plot_df.reorder_levels(idx_names)

        # # group and sum densities within adsorption shell bins
        # plot_df = plot_df.groupby(plot_df.index.names).agg(
        #     values=pd.NamedAgg("values", "sum"), colours=pd.NamedAgg("colours", "first")
        # )

        # separate colour column from plot_df -> yields 2 Series
        colour_df = plot_df["colours"]
        plot_df = plot_df["values"]

        # add missing atom probabilities from bulk to the largest bin
        # (bin.left, cutoff] -> (bin.left, all bulk])
        # inner_sum = plot_df.groupby(plot_df.index.droplevel("x_bins").names).sum()
        # extra = 1 - inner_sum
        # plot_df.where(
        #     np.rint(plot_df.index.get_level_values("x_bins").right)
        #     != int(self.data.cutoff),
        #     lambda x: x + extra,
        #     inplace=True,
        # )
        # setattr(self, "_plot_df", plot_df.copy())
        self._plot_df, self._colour_df = plot_df.copy(), colour_df.copy()
        logger.info(
            f"{self._plot_df}\n{self._colour_df}"
        )  # print("sel list", self.sel_list)

        # for line in self.sel_list:  #  #     line_plot_df = plot_df.copy()  #     # get values for atom types (including separate ions)  #     if line == "_atoms":  #         line = "atoms"  #     lines = line_plot_df.index.get_level_values(line)  #  #     # make new DataFrame from atom_type index level and values  #     line_plot_df.index = line_plot_df.index.droplevel(["x_bins"])  #  #     # plot_df.index.names = [name.strip('_') for name in plot_df.index.names]  #  #     line_plot_df = pd.DataFrame({"values": line_plot_df, "colours": lines})  #  #     # map unique atom types to colour map  #     line_types = lines.unique()  #     colour_dict = dict(zip(line_types, colours[: len(line_types)]))  #     print(colour_dict)  #     line_plot_df["colours"] = line_plot_df["colours"].transform(  #         lambda x: colour_dict[x]  #     )  #  #     # reorder index for grouping  #     line_plot_df = line_plot_df.reorder_levels(idx_names)  #  #     # group and sum densities within adsorption shell bins  #     colour_df = line_plot_df.groupby(line_plot_df.index.names).agg(  #         colours=pd.NamedAgg("colours", "first")  #     )  #  #     # separate colour column from plot_df -> yields 2 Series  #     colour_df = colour_df["colours"]  #     # print(colour_df)  #     line_plot_df = line_plot_df["values"]  #  #     # add missing atom probabilities from bulk to the largest bin  #     # (bin.left, cutoff] -> (bin.left, all bulk])  #     # inner_sum = plot_df.groupby(plot_df.index.droplevel("x_bins").names).sum()  #     # extra = 1 - inner_sum  #     # plot_df.where(  #     #     np.rint(plot_df.index.get_level_values("x_bins").right)  #     #     != int(self.data.cutoff),  #     #     lambda x: x + extra,  #     #     inplace=True,  #     # )  #     # setattr(self, f"_plot_df_{line}", line_plot_df.copy())  #     self._plot_df[line], self._colour_df[line] = (  #         line_plot_df.copy(),  #         colour_df.copy(),  #     )

    @redirect_tqdm
    def _split_plot_dfs(self):
        logger.info(f"Splitting plot and colour dfs")
        bin_df = self._plot_df.copy()
        plot_df = self._colour_df.copy()
        dfs = {"bin": bin_df, "colour": plot_df}
        for key, df in dfs.items():
            if "ions" in self.data.atoms:
                not_ions = [a for a in self.data.atoms if a != "ions"]
                ion_view = df.xs("ions", level="atoms")
                ion_view.index = ion_view.index.droplevel(["ions"])
                ion_view.index.names = [
                    idstr.strip("_") for idstr in ion_view.index.names
                ]
                setattr(self, f"_ions_{key}_df", ion_view.copy())
            else:
                not_ions = np.unique(self.data.atoms).tolist()

            other_view = df.loc[pd.IndexSlice[:, :, :, not_ions, :]]
            other_view.index = other_view.index.droplevel("_atoms")

            setattr(
                self, f"_other_{key}_df", other_view.copy()
            )  # dfs = {}  # # print(self._plot_df.keys())  # for line in self.sel_list:  #     if line == "_atoms":  #         line = "atoms"  #     bin_df = self._plot_df[line].copy()  #     plot_df = self._colour_df[line].copy()  #     dfs[line] = {"bin": bin_df.copy(), "colour": plot_df.copy()}  #     for key, df in dfs[line].items():  #         not_ions = [a for a in self.data.atoms if a != "ions"]  #         ion_view = df.xs("ions", level="atoms")  #         ion_view.index = ion_view.index.droplevel(["ions"])  #         ion_view.index.names = [  #             idstr.strip("_") for idstr in ion_view.index.names  #         ]  #  #         other_view = df.loc[pd.IndexSlice[:, :, :, not_ions, :]]  #         other_view.index = other_view.index.droplevel("_atoms")  #         print(f"_ions_{key}_df_{line}")  #         setattr(self, f"_ions_{key}_df_{line}", ion_view.copy())  #         setattr(self, f"_other_{key}_df_{line}", other_view.copy())

    def _process_data(self):
        self._get_binned_plot_colour_dfs_1d()
        self._split_plot_dfs()

    def get_suptitle(self, pl, separate):
        self.fig.suptitle(
            (
                ", ".join([self.title_dict[s].upper() for s in separate])
                + f": {self.label_mod(list(tuple(zip(pl, separate))))}"
            ),
            size=16,
            weight="bold",
        )

    @redirect_tqdm
    def plot(
        self,
        lines: Literal["clays", "ions", "aas", "atoms", "other"],
        x: Literal["clays", "aas", "ions", "atoms", "other"],
        y: Literal["clays", "ions", "aas", "atoms", "other"],
        rowlabel: str = "y",
        columnlabel: str = "x",
        figsize=None,
        dpi=None,
        diff=False,
        xmax=50,
        ymax=50,
        plsave=False,
        xlim=None,
        ylim=None,
        odir=".",
        plot_table=None,
        edges=False,
        sparse=None,
        contract_binsize=None,
        antialiased=True,
        smooth_line=0.01,
        colours=None,
    ):
        # line style cycler
        linestyle_list = [
            (0, ()),
            (0, (1, 2)),
            (0, (5, 2)),
            (0, (5, 2, 1, 2)),
            (0, (5, 2, 1, 2, 1, 2)),
            (0, (1, 4)),
            (0, (3, 4, 1, 4)),
            (0, (3, 4, 1, 4, 1, 4)),
            (0, (7, 6)),
            (0, (7, 6, 3, 6)),
            (0, (7, 6, 3, 6, 1, 2)),
        ]
        if sparse is True:
            pass
        self._init_plot(x, y, lines)

        plot_df = self.plot_df.copy()
        plot_df = plot_df.groupby(plot_df.index.names).sum()
        colour_df = self.colour_df.copy()
        colour_df = colour_df.groupby(colour_df.index.names).sum()
        # linestyle_dict = {}
        line_cycle = itertools.cycle(linestyle_list)
        # line_idx = colour_df.index.droplevel('x')
        line_grouped = (
            colour_df.groupby("atoms").first().duplicated(False).to_dict()
        )
        for at_type, duplicate in line_grouped.items():
            if duplicate is True:
                line_grouped[at_type] = next(line_cycle)
            else:
                line_cycle = itertools.cycle(linestyle_list)
                line_grouped[at_type] = linestyle_list[0]
        del at_type, duplicate, line_cycle

        # logger.info(f'{plot_df}')

        avail_attrs = plot_df.index.names
        logger.info(f"Available attributes: {avail_attrs}")

        logger.info("Setting x:")
        self.x = Axis(self, x)
        logger.info("Setting y:")
        self.y = Axis(self, y)
        logger.info("Setting lines:")
        self.line = Axis(self, lines)

        separate = [
            s
            for s in avail_attrs
            if s not in [self.x, self.y, self.line, self.bins]
        ]

        if len(separate) != 0:
            logger.info(f"Separate plots: {separate}")

        idx = pd.Index(
            [
                s
                for s in avail_attrs
                if (
                    s != self.x
                    and s != self.line
                    and s not in [*separate, self.bins]
                )
            ]
        )

        if colours is not None:
            from seaborn import saturate, set_hls_values

            colours = [
                set_hls_values(saturate(cval), l=0.4) for cval in colours
            ]
            colour_dict = dict(zip(self.line.v, colours))

        logger.info(f"Iteration index: {idx.values}")

        sep = pd.Index(separate)

        yid = np.ravel(np.where(np.array(idx) == self.y))[0]

        if figsize is None:
            logger.info(f"{xmax}, {ymax} figsize")
            figsize = self.get_figsize(xmax=xmax, ymax=ymax)

        # set resultion
        if dpi is None:
            dpi = 100

        # get plotting iter from index
        iters = self._get_idx_iter(idx=idx)

        logger.info(
            f"Printing line plots for {self.line}\nColumns: {self.x.v}\nRows: {self.y.v}"
        )

        # set label modifier function
        self.label_mod = self.modify_plot_labels

        try:
            # iterator for more than one plot
            sep_it = self._get_idx_iter(idx=sep)
        except ValueError:
            # only one plot
            sep_it = [None]
        # logger.info(f'{sep_it}\n{plot_df}')

        # iterate over separate plots
        for pl in sep_it:
            y_max = 0
            if pl is None:
                pl_str = ""
            else:
                pl_str = f"_{pl[0]}"
            self.name = (
                f"{self.data.name}_{self.data.analysis}_{self.select}"
                f"_{x}_{y}_{lines}{pl_str}_{self.data.cutoff}_{self.data.bins}"
            )
            if xlim is not None:
                self.name += f"_{xlim:.0f}"
            logger.info(f"plot {self.name}")

            # index map for y values
            # y_dict: dict = dict(zip(vy, np.arange(ly)))

            self.init_legend(ly=self.y.l, lx=self.x.l)

            self.fig, self.ax = plt.subplots(
                nrows=self.y.l,
                ncols=self.x.l,
                figsize=tuple(figsize),
                sharey=True,
                sharex=True,
                dpi=dpi,
                constrained_layout=True,
            )

            if pl is None:
                logger.info(f"Generating plot")
                sepview = plot_df.view()
                save_str = ""

            # multiple plots
            else:
                logger.info(f"Generating {pl} plot:")
                # print(plot_df.head(20), "\n", separate, pl)
                sepview = plot_df.xs((pl), level=separate, drop_level=False)
                save_str = pl  # self.get_suptitle(pl, separate)

            # set plot index
            pi = 0

            # iterate over subplot columns
            for col in self.x.v:
                # logger.info(col)
                try:
                    # print(sepview.dropna().index)
                    view = sepview.xs(
                        col, level=self.x, axis=0, drop_level=False
                    )
                    pi = 1
                except ValueError:
                    view = sepview
                    col = self.x.v
                    pi += 1
                except KeyError:
                    continue
                for it in iters:
                    try:
                        values = view.xs(
                            tuple(it), level=idx.tolist(), drop_level=False
                        )
                        v = values.index.get_level_values("atoms").unique()
                        logger.info(v)
                        # print(values)
                        # x_labels = []

                        # logging.info(f"it: {it}")  # , {self.line.v}, {values}")
                        x_id, y_id = self.x.dict[col], self.y.dict[it[yid]]
                        # if edges is True:
                        #     atom_type = values.index.get_level_values("atoms").unique()[
                        #         0
                        #     ]
                        #     edge_list = self.data.edges[atom_type]
                        #     p = self.ax[y_id, x_id].vlines(
                        #         edge_list, 0, ylim, color="gray", linestyles=(0, (1, 4))
                        #     )
                        line_cycle = itertools.cycle(linestyle_list)
                        for line_num, vline in enumerate(self.line.v):
                            logging.info(
                                f"line {vline} ({line_num + 1}/{self.line.l})"
                            )
                            # set linestyle
                            if self.line != "atoms":
                                # line_cycle = itertools.cycle(linestyle_list)
                                # linestyle_dict = {}
                                # for atom_type in np.unique(self.data.atoms):
                                #     linestyle_dict[atom_type] = next(line_cycle)
                                linestyle = next(
                                    line_cycle
                                )  # linestyle_list[line_num % len(linestyle_list)]
                            else:
                                linestyle = line_grouped[vline]

                            try:
                                line_vals: pd.Series = values.xs(
                                    vline, level=self.line, drop_level=False
                                )
                                if contract_binsize is not None:
                                    line_vals = (
                                        line_vals.rolling(
                                            contract_binsize,
                                            closed="right",
                                            center=True,
                                            step=contract_binsize,
                                        )
                                        .sum()
                                        .fillna(0)
                                    )
                            except KeyError:
                                logger.info(f"No data for {vline}")
                                continue
                            if colours is None:
                                colour = colour_df.loc[line_vals.index].values[
                                    0
                                ]
                            else:
                                colour = colour_dict[vline]
                            # if sparse is True:
                            #     line_vals = line_vals.rolling(10, min_periods=1, step=10).sum()
                            bin_list = line_vals.index.get_level_values("x")
                            if (
                                smooth_line is not None
                                and smooth_line is not False
                            ):
                                if isinstance(smooth_line, float):
                                    if self.data.bins.num < smooth_line:
                                        break
                                elif isinstance(smooth_line, int):
                                    smooth_line = (
                                        self.data.bins.num / smooth_line
                                    )
                                else:
                                    smooth_line = self.data.bins.num / 10
                                dense_bin_list = np.arange(
                                    bin_list[0], bin_list[-1], smooth_line
                                )
                            line_vals.index = bin_list
                            # logger.info(f'{vline}, {self.line}')
                            # print(line_vals.hasnans, np.max(line_vals.values), vline, line_num, y_id, x_id)
                            summed = np.round(np.sum(line_vals.values), 4)
                            if (
                                not line_vals.hasnans and summed > 0.0
                            ):  # and vline != 'C':
                                if np.max(line_vals) > y_max:
                                    y_max = np.max(line_vals)
                                    logger.info(
                                        f"adjusting max value: {y_max}"
                                    )
                                try:
                                    label = f"{self.label_mod([(vline, self.line)]):<2}"
                                    if hasattr(
                                        self.data, "ignore_density_sum"
                                    ):
                                        logger.debug(
                                            f"Ignoring line value sum check sum={summed:.2f}"
                                        )
                                        if summed != 1.0:
                                            logger.info(
                                                f"labelling sum {summed}"
                                            )
                                            label_add = (
                                                rf" ({summed * 100:>.0f} \%)"
                                            )
                                            label = f"{label:<3}{label_add:>7}"
                                    else:
                                        assert (
                                            np.round(summed) == 1.00
                                        ), f"Densities ({np.round(summed)}) do not sum to 1.00"
                                    if pi == 1:
                                        self.legends[(y_id, x_id)].append(
                                            label
                                        )
                                    # if sparse is True:
                                    #     line_vals_mask = line_vals.values != 0
                                    # spline = splrep(bin_list,#[line_vals_mask],
                                    #                               line_vals.values)#[line_vals_mask])
                                    # line_vals = splev(bin_list, spline)
                                    # else:
                                    #     line_vals = line_vals.values
                                    # print(x_id)
                                    if (
                                        smooth_line is not None
                                        and smooth_line is not False
                                    ):
                                        bin_list_plot = dense_bin_list
                                        line_vals_plot = make_smoothing_spline(
                                            bin_list, line_vals.values
                                        )(bin_list_plot)
                                    else:
                                        bin_list_plot = bin_list
                                        line_vals_plot = line_vals.values
                                    try:
                                        # f, ax = plt.subplots()
                                        # ax.plot(bin_list, line_vals)
                                        # f.suptitle(vline)
                                        # f.show()
                                        logger.info(
                                            f"{len(bin_list)}, {len(line_vals)}"
                                        )
                                        # p = self.ax[y_id, x_id].plot([1,1], [0,1], label='a')
                                        try:
                                            p = self.ax[y_id, x_id].plot(
                                                bin_list_plot,
                                                line_vals_plot,
                                                label=label,
                                                color=colour,
                                                linestyle=linestyle,
                                                antialiased=antialiased,
                                            )

                                        except IndexError:
                                            xy_id = np.max([x_id, y_id])
                                            p = self.ax[xy_id].plot(
                                                bin_list_plot,
                                                line_vals_plot,
                                                label=label,
                                                color=colour,
                                                linestyle=linestyle,
                                                antialiased=antialiased,
                                            )
                                        except TypeError:
                                            p = self.ax.plot(
                                                bin_list_plot,
                                                line_vals_plot,
                                                label=label,
                                                color=colour,
                                                linestyle=linestyle,
                                                antialiased=antialiased,
                                            )  # self.fig.show()  # self.fig.show()  # self.fig.savefig('a.png')  # logger.info('wrote file')  # logger.info(f'{pl} {col} {vline} {it[yid]}')  # handles[(y_id, x_id)] = p
                                    except Exception as e:
                                        # del p
                                        print(e, y_id, x_id, line_num, vline)

                                    # self.ax[y_id, x_id].bar_label(p, labels=[label],  #                          fmt='%s',  #                          label_type='center')
                                except IndexError:
                                    p = self.ax[x_id].plot(
                                        bin_list,
                                        line_vals.values,
                                        label=label,
                                        color=colour,
                                        linestyle=linestyle,
                                        antialiased=antialiased,
                                    )
                            else:
                                logger.info(
                                    f"nan or 0: {vline}"
                                )  # self.ax[y_id, x_id].legend()
                    # except FileNotFoundError:
                    except KeyError:
                        logger.info(
                            f"No data for {pl}, {self.x}, {it} {it[yid]}"
                        )
            i = 0
            j = 0
            for i in range(self.y.l):
                y_ax_label = f"{self.label_mod([(self.y.v[i], self.y)])}\n"
                if self.y.l > 1:
                    if i == 0:
                        self.fig.supylabel(f"{self.title_dict[y]}s", size=14)
                else:
                    if i == 0 and not re.match(
                        y_ax_label, "\nctl", flags=re.IGNORECASE
                    ):
                        self.fig.supylabel(
                            f"{self.title_dict[y]}: {y_ax_label}", size=14
                        )
                    self.name = re.sub(
                        rf"(.*)_{y}_(.*)",
                        r"\1_" + y_ax_label.strip("\n") + r"_\2",
                        self.name,
                    )
                    y_ax_label = (
                        ""  # f"{self.label_mod([(self.y.v[i], self.y)])}\n"
                    )
                try:
                    self.ax[i, 0].set_ylabel(y_ax_label + rowlabel)
                except IndexError:
                    self.ax[i].set_ylabel(y_ax_label + rowlabel)
                except TypeError:
                    self.ax.set_ylabel(y_ax_label + rowlabel)
                for j in range(self.x.l):
                    x_ax_label = f"\n{self.label_mod([(self.x.v[j], self.x)])}"
                    if self.x.l > 1:
                        if j == 0:
                            self.fig.supxlabel(
                                f"{self.title_dict[x]}s", size=14
                            )
                    else:
                        if j == 0 and not re.match(
                            x_ax_label, "\nctl", flags=re.IGNORECASE
                        ):
                            self.fig.supxlabel(
                                f"{self.title_dict[x]}: {x_ax_label}", size=14
                            )

                        self.name = re.sub(
                            rf"(.*)_{x}_(.*)",
                            r"\1_" + x_ax_label.strip("\n") + r"_\2",
                            self.name,
                        )
                        x_ax_label = ""  # f"\n{self.label_mod([(self.x.v[j], self.x)])}"
                    label = [
                        self.label_mod([(leg, self.line)])
                        for leg in self.legends[i, j]
                    ]
                    if len(label) % 3 == 0:
                        ncol = 3
                    else:
                        ncol = 2
                    try:
                        self.ax[i, j].legend(
                            label,
                            ncol=ncol,
                            title=self.title_dict[lines],
                            frameon=False,
                            loc="upper right",
                        )
                        if xlim is not None:
                            self.ax[i, j].set_xlim((0.0, float(xlim)))
                        if ylim is not None:
                            self.ax[i, j].set_ylim((0.0, float(ylim)))
                        else:
                            self.ax[i, j].set_ylim((0.0, float(y_max)))
                        self.ax[self.y.l - 1, j].set_xlabel(
                            columnlabel + x_ax_label
                        )
                        self.ax[i, j].spines[["top", "right"]].set_visible(
                            False
                        )
                    except IndexError:
                        try:
                            max_id = np.max(i, j)
                        except IndexError:
                            max_id = np.max(i, j)[0]
                        self.ax[max_id].legend(
                            label,
                            ncol=ncol,
                            title=self.title_dict[lines],
                            frameon=False,
                            loc="upper right",
                        )
                        if xlim is not None:
                            self.ax[j].set_xlim((0.0, float(xlim)))
                        if ylim is not None:
                            self.ax[i].set_ylim((0.0, float(ylim)))
                        else:
                            self.ax[i].set_ylim(
                                (0.0, np.round(float(y_max + 0.05 * y_max), 2))
                            )
                        self.ax[self.y.l - 1].set_xlabel(
                            columnlabel + x_ax_label
                        )
                        self.ax[j].spines[["top", "right"]].set_visible(False)
                        self.ax[i].spines[["top", "right"]].set_visible(False)
                    except TypeError:
                        self.ax.legend(
                            label,
                            ncol=ncol,
                            title=self.title_dict[lines],
                            frameon=False,
                            loc="upper right",
                        )
                        if xlim is not None:
                            self.ax.set_xlim((0.0, float(xlim)))
                        if ylim is not None:
                            self.ax.set_ylim((0.0, float(ylim)))
                        else:
                            self.ax.set_ylim(
                                (0.0, np.round(float(y_max + 0.05 * y_max), 2))
                            )
                        self.ax.set_xlabel(columnlabel + x_ax_label)
                        self.ax.spines[["top", "right"]].set_visible(False)
                    if j == 0:
                        try:
                            y_ticks = self.ax[i, 0].get_yticks()
                            x_ticks = self.ax[i, j].get_xticks()
                        except IndexError:
                            y_ticks = self.ax[i].get_yticks()
                            x_ticks = self.ax[j].get_xticks()
                        except TypeError:
                            y_ticks = self.ax.get_yticks()
                            x_ticks = self.ax.get_xticks()
                        finally:
                            y_int = np.rint(len(y_ticks) / 5).astype(int)
                            y_ticks = y_ticks[::y_int]
                            x_int = np.rint(len(x_ticks) / 5).astype(int)
                            x_ticks = x_ticks[::x_int]
                            try:
                                self.ax[i, j].set_xticks(x_ticks, x_ticks)
                            except IndexError:
                                self.ax[j].set_xticks(
                                    x_ticks, x_ticks.astype(int)
                                )
                            except TypeError:
                                self.ax.set_xticks(
                                    x_ticks, x_ticks.astype(int)
                                )
                            if i == 0:
                                try:
                                    self.ax[i, 0].set_yticks(y_ticks, y_ticks)
                                except IndexError:
                                    self.ax[i].set_yticks(y_ticks, y_ticks)
                                except TypeError:
                                    self.ax.set_yticks(
                                        y_ticks, y_ticks
                                    )  # self.fig.supxlabel(f"{self.title_dict[x]}s", size=14)  # self.fig.supylabel(f"{self.title_dict[y]}s", size=14)

                # except IndexError:  #     self.ax[i].set_ylabel(  #         f"{self.label_mod([(self.y.v[i], y)])}\n" + rowlabel  #     )  #     # self.ax[i].legend([self.label_mod([(leg, self.label_key)]) for leg in self.legends[i, 0]], ncol=3)  #     self.ax[self.y.l - 1].set_xlabel(  #         columnlabel + f"\n{self.label_mod([(self.x.v[0], self.x)])}"  #     )
            if edges is True:
                if self.line != "atoms":
                    max_y = plot_df.max()
                    for col in self.x.v:
                        # logger.info(col)
                        try:
                            view = sepview.xs(
                                col, level=self.x, axis=0, drop_level=False
                            )
                            pi = 1
                        except ValueError:
                            view = sepview
                            col = self.x.v
                            pi += 1
                        for it in iters:
                            try:
                                values = view.xs(
                                    tuple(it),
                                    level=idx.tolist(),
                                    drop_level=False,
                                )
                                # print(values)
                                # x_labels = []

                                logging.info(
                                    f"it: {it}"
                                )  # , {self.line.v}, {values}")
                                x_id, y_id = (
                                    self.x.dict[col],
                                    self.y.dict[it[yid]],
                                )

                                atom_type = values.index.get_level_values(
                                    "atoms"
                                ).unique()[0]
                                edge_list = self.data.edges[atom_type]
                                try:
                                    p = self.ax[y_id, x_id].vlines(
                                        edge_list,
                                        0,  # np.max(line_vals),
                                        max_y,
                                        color="gray",
                                        linestyles=(0, (1, 4)),
                                    )
                                except IndexError:
                                    xy_id = np.max([x_id, y_id])
                                    p = self.ax[xy_id].vlines(
                                        edge_list,
                                        0,  # np.max(line_vals),
                                        max_y,
                                        color="gray",
                                        linestyles=(0, (1, 4)),
                                    )
                            except TypeError:
                                xy_id = np.max([x_id, y_id])
                                p = self.ax.vlines(
                                    edge_list,
                                    0,  # np.max(line_vals),
                                    max_y,
                                    color="gray",
                                    linestyles=(0, (1, 4)),
                                )
                            except KeyError:
                                logger.info(
                                    f"No data for {pl}, {self.x}, {it}"
                                )

            # self.fig.supxlabel(f"{self.title_dict[x]}s", size=14)
            # self.fig.supylabel(f"{self.title_dict[y]}s", size=14)

            if plsave != False:
                logger.info("Saving plot")
                if type(plsave) == str:
                    outname = f"{plsave}.png"
                else:
                    outname = f"{self.name}.png"
                odir = Path(odir).absolute()
                logger.info(f"output to {odir.absolute()}")
                if not odir.is_dir():
                    os.makedirs(odir)
                logger.info(odir / outname)
                self.fig.savefig(str(odir / outname))
            else:
                if type(plsave) == str:
                    outname = f"{plsave}.p"
                else:
                    outname = f"{self.name}.p"
                odir = Path(odir).absolute()
                logger.info(f"output to {odir.absolute()}")
                if not odir.is_dir():
                    os.makedirs(odir)
                logger.info(odir / outname)
                with open(odir / outname, "wb") as pklfile:
                    pkl.dump(
                        {"fig": self.fig, "ax": self.ax, "data": self.data.df},
                        pklfile,
                    )
                self.fig.show()
            self.fig.clear()

        def _get_bin_label(self, x_bin, bin_list):
            if x_bin.right < np.max(bin_list.right):
                label = f"${x_bin.left} - {x_bin.right}$ \AA"  # barwidth = x_bin.right - x_bin.left

            else:
                label = f"$ > {x_bin.left}$ \AA"
            return label

    # @property
    # def plot_df(self):
    #     try:
    #         return getattr(self, f"_{self.select}_bin_df")
    #     except AttributeError:
    #         logging.error(f"Wrong usage of property, line not assigned yet.")
    #
    # @property
    # def colour_df(self):
    #     try:
    #         return getattr(self, f"_{self.select}_colour_df")
    #     except AttributeError:
    #         logging.error(f"Wrong usage of property, line not assigned yet.")

    def _get_attr(self, attr: Literal["clays", "aas", "atoms", "ions"]):
        return pd.unique(self.plot_df.index.get_level_values(attr))

    get_clays = partialmethod(_get_attr, attr="clays")
    get_aas = partialmethod(_get_attr, attr="aas")
    get_ions = partialmethod(_get_attr, attr="ions")
    get_atoms = partialmethod(_get_attr, attr="atoms")
    get_other = partialmethod(_get_attr, attr="other")

    def save(self, **kwargs):
        ...

    @plot_args_decorator(select="ions")
    def plot_ions(
        self,
        lines: Literal["clays", "ions", "aas", "atoms", "other"],
        x: Literal["clays", "aas", "ions", "atoms", "other"],
        y: Literal["clays", "ions", "aas", "atoms", "other"],
        rowlabel: str = "y",
        columnlabel: str = "x",
        figsize=None,
        dpi=None,
        diff=False,
        xmax=50,
        ymax=50,
        plsave=False,
        xlim=None,
        ylim=None,
        odir=".",
        plot_table=None,
        edges=True,
        sparse=None,
        antialiased=True,
        smooth_line=0.01,
        colours=None,
    ):
        self.plot(
            lines=lines,
            x=x,
            y=y,
            rowlabel=rowlabel,
            columnlabel=columnlabel,
            figsize=figsize,
            dpi=dpi,
            diff=diff,
            xmax=xmax,
            ymax=ymax,
            plsave=plsave,
            xlim=xlim,
            ylim=ylim,
            odir=odir,
            plot_table=plot_table,
            edges=edges,
            sparse=sparse,
            antialiased=antialiased,
            smooth_line=smooth_line,
            colours=colours,
        )

    @plot_args_decorator(select="other")
    def plot_other(
        self,
        lines: Literal["clays", "ions", "aas", "atoms", "other"],
        x: Literal["clays", "aas", "ions", "atoms", "other"],
        y: Literal["clays", "ions", "aas", "atoms", "other"],
        rowlabel: str = "y",
        columnlabel: str = "x",
        figsize=None,
        dpi=None,
        diff=False,
        xmax=50,
        ymax=50,
        plsave=False,
        xlim=None,
        ylim=None,
        odir=".",
        plot_table=None,
        edges=True,
        sparse=None,
        contract_binsize=None,
        antialiased=True,
        smooth_line=0.01,
        colours=None,
    ):
        self.plot(
            lines=lines,
            x=x,
            y=y,
            rowlabel=rowlabel,
            columnlabel=columnlabel,
            figsize=figsize,
            dpi=dpi,
            diff=diff,
            xmax=xmax,
            ymax=ymax,
            plsave=plsave,
            xlim=xlim,
            ylim=ylim,
            odir=odir,
            plot_table=plot_table,
            edges=edges,
            sparse=sparse,
            contract_binsize=contract_binsize,
            antialiased=antialiased,
            smooth_line=smooth_line,
            colours=colours,
        )


class SDevLinePlot(LinePlot):
    def __init__(
        self, data: List[Data] = None, indirs=None, select="other", **kwargs
    ):
        if data is None and indirs is not None:
            data = [Data(indir=indir, **kwargs) for indir in indirs]
            for d in data:
                d.df.sort_index(inplace=True)
                d.select = select
        elif data is not None and indirs is None:
            pass
        else:
            raise ValueError(f"Specify either data or indirs")
        new_data = copy.deepcopy(data[0])
        new_data.var_df = new_data.bin_df.copy()
        new_data.var_df = np.NaN
        if len(data) > 1:
            for d in data:
                d.__class__.__name__ == "Data", f"Expected Data, found {d.__class__.__name__}"
            props = {}
            for prop_name in [
                "cutoff",
                "bins",
                "analysis",
                "_arr_col",
                "atomnames",
                "select",
            ]:
                prop = np.unique(
                    [
                        d.__getattribute__(prop_name)
                        for d in data
                        if d.__getattribute__(prop_name) is not None
                    ]
                )
                if not prop:
                    props[prop_name] = prop
                else:
                    assert (
                        len(prop) == 1
                    ), "Incompatible Data objects, cutoff doesn't match"
                    props[prop_name] = prop[0]
            for d in data[1:]:
                # assert d.df.index.identical(new_data.df.index), 'Non-matching indices'
                assert d.bin_df.index.identical(
                    new_data.bin_df.index
                ), "Non-matching indices"
                new_data.df += d.bin_df
            new_data.df = new_data.bin_df / len(data)
            new_data.var_df = (new_data.bin_df - new_data.bin_df) ** 2
            for d in data[1:]:
                new_data.var_df += (d.bin_df - new_data.bin_df) ** 2
            new_data.var_df = new_data.var_df.divide(len(data))
            new_data.std_df = np.sqrt(new_data.var_df)
        # self.data = new_data
        # for attr in ['clays', 'ions', 'aas', 'atoms']:
        #     self.__setattr__(attr, new_data.__getattribute__(attr))
        super().__init__(new_data)

    def _get_binned_plot_colour_dfs_1d(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info(f"Getting plot and colour dfs")
        # from seaborn import set_hls_values, saturate
        #
        # # atom type colour cycle
        # colours = [
        #     "#8dd3c7",  # cyan (C)
        #     "#ffffb3",  # yellow (S)
        #     "#bebada",  # blue (N)
        #     "#fb8072",  # red (O)
        #     "#80b1d3",  # light blue (Mg)
        #     "#fdb462",  # orange (K)
        #     "#b3de69",  # light green (Ca)
        #     "#fccde5",  # pink (Na)
        # ]
        # colours = [set_hls_values(saturate(cval), l=0.4) for cval in colours]
        # colour_keys = ["C", "S", "N", "O", "Mg", "K", "Ca", "Na"]
        colour_dict = (
            self._get_colour_dict()
        )  # dict(zip(colour_keys, colours))

        sel = self.data.clays
        for df_name in ["bin", "std"]:
            # get data for plotting
            try:
                # clays still in columns
                plot_df = self.data.__getattribute__(f"{df_name}_df")[
                    sel
                ].copy()

                # move clays category from columns to index
                idx_names = [
                    "clays",
                    *plot_df.index.droplevel(["x_bins"]).names,
                ]
                # DataFrame -> Series
                plot_df = plot_df.stack()
            except KeyError:
                plot_df = self.data.bin_df.copy()
                idx_names = plot_df.index.droplevel(["x_bins"]).names

            # get values for atom types (including separate ions)
            atoms: np.array = plot_df.index.get_level_values(
                "_atoms"
            ).to_numpy()

            transform_atom_names_vec = np.vectorize(
                self.get_atom_colour_codes_from_names, otypes=[str]
            )

            atoms = transform_atom_names_vec(atoms)

            # make new DataFrame from atom_type index level and values
            plot_df.index = plot_df.index.droplevel(["x_bins"])

            try:
                plot_df = pd.DataFrame({"values": plot_df, "colours": atoms})
            except ValueError:
                plot_df["colours"] = atoms
                plot_df["values"] = plot_df[self.data.analysis]

            try:
                plot_df["colours"] = plot_df["colours"].apply(
                    lambda x: colour_dict[x]
                )
            except KeyError:
                plot_df["colours"] = self._default_colour

            # reorder index for grouping
            plot_df = plot_df.reorder_levels(idx_names)
            colour_df = plot_df["colours"]
            plot_df = plot_df["values"]
            setattr(self, f"_plot_{df_name}_df", plot_df.copy())
            setattr(
                self, f"_colour_{df_name}_df", colour_df.copy()
            )  # self._plot_df, self._colour_df = plot_df.copy(), colour_df.copy()  # logger.info(f"{self._plot_df}\n{self._colour_df}")

    @redirect_tqdm
    def _split_plot_dfs(self):
        logger.info(f"Splitting plot and colour dfs")
        for df_name in ["bin", "std"]:
            bin_df = self.__getattribute__(f"_plot_{df_name}_df").copy()
            plot_df = self.__getattribute__(f"_colour_{df_name}_df").copy()
            dfs = {"bin": bin_df, "colour": plot_df}
            for key, df in dfs.items():
                if "ions" in self.data.atoms:
                    not_ions = [a for a in self.data.atoms if a != "ions"]
                    ion_view = df.xs("ions", level="atoms")
                    ion_view.index = ion_view.index.droplevel(["ions"])
                    ion_view.index.names = [
                        idstr.strip("_") for idstr in ion_view.index.names
                    ]
                    setattr(self, f"_ions_{key}_{df_name}_df", ion_view.copy())
                else:
                    not_ions = np.unique(self.data.atoms).tolist()

                other_view = df.loc[pd.IndexSlice[:, :, :, not_ions, :]]
                other_view.index = other_view.index.droplevel("_atoms")

                setattr(self, f"_other_{key}_{df_name}_df", other_view.copy())

    @property
    def plot_df(self):
        try:
            return getattr(
                self, f"_{self.select}_bin_bin_df"
            )  # _{self.line}")
        except AttributeError:
            logging.error(f"Wrong usage of property, select not assigned yet.")

    @property
    def colour_df(self):
        try:
            return getattr(
                self, f"_{self.select}_colour_bin_df"
            )  # _{self.line}")
        except AttributeError:
            logging.error(f"Wrong usage of property, select not assigned yet.")

    @property
    def plot_sdev_df(self):
        try:
            return getattr(
                self, f"_{self.select}_bin_std_df"
            )  # _{self.line}")
        except AttributeError:
            logging.error(f"Wrong usage of property, select not assigned yet.")

    @property
    def colour_sdev_df(self):
        try:
            return getattr(
                self, f"_{self.select}_colour_std_df"
            )  # _{self.line}")
        except AttributeError:
            logging.error(f"Wrong usage of property, select not assigned yet.")

    def compare(
        self, first, second, category, df=None, sdev=None, select="other"
    ):
        if not self.select:
            sel_reset = True
            self.select = select
        else:
            sel_reset = False
        if df is not None:
            pass
        else:
            df = self.plot_df.copy()
            sdev = self.plot_sdev_df.copy()
        avail_attrs = df.index.names
        group_attrs = [a for a in avail_attrs if a != category]
        df_slice_1 = df.xs(first, level=category)
        df_slice_2 = df.xs(second, level=category)
        df_diff = df_slice_1 - df_slice_2
        sdev_diff = None
        if sdev is not None:
            sdev_slice_1 = sdev.xs(first, level=category)
            sdev_slice_2 = sdev.xs(second, level=category)
            sdev_diff = (sdev_slice_1**2 - sdev_slice_2**2).apply(np.sqrt)
        if sel_reset:
            self.select = None
        return df_diff, sdev_diff

    @redirect_tqdm
    def plot(
        self,
        lines: Literal["clays", "ions", "aas", "atoms", "other"],
        x: Literal["clays", "aas", "ions", "atoms", "other"],
        y: Literal["clays", "ions", "aas", "atoms", "other"],
        rowlabel: str = "y",
        columnlabel: str = "x",
        figsize=None,
        dpi=None,
        diff=False,
        xmax=50,
        ymax=50,
        plsave=False,
        xlim=None,
        ylim=None,
        odir=".",
        plot_table=None,
        edges=False,
        sparse=None,
        contract_binsize=None,
        smooth_line=0.01,
        antialiased=True,
        colours=None,
    ):
        # line style cycler
        linestyle_list = [
            (0, ()),
            (0, (1, 2)),
            (0, (5, 2)),
            (0, (5, 2, 1, 2)),
            (0, (5, 2, 1, 2, 1, 2)),
            (0, (1, 4)),
            (0, (3, 4, 1, 4)),
            (0, (3, 4, 1, 4, 1, 4)),
            (0, (7, 6)),
            (0, (7, 6, 3, 6)),
            (0, (7, 6, 3, 6, 1, 2)),
        ]
        if sparse is True:
            pass
        self._init_plot(x, y, lines)

        plot_df = self.plot_df.copy()
        plot_df = plot_df.groupby(plot_df.index.names).sum()
        plot_sdev_df = self.plot_sdev_df.copy()
        plot_sdev_df = plot_sdev_df.groupby(plot_sdev_df.index.names).sum()
        colour_df = self.colour_df.copy()
        colour_df = colour_df.groupby(colour_df.index.names).sum()
        colour_sdev_df = self.colour_sdev_df.copy()
        colour_sdev_df = colour_sdev_df.groupby(
            colour_sdev_df.index.names
        ).sum()
        # linestyle_dict = {}
        line_cycle = itertools.cycle(linestyle_list)
        # line_idx = colour_df.index.droplevel('x')
        line_grouped = (
            colour_df.groupby("atoms").first().duplicated(False).to_dict()
        )
        for at_type, duplicate in line_grouped.items():
            if duplicate is True:
                line_grouped[at_type] = next(line_cycle)
            else:
                line_cycle = itertools.cycle(linestyle_list)
                line_grouped[at_type] = linestyle_list[0]
        del at_type, duplicate, line_cycle

        # logger.info(f'{plot_df}')

        avail_attrs = plot_df.index.names
        logger.info(f"Available attributes: {avail_attrs}")

        logger.info("Setting x:")
        self.x = Axis(self, x)
        logger.info("Setting y:")
        self.y = Axis(self, y)
        logger.info("Setting lines:")
        self.line = Axis(self, lines)

        separate = [
            s
            for s in avail_attrs
            if s not in [self.x, self.y, self.line, self.bins]
        ]

        if len(separate) != 0:
            logger.info(f"Separate plots: {separate}")

        idx = pd.Index(
            [
                s
                for s in avail_attrs
                if (
                    s != self.x
                    and s != self.line
                    and s not in [*separate, self.bins]
                )
            ]
        )

        logger.info(f"Iteration index: {idx.values}")

        sep = pd.Index(separate)

        yid = np.ravel(np.where(np.array(idx) == self.y))[0]

        if figsize is None:
            logger.info(f"{xmax}, {ymax} figsize")
            figsize = self.get_figsize(xmax=xmax, ymax=ymax)

        # set resultion
        if dpi is None:
            dpi = 100

        # get plotting iter from index
        iters = self._get_idx_iter(idx=idx)

        logger.info(
            f"Printing line plots for {self.line}\nColumns: {self.x.v}\nRows: {self.y.v}"
        )

        # set label modifier function
        self.label_mod = self.modify_plot_labels

        try:
            # iterator for more than one plot
            sep_it = self._get_idx_iter(idx=sep)
        except ValueError:
            # only one plot
            sep_it = [None]
        # logger.info(f'{sep_it}\n{plot_df}')

        # iterate over separate plots
        for pl in sep_it:
            y_max = 0
            if pl is None:
                pl_str = ""
            else:
                pl_str = f"_{pl[0]}"
            self.name = (
                f"{self.data.name}_{self.data.analysis}_{self.select}"
                f"_{x}_{y}_{lines}{pl_str}_{self.data.cutoff}_{self.data.bins}"
            )
            if xlim is not None:
                self.name += f"_{xlim:.0f}"
            logger.info(f"plot {self.name}")

            # index map for y values
            # y_dict: dict = dict(zip(vy, np.arange(ly)))

            self.init_legend(ly=self.y.l, lx=self.x.l)

            self.fig, self.ax = plt.subplots(
                nrows=self.y.l,
                ncols=self.x.l,
                figsize=tuple(figsize),
                sharey=True,
                sharex=True,
                dpi=dpi,
                constrained_layout=True,
            )

            if pl is None:
                logger.info(f"Generating plot")
                sepview = plot_df.view()
                sepview_sdev = plot_sdev_df.view()
                save_str = ""

            # multiple plots
            else:
                logger.info(f"Generating {pl} plot:")
                # print(plot_df.head(20), "\n", separate, pl)
                sepview = plot_df.xs((pl), level=separate, drop_level=False)
                sepview_sdev = plot_sdev_df.xs(
                    (pl), level=separate, drop_level=False
                )
                save_str = pl  # self.get_suptitle(pl, separate)

            # set plot index
            pi = 0

            # iterate over subplot columns
            for col in self.x.v:
                # logger.info(col)
                try:
                    # print(sepview.dropna().index)
                    view = sepview.xs(
                        col, level=self.x, axis=0, drop_level=False
                    )
                    view_sdev = sepview_sdev.xs(
                        col, level=self.x, axis=0, drop_level=False
                    )
                    pi = 1
                except ValueError:
                    view = sepview
                    view_sdev = sepview_sdev
                    col = self.x.v
                    pi += 1
                except KeyError:
                    continue
                for it in iters:
                    try:
                        values = view.xs(
                            tuple(it), level=idx.tolist(), drop_level=False
                        )
                        values_sdev = view_sdev.xs(
                            tuple(it), level=idx.tolist(), drop_level=False
                        )
                        v = values.index.get_level_values("atoms").unique()
                        v_sdev = values_sdev.index.get_level_values(
                            "atoms"
                        ).unique()
                        logger.info(v)
                        # print(values)
                        # x_labels = []

                        # logging.info(f"it: {it}")  # , {self.line.v}, {values}")
                        x_id, y_id = self.x.dict[col], self.y.dict[it[yid]]
                        # if edges is True:
                        #     atom_type = values.index.get_level_values("atoms").unique()[
                        #         0
                        #     ]
                        #     edge_list = self.data.edges[atom_type]
                        #     p = self.ax[y_id, x_id].vlines(
                        #         edge_list, 0, ylim, color="gray", linestyles=(0, (1, 4))
                        #     )
                        line_cycle = itertools.cycle(linestyle_list)
                        for line_num, vline in enumerate(self.line.v):
                            logging.info(
                                f"line {vline} ({line_num + 1}/{self.line.l})"
                            )
                            # set linestyle
                            if self.line != "atoms":
                                # line_cycle = itertools.cycle(linestyle_list)
                                # linestyle_dict = {}
                                # for atom_type in np.unique(self.data.atoms):
                                #     linestyle_dict[atom_type] = next(line_cycle)
                                linestyle = next(
                                    line_cycle
                                )  # linestyle_list[line_num % len(linestyle_list)]
                            else:
                                linestyle = line_grouped[vline]

                            try:
                                line_vals: pd.Series = values.xs(
                                    vline, level=self.line, drop_level=False
                                )
                                line_vals_sdev: pd.Series = values_sdev.xs(
                                    vline, level=self.line, drop_level=False
                                )
                                if contract_binsize is not None:
                                    line_vals = (
                                        line_vals.rolling(
                                            contract_binsize,
                                            closed="right",
                                            center=True,
                                            step=contract_binsize,
                                        )
                                        .sum()
                                        .fillna(0)
                                    )
                                    line_vals_sdev = (
                                        line_vals.rolling(
                                            contract_binsize,
                                            closed="right",
                                            center=True,
                                            step=contract_binsize,
                                        )
                                        .sum()
                                        .fillna(0)
                                    )
                            except KeyError:
                                logger.info(f"No data for {vline}")
                                continue

                            colour = colour_df.loc[line_vals.index].values[0]
                            # if sparse is True:
                            #     line_vals = line_vals.rolling(10, min_periods=1, step=10).sum()
                            bin_list = line_vals.index.get_level_values("x")
                            line_vals.index = bin_list
                            line_vals_sdev.index = bin_list
                            # logger.info(f'{vline}, {self.line}')
                            # print(line_vals.hasnans, np.max(line_vals.values), vline, line_num, y_id, x_id)
                            summed = np.round(np.sum(line_vals.values), 4)
                            if (
                                not line_vals.hasnans and summed > 0.0
                            ):  # and vline != 'C':
                                if np.max(line_vals) > y_max:
                                    y_max = (
                                        np.max(line_vals)
                                        + line_vals_sdev.values[
                                            np.argmax(line_vals)
                                        ]
                                    )
                                    logger.info(
                                        f"adjusting max value: {y_max}"
                                    )
                                try:
                                    label = f"{self.label_mod([(vline, self.line)]):<2}"
                                    if hasattr(
                                        self.data, "ignore_density_sum"
                                    ):
                                        logger.debug(
                                            f"Ignoring line value sum check sum={summed:.2f}"
                                        )
                                        if summed != 1.0:
                                            logger.info(
                                                f"labelling sum {summed}"
                                            )  # label_add = rf" ({summed*100:>.0f} \%)"  # label = f"{label:<3}{label_add:>7}"
                                    else:
                                        assert (
                                            np.round(summed) == 1.00
                                        ), f"Densities ({np.round(summed)}) do not sum to 1.00"
                                    if pi == 1:
                                        self.legends[(y_id, x_id)].append(
                                            label
                                        )
                                    # if sparse is True:
                                    #     line_vals_mask = line_vals.values != 0
                                    # spline = splrep(bin_list,#[line_vals_mask],
                                    #                               line_vals.values)#[line_vals_mask])
                                    # line_vals = splev(bin_list, spline)
                                    # else:
                                    #     line_vals = line_vals.values
                                    # print(x_id)
                                    try:
                                        # f, ax = plt.subplots()
                                        # ax.plot(bin_list, line_vals)
                                        # f.suptitle(vline)
                                        # f.show()
                                        logger.info(
                                            f"{len(bin_list)}, {len(line_vals)}, {len(line_vals_sdev)}"
                                        )
                                        # p = self.ax[y_id, x_id].plot([1,1], [0,1], label='a')
                                        try:
                                            p1 = self.ax[
                                                y_id, x_id
                                            ].fill_between(
                                                bin_list,
                                                (
                                                    line_vals - line_vals_sdev
                                                ).astype(float),
                                                (
                                                    line_vals + line_vals_sdev
                                                ).astype(float),
                                                alpha=0.25,
                                                color=colour,
                                            )
                                            p = self.ax[y_id, x_id].plot(
                                                bin_list,
                                                line_vals,
                                                label=label,
                                                color=colour,
                                                linestyle=linestyle,
                                                antialiased=antialiased,
                                            )

                                        except IndexError:
                                            xy_id = np.max([x_id, y_id])
                                            p1 = self.ax[xy_id].fill_between(
                                                bin_list,
                                                (
                                                    line_vals - line_vals_sdev
                                                ).astype(float),
                                                (
                                                    line_vals + line_vals_sdev
                                                ).astype(float),
                                                alpha=0.25,
                                                color=colour,
                                            )
                                            p = self.ax[xy_id].plot(
                                                bin_list,
                                                line_vals,
                                                label=label,
                                                color=colour,
                                                linestyle=linestyle,
                                                antialiased=antialiased,
                                            )

                                        except TypeError:
                                            p1 = self.ax.fill_between(
                                                bin_list,
                                                (
                                                    line_vals - line_vals_sdev
                                                ).astype(float),
                                                (
                                                    line_vals + line_vals_sdev
                                                ).astype(float),
                                                alpha=0.25,
                                                color=colour,
                                            )
                                            p = self.ax.plot(
                                                bin_list,
                                                line_vals,
                                                label=label,
                                                color=colour,
                                                linestyle=linestyle,
                                            )

                                        # self.fig.show()  # self.fig.show()  # self.fig.savefig('a.png')  # logger.info('wrote file')  # logger.info(f'{pl} {col} {vline} {it[yid]}')
                                    except Exception as e:
                                        # del p
                                        print(e, y_id, x_id, line_num, vline)

                                    # self.ax[y_id, x_id].bar_label(p, labels=[label],  #                          fmt='%s',  #                          label_type='center')
                                except IndexError:
                                    p1 = self.ax[x_id].fill_between(
                                        bin_list,
                                        (line_vals + line_vals_sdev).astype(
                                            float
                                        ),
                                        (line_vals - line_vals_sdev).astype(
                                            float
                                        ),
                                        alpha=0.25,  # label=label,
                                        color=colour,
                                    )
                                    p = self.ax[x_id].plot(
                                        bin_list,
                                        line_vals.values,
                                        label=label,
                                        color=colour,
                                        linestyle=linestyle,
                                        antialiased=antialiased,
                                    )
                                finally:
                                    self.handles[(y_id, x_id)].extend(p)
                            else:
                                logger.info(f"nan or 0: {vline}")

                        # self.ax[y_id, x_id].legend()
                    # except FileNotFoundError:
                    except KeyError:
                        logger.info(
                            f"No data for {pl}, {self.x}, {self.y}, {self.line}, {it} {it[yid]}"
                        )
            i = 0
            j = 0
            for i in range(self.y.l):
                y_ax_label = f"{self.label_mod([(self.y.v[i], self.y)])}\n"
                if self.y.l > 1:
                    if i == 0:
                        pass  # self.fig.supylabel(f"{self.title_dict[y]}s", size=14)
                else:
                    if i == 0:
                        pass  # self.fig.supylabel(  #     f"{self.title_dict[y]}: {y_ax_label}", size=14  # )
                    y_ax_label = (
                        ""  # f"{self.label_mod([(self.y.v[i], self.y)])}\n"
                    )
                try:
                    self.ax[i, 0].set_ylabel(y_ax_label + rowlabel)
                except IndexError:
                    self.ax[i].set_ylabel(y_ax_label + rowlabel)
                except TypeError:
                    self.ax.set_ylabel(y_ax_label + rowlabel)
                for j in range(self.x.l):
                    x_ax_label = f"\n{self.label_mod([(self.x.v[j], self.x)])}"
                    if self.x.l > 1:
                        if j == 0:
                            pass  # self.fig.supxlabel(f"{self.title_dict[x]}s", size=14)
                    else:
                        if j == 0:
                            pass
                        # self.fig.supxlabel(
                        #     f"{self.title_dict[x]}: {x_ax_label}", size=14
                        # )
                        x_ax_label = ""  # f"\n{self.label_mod([(self.x.v[j], self.x)])}"
                    label = [
                        self.label_mod([(leg, self.line)])
                        for leg in self.legends[i, j]
                    ]
                    handle = np.ravel(self.handles[(i, j)]).tolist()
                    if len(label) % 3 == 0:
                        ncol = 3
                    else:
                        ncol = 2
                    try:
                        if j == self.x.l - 1 and i == 0:
                            self.ax[i, j].legend(
                                handles=handle,
                                labels=label,
                                ncol=ncol,
                                title=lines,
                                frameon=False,
                                loc="upper right",
                            )
                        if xlim is not None:
                            self.ax[i, j].set_xlim((0.0, float(xlim)))
                        if ylim is not None:
                            self.ax[i, j].set_ylim((0.0, float(ylim)))
                        else:
                            self.ax[i, j].set_ylim((0.0, float(y_max)))
                        self.ax[self.y.l - 1, j].set_xlabel(
                            columnlabel + x_ax_label
                        )
                        self.ax[i, j].spines[["top", "right"]].set_visible(
                            False
                        )
                    except IndexError:
                        try:
                            max_id = np.max([i, j])
                        except IndexError:
                            max_id = np.max([i, j])[0]
                        self.ax[max_id].legend(
                            handles=handle,
                            labels=label,  # label,
                            ncol=ncol,
                            title=lines,
                            frameon=False,
                            loc="upper right",
                        )
                        if xlim is not None:
                            self.ax[j].set_xlim((0.0, float(xlim)))
                        if ylim is not None:
                            self.ax[i].set_ylim((0.0, float(ylim)))
                        else:
                            self.ax[i].set_ylim(
                                (0.0, np.round(float(y_max + 0.05 * y_max), 2))
                            )
                        self.ax[max_id].set_xlabel(columnlabel + x_ax_label)
                        self.ax[j].spines[["top", "right"]].set_visible(False)
                        self.ax[i].spines[["top", "right"]].set_visible(False)
                    except TypeError:
                        self.ax.legend(
                            label,
                            ncol=ncol,
                            title=lines,
                            frameon=False,
                            loc="upper right",
                        )
                        if xlim is not None:
                            self.ax.set_xlim((0.0, float(xlim)))
                        if ylim is not None:
                            self.ax.set_ylim((0.0, float(ylim)))
                        else:
                            self.ax.set_ylim(
                                (0.0, np.round(float(y_max + 0.05 * y_max), 2))
                            )
                        self.ax.set_xlabel(columnlabel + x_ax_label)
                        self.ax.spines[["top", "right"]].set_visible(False)
                    if j == 0:
                        try:
                            y_ticks = self.ax[i, 0].get_yticks()
                            x_ticks = self.ax[i, j].get_xticks()
                        except IndexError:
                            y_ticks = self.ax[i].get_yticks()
                            x_ticks = self.ax[j].get_xticks()
                        except TypeError:
                            y_ticks = self.ax.get_yticks()
                            x_ticks = self.ax.get_xticks()
                        finally:
                            y_int = np.rint(len(y_ticks) / 5).astype(int)
                            y_ticks = y_ticks[::y_int]
                            x_int = np.rint(len(x_ticks) / 5).astype(int)
                            x_ticks = x_ticks[::x_int]
                            try:
                                self.ax[i, j].set_xticks(
                                    x_ticks.astype(int), x_ticks.astype(int)
                                )
                            except IndexError:
                                self.ax[j].set_xticks(
                                    x_ticks, x_ticks.astype(int)
                                )
                            except TypeError:
                                self.ax.set_xticks(
                                    x_ticks, x_ticks.astype(int)
                                )
                            if i == 0:
                                try:
                                    self.ax[i, 0].set_yticks(
                                        y_ticks, [f"{t:.3f}" for t in y_ticks]
                                    )
                                except IndexError:
                                    self.ax[i].set_yticks(y_ticks, y_ticks)
                                except TypeError:
                                    self.ax.set_yticks(y_ticks, y_ticks)
                    self.fig.supxlabel(f"{self.title_dict[x]}s", size=14)
                    self.fig.supylabel(f"{self.title_dict[y]}s", size=14)

                # except IndexError:  #     self.ax[i].set_ylabel(  #         f"{self.label_mod([(self.y.v[i], y)])}\n" + rowlabel  #     )  #     # self.ax[i].legend([self.label_mod([(leg, self.label_key)]) for leg in self.legends[i, 0]], ncol=3)  #     self.ax[self.y.l - 1].set_xlabel(  #         columnlabel + f"\n{self.label_mod([(self.x.v[0], self.x)])}"  #     )
            if edges is True:
                if self.line != "atoms":
                    max_y = plot_df.max()
                    for col in self.x.v:
                        # logger.info(col)
                        try:
                            view = sepview.xs(
                                col, level=self.x, axis=0, drop_level=False
                            )
                            view_sdev = sepview_sdev.xs(
                                col, level=self.x, axis=0, drop_level=False
                            )
                            pi = 1
                        except ValueError:
                            view = sepview
                            view_sdev = sepview_sdev
                            col = self.x.v
                            pi += 1
                        for it in iters:
                            try:
                                values = view.xs(
                                    tuple(it),
                                    level=idx.tolist(),
                                    drop_level=False,
                                )
                                # print(values)
                                # x_labels = []

                                logging.info(
                                    f"it: {it}"
                                )  # , {self.line.v}, {values}")
                                x_id, y_id = (
                                    self.x.dict[col],
                                    self.y.dict[it[yid]],
                                )

                                atom_type = values.index.get_level_values(
                                    "atoms"
                                ).unique()[0]
                                edge_list = self.data.edges[atom_type]
                                try:
                                    p = self.ax[y_id, x_id].vlines(
                                        edge_list,
                                        0,  # np.max(line_vals),
                                        max_y,
                                        color="gray",
                                        linestyles=(0, (1, 4)),
                                    )
                                except IndexError:
                                    xy_id = np.max([x_id, y_id])
                                    p = self.ax[xy_id].vlines(
                                        edge_list,
                                        0,  # np.max(line_vals),
                                        max_y,
                                        color="gray",
                                        linestyles=(0, (1, 4)),
                                    )
                            except TypeError:
                                xy_id = np.max([x_id, y_id])
                                p = self.ax.vlines(
                                    edge_list,
                                    0,  # np.max(line_vals),
                                    max_y,
                                    color="gray",
                                    linestyles=(0, (1, 4)),
                                )
                            except KeyError:
                                logger.info(
                                    f"No data for {pl}, {self.x}, {it}"
                                )

            # self.fig.supxlabel(f"{self.title_dict[x]}s", size=14)
            # self.fig.supylabel(f"{self.title_dict[y]}s", size=14)

            if plsave != False:
                logger.info("Saving plot")
                if type(plsave) == str:
                    outname = f"{plsave}.png"
                else:
                    outname = f"{self.name}.png"
                odir = Path(odir).absolute()
                logger.info(f"output to {odir.absolute()}")
                if not odir.is_dir():
                    os.makedirs(odir)
                logger.info(odir / outname)
                self.fig.savefig(str(odir / outname))
            else:
                if type(plsave) == str:
                    outname = f"{plsave}.p"
                else:
                    outname = f"{self.name}.p"
                odir = Path(odir).absolute()
                logger.info(f"output to {odir.absolute()}")
                if not odir.is_dir():
                    os.makedirs(odir)
                logger.info(odir / outname)
                with open(odir / outname, "wb") as pklfile:
                    pkl.dump(
                        {"fig": self.fig, "ax": self.ax, "data": self.data.df},
                        pklfile,
                    )
                self.fig.show()
            self.fig.clear()

        def _get_bin_label(self, x_bin, bin_list):
            if x_bin.right < np.max(bin_list.right):
                label = f"${x_bin.left} - {x_bin.right}$ \AA"  # barwidth = x_bin.right - x_bin.left

            else:
                label = f"$ > {x_bin.left}$ \AA"
            return label


class StackPlot(Plot):
    sel_list = ["clays", "ions", "aas", "_atoms"]
    title_dict = {
        "clays": "Clay type",
        "ions": "Ion type",
        "aas": "Amino acid",
        "atoms": "Atom type",
        "other": "Other atom type",
    }
    _init_args = {"x", "y", "lines"}
    _plot_args = {
        "rowlabel": "y",
        "columnlabel": "x",
        "dpi": None,
        "figsize": None,
    }

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.bins = "x"
        self.line = None

    def plot_args_decorator(select: str):
        def plot_decorator(plot_f):
            def wrapper(self, **kwargs):
                self.select = select
                plot_f(self, **kwargs)
                self.select = None

            return wrapper

        return plot_decorator

    @property
    def plot_df(self):
        try:
            return getattr(self, f"_{self.select}_bin_df")  # _{self.line}")
        except AttributeError:
            logging.error(f"Wrong usage of property, select not assigned yet.")

    @property
    def colour_df(self):
        try:
            return getattr(self, f"_{self.select}_colour_df")  # _{self.line}")
        except AttributeError:
            logging.error(f"Wrong usage of property, select not assigned yet.")

    def get_atom_colour_codes_from_names(self, atom_name):
        # atom_name = AtomHistPlot.get_atom_type_group(atom_name, group_all_atoms=True)
        if atom_name not in self.data.ions:
            atom_name = atom_name[0]
        return atom_name

    def _get_binned_plot_colour_dfs_1d(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info(f"Getting plot and colour dfs")
        from seaborn import saturate, set_hls_values

        # atom type colour cycle
        colours = [
            "#8dd3c7",  # cyan (C)
            "#ffffb3",  # yellow (S)
            "#bebada",  # blue (N)
            "#fb8072",  # red (O)
            "#80b1d3",  # light blue (Mg)
            "#fdb462",  # orange (K)
            "#b3de69",  # light green (Ca)
            "#fccde5",  # pink (Na)
        ]
        colours = [set_hls_values(saturate(cval), l=0.4) for cval in colours]
        colour_keys = ["C", "S", "N", "O", "Mg", "K", "Ca", "Na"]
        colour_dict = dict(zip(colour_keys, colours))
        # colours = color_palette('dark').as_hex()
        sel = self.data.clays

        # get data for plotting
        try:
            # clays still in columns
            plot_df = self.data.bin_df[sel].copy()

            # move clays category from columns to index
            idx_names = ["clays", *plot_df.index.droplevel(["x_bins"]).names]
            # DataFrame -> Series
            plot_df = plot_df.stack()
        except KeyError:
            plot_df = self.data.bin_df.copy()
            idx_names = plot_df.index.droplevel(["x_bins"]).names

        # get values for atom types (including separate ions)
        atoms: np.array = plot_df.index.get_level_values("_atoms").to_numpy()
        # atom_type_groups = np.unique(atoms)
        transform_atom_names_vec = np.vectorize(
            self.get_atom_colour_codes_from_names
        )
        # transform_dict = dict(zip(atom_type_groups, transform_atom_names_vec(atom_type_groups)))
        atoms = transform_atom_names_vec(atoms)
        # atom_type_groups = transform_atom_names_vec(atom_type_groups)

        # make new DataFrame from atom_type index level and values
        plot_df.index = plot_df.index.droplevel(["x_bins"])
        # plot_df.index.names = [name.strip('_') for name in plot_df.index.names]

        try:
            plot_df = pd.DataFrame({"values": plot_df, "colours": atoms})
        except ValueError:
            #     if self.select == 'ions':
            plot_df["colours"] = atoms
            plot_df["values"] = plot_df[
                self.data.analysis
            ]  # else:  #     plot_df['colours'] = atoms.transform(lambda x: x[0])

        # self._plot_df = {}
        # self._colour_df = {}

        color_cycle = itertools.cycle(colours)
        # colour_dict = {}
        for atom_type in np.unique(atoms):
            if atom_type not in colour_dict.keys():
                colour_dict[atom_type] = next(color_cycle)
        # colour_dict = dict(zip(atom_type_groups, colours[:len(atom_type_groups)]))
        plot_df["colours"] = plot_df["colours"].apply(lambda x: colour_dict[x])

        # reorder index for grouping
        plot_df = plot_df.reorder_levels(idx_names)

        # # group and sum densities within adsorption shell bins
        # plot_df = plot_df.groupby(plot_df.index.names).agg(
        #     values=pd.NamedAgg("values", "sum"), colours=pd.NamedAgg("colours", "first")
        # )

        # separate colour column from plot_df -> yields 2 Series
        colour_df = plot_df["colours"]
        plot_df = plot_df["values"]

        # add missing atom probabilities from bulk to the largest bin
        # (bin.left, cutoff] -> (bin.left, all bulk])
        # inner_sum = plot_df.groupby(plot_df.index.droplevel("x_bins").names).sum()
        # extra = 1 - inner_sum
        # plot_df.where(
        #     np.rint(plot_df.index.get_level_values("x_bins").right)
        #     != int(self.data.cutoff),
        #     lambda x: x + extra,
        #     inplace=True,
        # )
        # setattr(self, "_plot_df", plot_df.copy())
        self._plot_df, self._colour_df = plot_df.copy(), colour_df.copy()
        logger.info(
            f"{self._plot_df}\n{self._colour_df}"
        )  # print("sel list", self.sel_list)

        # for line in self.sel_list:  #  #     line_plot_df = plot_df.copy()  #     # get values for atom types (including separate ions)  #     if line == "_atoms":  #         line = "atoms"  #     lines = line_plot_df.index.get_level_values(line)  #  #     # make new DataFrame from atom_type index level and values  #     line_plot_df.index = line_plot_df.index.droplevel(["x_bins"])  #  #     # plot_df.index.names = [name.strip('_') for name in plot_df.index.names]  #  #     line_plot_df = pd.DataFrame({"values": line_plot_df, "colours": lines})  #  #     # map unique atom types to colour map  #     line_types = lines.unique()  #     colour_dict = dict(zip(line_types, colours[: len(line_types)]))  #     print(colour_dict)  #     line_plot_df["colours"] = line_plot_df["colours"].transform(  #         lambda x: colour_dict[x]  #     )  #  #     # reorder index for grouping  #     line_plot_df = line_plot_df.reorder_levels(idx_names)  #  #     # group and sum densities within adsorption shell bins  #     colour_df = line_plot_df.groupby(line_plot_df.index.names).agg(  #         colours=pd.NamedAgg("colours", "first")  #     )  #  #     # separate colour column from plot_df -> yields 2 Series  #     colour_df = colour_df["colours"]  #     # print(colour_df)  #     line_plot_df = line_plot_df["values"]  #  #     # add missing atom probabilities from bulk to the largest bin  #     # (bin.left, cutoff] -> (bin.left, all bulk])  #     # inner_sum = plot_df.groupby(plot_df.index.droplevel("x_bins").names).sum()  #     # extra = 1 - inner_sum  #     # plot_df.where(  #     #     np.rint(plot_df.index.get_level_values("x_bins").right)  #     #     != int(self.data.cutoff),  #     #     lambda x: x + extra,  #     #     inplace=True,  #     # )  #     # setattr(self, f"_plot_df_{line}", line_plot_df.copy())  #     self._plot_df[line], self._colour_df[line] = (  #         line_plot_df.copy(),  #         colour_df.copy(),  #     )

    @redirect_tqdm
    def _split_plot_dfs(self):
        logger.info(f"Splitting plot and colour dfs")
        bin_df = self._plot_df.copy()
        plot_df = self._colour_df.copy()
        dfs = {"bin": bin_df, "colour": plot_df}
        for key, df in dfs.items():
            if "ions" in self.data.atoms:
                not_ions = [a for a in self.data.atoms if a != "ions"]
                ion_view = df.xs("ions", level="atoms")
                ion_view.index = ion_view.index.droplevel(["ions"])
                ion_view.index.names = [
                    idstr.strip("_") for idstr in ion_view.index.names
                ]
                setattr(self, f"_ions_{key}_df", ion_view.copy())
            else:
                not_ions = np.unique(self.data.atoms).tolist()

            other_view = df.loc[pd.IndexSlice[:, :, :, not_ions, :]]
            other_view.index = other_view.index.droplevel("_atoms")

            setattr(
                self, f"_other_{key}_df", other_view.copy()
            )  # dfs = {}  # # print(self._plot_df.keys())  # for line in self.sel_list:  #     if line == "_atoms":  #         line = "atoms"  #     bin_df = self._plot_df[line].copy()  #     plot_df = self._colour_df[line].copy()  #     dfs[line] = {"bin": bin_df.copy(), "colour": plot_df.copy()}  #     for key, df in dfs[line].items():  #         not_ions = [a for a in self.data.atoms if a != "ions"]  #         ion_view = df.xs("ions", level="atoms")  #         ion_view.index = ion_view.index.droplevel(["ions"])  #         ion_view.index.names = [  #             idstr.strip("_") for idstr in ion_view.index.names  #         ]  #  #         other_view = df.loc[pd.IndexSlice[:, :, :, not_ions, :]]  #         other_view.index = other_view.index.droplevel("_atoms")  #         print(f"_ions_{key}_df_{line}")  #         setattr(self, f"_ions_{key}_df_{line}", ion_view.copy())  #         setattr(self, f"_other_{key}_df_{line}", other_view.copy())

    def _process_data(self):
        self._get_binned_plot_colour_dfs_1d()
        self._split_plot_dfs()

    def get_suptitle(self, pl, separate):
        self.fig.suptitle(
            (
                ", ".join([self.title_dict[s].upper() for s in separate])
                + f": {self.label_mod(list(tuple(zip(pl, separate))))}"
            ),
            size=16,
            weight="bold",
        )

    @redirect_tqdm
    def plot(
        self,
        lines: Literal["clays", "ions", "aas", "atoms", "other"],
        x: Literal["clays", "aas", "ions", "atoms", "other"],
        y: Literal["clays", "ions", "aas", "atoms", "other"],
        rowlabel: str = "y",
        columnlabel: str = "x",
        figsize=None,
        dpi=None,
        diff=False,
        xmax=50,
        ymax=50,
        plsave=False,
        xlim=None,
        ylim=None,
        odir=".",
        plot_table=None,
        edges=False,
        sparse=None,
        antialiased=True,
    ):
        # line style cycler
        linestyle_list = [
            (0, ()),
            (0, (1, 2)),
            (0, (5, 2)),
            (0, (5, 2, 1, 2)),
            (0, (5, 2, 1, 2, 1, 2)),
            (0, (1, 4)),
            (0, (3, 4, 1, 4)),
            (0, (3, 4, 1, 4, 1, 4)),
            (0, (7, 6)),
            (0, (7, 6, 3, 6)),
            (0, (7, 6, 3, 6, 1, 2)),
        ]
        if sparse is True:
            pass
        self._init_plot(x, y, lines)

        plot_df = self.plot_df.copy()
        plot_df = plot_df.groupby(plot_df.index.names).sum()
        colour_df = self.colour_df.copy()
        colour_df = colour_df.groupby(colour_df.index.names).sum()
        # linestyle_dict = {}
        line_cycle = itertools.cycle(linestyle_list)
        # line_idx = colour_df.index.droplevel('x')
        line_grouped = colour_df.groupby("atoms").first().duplicated(False)
        for at_type, duplicate in line_grouped.items():
            if duplicate is True:
                line_grouped[at_type] = next(line_cycle)
            else:
                line_cycle = itertools.cycle(linestyle_list)
                line_grouped[at_type] = linestyle_list[0]
        del at_type, duplicate, line_cycle

        # logger.info(f'{plot_df}')

        avail_attrs = plot_df.index.names
        logger.info(f"Available attributes: {avail_attrs}")

        logger.info("Setting x:")
        self.x = Axis(self, x)
        logger.info("Setting y:")
        self.y = Axis(self, y)
        logger.info("Setting lines:")
        self.line = Axis(self, lines)

        separate = [
            s
            for s in avail_attrs
            if s not in [self.x, self.y, self.line, self.bins]
        ]

        if len(separate) != 0:
            logger.info(f"Separate plots: {separate}")

        idx = pd.Index(
            [
                s
                for s in avail_attrs
                if (
                    s != self.x
                    and s != self.line
                    and s not in [*separate, self.bins]
                )
            ]
        )

        logger.info(f"Iteration index: {idx.values}")

        sep = pd.Index(separate)

        yid = np.ravel(np.where(np.array(idx) == self.y))[0]

        if figsize is None:
            logger.info(f"{xmax}, {ymax} figsize")
            figsize = self.get_figsize(xmax=xmax, ymax=ymax)

        # set resultion
        if dpi is None:
            dpi = 100

        # get plotting iter from index
        iters = self._get_idx_iter(idx=idx)

        logger.info(
            f"Printing line plots for {self.line}\nColumns: {self.x.v}\nRows: {self.y.v}"
        )

        # set label modifier function
        self.label_mod = self.modify_plot_labels

        try:
            # iterator for more than one plot
            sep_it = self._get_idx_iter(idx=sep)
        except ValueError:
            # only one plot
            sep_it = [None]
        # logger.info(f'{sep_it}\n{plot_df}')

        # iterate over separate plots
        for pl in sep_it:
            y_max = 0
            if pl is None:
                pl_str = ""
            else:
                pl_str = f"_{pl[0]}"
            self.name = f"{self.data.name}_{self.data.analysis}_{self.select}_{x}_{y}_{lines}{pl_str}_{self.data.cutoff}_{self.data.bins}"
            if xlim is not None:
                self.name += f"_{xlim:.0f}"
            logger.info(f"plot {self.name}")

            # index map for y values
            # y_dict: dict = dict(zip(vy, np.arange(ly)))

            self.init_legend(ly=self.y.l, lx=self.x.l)

            self.fig, self.ax = plt.subplots(
                nrows=self.y.l,
                ncols=self.x.l,
                figsize=figsize,
                sharey=True,
                sharex=True,
                dpi=dpi,
                constrained_layout=True,
            )

            if pl is None:
                logger.info(f"Generating plot")
                sepview = plot_df.view()
                save_str = ""

            # multiple plots
            else:
                logger.info(f"Generating {pl} plot:")
                # print(plot_df.head(20), "\n", separate, pl)
                sepview = plot_df.xs((pl), level=separate, drop_level=False)
                save_str = pl
                self.get_suptitle(pl, separate)

            # set plot index
            pi = 0

            # iterate over subplot columns
            for col in self.x.v:
                # logger.info(col)
                try:
                    # print(sepview.dropna().index)
                    view = sepview.xs(
                        col, level=self.x, axis=0, drop_level=False
                    )
                    pi = 1
                except ValueError:
                    view = sepview
                    col = self.x.v
                    pi += 1
                except KeyError:
                    continue
                for it in iters:
                    try:
                        values = view.xs(
                            tuple(it), level=idx.tolist(), drop_level=False
                        )
                        v = values.index.get_level_values("atoms").unique()
                        logger.info(v)
                        # print(values)
                        # x_labels = []

                        # logging.info(f"it: {it}")  # , {self.line.v}, {values}")
                        x_id, y_id = self.x.dict[col], self.y.dict[it[yid]]
                        # if edges is True:
                        #     atom_type = values.index.get_level_values("atoms").unique()[
                        #         0
                        #     ]
                        #     edge_list = self.data.edges[atom_type]
                        #     p = self.ax[y_id, x_id].vlines(
                        #         edge_list, 0, ylim, color="gray", linestyles=(0, (1, 4))
                        #     )
                        line_cycle = itertools.cycle(linestyle_list)
                        for line_num, vline in enumerate(self.line.v):
                            logging.info(
                                f"line {vline} ({line_num + 1}/{self.line.l})"
                            )
                            # set linestyle
                            if self.line != "atoms":
                                # line_cycle = itertools.cycle(linestyle_list)
                                # linestyle_dict = {}
                                # for atom_type in np.unique(self.data.atoms):
                                #     linestyle_dict[atom_type] = next(line_cycle)
                                linestyle = next(
                                    line_cycle
                                )  # linestyle_list[line_num % len(linestyle_list)]
                            else:
                                linestyle = line_grouped[vline]

                            try:
                                line_vals = values.xs(
                                    vline, level=self.line, drop_level=False
                                )
                            except KeyError:
                                logger.info(f"No data for {vline}")
                                continue

                            colour = colour_df.loc[line_vals.index].values[0]
                            # if sparse is True:
                            #     line_vals = line_vals.rolling(10, min_periods=1, step=10).sum()
                            bin_list = line_vals.index.get_level_values("x")
                            line_vals.index = bin_list
                            # logger.info(f'{vline}, {self.line}')
                            # print(line_vals.hasnans, np.max(line_vals.values), vline, line_num, y_id, x_id)
                            summed = np.round(np.sum(line_vals.values), 4)
                            if (
                                not line_vals.hasnans and summed > 0.0
                            ):  # and vline != 'C':
                                if np.max(line_vals) > y_max:
                                    y_max = np.max(line_vals)
                                    logger.info(
                                        f"adjusting max value: {y_max}"
                                    )
                                try:
                                    label = f"{self.label_mod([(vline, self.line)]):<2}"
                                    try:
                                        if (
                                            self.data.ignore_density_sum
                                            is True
                                        ):
                                            logger.debug(
                                                f"Ignoring line value sum check sum={summed:.2f}"
                                            )
                                            if summed != 1.0:
                                                logger.info(
                                                    f"labelling sum {summed}"
                                                )  # label_add = rf" ({summed*100:>.0f} \%)"  # label = f"{label:<3}{label_add:>7}"
                                        else:
                                            assert (
                                                np.round(summed) == 1.00
                                            ), f"Densities ({np.round(summed)}) do not sum to 1.00"
                                    except AttributeError:
                                        assert (
                                            np.round(summed) == 1.00
                                        ), f"Densities ({np.round(summed)}) do not sum to 1.00"
                                    if pi == 1:
                                        self.legends[(y_id, x_id)].append(
                                            label
                                        )
                                    # if sparse is True:
                                    #     line_vals_mask = line_vals.values != 0
                                    # spline = splrep(bin_list,#[line_vals_mask],
                                    #                               line_vals.values)#[line_vals_mask])
                                    # line_vals = splev(bin_list, spline)
                                    # else:
                                    #     line_vals = line_vals.values
                                    # print(x_id)
                                    try:
                                        # f, ax = plt.subplots()
                                        # ax.plot(bin_list, line_vals)
                                        # f.suptitle(vline)
                                        # f.show()
                                        logger.info(
                                            f"{len(bin_list)}, {len(line_vals)}"
                                        )
                                        # p = self.ax[y_id, x_id].plot([1,1], [0,1], label='a')
                                        p = self.ax[y_id, x_id].plot(
                                            bin_list,
                                            line_vals,
                                            label=label,
                                            color=colour,
                                            linestyle=linestyle,
                                            antialiased=antialiased,
                                        )  # self.fig.show()  # self.fig.show()  # self.fig.savefig('a.png')  # logger.info('wrote file')  # logger.info(f'{pl} {col} {vline} {it[yid]}')
                                    except Exception as e:
                                        # del p
                                        print(e, y_id, x_id, line_num, vline)

                                    # self.ax[y_id, x_id].bar_label(p, labels=[label],  #                          fmt='%s',  #                          label_type='center')
                                except IndexError:
                                    p = self.ax[x_id].plot(
                                        bin_list,
                                        line_vals.values,
                                        label=label,
                                        color=colour,
                                        linestyle=linestyle,
                                        antialiased=antialiased,
                                    )
                            else:
                                logger.info(f"nan or 0: {vline}")
                        self.ax[y_id, x_id].legend()
                    # except FileNotFoundError:
                    except KeyError:
                        logger.info(
                            f"No data for {pl}, {self.x}, {it} {it[yid]}"
                        )

            for i in range(self.y.l):
                try:
                    self.ax[i, 0].set_ylabel(
                        f"{self.label_mod([(self.y.v[i], self.y)])}\n"
                        + rowlabel
                    )
                    for j in range(self.x.l):
                        label = [
                            self.label_mod([(leg, self.line)])
                            for leg in self.legends[i, j]
                        ]
                        if len(label) % 3 == 0:
                            ncol = 3
                        else:
                            ncol = 2
                        self.ax[i, j].legend(label, ncol=ncol, title=lines)
                        if xlim is not None:
                            self.ax[i, j].set_xlim((0.0, float(xlim)))
                        if ylim is not None:
                            self.ax[i, j].set_ylim((0.0, float(ylim)))
                        else:
                            self.ax[i, j].set_ylim((0.0, float(y_max)))
                        self.ax[self.y.l - 1, j].set_xlabel(
                            columnlabel
                            + f"\n{self.label_mod([(self.x.v[j], self.x)])}"
                        )

                        self.ax[i, j].spines[["top", "right"]].set_visible(
                            False
                        )
                    self.fig.supxlabel(f"{self.title_dict[x]}s", size=14)
                    self.fig.supylabel(f"{self.title_dict[y]}s", size=14)
                except IndexError:
                    self.ax[i].set_ylabel(
                        f"{self.label_mod([(self.y.v[i], y)])}\n" + rowlabel
                    )
                    # self.ax[i].legend([self.label_mod([(leg, self.label_key)]) for leg in self.legends[i, 0]], ncol=3)
                    self.ax[self.y.l - 1].set_xlabel(
                        columnlabel
                        + f"\n{self.label_mod([(self.x.v[0], self.x)])}"
                    )
            if edges is True:
                for col in self.x.v:
                    # logger.info(col)
                    try:
                        view = sepview.xs(
                            col, level=self.x, axis=0, drop_level=False
                        )
                        pi = 1
                    except ValueError:
                        view = sepview
                        col = self.x.v
                        pi += 1
                    for it in iters:
                        try:
                            values = view.xs(
                                tuple(it), level=idx.tolist(), drop_level=False
                            )
                            # print(values)
                            # x_labels = []

                            logging.info(
                                f"it: {it}"
                            )  # , {self.line.v}, {values}")
                            x_id, y_id = self.x.dict[col], self.y.dict[it[yid]]

                            atom_type = values.index.get_level_values(
                                "atoms"
                            ).unique()[0]
                            edge_list = self.data.edges[atom_type]
                            p = self.ax[y_id, x_id].vlines(
                                edge_list,
                                0,
                                np.max(line_vals),
                                color="gray",
                                linestyles=(0, (1, 4)),
                            )
                        except KeyError:
                            logger.info(f"No data for {pl}, {self.x}, {it}")

            # self.fig.supxlabel(f"{self.title_dict[x]}s", size=14)
            # self.fig.supylabel(f"{self.title_dict[y]}s", size=14)

            if plsave != False:
                logger.info("Saving plot")
                if type(plsave) == str:
                    outname = f"{plsave}.png"
                else:
                    outname = f"{self.name}.png"
                odir = Path(odir).absolute()
                logger.info(f"output to {odir.absolute()}")
                if not odir.is_dir():
                    os.makedirs(odir)
                logger.info(odir / outname)
                self.fig.savefig(str(odir / outname))
            else:
                plt.show()
            self.fig.clear()

        def _get_bin_label(self, x_bin, bin_list):
            if x_bin.right < np.max(bin_list.right):
                label = f"${x_bin.left} - {x_bin.right}$ \AA"  # barwidth = x_bin.right - x_bin.left

            else:
                label = f"$ > {x_bin.left}$ \AA"
            return label

    # @property
    # def plot_df(self):
    #     try:
    #         return getattr(self, f"_{self.select}_bin_df")
    #     except AttributeError:
    #         logging.error(f"Wrong usage of property, line not assigned yet.")
    #
    # @property
    # def colour_df(self):
    #     try:
    #         return getattr(self, f"_{self.select}_colour_df")
    #     except AttributeError:
    #         logging.error(f"Wrong usage of property, line not assigned yet.")

    def _get_attr(self, attr: Literal["clays", "aas", "atoms", "ions"]):
        return pd.unique(self.plot_df.index.get_level_values(attr))

    get_clays = partialmethod(_get_attr, attr="clays")
    get_aas = partialmethod(_get_attr, attr="aas")
    get_ions = partialmethod(_get_attr, attr="ions")
    get_atoms = partialmethod(_get_attr, attr="atoms")
    get_other = partialmethod(_get_attr, attr="other")

    def save(self, **kwargs):
        ...

    @plot_args_decorator(select="ions")
    def plot_ions(
        self,
        lines: Literal["clays", "ions", "aas", "atoms", "other"],
        x: Literal["clays", "aas", "ions", "atoms", "other"],
        y: Literal["clays", "ions", "aas", "atoms", "other"],
        rowlabel: str = "y",
        columnlabel: str = "x",
        figsize=None,
        dpi=None,
        diff=False,
        xmax=50,
        ymax=50,
        plsave=False,
        xlim=None,
        ylim=None,
        odir=".",
        plot_table=None,
        edges=True,
        sparse=None,
        antialiased=True,
    ):
        self.plot(
            lines,
            x,
            y,
            rowlabel,
            columnlabel,
            figsize,
            dpi,
            diff,
            xmax,
            ymax,
            plsave,
            xlim,
            ylim,
            odir,
            plot_table,
            edges,
            sparse,
            antialiased,
        )

    @plot_args_decorator(select="other")
    def plot_other(
        self,
        lines: Literal["clays", "ions", "aas", "atoms", "other"],
        x: Literal["clays", "aas", "ions", "atoms", "other"],
        y: Literal["clays", "ions", "aas", "atoms", "other"],
        rowlabel: str = "y",
        columnlabel: str = "x",
        figsize=None,
        dpi=None,
        diff=False,
        xmax=50,
        ymax=50,
        plsave=False,
        xlim=None,
        ylim=None,
        odir=".",
        plot_table=None,
        edges=True,
        sparse=None,
        antialiased=True,
    ):
        self.plot(
            lines,
            x,
            y,
            rowlabel,
            columnlabel,
            figsize,
            dpi,
            diff,
            xmax,
            ymax,
            plsave,
            xlim,
            ylim,
            odir,
            plot_table,
            edges,
            sparse,
            antialiased,
        )


# https://colorbrewer2.org/?type=qualitative&scheme=Set3&n=5


class HistPlot(Plot):
    # __slots__ = ()
    sel_list = ["clays", "ions", "aas", "_atoms"]
    title_dict = {
        "clays": "Clay type",
        "ions": "Ion type",
        "aas": "Amino acid",
        "atoms": "Atom type",
        "other": "Other atom type",
    }
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
    _init_args = {"x", "y", "bars", "add_missing_bulk"}
    _plot_args = {
        "rowlabel": "y",
        "columnlabel": "x",
        "dpi": None,
        "figsize": None,
    }

    def __init__(self, data, add_missing_bulk=True, **kwargs):
        kwargs["add_missing_bulk"] = add_missing_bulk
        super().__init__(data, **kwargs)
        self.bins = "x_bins"
        self.group = "atoms"
        self.bars = None

    def plot_args_decorator(select: str):
        def plot_decorator(plot_f):
            def wrapper(self, **kwargs):
                self.select = select
                plot_f(self, **kwargs)
                self.select = None

            return wrapper

        return plot_decorator

    def _get_binned_plot_colour_dfs_1d(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info(f"Getting binned plot and colour dfs")

        # list of matplotlib sequential cmaps
        cmaps = [  # 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
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

        sel = self.data.clays

        # get data for plotting
        plot_df = self.data.bin_df[sel].copy()

        # move clays category from columns to index
        idx_names = ["clays", *plot_df.index.droplevel(["x"]).names]
        # DataFrame -> Series
        plot_df = plot_df.stack()

        # get values for atom types (including separate ions)
        atoms = plot_df.index.get_level_values("_atoms")

        # make new DataFrame from atom_type index level and values
        plot_df.index = plot_df.index.droplevel(["x"])
        # plot_df.index.names = [name.strip('_') for name in plot_df.index.names]

        plot_df = pd.DataFrame({"values": plot_df, "colours": atoms})

        # map unique atom types to colour map
        atom_types = atoms.unique()
        colour_dict = dict(zip(atom_types, cmaps[: len(atom_types)]))
        plot_df["colours"] = plot_df["colours"].transform(
            lambda x: colour_dict[x]
        )

        # reorder index for grouping
        plot_df = plot_df.reorder_levels(idx_names)

        plot_df = self._get_peak_area_df(plot_df)

        # separate colour column from plot_df -> yields 2 Series
        colour_df = plot_df["colours"]
        plot_df = plot_df["values"]

        # add missing atom probabilities from bulk to the largest bin
        # (bin.left, cutoff] -> (bin.left, all bulk])
        if self.add_missing_bulk is True:
            inner_sum = plot_df.groupby(
                plot_df.index.droplevel("x_bins").names
            ).sum()
            extra = 1 - inner_sum
            # print(
            #     extra,
            #     self.data.cutoff.num,
            #     np.rint(plot_df.index.get_level_values("x_bins").right),
            # )
            plot_df.where(
                np.rint(plot_df.index.get_level_values("x_bins").right)
                != self.data.cutoff.num,
                lambda x: x + extra,
                inplace=True,
            )
        # setattr(self, "_plot_df", plot_df.copy())
        self._plot_df, self._colour_df = plot_df.copy(), colour_df.copy()

    def _get_peak_area_df(self, plot_df):
        # group and sum densities within adsorption shell bins
        plot_df = plot_df.groupby(plot_df.index.names).agg(
            values=pd.NamedAgg("values", "sum"),
            colours=pd.NamedAgg("colours", "first"),
        )
        return plot_df

    def _split_plot_dfs(self):
        logger.info(f"Splitting binned plot and colour dfs")
        bin_df = self._plot_df.copy()
        plot_df = self._colour_df.copy()
        dfs = {"bin": bin_df, "colour": plot_df}
        for key, df in dfs.items():
            if "ions" in self.data.atoms:
                not_ions = [a for a in self.data.atoms if a != "ions"]
                ion_view = df.xs("ions", level="atoms")
                ion_view.index = ion_view.index.droplevel(["ions"])
                ion_view.index.names = [
                    idstr.strip("_") for idstr in ion_view.index.names
                ]
                setattr(self, f"_ions_{key}_df", ion_view.copy())
            else:
                not_ions = np.unique(self.data.atoms).tolist()
            # not_ions = [a for a in self.data.atoms if a != "ions"]
            # ion_view = df.xs("ions", level="atoms")
            # ion_view.index = ion_view.index.droplevel(["ions"])
            # ion_view.index.names = [idstr.strip("_") for idstr in ion_view.index.names]
            # logger.info(f'\n{not_ions}, {df.index.names}')
            other_view = df.loc[pd.IndexSlice[:, :, :, :, not_ions, :]]
            other_view.index = other_view.index.droplevel("_atoms")
            # logger.info(f'{other_view.index.get_level_values("atoms").unique()}, {other_view.index.names}')
            # setattr(self, f"_ions_{key}_df", ion_view.copy())
            setattr(self, f"_other_{key}_df", other_view.copy())

    def _process_data(self, **kwargs):
        self._get_binned_plot_colour_dfs_1d()
        self._split_plot_dfs()

    @property
    def plot_df(self):
        try:
            plot_df = getattr(self, f"_{self.select}_bin_df")
            # logger.info(f'{plot_df.index.get_level_values("atoms").unique()}')
            return plot_df
        except AttributeError:
            logging.error(f"Wrong usage of property, select not assigned yet.")

    @property
    def colour_df(self):
        try:
            return getattr(self, f"_{self.select}_colour_df")
        except AttributeError:
            logging.error(f"Wrong usage of property, select not assigned yet.")

    def save(self, **kwargs):
        ...

    @cached_property
    def cnorm(self) -> mpc.Normalize:
        # normalise colour map from 0 to max_edge
        return mpc.Normalize(vmin=0, vmax=self.data.max_shell_edge, clip=False)

    def _get_attr(self, attr: Literal["clays", "aas", "atoms", "ions"]):
        return pd.unique(self.plot_df.index.get_level_values(attr))

    get_clays = partialmethod(_get_attr, attr="clays")
    get_aas = partialmethod(_get_attr, attr="aas")
    get_ions = partialmethod(_get_attr, attr="ions")
    get_atoms = partialmethod(_get_attr, attr="atoms")
    get_other = partialmethod(_get_attr, attr="other")

    @plot_args_decorator(select="ions")
    def plot_ions(
        self,
        bars: Literal["clays", "aas", "ions", "other"],
        x: Literal["clays", "aas", "ions", "other"],
        y: Literal["clays", "aas", "ions", "other"],
        rowlabel: str = "y",
        columnlabel: str = "x",
        figsize=None,
        dpi=None,
        xmax=50,
        ymax=50,
        plsave=False,
        ylim=None,
        odir=".",
        barwidth=0.75,
        xpad=0.25,
        cmap="winter",
        plot_table=False,
    ):
        self.plot(
            bars,
            x,
            y,
            rowlabel,
            columnlabel,
            figsize,
            dpi,
            xmax,
            ymax,
            plsave,
            ylim,
            odir,
            barwidth,
            xpad,
            cmap,
            plot_table=plot_table,
        )

    @plot_args_decorator(select="other")
    def plot_other(
        self,
        bars: Literal["clays", "aas", "ions", "other"],
        x: Literal["clays", "aas", "ions", "other"],
        y: Literal["clays", "aas", "ions", "other"],
        rowlabel: str = "y",
        columnlabel: str = "x",
        figsize=None,
        dpi=None,
        xmax=50,
        ymax=50,
        plsave=False,
        ylim=None,
        odir=".",
        barwidth=0.75,
        xpad=0.25,
        cmap="winter",
        plot_table=None,
    ):
        self.plot(
            bars,
            x,
            y,
            rowlabel,
            columnlabel,
            figsize,
            dpi,
            xmax,
            ymax,
            plsave,
            ylim,
            odir,
            barwidth,
            xpad,
            cmap,
            plot_table=plot_table,
        )

    def get_suptitle(self, pl, separate):
        self.fig.suptitle(
            (
                ", ".join([self.title_dict[s].upper() for s in separate])
                + f": {self.label_mod(list(tuple(zip(pl, separate))))}"
            ),
            size=16,
            weight="bold",
        )

    def plot(
        self,
        bars: Literal["clays", "aas", "ions", "other"],
        x: Literal["clays", "aas", "ions", "other"],
        y: Literal["clays", "aas", "ions", "other"],
        rowlabel: str = "y",
        columnlabel: str = "x",
        figsize=None,
        dpi=None,  # diff=False,
        xmax=50,
        ymax=50,
        plsave=True,
        ylim=None,
        odir=".",
        barwidth=0.75,
        xpad=0.25,
        cmap="winter",
        tab_fontsize=12,
        plot_table=False,
    ):
        """Create stacked Histogram adsorption shell populations."""

        logger.info(f"plot table {plot_table}")

        self._init_plot(x, y, bars)

        plot_df = self.plot_df.copy()
        colour_df = self.colour_df.copy()

        avail_attrs = plot_df.index.names
        logger.info(f"Available attributes: {avail_attrs}")

        logger.info("Setting x:")
        self.x = Axis(self, x)
        logger.info("Setting y:")
        self.y = Axis(self, y)
        logger.info("Setting bars:")
        self.bars = Axis(self, bars)

        # print(avail_attrs)
        separate = [
            s
            for s in avail_attrs
            if s not in [self.x, self.y, self.bars, self.bins]
        ]

        if len(separate) != 0:
            logger.info(f"Separate plots: {separate}")

        idx = pd.Index(
            [
                s
                for s in avail_attrs
                if (
                    s != self.x
                    and s != self.bars
                    and s not in [*separate, self.bins]
                )
            ]
        )

        logger.info(f"Iteration index: {idx.values}")

        sep = pd.Index(separate)
        # logger.info(f'{sep}')

        yid = np.ravel(np.where(np.array(idx) == self.y))[0]

        if figsize is None:
            if plot_table is True:
                figsize = self.get_figsize(xmax=xmax, ymax=ymax * 2)
            else:
                figsize = self.get_figsize(xmax=xmax, ymax=ymax)

        # set resultion
        if dpi is None:
            dpi = 100

        # get plotting iter from index
        iters = self._get_idx_iter(idx=idx)

        logger.info(
            f"Printing bar plots for {self.bars}\nColumns: {self.x.v}\nRows: {self.y.v}"
        )

        # set label modifier function
        self.label_mod = self.modify_plot_labels

        try:
            # iterator for more than one plot
            sep_it = self._get_idx_iter(idx=sep)  # logger.info(f'{sep_it}')
        except ValueError:
            # only one plot
            sep_it = [None]

        # iterate over separate plots
        for pl in sep_it:
            if pl is None:
                pl_str = ""
            else:
                pl_str = f"_{pl[0]}"
            self.name = f"{self.data.name}_{self.data.analysis}_{self.select}_{x}_{y}_{bars}{pl_str}_{self.data.cutoff}_{self.data.bins}"
            if plot_table is True:
                self.name += "_table"
            logger.info(f"plot {self.name}")
            # index map for y values
            # y_dict: dict = dict(zip(vy, np.arange(ly)))

            self.init_legend(ly=self.y.l, lx=self.x.l)

            # if plot_table is True:
            #     plt_nrows = self.y.l * 2
            #     logger.info('Has table')
            # else:
            #     plt_nrows = self.y.l
            # print('rows ', plt_nrows)

            # generate figure and axes array
            if plot_table is True:
                self.fig = plt.figure(figsize=figsize, dpi=dpi)
            else:
                self.fig = plt.figure(
                    figsize=figsize, dpi=dpi, constrained_layout=True
                )

            # xmul = 2

            self.gs = gridspec.GridSpec(
                nrows=self.y.l,
                ncols=self.x.l,
                figure=self.fig,  # width_ratios=[2,1]
                # sharey=True,
                # sharex=True
            )

            self.ax = {}

            if plot_table is True:
                self.gs = gridspec.GridSpec(
                    nrows=self.y.l,
                    ncols=self.x.l,
                    figure=self.fig,
                    wspace=2 / self.x.l,
                    hspace=0.5,  # width_ratios=
                    # sharey=True,
                    # sharex=True
                )
                self.subax = {}
                # self.tabax = {}
                for i in range(self.y.l):
                    for j in range(self.x.l):
                        #     if i == 0 and j == 0:
                        #         self.subax[i, j] = gridspec.GridSpecFromSubplotSpec(
                        #     2, 1, subplot_spec=self.gs[i, j]
                        # )
                        #     else:
                        self.subax[i, j] = self.gs[i, j].subgridspec(
                            10, self.x.l, wspace=1.5, hspace=1.5
                        )
                        self.ax[i, j] = self.fig.add_subplot(
                            self.subax[i, j][:8, :]
                        )

                        # if i == (self.y.l - 1) or j == 0:

                        # else:  #     self.ax[i, j] = self.fig.add_subplot(self.subax[i, j][:8, :])#,  #                                          # sharex=self.ax[0, i],  #                                          # sharey=self.ax[j, (self.y.l - 1)])  # if i != (self.y.l - 1):  #     self.ax[i, j].tick_params(labelbottom=False)  # if j != 0:  #     self.ax[i, j].tick_params(labelleft=False)

                        # self.tabax[i, j] = self.fig.add_subplot(self.subax[i, j][1])  # self.tabax[i, j].set

            else:
                for i in range(self.y.l):
                    for j in range(self.x.l):
                        self.ax[i, j] = self.fig.add_subplot(self.gs[i, j])

            # only one plot
            if pl is None:
                logger.info(f"Generating plot")
                sepview = plot_df.view()
                plsave = ""

            # multiple plots
            else:
                logger.info(f"Generating {pl} plot:")
                # print(plot_df.head(20), "\n", separate, pl)
                sepview = plot_df.xs((pl), level=separate, drop_level=False)
                plsave = pl
                self.get_suptitle(pl, separate)

            # set plot index
            pi = 0

            # iterate over subplot columns
            for col in self.x.v:
                logger.info(f"Column {col}")
                try:
                    view = sepview.xs(
                        col, level=self.x, axis=0, drop_level=False
                    )
                    pi = 1
                except ValueError:
                    view = sepview
                    col = self.x.v
                    pi += 1

                for it in iters:
                    try:
                        values = view.xs(
                            tuple(it), level=idx.tolist(), drop_level=False
                        )

                        x_labels = []
                        x_ticks = []
                        for bar_num, vbar in enumerate(self.bars.v):
                            tab_colours = []
                            tab_rows = []
                            if plot_table is True and bar_num == 0:
                                # tab_colours = []
                                # tab_rows = []
                                try:
                                    tab_text = values.values.reshape(
                                        (self.bars.l, -1)
                                    ).T
                                except ValueError:
                                    tab_text = values.values
                                # tab_text = tab_text.astype(str)
                                tab_format = lambda x: rf"${x * 100:3.1f} \%$"
                                tab_format_vec = np.vectorize(tab_format)
                                tab_text = tab_format_vec(tab_text)
                                tab_cols = list(
                                    self.bars.v[
                                        : np.atleast_2d(tab_text).shape[0]
                                    ]
                                )

                            x_ticks.append(
                                self.bars.dict[vbar] * (barwidth + xpad)
                            )
                            x_labels.append(vbar)
                            bottom = 0.0
                            bar_vals = values.xs(
                                vbar, level=self.bars, drop_level=False
                            )
                            # tab_text.append(bar_vals.values)

                            cmap = colormaps[
                                colour_df.loc[bar_vals.index].values[0]
                            ]

                            peaks = self.data.get_bar_peaks(
                                atom_type=col, other=None
                            )
                            if np.all(bar_vals.values) >= 0:
                                # table_text.append([f'${v * 100:3.1f} %$' for v in bar_vals.values])
                                x_id, y_id = (
                                    self.x.dict[col],
                                    self.y.dict[it[yid]],
                                )
                                # if plot_table is True:
                                #     # logger.info(y_id)
                                #     logger.info("Has table")
                                #     y_id *= 2
                                #     logger.info(y_id)
                                # print(y_id)

                                bar_val_view = bar_vals
                                bin_list = bar_val_view.index.get_level_values(
                                    "x_bins"
                                )
                                bar_val_view.index = bin_list

                                x_tick = x_ticks[-1]

                                for bar_id, bar_val in enumerate(
                                    bar_val_view.items()
                                ):
                                    x_bin, y_val = bar_val

                                    try:
                                        peak = peaks[bar_id]
                                    except IndexError:
                                        peak = x_bin.right
                                    colour = cmap(self.cnorm(peak))
                                    # if colour not in table_col and cmap == table_cmap:
                                    #     print('colour', colour)
                                    #     table_col.append(colour)

                                    label = self._get_bin_label(
                                        x_bin, bin_list
                                    )

                                    if bar_num == 0:
                                        tab_colours.append(list(colour))
                                        tab_rows.append(
                                            label
                                        )  # print(tab_colours)

                                    # if x_bin.right < np.max(self.x):
                                    #     label = f'${x_bin.left} - {x_bin.right}$ \AA'
                                    # else:
                                    #     label = f'$ > {x_bin.left}$ \AA'
                                    # if label not in table_rows and cmap == table_cmap:
                                    #     table_rows.append(label)
                                    if y_val >= 0.001:
                                        # barwidth = bulk_edge - x_bin.left
                                        # try:
                                        # x_tick = x_ticks[-1] + barwidth
                                        # x_ticks.append(x_tick)
                                        # except IndexError:
                                        #     x_tick = x_bin.left
                                        try:
                                            p = self.ax[y_id, x_id].bar(
                                                x_tick,
                                                y_val,
                                                label=label,
                                                bottom=bottom,
                                                width=barwidth,
                                                align="edge",
                                                color=colour,
                                            )  # self.ax[y_id, x_id].bar_label(p, labels=[label],  #                          fmt='%s',  #                          label_type='center')
                                        except IndexError:
                                            p = self.ax[y_id].bar(
                                                x_tick,
                                                y_val,
                                                label=label,
                                                bottom=bottom,
                                                width=barwidth,
                                                align="edge",
                                                color=colour,
                                            )  # self.ax[y_id, x_id].bar_label(p, labels=[label],  #                          fmt='%s',  #                          label_type='center')
                                        # finally:
                                        bottom += y_val
                                # try:
                                #     self.fig.subplots_adjust(left=0.2, bottom=0.2)
                                # except IndexError:
                                #     self.ax[y_id].subplots_adjust(left=0.2, bottom=0.2)
                                if plot_table is True:
                                    # y_id += 1
                                    # print(tab_colours)
                                    logger.info(f"Has table, {y_id}")
                                    has_table = False
                                    try:
                                        # print("rows", tab_rows)
                                        tab_colours = np.array(tab_colours)
                                        print("colours", tab_colours)
                                        # self.tabax[y_id, x_id].clear()
                                        the_table: table = table(
                                            self.ax[y_id, x_id],
                                            cellText=tab_text,
                                            rowLabels=tab_rows,
                                            rowColours=tab_colours,
                                            colLabels=tab_cols,
                                            loc="bottom",
                                            edges="horizontal",
                                        )
                                        has_table = True
                                    except IndexError:
                                        try:
                                            # self.tabax[y_id].clear()
                                            the_table: table = table(
                                                self.ax[y_id],
                                                cellText=tab_text,
                                                rowLabels=tab_rows,
                                                rowColours=tab_colours,
                                                colLabels=tab_cols,
                                                loc="bottom",
                                                edges="horizontal",
                                            )
                                            has_table = True
                                        except ValueError:
                                            logger.info("could not plot table")
                                    except ValueError:
                                        logger.info("could not plot table")
                                    if has_table is True:
                                        logger.info("Plotting table")
                                        # renderer = self.fig.canvas.get_renderer()
                                        the_table.auto_set_font_size(False)
                                        the_table.set_fontsize(12)
                                        h = the_table.get_celld()[
                                            (0, 0)
                                        ].get_height()
                                        w = the_table.get_celld()[
                                            (0, 0)
                                        ].get_width()
                                        table_cells = the_table.properties()[
                                            "children"
                                        ]
                                        for cell in table_cells:
                                            max_width = w
                                            if cell.get_width() > max_width:
                                                max_width = cell.get_width()
                                            # print(cell.get_y())
                                            # cell.set_y(5)
                                            # print(cell.get_y())
                                            cell.set_width(barwidth + xpad)
                                            cell.set_height(0.13)
                                            # cell.set_fontsize(12)
                                            cell.set_text_props(
                                                verticalalignment="center"
                                            )
                                        for cell in table_cells:
                                            cell.set_width(
                                                max_width
                                            )  # the_table.set_fontsize(tab_fontsize)  # has_table = True  # the_table.scale(1, 1.2)  # the_table.stale = True  # the_table.set_clip_on(False)  # the_table._update_positions(renderer)  # bbox = the_table.get_window_extent(renderer)  # the_table.set(clip_bbox=bbox)  # self.ax[y_id, x_id].stale = True  # renderer = self.fig.canvas.get_renderer()  #  #  # # text_height = the_table._approx_text_height()  #  # # the_table.auto_set_font_size(False)  # # new_font_size = the_table.FONTSIZE  # # the_table.FONTSIZE = tab_fontsize  # the_table.set_fontsize(tab_fontsize)  #  # cell = the_table[0, 0]  # # cell_height = cell.get_text()  # win_ext = the_table.get_window_extent(renderer=renderer)  # text_height = cell.get_text_bounds(renderer=renderer)[-1]  # tab_font = the_table.FONTSIZE  # text_height = cell.get_text_bounds()  # # the_table._do_  # tab_font_new = tab_fontsize  # # sys.exit(1)  # # text_height = cells.get_text_bounds()  #  # font_ratio = tab_fontsize / the_table.FONTSIZE  #  # # # the_table.set_fontsize(tab_font_new)  # # # tab_dict = the_table.get_celld()  # #  # # # for col_id in range(0, len(tab_cols)):  # # #     tab_dict[(0, col_id)].set_height(text_height + 0.1)  # # #     for row_id in range(1, len(tab_rows)):  # # #         tab_dict[(row_id, col_id)].set_height(text_height + 0.1)  # the_table.scale(1, font_ratio + 0.3)  # the_table._stale = True  # the_table[...].set_text_props(linespacing=1.5,  #                          verticalalignment='bottom')  # the_table.scale(1, 1.2)  # the_table.stale = True  # the_table.set_clip_on(False)  # the_table._update_positions(renderer)

                                    else:
                                        logger.info(
                                            "No table"
                                        )  # else:  # print(y_val)
                    # print("table false")
                    # sys.exit(1)
                    # tab_dict = the_table.get_celld()
                    # table = ax[y_id, x_id].table(cellText=table_text,
                    #                              rowColours=table_col,
                    #                              rowLabels=table_rows,
                    #                              # colLables=...,
                    #                              loc='bottom')
                    # x_ticks = x_ticks[:-1]
                    # x_ticks.append(bar_dict[vbar] * (barwidth + xpad))
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
                        logger.info(f"No data for {pl}, {self.x}, {it}")

            # x_ticks = [np.linspace(n_bar * bulk_edge + xpad,
            #                        n_bar * bulk_edge + bulk_edge, int(bulk_edge)) for n_bar in range(lbars)]
            # x_ticks = np.ravel(x_ticks)
            # x_labels = np.tile(np.arange(0, bulk_edge, 1), lbars)
            for i in range(self.y.l):
                # if plot_table is True:
                #     ax_multi = 2
                # else:
                #     ax_multi = 1
                # ax_i = i * ax_multi
                # print(f"Axis index: {ax_i}, multi: {ax_multi}")

                try:
                    # self.ax[]
                    # self.ax[i, 0].set_ylabel(
                    #     f"{self.label_mod([(self.y.v[i], self.y)])}\n" + rowlabel
                    # )
                    for j in range(self.x.l):
                        # if plot_table is True:
                        #     self.ax[i, j].subplots_adjust(bottom=0.2)
                        #     f"{self.label_mod([(self.y.v[i], self.y)])}\n" + rowlabel
                        # )
                        if j == 0:
                            self.ax[i, 0].set_ylabel(
                                f"{self.label_mod([(self.y.v[i], self.y)])}\n"
                                + rowlabel
                            )
                            self.ax[i, j].set_yticks(np.arange(0.0, 1.1, 0.2))
                        else:
                            self.ax[i, j].set_yticks(np.arange(0.0, 1.1, 0.2))
                            self.ax[i, j].set_yticklabels(
                                []
                            )  # np.arange(0.0, 1.1, 0.2))
                        self.ax[i, j].spines[["top", "right"]].set_visible(
                            False
                        )
                        self.ax[i, j].hlines(
                            1.0,
                            -xpad,
                            self.bars.l * (barwidth + xpad) + xpad,
                            linestyle="--",
                        )
                        # self.ax[i, j].legend(ncol=2, loc='lower center')#[leg for leg in legends[i, j]], ncol=3)
                        #                  if xlim != None:
                        self.ax[i, j].set_xlim(
                            (-xpad, self.bars.l * (barwidth + xpad))
                        )
                        self.ax[i, j].set_xticks([], [])
                        #                 if ylim != None:
                        self.ax[i, j].set_ylim((0.0, 1.2))
                        self.ax[self.y.l - 1, j].set_xticks(
                            np.array(x_ticks) + 0.5 * barwidth, x_labels
                        )
                        self.ax[self.y.l - 1, j].set_xlabel(
                            bars
                            + f"\n{self.label_mod([(self.x.v[j], self.x)])}"
                        )  # self.ax[i, j].set_yticklabels(np.arange(0.0, 1.1, 0.2))
                except IndexError:
                    self.ax[i].set_ylabel(
                        f"{self.label_mod([(self.y.v[i], y)])}\n" + rowlabel
                    )
                    # self.ax[i].legend([self.label_mod([(leg, self.label_key)]) for leg in self.legends[i, 0]], ncol=3)
                    self.ax[self.y.l - 1].set_xlabel(
                        columnlabel
                        + f"\n{self.label_mod([(self.x.v[0], self.x)])}"
                    )
            # # #
            self.fig.supxlabel(f"{self.title_dict[x]}s", size=14)
            self.fig.supylabel(f"{self.title_dict[y]}s", size=14)
            # if plsave is True:
            logger.info("Saving plot")
            if type(plsave) == str and plsave != "":
                outname = f"{plsave}.png"
            else:
                outname = f"{self.name}.png"
            odir = Path(odir).absolute()
            logger.info(f"output to {odir.absolute()}")
            if not odir.is_dir():
                os.makedirs(odir)
            logger.info(odir)
            logger.info(outname)
            self.fig.savefig(str(odir / outname))
            # else:
            #     plt.show()
            self.fig.clear()

    def _get_bin_label(self, x_bin, bin_list):
        if x_bin.right < np.max(bin_list.right):
            label = f"${x_bin.left:>3.1f} - {x_bin.right:3.1f}$ \AA"  # barwidth = x_bin.right - x_bin.left

        else:
            label = f"$ > {x_bin.left:>3.1f}$ \AA"
        return label


class GaussHistPlot(Plot):
    # __slots__ = ()
    sel_list = ["clays", "ions", "aas", "_atoms"]
    title_dict = {
        "clays": "Clay type",
        "ions": "Ion type",
        "aas": "Amino acid",
        "atoms": "Atom type",
        "other": "Other atom type",
    }
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
    _init_args = {"x", "y", "bars", "add_missing_bulk"}
    _plot_args = {
        "rowlabel": "y",
        "columnlabel": "x",
        "dpi": None,
        "figsize": None,
    }

    def __init__(self, data, add_missing_bulk=True, **kwargs):
        kwargs["add_missing_bulk"] = add_missing_bulk
        super().__init__(data, **kwargs)
        self.bins = "x_bins"
        self.group = "atoms"
        self.bars = None

    def plot_args_decorator(select: str):
        def plot_decorator(plot_f):
            def wrapper(self, **kwargs):
                self.select = select
                plot_f(self, **kwargs)
                self.select = None

            return wrapper

        return plot_decorator

    @staticmethod
    def _get_peak_y(group):
        ...

    @staticmethod
    def _get_peak_area_masked(group: pd.Series):
        # if group[0] == 0.0 and group[-1] == 0.0:
        #     sum = np.sum(group.values)
        # else:
        _idx = group.index
        _idx_names = group.index.names
        group_idx: pd.MultiIndex = group.index.droplevel("x").names
        y = group.values
        x = group.index.get_level_values("x")
        interval = group.index.get_level_values("x_bins").unique()
        max_y = group.groupby(group_idx, group_keys=True).max().values[:-1]
        # y = y.values
        group = group.groupby(group_idx, group_keys=True).first()
        edges = interval[:-1]
        mean = interval[:-1].mid
        sigma = interval[:-1].right - mean
        accept = False
        while accept is False:
            # if len(edges) < 1:
            #     plt.close()
            #     plt.plot(x, y)
            #     plt.show()
            #     peak_mask[:] = False
            #     break
            popt, _ = scipy.optimize.curve_fit(
                n_gauss_peaks,
                x,
                y,
                [
                    *max_y,
                    *mean,
                    *(0.34 * sigma),
                    edges[-1].right,
                    interval[-1].right,
                    1,
                    1,
                    0,
                ],
                bounds=(
                    [
                        *(0.8 * max_y),
                        *edges.left,
                        *np.full_like(mean, 0),
                        edges[-1].right,
                        edges[-1].right,
                        0.5,
                        1,
                        0,
                    ],
                    [
                        *(1.2 * max_y),
                        *edges.right,
                        *(0.34 * sigma),
                        interval[-1].right,
                        interval[-1].right,
                        100,
                        100,
                        0.1,
                    ],
                ),
            )
            peak_prms = popt[:-5].reshape(3, len(mean)).T
            bulk_prms = popt[
                -5:
            ]  # peak_mask = peak_prms[:, 0] > 1.1 * bulk_prms[-1]  # peak_idxs = np.ravel(np.argwhere(peak_mask))  # if np.all(peak_mask):  #     accept = True  # else:  #     max_y = max_y[peak_mask]  #     mean = mean[peak_mask]  #     sigma = sigma[peak_mask]  #     drop_val = edges[~peak_mask]  #     edges = edges[peak_mask]  #     group.drop(drop_val, level="x_bins", inplace=True)

        peak_groups = group.groupby(group_idx, group_keys=True)
        for peak, peak_group in zip(peak_prms, peak_groups):
            idx, series = peak_group
            gauss_y = gauss_peak(x, *peak)
            gauss_sum = np.sum(gauss_y)
            group.loc[series.index] = gauss_sum
            plt.plot(x, gauss_y, linestyle="dashed")
        plt.plot(x, n_gauss_peaks(x, *popt), linestyle="dotted")
        gauss_y = n_gauss_peaks(x, *popt)
        bulk_y = heavyside_func(x, *bulk_prms)
        bulk_sum = np.sum(bulk_y)
        # bulk_sum += np.sum(y) - np.sum(gauss_y)
        # plt.plot(x, bulk_y, linestyle='dashed')
        # plt.title(idx)
        # plt.show()
        # plt.close()
        # bulk_idx = peak_groups.grops[-1].index
        group.iloc[-1] = bulk_sum
        # print(np.sum(bulk_y))
        _idx = _idx.to_frame()
        # _idx, inplace=True)
        _idx = (
            _idx.set_index(group_idx)
            .groupby(group_idx, group_keys=False)
            .first()
        )
        _idx["x"] = float(len(mean))
        _idx["values"] = 0
        _idx["A"] = np.NaN
        _idx["mean"] = np.NaN
        _idx["sigma"] = np.NaN
        # _idx["mean"].iloc[peak_idxs] = peak_prms[:, 0]
        # _idx["mean"].iloc[peak_idxs] = peak_prms[:, 1]
        # _idx["sigma"].iloc[peak_idxs] = peak_prms[:, 2]
        _idx["mean"] = peak_prms[:, 0]
        _idx["mean"] = peak_prms[:, 1]
        _idx["sigma"] = peak_prms[:, 2]
        _idx["values"].update(group)
        # _idx.set_index('x', append=True, inplace=True)
        _idx = _idx.reset_index().set_index(["x", "x_bins"])
        group = _idx.loc[:, ["values", "mean", "sigma"]]
        # for peak in popt[:-5].reshape(3, len(mean)).T:
        #     group.loc[pd.IndexSlice[:,:]] = gauss_peak(x, *peak)
        #     peak_sum = np.sum(peak)
        # gauss_y =
        # sum = np.sum(gauss_y)
        # plt.plot(x, y)
        # plt.plot(full_x, gauss_y)
        # plt.show()
        # plt.close()
        # if self.ignore_density_sum is True:
        #     assert np.round(np.sum(group['values']), 2) == 1.00
        return group

    @staticmethod
    def _get_peak_area(group: pd.Series, name: str, grouped=""):
        # if group[0] == 0.0 and group[-1] == 0.0:
        #     sum = np.sum(group.values)
        # else:
        _idx = group.index
        _idx_names = group.index.names
        group_idx: pd.MultiIndex = group.index.droplevel("x").names
        y = group.values
        x = group.index.get_level_values("x")
        interval = group.index.get_level_values("x_bins").unique()
        max_y = group.groupby(group_idx, group_keys=True).max().values[:-1]
        max_x = x[y.isin(max_y)]
        # y = y.values
        group = group.groupby(group_idx, group_keys=True).first()
        fname = "_".join(group.index.droplevel("x_bins").values[0])
        savedir = Path(".gauss_peaks")
        fname = savedir / f"{name}_{fname}{grouped}"
        fname = fname.with_suffix(".p")
        if fname.is_file():
            with open(fname, "rb") as file:
                group = pkl.load(file)
        else:
            edges = interval[:-1]
            mean = interval[:-1].mid
            sigma = interval[:-1].right - mean
            if len(edges) != 0:
                try:
                    popt, _ = scipy.optimize.curve_fit(
                        n_gauss_peaks,
                        x,
                        y,
                        [
                            *max_y,
                            *max_x,
                            *(0.34 * sigma),
                            edges[-1].right,
                            interval[-1].right,
                            1,
                            1,
                            0,
                        ],
                        bounds=(
                            [
                                *(0.9 * max_y),
                                *(max_x - 0.1 * sigma),
                                *np.full_like(mean, 0),
                                edges[-1].right,
                                edges[-1].right,
                                0.5,
                                1,
                                0,
                            ],
                            [
                                *(max_y),
                                *(max_x + 0.1 * sigma),
                                *(0.34 * sigma),
                                interval[-1].right,
                                interval[-1].right,
                                100,
                                100,
                                0.1,
                            ],
                        ),
                    )

                    peak_prms = popt[:-5].reshape(3, len(mean)).T
                    bulk_prms = popt[-5:]
                except ValueError:
                    plt.plot(
                        x,
                        y,
                        label="_".join(
                            *group.index.droplevel("x_bins").unique()
                        ),
                    )
                    plt.vlines(edges.right, 0, max_y, label=edges.right)
                    plt.vlines(
                        max_x, 0, max_y, label=max_x, linestyle="dashed"
                    )
                    plt.legend()
                    plt.show()
                    plt.close()
                # peak_mask = peak_prms[:, 0] > 1.1 * bulk_prms[-1]
                # peak_idxs = np.ravel(np.argwhere(peak_mask))
                # if np.all(peak_mask):
                #     accept = True
                # else:
                #     max_y = max_y[peak_mask]
                #     mean = mean[peak_mask]
                #     sigma = sigma[peak_mask]
                #     drop_val = edges[~peak_mask]
                #     edges = edges[peak_mask]
                #     group.drop(drop_val, level="x_bins", inplace=True)

                peak_groups = group.groupby(group_idx, group_keys=True)
                for peak, peak_group in zip(peak_prms, peak_groups):
                    idx, series = peak_group
                    gauss_y = gauss_peak(x, *peak)
                    gauss_sum = np.sum(gauss_y)
                    group.loc[series.index] = gauss_sum
                    plt.plot(x, gauss_y, linestyle="dashed", label=peak[1])
                plt.plot(
                    x,
                    n_gauss_peaks(x, *popt),
                    linestyle="dotted",
                    label="_".join(*group.index.droplevel("x_bins").unique()),
                )
                plt.vlines(edges.right, 0, peak_prms[:, 0])
                gauss_y = n_gauss_peaks(x, *popt)
                bulk_y = heavyside_func(x, *bulk_prms)
                bulk_sum = np.sum(bulk_y)
                bulk_sum += np.sum(y) - np.sum(gauss_y)
                group.iloc[-1] = bulk_sum
                plt.plot(x, bulk_y, linestyle="dashed")
            else:
                # peak_prms = np.array([[np.NaN, np.NaN, np.NaN]])

                # plt.title(idx)
                # plt.show()
                # plt.close()
                # bulk_idx = peak_groups.grops[-1].index
                group.iloc[-1] = np.sum(y)
            # print(np.sum(bulk_y))
            _idx = _idx.to_frame()
            # _idx, inplace=True)
            _idx = (
                _idx.set_index(group_idx)
                .groupby(group_idx, group_keys=False)
                .first()
            )
            _idx["x"] = float(len(mean))
            _idx["values"] = 0
            _idx["A"] = np.NaN
            _idx["mean"] = np.NaN
            _idx["sigma"] = np.NaN
            try:
                _idx["mean"][:-1] = peak_prms[:, 0]
                _idx["mean"][:-1] = peak_prms[:, 1]
                _idx["sigma"][:-1] = peak_prms[:, 2]
            except UnboundLocalError:
                pass
            _idx["values"].update(group)
            # _idx.set_index('x', append=True, inplace=True)
            _idx = _idx.reset_index().set_index(["x", "x_bins"])
            group = _idx.loc[:, ["values", "mean", "sigma"]]
            # for peak in popt[:-5].reshape(3, len(mean)).T:
            #     group.loc[pd.IndexSlice[:,:]] = gauss_peak(x, *peak)
            #     peak_sum = np.sum(peak)
            # gauss_y =
            # sum = np.sum(gauss_y)
            plt.plot(x, y)
            # plt.plot(x, gauss_y)
            plt.legend()
            plt.show()
            plt.close()
            # if self.ignore_density_sum is True:
            #     assert np.round(np.sum(group['values']), 2) == 1.00
            if not savedir.is_dir():
                os.makedirs(savedir)
            with open(fname, "wb") as file:
                pkl.dump(group, file)
        return group

    # def _get_colour_dict(self, group_atoms=True):
    #     from seaborn import set_hls_values, saturate
    #
    #     # # atom type colour cycle
    #     # colours = [
    #     #     "#8dd3c7",
    #     #     "#ffffb3",
    #     #     "#bebada",
    #     #     "#fb8072",
    #     #     "#80b1d3",
    #     #     "#fdb462",
    #     #     "#b3de69",
    #     #     "#fccde5",
    #     # ]
    #     #
    #     # colours = [set_hls_values(saturate(cval), l=0.4) for cval in colours]
    #     # plot_df = self.data.df.copy()
    #     # plot_df.reset_index("_atoms", inplace=True)
    #     # plot_df['_atoms'] = plot_df["_atoms"].apply(
    #     #     lambda x: self.get_atom_type_group(x, group_all_atoms=group_atoms)
    #     # )
    #     # plot_df.set_index("_atoms", append=True, inplace=True)
    #     # plot_df = plot_df.groupby(plot_df.index.names).sum()
    #     # # self.atomname = plot_df.index.get_level_values("atoms").unique()[0]
    #     # # atoms = plot_df.index.get_level_values("_atoms")
    #     # # plot_df["colours"] = atoms
    #     # # plot_df["colours"] = plot_df["colours"].transform(lambda x: x[0])
    #     # # plot_df = pd.DataFrame({'a-types': atoms, 'values': plot_df, 'colours': atoms})
    #     # atom_types = atoms.unique()
    #     # colour_dict = dict(zip(atom_types, colours[: len(atom_types)]))
    #     # plot_df["colours"] = plot_df["colours"].transform(lambda x: colour_dict[x])
    #     colours = [
    #         "#8dd3c7",  # cyan (C)
    #         "#ffffb3",  # yellow (S)
    #         "#bebada",  # blue (N)
    #         "#fb8072",  # red (O)
    #         "#80b1d3",  # light blue (Mg)
    #         "#fdb462",  # orange (K)
    #         "#b3de69",  # light green (Ca)
    #         "#fccde5",  # pink (Na)
    #     ]
    #     colours = [set_hls_values(saturate(cval), l=0.4) for cval in colours]
    #     colour_keys = ["C", "S", "N", "O", "Mg", "K", "Ca", "Na"]
    #     colour_dict = dict(zip(colour_keys, colours))
    #     return colour_dict

    @redirect_tqdm
    def _get_binned_plot_colour_dfs_1d(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info(f"Getting binned plot and colour dfs")

        # # list of matplotlib sequential cmaps
        # cmaps = [
        #     # 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
        #     "spring",
        #     "summer",
        #     "autumn",
        #     "winter",
        #     "cool",
        #     "Wistia",
        #     "hot",
        #     "afmhot",
        #     "gist_heat",
        #     "copper",
        # ]
        colour_dict = self._get_colour_dict()
        colour_dict = {k: [v, "w"] for k, v in colour_dict.items()}

        sel = self.data.clays

        # get data for plotting
        plot_df = self.data.bin_df[sel].copy()

        # move clays category from columns to index
        idx_names = ["clays", *plot_df.index.names]
        # DataFrame -> Series
        plot_df = plot_df.stack()

        # get values for atom types (including separate ions)
        atoms = plot_df.index.get_level_values("_atoms")
        # atoms = atoms.
        # )

        # make new DataFrame from atom_type index level and values
        # plot_df.index = plot_df.index.droplevel(["x"])
        # plot_df.index.names = [name.strip('_') for name in plot_df.index.names]

        plot_df = pd.DataFrame({"values": plot_df, "colours": atoms})

        # map unique atom types to colour map
        # atom_types = atoms.unique()
        # colour_dict = dict(zip(atom_types, cmaps[: len(atom_types)]))
        # plot_df['colours'] = plot_df['colours']

        # reorder index for grouping
        # plot_df = plot_df.reorder_levels(idx_names)

        # separate colour column from plot_df -> yields 2 Series
        colour_df = plot_df["colours"]
        plot_df = plot_df["values"]

        plot_df = self._get_peak_area_df(plot_df)

        plot_df.index = plot_df.index.reorder_levels(idx_names)
        plot_df.reset_index("x", inplace=True, drop=True)
        colour_df = (
            colour_df.reset_index(["x", "x_bins"], drop=True)
            .reset_index()
            .drop_duplicates()
            .set_index(plot_df.index.droplevel("x_bins").names)
        )
        colour_df = colour_df["colours"].apply(
            lambda x: self.get_atom_colour_codes_from_names(x)
        )
        colour_df = colour_df.progress_apply(
            lambda x: mpc.LinearSegmentedColormap.from_list(
                x,
                colour_dict[x],
                N=len(self.data.df.index.get_level_values("x").unique()),
            )
        )

        # add missing atom probabilities from bulk to the largest bin
        # (bin.left, cutoff] -> (bin.left, all bulk])
        if self.add_missing_bulk is True:
            inner_sum = plot_df.groupby(
                plot_df.index.droplevel("x_bins").names
            ).sum()
            extra = 1 - inner_sum
            plot_df.where(
                np.rint(plot_df.index.get_level_values("x_bins").right)
                != int(self.data.cutoff),
                lambda x: x + extra,
                inplace=True,
            )
        # setattr(self, "_plot_df", plot_df.copy())
        self._plot_df, self._colour_df = plot_df.copy(), colour_df.copy()

    @redirect_tqdm
    def _get_peak_area_df(self, plot_df):
        # group and sum densities within adsorption shell bins
        plot_df = plot_df.groupby(
            plot_df.index.droplevel(["x", "x_bins"]).names, group_keys=True
        ).progress_apply(
            lambda x: self._get_peak_area(
                x, name=self.data.analysis, grouped=self.data.group_all
            )
            # values=pd.NamedAgg("values", self._get_peak_area), colours=pd.NamedAgg("colours", "first")
        )
        return plot_df

    def _split_plot_dfs(self):
        logger.info(f"Splitting binned plot and colour dfs")
        bin_df = self._plot_df["values"].copy()
        plot_df = self._colour_df.copy()
        dfs = {"bin": bin_df, "colour": plot_df}
        for key, df in dfs.items():
            if "ions" in self.data.atoms:
                not_ions = [a for a in self.data.atoms if a != "ions"]
                ion_view = df.xs("ions", level="atoms")
                ion_view.index = ion_view.index.droplevel(["ions"])
                ion_view.index.names = [
                    idstr.strip("_") for idstr in ion_view.index.names
                ]
                setattr(self, f"_ions_{key}_df", ion_view.copy())
            else:
                not_ions = np.unique(self.data.atoms).tolist()
            # not_ions = [a for a in self.data.atoms if a != "ions"]
            # ion_view = df.xs("ions", level="atoms")
            # ion_view.index = ion_view.index.droplevel(["ions"])
            # ion_view.index.names = [idstr.strip("_") for idstr in ion_view.index.names]
            # logger.info(f'\n{not_ions}, {df.index.names}')
            other_view = df.loc[pd.IndexSlice[:, :, :, :, not_ions, :]]
            other_view.index = other_view.index.droplevel("_atoms")
            # logger.info(f'{other_view.index.get_level_values("atoms").unique()}, {other_view.index.names}')
            # setattr(self, f"_ions_{key}_df", ion_view.copy())
            setattr(self, f"_other_{key}_df", other_view.copy())

    def _process_data(self, **kwargs):
        self._get_binned_plot_colour_dfs_1d()
        self._split_plot_dfs()

    @property
    def plot_df(self):
        try:
            plot_df = getattr(self, f"_{self.select}_bin_df")
            # logger.info(f'{plot_df.index.get_level_values("atoms").unique()}')
            return plot_df
        except AttributeError:
            logging.error(f"Wrong usage of property, select not assigned yet.")

    @property
    def colour_df(self):
        try:
            return getattr(self, f"_{self.select}_colour_df")
        except AttributeError:
            logging.error(f"Wrong usage of property, select not assigned yet.")

    def save(self, **kwargs):
        ...

    @cached_property
    def cnorm(self) -> mpc.Normalize:
        # normalise colour map from 0 to max_edge
        return mpc.Normalize(
            vmin=0, vmax=1.1 * self.data.max_shell_edge, clip=False
        )

    def _get_attr(self, attr: Literal["clays", "aas", "atoms", "ions"]):
        return pd.unique(self.plot_df.index.get_level_values(attr))

    get_clays = partialmethod(_get_attr, attr="clays")
    get_aas = partialmethod(_get_attr, attr="aas")
    get_ions = partialmethod(_get_attr, attr="ions")
    get_atoms = partialmethod(_get_attr, attr="atoms")
    get_other = partialmethod(_get_attr, attr="other")

    @plot_args_decorator(select="ions")
    def plot_ions(
        self,
        bars: Literal["clays", "aas", "ions", "other"],
        x: Literal["clays", "aas", "ions", "other"],
        y: Literal["clays", "aas", "ions", "other"],
        rowlabel: str = "y",
        columnlabel: str = "x",
        figsize=None,
        dpi=None,
        xmax=50,
        ymax=50,
        plsave=False,
        ylim=None,
        odir=".",
        barwidth=0.75,
        xpad=0.25,
        cmap="winter",
        plot_table=False,
    ):
        self.plot(
            bars,
            x,
            y,
            rowlabel,
            columnlabel,
            figsize,
            dpi,
            xmax,
            ymax,
            plsave,
            ylim,
            odir,
            barwidth,
            xpad,
            cmap,
            plot_table=plot_table,
        )

    @plot_args_decorator(select="other")
    def plot_other(
        self,
        bars: Literal["clays", "aas", "ions", "other"],
        x: Literal["clays", "aas", "ions", "other"],
        y: Literal["clays", "aas", "ions", "other"],
        rowlabel: str = "y",
        columnlabel: str = "x",
        figsize=None,
        dpi=None,
        xmax=50,
        ymax=50,
        plsave=False,
        ylim=None,
        odir=".",
        barwidth=0.75,
        xpad=0.25,
        cmap="winter",
        plot_table=None,
    ):
        self.plot(
            bars,
            x,
            y,
            rowlabel,
            columnlabel,
            figsize,
            dpi,
            xmax,
            ymax,
            plsave,
            ylim,
            odir,
            barwidth,
            xpad,
            cmap,
            plot_table=plot_table,
        )

    def get_suptitle(self, pl, separate):
        self.fig.suptitle(
            (
                ", ".join([self.title_dict[s].upper() for s in separate])
                + f": {self.label_mod(list(tuple(zip(pl, separate))))}"
            ),
            size=16,
            weight="bold",
        )

    def plot(
        self,
        bars: Literal["clays", "aas", "ions", "other"],
        x: Literal["clays", "aas", "ions", "other"],
        y: Literal["clays", "aas", "ions", "other"],
        rowlabel: str = "y",
        columnlabel: str = "x",
        figsize=None,
        dpi=None,  # diff=False,
        xmax=50,
        ymax=50,
        plsave=False,
        ylim=None,
        odir=".",
        barwidth=0.75,
        xpad=0.25,
        cmap="winter",
        tab_fontsize=12,
        plot_table=False,
    ):
        """Create stacked Histogram adsorption shell populations."""

        logger.info(f"plot table {plot_table}")

        self._init_plot(x, y, bars)

        plot_df = self.plot_df.copy()
        colour_df = self.colour_df.copy()

        avail_attrs = plot_df.index.names
        logger.info(f"Available attributes: {avail_attrs}")

        logger.info("Setting x:")
        self.x = Axis(self, x)
        logger.info("Setting y:")
        self.y = Axis(self, y)
        logger.info("Setting bars:")
        self.bars = Axis(self, bars)

        print(avail_attrs)
        separate = [
            s
            for s in avail_attrs
            if s not in [self.x, self.y, self.bars, self.bins]
        ]

        if len(separate) != 0:
            logger.info(f"Separate plots: {separate}")

        idx = pd.Index(
            [
                s
                for s in avail_attrs
                if (
                    s != self.x
                    and s != self.bars
                    and s not in [*separate, self.bins]
                )
            ]
        )

        logger.info(f"Iteration index: {idx.values}")

        sep = pd.Index(separate)
        # logger.info(f'{sep}')

        yid = np.ravel(np.where(np.array(idx) == self.y))[0]

        if figsize is None:
            if plot_table is True:
                figsize = self.get_figsize(xmax=xmax, ymax=ymax * 2)
            else:
                figsize = self.get_figsize(xmax=xmax, ymax=ymax)

        # set resultion
        if dpi is None:
            dpi = 100

        # get plotting iter from index
        iters = self._get_idx_iter(idx=idx)

        logger.info(
            f"Printing bar plots for {self.bars}\nColumns: {self.x.v}\nRows: {self.y.v}"
        )

        # set label modifier function
        self.label_mod = self.modify_plot_labels

        try:
            # iterator for more than one plot
            sep_it = self._get_idx_iter(idx=sep)  # logger.info(f'{sep_it}')
        except ValueError:
            # only one plot
            sep_it = [None]

        # iterate over separate plots
        for pl in sep_it:
            if pl is None:
                pl_str = ""
            else:
                pl_str = f"_{pl[0]}"
            self.name = f"{self.data.name}_{self.data.analysis}_{self.select}_{x}_{y}_{bars}{pl_str}_{self.data.cutoff}_{self.data.bins}"
            if plot_table is True:
                self.name += "_table"
            logger.info(f"plot {self.name}")
            # index map for y values
            # y_dict: dict = dict(zip(vy, np.arange(ly)))

            self.init_legend(ly=self.y.l, lx=self.x.l)

            # if plot_table is True:
            #     plt_nrows = self.y.l * 2
            #     logger.info('Has table')
            # else:
            #     plt_nrows = self.y.l
            # print('rows ', plt_nrows)

            # generate figure and axes array
            if plot_table is True:
                self.fig = plt.figure(figsize=figsize, dpi=dpi)
            else:
                self.fig = plt.figure(
                    figsize=figsize, dpi=dpi, constrained_layout=True
                )

            # xmul = 2

            self.gs = gridspec.GridSpec(
                nrows=self.y.l,
                ncols=self.x.l,
                figure=self.fig,  # width_ratios=[2,1]
                # sharey=True,
                # sharex=True
            )

            self.ax = {}

            if plot_table is True:
                self.gs = gridspec.GridSpec(
                    nrows=self.y.l,
                    ncols=self.x.l,
                    figure=self.fig,
                    wspace=2 / self.x.l,
                    hspace=0.5,  # width_ratios=
                    # sharey=True,
                    # sharex=True
                )
                self.subax = {}
                # self.tabax = {}
                for i in range(self.y.l):
                    for j in range(self.x.l):
                        #     if i == 0 and j == 0:
                        #         self.subax[i, j] = gridspec.GridSpecFromSubplotSpec(
                        #     2, 1, subplot_spec=self.gs[i, j]
                        # )
                        #     else:
                        self.subax[i, j] = self.gs[i, j].subgridspec(
                            10, self.x.l, wspace=1.5, hspace=1.5
                        )
                        self.ax[i, j] = self.fig.add_subplot(
                            self.subax[i, j][:8, :]
                        )

                        # if i == (self.y.l - 1) or j == 0:

                        # else:  #     self.ax[i, j] = self.fig.add_subplot(self.subax[i, j][:8, :])#,  #                                          # sharex=self.ax[0, i],  #                                          # sharey=self.ax[j, (self.y.l - 1)])  # if i != (self.y.l - 1):  #     self.ax[i, j].tick_params(labelbottom=False)  # if j != 0:  #     self.ax[i, j].tick_params(labelleft=False)

                        # self.tabax[i, j] = self.fig.add_subplot(self.subax[i, j][1])  # self.tabax[i, j].set

            else:
                for i in range(self.y.l):
                    for j in range(self.x.l):
                        self.ax[i, j] = self.fig.add_subplot(self.gs[i, j])

            # only one plot
            if pl is None:
                logger.info(f"Generating plot")
                sepview = plot_df.view()
                plsave = ""

            # multiple plots
            else:
                logger.info(f"Generating {pl} plot:")
                # print(plot_df.head(20), "\n", separate, pl)
                sepview = plot_df.xs((pl), level=separate, drop_level=False)
                plsave = pl
                self.get_suptitle(pl, separate)

            # set plot index
            pi = 0

            # iterate over subplot columns
            for col in self.x.v:
                logger.info(f"Column {col}")
                try:
                    view = sepview.xs(
                        col, level=self.x, axis=0, drop_level=False
                    )
                    pi = 1
                except ValueError:
                    view = sepview
                    col = self.x.v
                    pi += 1

                for it in iters:
                    try:
                        values = view.xs(
                            tuple(it), level=idx.tolist(), drop_level=False
                        )

                        x_labels = []
                        x_ticks = []
                        for bar_num, vbar in enumerate(self.bars.v):
                            tab_colours = []
                            tab_rows = []
                            if plot_table is True and bar_num == 0:
                                # tab_colours = []
                                # tab_rows = []
                                try:
                                    tab_text = values.values.reshape(
                                        (self.bars.l, -1)
                                    ).T
                                except ValueError:
                                    tab_text = values.values
                                # tab_text = tab_text.astype(str)
                                tab_format = lambda x: rf"${x * 100:3.1f} \%$"
                                tab_format_vec = np.vectorize(tab_format)
                                tab_text = tab_format_vec(tab_text)
                                tab_cols = list(
                                    self.bars.v[
                                        : np.atleast_2d(tab_text).shape[0]
                                    ]
                                )

                            x_ticks.append(
                                self.bars.dict[vbar] * (barwidth + xpad)
                            )
                            x_labels.append(vbar)
                            bottom = 0.0
                            bar_vals = values.xs(
                                vbar, level=self.bars, drop_level=False
                            )
                            # tab_text.append(bar_vals.values)

                            cmap = colour_df.loc[bar_vals.index].values[0]

                            peaks = self.data.get_bar_peaks(
                                atom_type=col, other=None
                            )
                            if np.all(bar_vals.values) >= 0:
                                # table_text.append([f'${v * 100:3.1f} %$' for v in bar_vals.values])
                                x_id, y_id = (
                                    self.x.dict[col],
                                    self.y.dict[it[yid]],
                                )
                                # if plot_table is True:
                                #     # logger.info(y_id)
                                #     logger.info("Has table")
                                #     y_id *= 2
                                #     logger.info(y_id)
                                # print(y_id)

                                bar_val_view = bar_vals
                                bin_list = bar_val_view.index.get_level_values(
                                    "x_bins"
                                )
                                bar_val_view.index = bin_list

                                x_tick = x_ticks[-1]

                                for bar_id, bar_val in enumerate(
                                    bar_val_view.items()
                                ):
                                    x_bin, y_val = bar_val

                                    try:
                                        peak = peaks[bar_id]
                                    except IndexError:
                                        peak = x_bin.right
                                    colour = cmap(self.cnorm(peak))
                                    # if colour not in table_col and cmap == table_cmap:
                                    #     print('colour', colour)
                                    #     table_col.append(colour)

                                    label = self._get_bin_label(
                                        x_bin, bin_list
                                    )

                                    if bar_num == 0:
                                        tab_colours.append(list(colour))
                                        tab_rows.append(label)
                                        print(tab_colours)

                                    # if x_bin.right < np.max(self.x):
                                    #     label = f'${x_bin.left} - {x_bin.right}$ \AA'
                                    # else:
                                    #     label = f'$ > {x_bin.left}$ \AA'
                                    # if label not in table_rows and cmap == table_cmap:
                                    #     table_rows.append(label)
                                    if y_val >= 0.001:
                                        # barwidth = bulk_edge - x_bin.left
                                        # try:
                                        # x_tick = x_ticks[-1] + barwidth
                                        # x_ticks.append(x_tick)
                                        # except IndexError:
                                        #     x_tick = x_bin.left
                                        try:
                                            p = self.ax[y_id, x_id].bar(
                                                x_tick,
                                                y_val,
                                                label=label,
                                                bottom=bottom,
                                                width=barwidth,
                                                align="edge",
                                                color=colour,
                                            )  # self.ax[y_id, x_id].bar_label(p, labels=[label],  #                          fmt='%s',  #                          label_type='center')
                                        except IndexError:
                                            p = self.ax[y_id].bar(
                                                x_tick,
                                                y_val,
                                                label=label,
                                                bottom=bottom,
                                                width=barwidth,
                                                align="edge",
                                                color=colour,
                                            )  # self.ax[y_id, x_id].bar_label(p, labels=[label],  #                          fmt='%s',  #                          label_type='center')
                                        # finally:
                                        bottom += y_val
                                # try:
                                #     self.fig.subplots_adjust(left=0.2, bottom=0.2)
                                # except IndexError:
                                #     self.ax[y_id].subplots_adjust(left=0.2, bottom=0.2)
                                if plot_table is True:
                                    # y_id += 1
                                    # print(tab_colours)
                                    logger.info(f"Has table, {y_id}")
                                    has_table = False
                                    try:
                                        print("rows", tab_rows)
                                        tab_colours = np.array(tab_colours)
                                        print("colours", tab_colours)
                                        # self.tabax[y_id, x_id].clear()
                                        the_table: table = table(
                                            self.ax[y_id, x_id],
                                            cellText=tab_text,
                                            rowLabels=tab_rows,
                                            rowColours=tab_colours,
                                            colLabels=tab_cols,
                                            loc="bottom",
                                            edges="horizontal",
                                        )
                                        has_table = True
                                    except IndexError:
                                        try:
                                            # self.tabax[y_id].clear()
                                            the_table: table = table(
                                                self.ax[y_id],
                                                cellText=tab_text,
                                                rowLabels=tab_rows,
                                                rowColours=tab_colours,
                                                colLabels=tab_cols,
                                                loc="bottom",
                                                edges="horizontal",
                                            )
                                            has_table = True
                                        except ValueError:
                                            logger.info("could not plot table")
                                    except ValueError:
                                        logger.info("could not plot table")
                                    if has_table is True:
                                        logger.info("Plotting table")
                                        # renderer = self.fig.canvas.get_renderer()
                                        the_table.auto_set_font_size(False)
                                        the_table.set_fontsize(12)
                                        h = the_table.get_celld()[
                                            (0, 0)
                                        ].get_height()
                                        w = the_table.get_celld()[
                                            (0, 0)
                                        ].get_width()
                                        table_cells = the_table.properties()[
                                            "children"
                                        ]
                                        for cell in table_cells:
                                            max_width = w
                                            if cell.get_width() > max_width:
                                                max_width = cell.get_width()
                                            # print(cell.get_y())
                                            # cell.set_y(5)
                                            # print(cell.get_y())
                                            cell.set_width(barwidth + xpad)
                                            cell.set_height(0.13)
                                            # cell.set_fontsize(12)
                                            cell.set_text_props(
                                                verticalalignment="center"
                                            )
                                        for cell in table_cells:
                                            cell.set_width(
                                                max_width
                                            )  # the_table.set_fontsize(tab_fontsize)  # has_table = True  # the_table.scale(1, 1.2)  # the_table.stale = True  # the_table.set_clip_on(False)  # the_table._update_positions(renderer)  # bbox = the_table.get_window_extent(renderer)  # the_table.set(clip_bbox=bbox)  # self.ax[y_id, x_id].stale = True  # renderer = self.fig.canvas.get_renderer()  #  #  # # text_height = the_table._approx_text_height()  #  # # the_table.auto_set_font_size(False)  # # new_font_size = the_table.FONTSIZE  # # the_table.FONTSIZE = tab_fontsize  # the_table.set_fontsize(tab_fontsize)  #  # cell = the_table[0, 0]  # # cell_height = cell.get_text()  # win_ext = the_table.get_window_extent(renderer=renderer)  # text_height = cell.get_text_bounds(renderer=renderer)[-1]  # tab_font = the_table.FONTSIZE  # text_height = cell.get_text_bounds()  # # the_table._do_  # tab_font_new = tab_fontsize  # # sys.exit(1)  # # text_height = cells.get_text_bounds()  #  # font_ratio = tab_fontsize / the_table.FONTSIZE  #  # # # the_table.set_fontsize(tab_font_new)  # # # tab_dict = the_table.get_celld()  # #  # # # for col_id in range(0, len(tab_cols)):  # # #     tab_dict[(0, col_id)].set_height(text_height + 0.1)  # # #     for row_id in range(1, len(tab_rows)):  # # #         tab_dict[(row_id, col_id)].set_height(text_height + 0.1)  # the_table.scale(1, font_ratio + 0.3)  # the_table._stale = True  # the_table[...].set_text_props(linespacing=1.5,  #                          verticalalignment='bottom')  # the_table.scale(1, 1.2)  # the_table.stale = True  # the_table.set_clip_on(False)  # the_table._update_positions(renderer)

                                    else:
                                        logger.info(
                                            "No table"
                                        )  # else:  # print(y_val)
                    # print("table false")
                    # sys.exit(1)
                    # tab_dict = the_table.get_celld()
                    # table = ax[y_id, x_id].table(cellText=table_text,
                    #                              rowColours=table_col,
                    #                              rowLabels=table_rows,
                    #                              # colLables=...,
                    #                              loc='bottom')
                    # x_ticks = x_ticks[:-1]
                    # x_ticks.append(bar_dict[vbar] * (barwidth + xpad))
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
                        logger.info(f"No data for {pl}, {self.x}, {it}")

            # x_ticks = [np.linspace(n_bar * bulk_edge + xpad,
            #                        n_bar * bulk_edge + bulk_edge, int(bulk_edge)) for n_bar in range(lbars)]
            # x_ticks = np.ravel(x_ticks)
            # x_labels = np.tile(np.arange(0, bulk_edge, 1), lbars)
            for i in range(self.y.l):
                # if plot_table is True:
                #     ax_multi = 2
                # else:
                #     ax_multi = 1
                # ax_i = i * ax_multi
                # print(f"Axis index: {ax_i}, multi: {ax_multi}")

                try:
                    # self.ax[]
                    # self.ax[i, 0].set_ylabel(
                    #     f"{self.label_mod([(self.y.v[i], self.y)])}\n" + rowlabel
                    # )
                    for j in range(self.x.l):
                        # if plot_table is True:
                        #     self.ax[i, j].subplots_adjust(bottom=0.2)
                        #     f"{self.label_mod([(self.y.v[i], self.y)])}\n" + rowlabel
                        # )
                        if j == 0:
                            self.ax[i, 0].set_ylabel(
                                f"{self.label_mod([(self.y.v[i], self.y)])}\n"
                                + rowlabel
                            )
                            self.ax[i, j].set_yticks(np.arange(0.0, 1.1, 0.2))
                        else:
                            self.ax[i, j].set_yticks(np.arange(0.0, 1.1, 0.2))
                            self.ax[i, j].set_yticklabels(
                                []
                            )  # np.arange(0.0, 1.1, 0.2))
                        self.ax[i, j].spines[["top", "right"]].set_visible(
                            False
                        )
                        self.ax[i, j].hlines(
                            1.0,
                            -xpad,
                            self.bars.l * (barwidth + xpad) + xpad,
                            linestyle="--",
                        )
                        # self.ax[i, j].legend(ncol=2, loc='lower center')#[leg for leg in legends[i, j]], ncol=3)
                        #                  if xlim != None:
                        self.ax[i, j].set_xlim(
                            (-xpad, self.bars.l * (barwidth + xpad))
                        )
                        self.ax[i, j].set_xticks([], [])
                        #                 if ylim != None:
                        self.ax[i, j].set_ylim((0.0, 1.2))
                        self.ax[self.y.l - 1, j].set_xticks(
                            np.array(x_ticks) + 0.5 * barwidth, x_labels
                        )
                        self.ax[self.y.l - 1, j].set_xlabel(
                            bars
                            + f"\n{self.label_mod([(self.x.v[j], self.x)])}"
                        )  # self.ax[i, j].set_yticklabels(np.arange(0.0, 1.1, 0.2))
                except IndexError:
                    self.ax[i].set_ylabel(
                        f"{self.label_mod([(self.y.v[i], y)])}\n" + rowlabel
                    )
                    # self.ax[i].legend([self.label_mod([(leg, self.label_key)]) for leg in self.legends[i, 0]], ncol=3)
                    self.ax[self.y.l - 1].set_xlabel(
                        columnlabel
                        + f"\n{self.label_mod([(self.x.v[0], self.x)])}"
                    )
            # # #
            self.fig.supxlabel(f"{self.title_dict[x]}s", size=14)
            self.fig.supylabel(f"{self.title_dict[y]}s", size=14)
            if plsave is not False:
                logger.info("Saving plot")
                if type(plsave) == str and plsave != "":
                    outname = f"{plsave}.png"
                else:
                    outname = f"{self.name}.png"
                odir = Path(odir).absolute()
                logger.info(f"output to {odir.absolute()}")
                if not odir.is_dir():
                    os.makedirs(odir)
                logger.info(odir)
                logger.info(outname)
                self.fig.savefig(str(odir / outname))
            else:
                plt.show()
            self.fig.clear()

    def _get_bin_label(self, x_bin, bin_list):
        if x_bin.right < np.max(bin_list.right):
            label = f"${x_bin.left:>3.1f} - {x_bin.right:3.1f}$ \AA"  # barwidth = x_bin.right - x_bin.left

        else:
            label = f"$ > {x_bin.left:>3.1f}$ \AA"
        return label


class AtomHistPlot(Plot):
    sel_list = ["clays", "ions", "aas", "atoms"]
    title_dict = {
        "clays": "Clay type",
        "ions": "Ion type",
        "aas": "Amino acid",
        "atoms": "Atom type",
        "other": "Other atom type",
    }
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
    id_dict = {
        "A": r"\alpha",
        "B": r"\beta",
        "G": r"\gamma",
        "D": r"\delta",
        "H": r"\eta",
        "E": r"\epsilon",
        "Z": r"\zeta",
        "R": "side",
    }
    _init_args = {"x", "y", "bars", "group_atoms"}
    _plot_args = {
        "rowlabel": "y",
        "columnlabel": "x",
        "dpi": None,
        "figsize": None,
    }

    def __init__(self, data, sdir=None, group_atoms=False, **kwargs):
        logger.info(f"Initialising {self.__class__.__name__}")
        super().__init__(data, sdir=sdir, group_atoms=group_atoms, **kwargs)
        self.bins = "x"
        self.group = "aas"
        self.bars = "x"

    def plot_args_decorator(select: str):
        def plot_decorator(plot_f):
            def wrapper(self, **kwargs):
                self.select = select
                plot_f(self, **kwargs)
                self.select = None

            return wrapper

        return plot_decorator

    def _split_plot_dfs(self):
        self._plot_df = ...
        logger.info(f"Splitting binned plot and colour dfs")
        bin_df = self._plot_df.copy()
        plot_df = self._colour_df.copy()
        dfs = {"bin": bin_df, "colour": plot_df}
        for key, df in dfs.items():
            not_ions = [a for a in self.data.atoms if a != "ions"]
            ion_view = df.xs("ions", level="atoms")
            ion_view.index = ion_view.index.droplevel(["ions"])
            ion_view.index.names = [
                idstr.strip("_") for idstr in ion_view.index.names
            ]
            # logger.info(f'\n{not_ions}, {df.index.names}')
            other_view = df.loc[pd.IndexSlice[:, :, :, :, not_ions, :]]
            other_view.index = other_view.index.droplevel("_atoms")
            # logger.info(f'{other_view.index.get_level_values("atoms").unique()}, {other_view.index.names}')
            setattr(self, f"_ions_{key}_df", ion_view.copy())
            setattr(self, f"_other_{key}_df", other_view.copy())

    def get_atom_type(self, atom_name):
        if atom_name in ["O", "OT", "OXT"]:
            atom_name = "OT"
        elif len(atom_name) == 1:
            pass
        else:
            atom_type = atom_name[0]
            atom_pos_id = atom_name[1]
            if atom_type != "C" and atom_pos_id == "H":
                pass
            else:
                atom_pos_id = self.id_dict[atom_pos_id]
            if atom_pos_id != "H":
                atom_pos_id = "_{" + atom_pos_id
            else:
                atom_pos_id = atom_pos_id + "_{"
            try:
                atom_id = re.search(r"[A-Z]+([0-9])", atom_name).group(1)
                atom_pos_id += atom_id
            except AttributeError:
                pass
            finally:
                atom_pos_id += "}"
            atom_name = rf"{atom_type}{atom_pos_id}"
        atom_name = r"$\mathrm{" + rf"{atom_name}" + "}$"
        return atom_name

    # def _get_colour_dict(self, group_atoms=True):
    #     from seaborn import set_hls_values, saturate
    #
    #     # atom type colour cycle
    #     colours = [
    #         "#8dd3c7",
    #         "#ffffb3",
    #         "#bebada",
    #         "#fb8072",
    #         "#80b1d3",
    #         "#fdb462",
    #         "#b3de69",
    #         "#fccde5",
    #     ]
    #     colours = [set_hls_values(saturate(cval), l=0.4) for cval in colours]
    #     plot_df = self.data.df.copy()
    #     plot_df.reset_index("x", inplace=True)
    #     plot_df['x'] = plot_df["x"].apply(
    #         lambda x: self.get_atom_type_group(x, group_all_atoms=group_atoms)
    #     )
    #     plot_df.set_index("x", append=True, inplace=True)
    #     plot_df = plot_df.groupby(plot_df.index.names).sum()
    #     atoms = plot_df.index.get_level_values("x")
    #     atoms.transform(lambda x: x[0])
    #     # plot_df = pd.DataFrame({'a-types': atoms, 'values': plot_df, 'colours': atoms})
    #     atom_types = atoms.unique()
    #     colour_dict = dict(zip(atom_types, colours[: len(atom_types)]))
    #     # plot_df["colours"] = plot_df["colours"].transform(lambda x: colour_dict[x])
    #     return colour_dict

    def _process_data(self, group_atoms=False, **kwargs):
        logger.info(f"Getting binned plot and colour dfs")
        logger.info(f"Getting plot and colour dfs")
        colour_dict = self._get_colour_dict()
        colour_dict = {k: [v, "k"] for k, v in colour_dict.items()}

        # atom type colour cycle
        # colours = [
        #     "#8dd3c7",
        #     "#ffffb3",
        #     "#bebada",
        #     "#fb8072",
        #     "#80b1d3",
        #     "#fdb462",
        #     "#b3de69",
        #     "#fccde5",
        # ]
        # colours = [set_hls_values(saturate(cval), l=0.4) for cval in colours]
        plot_df = self.data.df.copy()
        # plot_df = plot_df.groupby(plot_df.index.names).sum()
        plot_df.reset_index("x", inplace=True)
        plot_df["x"] = plot_df["x"].apply(
            lambda x: self.get_atom_type_group(x, group_all_atoms=group_atoms)
        )
        plot_df.set_index("x", append=True, inplace=True)
        plot_df = plot_df.groupby(plot_df.index.names).sum()
        self.atomname = plot_df.index.get_level_values("atoms").unique()[0]
        plot_df.index = plot_df.index.droplevel("atoms")
        atoms = plot_df.index.get_level_values("x")

        plot_df["colours"] = atoms
        plot_df["colours"] = plot_df["colours"].transform(lambda x: x[0])
        # plot_df = pd.DataFrame({'a-types': atoms, 'values': plot_df, 'colours': atoms})
        # atom_types = plot_df["colours"].unique()
        # colour_dict = dict(zip(atom_types, colours[: len(atom_types)]))
        plot_df["colours"] = plot_df["colours"].transform(
            lambda x: colour_dict[x]
        )

        # separate colour column from plot_df -> yields 2 Series
        colour_df = plot_df["colours"]
        plot_df = plot_df["values"]
        self._plot_df, self._colour_df = plot_df.copy(), colour_df.copy()
        logger.info(f"{self._plot_df}\n{self._colour_df}")

    @property
    def plot_df(self):
        try:
            plot_df = getattr(self, f"_plot_df")
            # logger.info(f'{plot_df.index.get_level_values("atoms").unique()}')
            return plot_df
        except AttributeError:
            logging.error(f"Wrong usage of property, select not assigned yet.")

    @property
    def colour_df(self):
        try:
            return getattr(self, f"_colour_df")
        except AttributeError:
            logging.error(f"Wrong usage of property, select not assigned yet.")

    def save(self, **kwargs):
        ...

    @cached_property
    def cnorm(self) -> mpc.Normalize:
        # normalise colour map from 0 to max_edge
        return mpc.Normalize(vmin=0, vmax=self.data.max_shell_edge, clip=False)

    def _get_attr(self, attr: Literal["clays", "aas", "atoms", "ions"]):
        return pd.unique(self.plot_df.index.get_level_values(attr))

    get_clays = partialmethod(_get_attr, attr="clays")
    get_aas = partialmethod(_get_attr, attr="aas")
    get_ions = partialmethod(_get_attr, attr="ions")
    # get_atoms = partialmethod(_get_attr, attr="atoms")
    get_other = partialmethod(_get_attr, attr="other")

    @redirect_tqdm
    @plot_args_decorator(select="other")
    def plot_other(
        self,
        bars: Literal["clays", "aas", "ions", "other"],
        x: Literal["clays", "aas", "ions", "other"],
        y: Literal["clays", "aas", "ions", "other"],
        rowlabel: str = "y",
        columnlabel: str = "x",
        figsize=None,
        dpi=None,
        xmax=50,
        ymax=50,
        plsave=False,
        ylim=None,
        odir=".",
        barwidth=0.75,
        xpad=0.25,
        cmap="winter",  # group_atoms=False,
        plot_table=None,
    ):
        self.plot(
            bars=bars,
            x=x,
            y=y,
            rowlabel=rowlabel,
            columnlabel=columnlabel,
            figsize=figsize,
            dpi=dpi,
            xmax=xmax,
            ymax=ymax,
            plsave=plsave,
            ylim=ylim,
            odir=odir,
            barwidth=barwidth,
            xpad=xpad,
            cmap=cmap,
            # group_atoms=group_atoms
        )

    def get_suptitle(self, pl, separate):
        self.fig.suptitle(
            (
                ", ".join([self.title_dict[s].upper() for s in separate])
                + f": {self.label_mod(list(tuple(zip(pl, separate))))}"
            ),
            size=16,
            weight="bold",
        )

    @redirect_tqdm
    def plot(
        self,
        bars: Literal["clays", "aas", "ions", "other"],
        x: Literal["clays", "aas", "ions", "other"],
        y: Literal["clays", "aas", "ions", "other"],
        rowlabel: str = "y",
        columnlabel: str = "x",
        figsize=None,
        dpi=None,  # diff=False,
        xmax=50,
        ymax=50,
        plsave=False,
        ylim=None,
        odir=".",
        barwidth=0.75,
        xpad=0.25,
        cmap="winter",
        # group_atoms=False,
        tab_fontsize=12,
        plot_table=False,
    ):
        """Create stacked Histogram adsorption shell populations."""

        logger.info(f"plot table {plot_table}")

        hatch_list = [
            r"//",
            r"--",
            r"\\",
            r"//",
            r".",
            r"///",
            r"...",
            "x",
            r"+",
            r"..",
        ]

        self._init_plot(x, y, bars)

        plot_df = self.plot_df.copy()
        colour_df = self.colour_df.copy()

        avail_attrs = plot_df.index.names
        logger.info(f"Available attributes: {avail_attrs}")

        logger.info("Setting x:")
        self.x = Axis(self, x)
        logger.info("Setting y:")
        self.y = Axis(self, y)
        logger.info("Setting bars:")
        self.bars = Axis(self, bars)

        print(avail_attrs)
        separate = [
            s
            for s in avail_attrs
            if s not in [self.x, self.y, self.bars, self.bins]
        ]

        if len(separate) != 0:
            logger.info(f"Separate plots: {separate}")

        idx = pd.Index(
            [
                s
                for s in avail_attrs
                if (
                    s != self.x
                    and s != self.bars
                    and s not in [*separate, self.bins]
                )
            ]
        )

        logger.info(f"Iteration index: {idx.values}")

        sep = pd.Index(separate)
        # logger.info(f'{sep}')

        yid = np.ravel(np.where(np.array(idx) == self.y))[0]

        if figsize is None:
            if plot_table is True:
                figsize = self.get_figsize(xmax=xmax, ymax=ymax * 2)
            else:
                figsize = self.get_figsize(xmax=xmax, ymax=ymax)

        # set resultion
        if dpi is None:
            dpi = 100

        # get plotting iter from index
        iters = self._get_idx_iter(idx=idx)

        logger.info(
            f"Printing bar plots for {self.bars}\nColumns: {self.x.v}\nRows: {self.y.v}"
        )

        # set label modifier function
        self.label_mod = self.modify_plot_labels

        try:
            # iterator for more than one plot
            sep_it = self._get_idx_iter(idx=sep)  # logger.info(f'{sep_it}')
        except ValueError:
            # only one plot
            sep_it = [None]

        # iterate over separate plots
        for pl in sep_it:
            if pl is None:
                pl_str = ""
            else:
                pl_str = f"_{pl[0]}"
            self.name = f"{self.data.name}_{self.data.analysis}_{x}_{y}_{bars}{pl_str}_{self.data.cutoff}_{self.data.bins}"
            if self.group_atoms != False:
                self.name += "_grouped"
            logger.info(f"Plot name: {self.name!r}")
            # index map for y values
            # y_dict: dict = dict(zip(vy, np.arange(ly)))

            self.init_legend(ly=self.y.l, lx=self.x.l)

            # if plot_table is True:
            #     plt_nrows = self.y.l * 2
            #     logger.info('Has table')
            # else:
            #     plt_nrows = self.y.l
            # print('rows ', plt_nrows)

            # generate figure and axes array
            self.ax = {}

            if plot_table is True:
                self.fig = plt.figure(figsize=figsize, dpi=dpi)
                self.gs = gridspec.GridSpec(
                    nrows=self.y.l,
                    ncols=self.x.l,
                    figure=self.fig,
                    wspace=2 / self.x.l,
                    hspace=0.5,  # width_ratios=
                    # sharey=True,
                    # sharex=True
                )
                self.subax = {}
                # self.tabax = {}
                for i in range(self.y.l):
                    for j in range(self.x.l):
                        #     if i == 0 and j == 0:
                        #         self.subax[i, j] = gridspec.GridSpecFromSubplotSpec(
                        #     2, 1, subplot_spec=self.gs[i, j]
                        # )
                        #     else:
                        self.subax[i, j] = self.gs[i, j].subgridspec(
                            10, self.x.l, wspace=1.5, hspace=1.5
                        )
                        self.ax[i, j] = self.fig.add_subplot(
                            self.subax[i, j][:8, :]
                        )

                        # if i == (self.y.l - 1) or j == 0:

                        # else:  #     self.ax[i, j] = self.fig.add_subplot(self.subax[i, j][:8, :])#,  #                                          # sharex=self.ax[0, i],  #                                          # sharey=self.ax[j, (self.y.l - 1)])  # if i != (self.y.l - 1):  #     self.ax[i, j].tick_params(labelbottom=False)  # if j != 0:  #     self.ax[i, j].tick_params(labelleft=False)

                        # self.tabax[i, j] = self.fig.add_subplot(self.subax[i, j][1])  # self.tabax[i, j].set

            else:
                self.fig = plt.figure(
                    figsize=figsize, dpi=dpi, constrained_layout=True
                )

                # xmul = 2

                self.gs = gridspec.GridSpec(
                    nrows=self.y.l,
                    ncols=self.x.l,
                    figure=self.fig,  # width_ratios=[2,1]
                    # sharey=True,
                    # sharex=True
                )
                for i in range(self.y.l):
                    for j in range(self.x.l):
                        self.ax[i, j] = self.fig.add_subplot(self.gs[i, j])

            # only one plot
            if pl is None:
                logger.info(f"Generating plot")
                sepview = plot_df.view()
                plsave = ""

            # multiple plots
            else:
                logger.info(f"Generating {pl} plot:")
                # print(plot_df.head(20), "\n", separate, pl)
                sepview = plot_df.xs((pl), level=separate, drop_level=False)
                plsave = pl
                self.get_suptitle(pl, separate)

            # set plot index
            pi = 0

            # iterate over subplot columns
            for col in self.x.v:
                logger.info(f"Column {col}")
                try:
                    view = sepview.xs(
                        col, level=self.x, axis=0, drop_level=False
                    )
                    pi = 1
                except ValueError:
                    view = sepview
                    col = self.x.v
                    pi += 1

                for it in iters:
                    x_id, y_id = self.x.dict[col], self.y.dict[it[yid]]
                    try:
                        values = view.xs(
                            tuple(it), level=idx.tolist(), drop_level=False
                        )

                        x_labels = []
                        x_label_ticks = []
                        for bar_num, vbar in enumerate(self.bars.v):
                            tab_colours = []
                            tab_rows = []
                            if plot_table is True and bar_num == 0:
                                # tab_colours = []
                                # tab_rows = []
                                try:
                                    tab_text = values.values.reshape(
                                        (self.bars.l, -1)
                                    ).T
                                except ValueError:
                                    tab_text = values.values
                                # tab_text = tab_text.astype(str)
                                tab_format = lambda x: rf"${x * 100:3.1f} \%$"
                                tab_format_vec = np.vectorize(tab_format)
                                tab_text = tab_format_vec(tab_text)
                                tab_cols = list(
                                    self.bars.v[
                                        : np.atleast_2d(tab_text).shape[0]
                                    ]
                                )

                            # x_ticks.append(self.bars.dict[vbar] * (barwidth + xpad))
                            # x_labels.append(vbar)
                            bottom = 0.0
                            bar_vals = values.xs(
                                vbar, level=self.bars, drop_level=False
                            )
                            start = bar_num * (barwidth + xpad)
                            step = barwidth + xpad
                            x_ticks = np.linspace(
                                start,
                                start + (len(bar_vals) * step * self.bars.l),
                                len(bar_vals),
                                endpoint=True,
                            ).tolist()
                            x_label_ticks.append([x_ticks])
                            x_labels = list(
                                map(
                                    lambda x: self.get_atom_type(x),
                                    bar_vals.index.get_level_values("x"),
                                )
                            )
                            bar_hatch = [
                                hatch_list[bar_num % len(hatch_list)]
                                for val in bar_vals
                            ]
                            # tab_text.append(bar_vals.values)

                            cmap = colour_df.loc[bar_vals.index]

                            # peaks = self.data.get_bar_peaks(
                            #     atom_type=col, other=None
                            # )
                            # plot if all values > 0
                            if np.all(bar_vals.values) >= 0:
                                # table_text.append([f'${v * 100:3.1f} %$' for v in bar_vals.values])

                                # if plot_table is True:
                                #     # logger.info(y_id)
                                #     logger.info("Has table")
                                #     y_id *= 2
                                #     logger.info(y_id)
                                # print(y_id)

                                bar_val_view = bar_vals
                                bin_list = bar_val_view.index.get_level_values(
                                    "x"
                                )
                                bar_val_view.index = bin_list

                                # x_tick = x_ticks[-1]

                                # for bar_id, bar_val in enumerate(bar_val_view.items()):
                                #
                                #     x_bin, y_val = bar_val

                                # try:
                                # peak = peaks[bar_id]
                                # except IndexError:
                                # peak = x_bin.right
                                # colour = cmap[bar_id] #(self.cnorm(peak))
                                # if colour not in table_col and cmap == table_cmap:
                                #     print('colour', colour)
                                #     table_col.append(colour)

                                # label = self._get_bin_label(x_bin, bin_list)

                                # if bar_num == 0:
                                #     tab_colours.append(list(colour))
                                #     tab_rows.append(label)
                                #     print(tab_colours)

                                # if x_bin.right < np.max(self.x):
                                #     label = f'${x_bin.left} - {x_bin.right}$ \AA'
                                # else:
                                #     label = f'$ > {x_bin.left}$ \AA'
                                # if label not in table_rows and cmap == table_cmap:
                                #     table_rows.append(label)
                                # if y_val >= 0.0010:
                                # barwidth = bulk_edge - x_bin.left
                                # try:
                                #     x_tick = x_ticks[-1] + barwidth #+ (barwidth + xpad) * bar_num
                                #     x_ticks.append(x_tick)
                                # except IndexError:
                                #     x_tick = 0
                                # plot
                                try:
                                    p = self.ax[y_id, x_id].bar(
                                        x_ticks,
                                        bar_val_view,  # label=x_labels,
                                        bottom=bottom,
                                        width=barwidth,
                                        align="center",
                                        color=cmap,
                                        hatch=bar_hatch,
                                    )  # self.ax[y_id, x_id].bar_label(p, labels=[label],  #                          fmt='%s',  #                          label_type='center')
                                except IndexError:
                                    p = self.ax[y_id].bar(
                                        x_ticks,
                                        bar_val_view,  # label=x_labels,
                                        bottom=bottom,
                                        width=barwidth,
                                        align="edge",
                                        color=cmap,
                                        hatch=bar_hatch,
                                    )  # self.ax[y_id, x_id].bar_label(p, labels=[label],  #                          fmt='%s',  #                          label_type='center')  # finally:  # bottom += y_val
                                # try:
                                #     self.fig.subplots_adjust(left=0.2, bottom=0.2)
                                # except IndexError:
                                #     self.ax[y_id].subplots_adjust(left=0.2, bottom=0.2)
                                if plot_table is True:
                                    # y_id += 1
                                    # print(tab_colours)
                                    logger.info(f"Has table, {y_id}")
                                    has_table = False
                                    try:
                                        print("rows", tab_rows)
                                        tab_colours = np.array(tab_colours)
                                        print("colours", tab_colours)
                                        # self.tabax[y_id, x_id].clear()
                                        the_table: table = table(
                                            self.ax[y_id, x_id],
                                            cellText=tab_text,
                                            rowLabels=tab_rows,
                                            rowColours=tab_colours,
                                            colLabels=tab_cols,
                                            loc="bottom",
                                            edges="horizontal",
                                        )
                                        has_table = True
                                    except IndexError:
                                        try:
                                            # self.tabax[y_id].clear()
                                            the_table: table = table(
                                                self.ax[y_id],
                                                cellText=tab_text,
                                                rowLabels=tab_rows,
                                                rowColours=tab_colours,
                                                colLabels=tab_cols,
                                                loc="bottom",
                                                edges="horizontal",
                                            )
                                            has_table = True
                                        except ValueError:
                                            logger.info("could not plot table")
                                    except ValueError:
                                        logger.info("could not plot table")
                                    if has_table is True:
                                        logger.info("Plotting table")
                                        # renderer = self.fig.canvas.get_renderer()
                                        the_table.auto_set_font_size(False)
                                        the_table.set_fontsize(12)
                                        h = the_table.get_celld()[
                                            (0, 0)
                                        ].get_height()
                                        w = the_table.get_celld()[
                                            (0, 0)
                                        ].get_width()
                                        table_cells = the_table.properties()[
                                            "children"
                                        ]
                                        for cell in table_cells:
                                            max_width = w
                                            if cell.get_width() > max_width:
                                                max_width = cell.get_width()
                                            # print(cell.get_y())
                                            # cell.set_y(5)
                                            # print(cell.get_y())
                                            cell.set_width(barwidth + xpad)
                                            cell.set_height(0.13)
                                            # cell.set_fontsize(12)
                                            cell.set_text_props(
                                                verticalalignment="center"
                                            )
                                        for cell in table_cells:
                                            cell.set_width(
                                                max_width
                                            )  # the_table.set_fontsize(tab_fontsize)  # has_table = True  # the_table.scale(1, 1.2)  # the_table.stale = True  # the_table.set_clip_on(False)  # the_table._update_positions(renderer)  # bbox = the_table.get_window_extent(renderer)  # the_table.set(clip_bbox=bbox)  # self.ax[y_id, x_id].stale = True  # renderer = self.fig.canvas.get_renderer()  #  #  # # text_height = the_table._approx_text_height()  #  # # the_table.auto_set_font_size(False)  # # new_font_size = the_table.FONTSIZE  # # the_table.FONTSIZE = tab_fontsize  # the_table.set_fontsize(tab_fontsize)  #  # cell = the_table[0, 0]  # # cell_height = cell.get_text()  # win_ext = the_table.get_window_extent(renderer=renderer)  # text_height = cell.get_text_bounds(renderer=renderer)[-1]  # tab_font = the_table.FONTSIZE  # text_height = cell.get_text_bounds()  # # the_table._do_  # tab_font_new = tab_fontsize  # # sys.exit(1)  # # text_height = cells.get_text_bounds()  #  # font_ratio = tab_fontsize / the_table.FONTSIZE  #  # # # the_table.set_fontsize(tab_font_new)  # # # tab_dict = the_table.get_celld()  # #  # # # for col_id in range(0, len(tab_cols)):  # # #     tab_dict[(0, col_id)].set_height(text_height + 0.1)  # # #     for row_id in range(1, len(tab_rows)):  # # #         tab_dict[(row_id, col_id)].set_height(text_height + 0.1)  # the_table.scale(1, font_ratio + 0.3)  # the_table._stale = True  # the_table[...].set_text_props(linespacing=1.5,  #                          verticalalignment='bottom')  # the_table.scale(1, 1.2)  # the_table.stale = True  # the_table.set_clip_on(False)  # the_table._update_positions(renderer)

                                    else:
                                        logger.info(
                                            "No table"
                                        )  # else:  # print(y_val)
                        x_label_ticks = np.mean(
                            np.array(x_label_ticks), axis=0
                        )
                        # self.ax[y_id, x_id].legend(ncol=2, loc='lower center')#[leg for leg in legends[i, j]], ncol=3)
                        # if xlim != None:
                        # self.ax[i, j].set_xlim((-xpad, self.bars.l * (barwidth + xpad)))
                        # self.ax[i, j].set_xticks([], [])
                        #                 if ylim != None:
                        # self.ax[i, j].set_ylim((0.0, 1.2))
                        self.ax[y_id, x_id].set_xticks(
                            np.ravel(x_label_ticks), np.ravel(x_labels)
                        )
                    # print("table false")
                    # sys.exit(1)
                    # tab_dict = the_table.get_celld()
                    # table = ax[y_id, x_id].table(cellText=table_text,
                    #                              rowColours=table_col,
                    #                              rowLabels=table_rows,
                    #                              # colLables=...,
                    #                              loc='bottom')
                    # x_ticks = x_ticks[:-1]
                    # x_ticks.append(bar_dict[vbar] * (barwidth + xpad))
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
                        logger.info(f"No data for {pl}, {self.x}, {it}")

            # x_ticks = [np.linspace(n_bar * bulk_edge + xpad,
            #                        n_bar * bulk_edge + bulk_edge, int(bulk_edge)) for n_bar in range(lbars)]
            # x_ticks = np.ravel(x_ticks)
            # x_labels = np.tile(np.arange(0, bulk_edge, 1), lbars)
            for i in range(self.y.l):
                # if plot_table is True:
                #     ax_multi = 2
                # else:
                #     ax_multi = 1
                # ax_i = i * ax_multi
                # print(f"Axis index: {ax_i}, multi: {ax_multi}")

                try:
                    # self.ax[]
                    # self.ax[i, 0].set_ylabel(
                    #     f"{self.label_mod([(self.y.v[i], self.y)])}\n" + rowlabel
                    # )
                    for j in range(self.x.l):
                        # if plot_table is True:
                        #     self.ax[i, j].subplots_adjust(bottom=0.2)
                        #     f"{self.label_mod([(self.y.v[i], self.y)])}\n" + rowlabel
                        # )
                        if j == 0:
                            self.ax[i, 0].set_ylabel(
                                f"{self.label_mod([(self.y.v[i], self.y)])}\n"
                                + rowlabel
                            )
                            self.ax[i, j].set_yticks(np.arange(0.0, 1.1, 0.2))
                        else:
                            self.ax[i, j].set_yticks(
                                np.arange(0.0, 1.1, 0.2)
                            )  # , label="")
                            self.ax[i, j].set_yticklabels(
                                []
                            )  # np.arange(0.0, 1.1, 0.2))
                        self.ax[i, j].spines[["top", "right"]].set_visible(
                            False
                        )
                        # self.ax[i, j].hlines(
                        #     1.0,
                        #     -xpad,
                        #     self.bars.l * (barwidth + xpad) + xpad,
                        #     linestyle="--",
                        # )
                        # legend = [matplotlib.patches.Patch(color = 'black',
                        #                                    hatch=hatch_list[xi],
                        #                                    label=x) for xi, x in enumerate(self.bars.v)]
                        # self.ax[i, j].legend(ncol = 2, handles = legend)
                        # self.ax[i, j].legend(ncol=2, loc='lower center')#[leg for leg in legends[i, j]], ncol=3)
                        #                  if xlim != None:
                        # self.ax[i, j].set_xlim((-xpad, self.bars.l * (barwidth + xpad)))
                        # self.ax[i, j].set_xticks([], [])
                        #                 if ylim != None:
                        self.ax[i, j].set_ylim((0.0, 1.1))
                        # self.ax[self.y.l - 1, j].set_xticks(
                        #     np.array(x_ticks) + 0.5 * barwidth, x_labels
                        # )
                        self.ax[self.y.l - 1, j].set_xlabel(
                            "atom types"
                            + f"\n{self.label_mod([(self.x.v[j], self.x)])}"
                        )  # self.ax[self.y.l-1, j].subplots_adjust(bottom=0.1)  # self.ax[i, j].set_yticklabels(np.arange(0.0, 1.1, 0.2))
                except IndexError:
                    self.ax[i].set_ylabel(
                        f"{self.label_mod([(self.y.v[i], y)])}\n" + rowlabel
                    )
                    # self.ax[i].legend([self.label_mod([(leg, self.label_key)]) for leg in self.legends[i, 0]], ncol=3)
                    self.ax[self.y.l - 1].set_xlabel(
                        columnlabel
                        + f"\n{self.label_mod([(self.x.v[0], self.x)])}"
                    )
            # # #
            legend_dict = {
                x: matplotlib.patches.Patch(
                    edgecolor="black",
                    hatch=hatch_list[xi],
                    label=x,
                    fill=False,
                )
                for xi, x in enumerate(self.bars.v)
            }
            self.fig.legend(
                ncol=2,
                handles=list(legend_dict.values()),
                labels=list(legend_dict.keys()),
                loc="outside upper center",
                title=bars,
                frameon=False,
            )
            self.fig.supxlabel(f"{self.title_dict[x]}s", size=14)
            self.fig.supylabel(f"{self.title_dict[y]}s", size=14)
            if plsave != False:
                logger.info("Saving plot")
                if type(plsave) == str and plsave != "":
                    outname = f"{self.atomname}_{plsave}.png"
                else:
                    outname = f"{self.atomname}_{self.name}.png"
                odir = Path(odir).absolute()
                logger.info(f"output to {odir.absolute()}")
                if not odir.is_dir():
                    os.makedirs(odir)
                logger.info(odir)
                logger.info(outname)
                self.fig.savefig(str(odir / outname))
            else:
                plt.show()
            self.fig.clear()

    def _get_bin_label(self, x_bin, bin_list):
        # if x_bin.right < np.max(bin_list.right):
        #     label = f"${x_bin.left:>3.1f} - {x_bin.right:3.1f}$ \AA"
        #     # barwidth = x_bin.right - x_bin.left
        #
        # else:
        #     label = f"$ > {x_bin.left:>3.1f}$ \AA"
        return x_bin


class HistPlot2D(Plot):
    # __slots__ = ()
    sel_list = ["clays", "ions", "aas", "_atoms"]
    title_dict = {
        "clays": "Clay type",
        "ions": "Ion type",
        "aas": "Amino acid",
        "atoms": "Atom type",
        "other": "Other atom type",
    }
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
    _init_args = {"x", "y", "bars"}
    _plot_args = {
        "rowlabel": "y",
        "columnlabel": "x",
        "dpi": None,
        "figsize": None,
    }

    def __init__(self, data, sdir=None, **kwargs):
        logger.info(f"Initialising {self.__class__.__name__}")
        super().__init__(data, **kwargs)
        self.bins = "x_bins"
        self.group = "atoms"
        self.bars = None

    def plot_args_decorator(select: str):
        def plot_decorator(plot_f):
            def wrapper(self, **kwargs):
                self.select = select
                plot_f(self, **kwargs)
                self.select = None

            return wrapper

        return plot_decorator

    # @plot_args_decorator(select="other")
    # def plot_other(
    #     self,
    #     bars: Literal["clays", "aas", "ions", "other"],
    #     x: Literal["clays", "aas", "ions", "other"],
    #     y: Literal["clays", "aas", "ions", "other"],
    #     col_sel: str,
    #     edges: Tuple[float, float] = None,
    #     nbins: int = None,
    #     rowlabel: str = "y",
    #     columnlabel: str = "x",
    #     figsize=None,
    #     dpi=None,
    #     xmax=50,
    #     ymax=50,
    #     plsave=False,
    #     ylim=None,
    #     odir=".",
    #     barwidth=0.75,
    #     xpad=0.25,
    #     cmap="winter",
    #     plot_table=None,
    #     get_new_data=False,
    # ):
    #     self.plot(
    #         bars,
    #         x,
    #         y,
    #         col_sel,
    #         edges,
    #         nbins,
    #         rowlabel,
    #         columnlabel,
    #         figsize,
    #         dpi,
    #         xmax,
    #         ymax,
    #         plsave,
    #         ylim,
    #         odir,
    #         barwidth,
    #         xpad,
    #         cmap,
    #         get_new_data=get_new_data,
    #     )

    # def get_suptitle(self, pl, separate):
    #     logger.info(self.title_dict)
    #     self.fig.suptitle(
    #         (
    #             ", ".join([self.title_dict[s].upper() for s in separate])
    #             + f": {self.label_mod(list(tuple(zip(pl, separate))))}"
    #         ),
    #         size=16,
    #         weight="bold",
    #     )

    # @redirect_tqdm
    # def plot(
    #     self,
    #     bars: Literal["clays", "aas", "ions", "other"],
    #     x: Literal["clays", "aas", "ions", "other"],
    #     y: Literal["clays", "aas", "ions", "other"],
    #     col_sel: Union[str, List[str]],
    #     edges: Tuple[float, float] = None,
    #     nbins: int = None,
    #     rowlabel: str = "y",
    #     columnlabel: str = "x",
    #     figsize=None,
    #     dpi=None,
    #     # diff=False,
    #     xmax=50,
    #     ymax=50,
    #     plsave=False,
    #     ylim=None,
    #     odir=".",
    #     barwidth=0.75,
    #     xpad=0.25,
    #     cmap="winter",
    #     tab_fontsize=12,
    #     plot_table=False,
    #     get_new_data=True,
    # ):
    #     """Create stacked Histogram adsorption shell populations."""
    #
    #     logger.info(f"plot table {plot_table}")
    #
    #     self._init_plot(x, y, bars)
    #     self.col_sel = col_sel
    #     self._process_plot_data(
    #         get_new_data=get_new_data,
    #         col_sel=self.col_sel,
    #         v_edges=edges,
    #         v_nbins=nbins,
    #     )

    @plot_args_decorator(select="ions")
    def plot_ions(
        self,
        bars: Literal["clays", "aas", "ions", "other"],
        x: Literal["clays", "aas", "ions", "other"],
        y: Literal["clays", "aas", "ions", "other"],
        col_sel,
        edges: Tuple[float, float] = None,
        nbins: int = None,
        rowlabel: str = "y",
        columnlabel: str = "x",
        figsize=None,
        dpi=None,
        xmax=50,
        ymax=50,
        plsave=False,
        ylim=None,
        odir=".",
        barwidth=0.75,
        xpad=0.25,
        cmap="winter",
        plot_table=False,
        get_new_data=False,
    ):
        self.plot(
            bars,
            x,
            y,
            col_sel,
            edges,
            nbins,
            rowlabel,
            columnlabel,
            figsize,
            dpi,
            xmax,
            ymax,
            plsave,
            ylim,
            odir,
            barwidth,
            xpad,
            cmap,
            plot_table=plot_table,
            get_new_data=get_new_data,
        )

    @plot_args_decorator(select="other")
    def plot_other(
        self,
        bars: Literal["clays", "aas", "ions", "other"],
        x: Literal["clays", "aas", "ions", "other"],
        y: Literal["clays", "aas", "ions", "other"],
        col_sel: str,
        edges: Tuple[float, float] = None,
        nbins: int = None,
        rowlabel: str = "y",
        columnlabel: str = "x",
        figsize=None,
        dpi=None,
        xmax=50,
        ymax=50,
        plsave=False,
        ylim=None,
        odir=".",
        barwidth=0.75,
        xpad=0.25,
        cmap="winter",
        plot_table=None,
        get_new_data=False,
    ):
        self.plot(
            bars,
            x,
            y,
            col_sel,
            edges,
            nbins,
            rowlabel,
            columnlabel,
            figsize,
            dpi,
            xmax,
            ymax,
            plsave,
            ylim,
            odir,
            barwidth,
            xpad,
            cmap,
            get_new_data=get_new_data,
        )

    def get_suptitle(self, pl, separate):
        logger.info(self.title_dict)
        self.fig.suptitle(
            (
                ", ".join([self.title_dict[s].upper() for s in separate])
                + f": {self.label_mod(list(tuple(zip(pl, separate))))}"
            ),
            size=16,
            weight="bold",
        )

    @redirect_tqdm
    def plot(
        self,
        bars: Literal["clays", "aas", "ions", "other"],
        x: Literal["clays", "aas", "ions", "other"],
        y: Literal["clays", "aas", "ions", "other"],
        col_sel: Union[str, List[str]],
        edges: Tuple[float, float] = None,
        nbins: int = None,
        rowlabel: str = "y",
        columnlabel: str = "x",
        figsize=None,
        dpi=None,  # diff=False,
        xmax=50,
        ymax=50,
        plsave=False,
        ylim=None,
        odir=".",
        barwidth=0.75,
        xpad=0.25,
        cmap="winter",
        tab_fontsize=12,
        plot_table=False,
        get_new_data=True,
    ):
        """Create stacked Histogram adsorption shell populations."""

        logger.info(f"plot table {plot_table}")

        self._init_plot(x, y, bars)
        self.col_sel = col_sel
        self._process_plot_data(
            get_new_data=get_new_data,
            col_sel=self.col_sel,
            v_edges=edges,
            v_nbins=nbins,
        )
        plot_df = self.plot_df.copy()
        # plot_df.sort_index(inplace=True, sort_remaining=True)
        colour_df = self.colour_df.copy()
        # colour_df.sort_index(inplace=True, sort_remaining=True)

        avail_attrs = plot_df.index.names
        logger.info(f"Available attributes: {avail_attrs}")

        logger.info("Setting x:")
        self.x = Axis(self, x)
        logger.info("Setting y:")
        self.y = Axis(self, y)
        logger.info("Setting bars:")
        self.bars = Axis(self, bars)

        separate = [
            s
            for s in avail_attrs
            if s not in [self.x, self.y, self.bars, self.bins]
        ]

        if len(separate) != 0:
            logger.info(f"Separate plots: {separate}")

        idx = pd.Index(
            [
                s
                for s in avail_attrs
                if (
                    s != self.x
                    and s != self.bars
                    and s not in [*separate, self.bins]
                )
            ]
        )

        logger.info(f"Iteration index: {idx.values}")

        sep = pd.Index(separate)
        logger.info(f"{sep}")

        yid = np.ravel(np.where(np.array(idx) == self.y))[0]

        if figsize is None:
            if plot_table is True:
                figsize = self.get_figsize(xmax=xmax, ymax=ymax * 2)
            else:
                figsize = self.get_figsize(xmax=xmax, ymax=ymax)

        # set resultion
        if dpi is None:
            dpi = 100

        # get plotting iter from index
        iters = self._get_idx_iter(idx=idx, plot_df=plot_df)
        plot_df.sort_index(
            level=[*sep, *idx.values], sort_remaining=True, inplace=True
        )
        # colour_df.sort_index(level=)
        logger.info(
            f"Printing bar plots for {self.bars}\nColumns: {self.x.v}\nRows: {self.y.v}"
        )

        # set label modifier function
        self.label_mod = self.modify_plot_labels

        try:
            # iterator for more than one plot
            sep_it = self._get_idx_iter(idx=sep, plot_df=plot_df)
            logger.info(f"{sep_it}")
        except ValueError:
            # only one plot
            sep_it = [None]
            logger.info(f"{sep_it}")

        # iterate over separate plots
        for pl in sep_it:
            if pl is None:
                pl_str = ""
            else:
                pl_str = f"_{pl[0]}"
            self.savename = f"{self.name}_{self.data.analysis}_{x}_{y}_{bars}{pl_str}_{self.data.cutoff}_{self.data.bins}"
            logger.info(f"plot {self.savename}")

            self.init_legend(ly=self.y.l, lx=self.x.l)

            # generate figure and axes array
            self.fig = plt.figure(
                figsize=figsize, dpi=dpi, constrained_layout=True
            )

            self.gs = gridspec.GridSpec(
                nrows=self.y.l,
                ncols=self.x.l,
                figure=self.fig,  # width_ratios=[2,1]
                # sharey=True,
                # sharex=True
            )

            self.ax = {}

            for i in range(self.y.l):
                for j in range(self.x.l):
                    self.ax[i, j] = self.fig.add_subplot(self.gs[i, j])

            # only one plot
            if pl is None:
                logger.info(f"Generating plot")
                sepview = plot_df
                plsave = ""

            else:
                logger.info(f"Generating {pl} plot:")
                # print(plot_df.head(20), "\n", separate, pl)
                sepview = plot_df.xs((pl), level=separate, drop_level=False)
                plsave = pl
                self.get_suptitle(pl, separate)

            # set plot index
            pi = 0

            # iterate over subplot columns
            for col in self.x.v:
                logger.info(f"Column {col}")

                try:
                    view = sepview.xs((col), level=self.x, drop_level=False)
                    pi = 1
                except ValueError:
                    view = sepview
                    col = self.x.v
                    pi += 1

                for it in iters:
                    try:
                        values = view.xs(
                            tuple(it), level=idx.tolist(), drop_level=False
                        )
                        # print(values.columns)
                        x_labels = []
                        x_ticks = []
                        # print(self.bars.v)
                        for bar_num, vbar in enumerate(self.bars.v):
                            logger.info(f"Bar {bar_num}: {vbar}")
                            tab_colours = []
                            tab_rows = []
                            if plot_table is True and bar_num == 0:
                                try:
                                    tab_text = values.values.reshape(
                                        (self.bars.l, -1)
                                    ).T
                                except ValueError:
                                    tab_text = values.values
                                # tab_text = tab_text.astype(str)
                                tab_format = lambda x: rf"${x * 100:3.1f} \%$"
                                tab_format_vec = np.vectorize(tab_format)
                                tab_text = tab_format_vec(tab_text)
                                tab_cols = list(
                                    self.bars.v[
                                        : np.atleast_2d(tab_text).shape[0]
                                    ]
                                )
                            print(self.bars.dict[vbar])
                            print(barwidth, xpad)
                            x_ticks.append(
                                self.bars.dict[vbar] * (barwidth + xpad)
                            )
                            x_labels.append(vbar)
                            bottom = 0.0
                            bar_vals = values.xs(
                                vbar, level=self.bars, drop_level=False
                            )
                            try:
                                col_1, col_2 = self.name.split("_")
                                hist = bar_vals[col_1].make_hist2d(
                                    other=bar_vals[col_2]
                                )
                            except:
                                bins = self.v_axis.bins
                                z_bins = self.z_axis.bins
                                bar_vals = np.ravel(bar_vals.values)[0]
                                print(bar_vals.__dict__)
                                print(bar_vals.timeseries)
                                print(bar_vals.zdata)
                                hist = bar_vals.make_hist2d(
                                    bins=self.v_axis.bins,
                                    z_bins=self.z_axis.bins,
                                )

                                # timeseries_z = bar_vals["timeseries_z"]  # bins_z = bar_vals["timeseries_z"]["edges_z"]  # timeseries = bar_vals["timeseries_v"]

                            # tab_text.append(bar_vals.values)
                            #
                            cmap = colormaps[
                                colour_df.loc[bar_vals.index].values[0]
                            ]
                            #
                            peaks = self.data.get_bar_peaks(
                                bar_vals=bar_vals, atom_type=col
                            )
                    #
                    #     if np.all(bar_vals.values) >= 0:
                    #         # table_text.append([f'${v * 100:3.1f} %$' for v in bar_vals.values])
                    #         x_id, y_id = self.x.dict[col], self.y.dict[it[yid]]
                    #         # if plot_table is True:
                    #         #     # logger.info(y_id)
                    #         #     logger.info("Has table")
                    #         #     y_id *= 2
                    #         #     logger.info(y_id)
                    #         # print(y_id)
                    #
                    #         bar_val_view = bar_vals
                    #         bin_list = bar_val_view.index.get_level_values("x_bins")
                    #         bar_val_view.index = bin_list
                    #
                    #         x_tick = x_ticks[-1]
                    #
                    #         for bar_id, bar_val in enumerate(bar_val_view.items()):
                    #
                    #             x_bin, y_val = bar_val
                    #
                    #             try:
                    #                 peak = peaks[bar_id]
                    #             except IndexError:
                    #                 peak = x_bin.right
                    #             colour = cmap(self.cnorm(peak))
                    #
                    #             # if colour not in table_col and cmap == table_cmap:
                    #             #     print('colour', colour)
                    #             #     table_col.append(colour)
                    #
                    #             label = self._get_bin_label(x_bin, bin_list)
                    #
                    #             if bar_num == 0:
                    #                 tab_colours.append(colour)
                    #                 tab_rows.append(label)
                    #             # if x_bin.right < np.max(self.x):
                    #             #     label = f'${x_bin.left} - {x_bin.right}$ \AA'
                    #             # else:
                    #             #     label = f'$ > {x_bin.left}$ \AA'
                    #             # if label not in table_rows and cmap == table_cmap:
                    #             #     table_rows.append(label)
                    #             if y_val >= 0.010:
                    #                 # barwidth = bulk_edge - x_bin.left
                    #                 # try:
                    #                 # x_tick = x_ticks[-1] + barwidth
                    #                 # x_ticks.append(x_tick)
                    #                 # except IndexError:
                    #                 #     x_tick = x_bin.left
                    #                 try:
                    #                     p = self.ax[y_id, x_id].bar(
                    #                         x_tick,
                    #                         y_val,
                    #                         label=label,
                    #                         bottom=bottom,
                    #                         width=barwidth,
                    #                         align="edge",
                    #                         color=colour,
                    #                     )
                    #                     # self.ax[y_id, x_id].bar_label(p, labels=[label],
                    #                     #                          fmt='%s',
                    #                     #                          label_type='center')
                    #                 except IndexError:
                    #                     p = self.ax[y_id].bar(
                    #                         x_tick,
                    #                         y_val,
                    #                         label=label,
                    #                         bottom=bottom,
                    #                         width=barwidth,
                    #                         align="edge",
                    #                         color=colour,
                    #                     )
                    #                     # self.ax[y_id, x_id].bar_label(p, labels=[label],
                    #                     #                          fmt='%s',
                    #                     #                          label_type='center')
                    #                 # finally:
                    #                 bottom += y_val
                    except:
                        print("except")

    def save(self):
        ...

    def df_apply(self, dfs, function):
        return [function(df) for df in dfs]

    def _process_data(self, sdir=None, **kwargs):
        self.sdir = sdir

    @redirect_tqdm
    def _process_plot_data(
        self,
        col_sel: Union[List[str], str],
        col_abs: Optional[Union[Tuple[bool, bool], bool]] = None,
        z_edges: Optional[Tuple[float, float]] = None,
        v_edges: Optional[Tuple[float, float]] = None,
        z_nbins: Optional[int] = None,
        v_nbins: Optional[int] = None,
        get_new_data=False,
        # df_name: Union[Literal["df"], Literal["zf"]] = "df",
        **kwargs,
    ):
        # data save directory
        while self.sdir is None:
            try:
                sdir = Path.home() / ".data"
            except ValueError:
                pass
            else:
                self.sdir = sdir
        else:
            sdir = Path(self.sdir)
        if not sdir.is_dir():
            os.makedirs(sdir)
        # list of matplotlib sequential cmaps
        cmaps = [  # 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
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
        # df_name='df'
        # for df_name in ['df', 'zf']:
        # logger.info(f'df name {df_name}')
        # plot_df = getattr(self.data, df_name).copy()
        logger.info(f"Processing {self.__class__.__name__} data")
        plot_df = self.data.df
        # logger.info(f"{df_name}, {plot_df.index.names}")
        idx_names = plot_df.index.names
        atoms = plot_df.index.get_level_values("_atoms")
        plot_df["colours"] = atoms
        atom_types = atoms.unique()
        colour_dict = dict(zip(atom_types, cmaps[: len(atom_types)]))
        print(plot_df)
        plot_df["colours"] = plot_df["colours"].transform(
            lambda x: colour_dict[x]
        )
        print(plot_df)
        # reorder index for grouping
        colour_df = plot_df.pop("colours")
        self._colour_df = colour_df
        plot_df = plot_df.reorder_levels(idx_names)

        # check column selection for values df
        # if df_name == 'df':
        # only one column
        if type(col_sel) == list and len(col_sel) == 1:
            col_sel = col_sel[0]
        if type(col_sel) == str:
            assert (
                col_sel in plot_df.columns
            ), f"Selected column {col_sel} does not exist"
            self.name = col_sel
            self.other_col = None  # plot_col = [plot_df[col_sel]]
        # 2 columns -> ratio between columns (col1 / col2)
        else:
            assert (
                type(col_sel) == list and len(col_sel) == 2
            ), f"Found wrong length ({len(col_sel)}) for plotting df columns"
            assert np.all(
                [c in plot_df.columns for c in col_sel]
            ), f"One of {col_sel} is not in plotting df columns"
            self.name = "_".join(col_sel)
            self.other_col = col_sel[
                -1
            ]  # plot_col = [plot_df[col_sel[0]], plot_df[col_sel[1]]]
        col_len = len(self.name.split("_"))
        # check if absolute values of columns should be used
        # default is absolute
        if col_abs is None:
            self.col_abs = tuple([True for col in self.name.split("_")])
        elif type(col_abs) == bool:
            self.col_abs = [col_abs]
            assert (
                len(col_abs) == col_len
            ), f"Length for absolute value specifications ({len(col_abs)}) does not match length of columns ({col_len})"

            # if df_name == 'zf':  #     # _plot_zf = plot_df.copy()  #     _plot_zf = pd.Series(index=plot_df.index, name='timeseries', dtype='object')  #     print(plot_df)  #     _plot_zf = plot_df[self.data.zname]  #     print(_plot_zf)  #     # _plot_zf.index = _plot_zf.index.to_flat_index()  # _plot_zf = dask.dataframe.from_pandas(_plot_zf, npartitions=5)
        logger.info("Setting z-density bins")
        if z_edges is None and z_nbins is None:
            self.z_axis = HistData(bins=self.data.z_bins.bins)
        else:
            self.z_axis = HistData(
                min=z_edges[0], max=z_edges[1], nbins=z_nbins
            )
        logger.info(f"{self.z_axis}")
        # save = (
        #         sdir
        #         / f"{df_name}_{self.data.cutoff}_{self.data.bins}.tar.xz"
        # )
        # if not save.is_file() or (save.is_file and get_new_data is True):
        #     for idx, row in plot_df[self.data.zname].items():
        #         fname = row
        #         with open(fname, "rb") as f:
        #             data = pkl.load(f)
        # _plot_zf.assign(timeseries=lambda x: [np.ravel(pkl.load(open(x[self.zname], 'rb')).timeseries)])
        #         _plot_zf.loc[idx] = _plot_zf.loc[idx].apply(lambda x: Timeseries(filename=x))
        # logger.info(f"Saving {save!r}")
        # _plot_zf.to_pickle(save)
        # else:
        #     logger.info(f"Reading {save!r}")
        #     _plot_zf = pd.read_pickle(save)
        # plot_df = _plot_zf.copy()
        # del _plot_zf
        # else:
        # _plot_df = plot_df.copy()
        # _plot_df.index = _plot_df.index.to_flat_index()
        # _plot_df = dask.dataframe.from_pandas(_plot_df,
        #     index=plot_df.index.to_flat_index(),
        # _plot_df = pd.DataFrame(
        #     index=plot_df.index,
        #     columns=[
        #         f"{x}_{y}"
        #         for x in ["timeseries"]
        #         for y in self.name.split("_")
        #     ]
        # )
        #     npartitions=10)
        logger.info(f"Setting histogram {self.name} bins")
        if v_edges is not None and v_nbins is not None:
            self.v_axis = HistData(
                min=v_edges[0], max=v_edges[1], nbins=v_nbins
            )
        elif col_len == 1:
            print(self.data.filelist[self.name][-1])
            with open(self.data.filelist[self.name][-1], "rb") as file:
                data = pkl.load(file)
            print(data)
            sys.exit(1)
            self.v_axis = HistData(edges=data.edges)
        elif col_len == 2 and v_nbins is not None:
            self.v_axis = HistData(min=0, max=1, nbins=v_nbins)
        else:
            self.v_axis = HistData(
                min=0,
                max=1,
                nbins=HistData(
                    bins=self.data.other_bins[self.col_sel[0]].bins
                ).nbins,
            )  # raise ValueError(f"No bins specified")
        logger.info(f"{self.v_axis}")
        # for col_name in name.split('_'):
        # save = (
        #         sdir
        #         / f"{col_name}_{self.data.cutoff}_{self.data.bins}.tar.xz"
        # )
        # if not save.is_file() or (save.is_file() and get_new_data is True):
        # col_ts = f"timeseries_{col_name}"
        # col = plot_df[col_name]
        # _plot_df[col_ts] = _plot_df[col_name].map(lambda x: pkl.load(open(Path(x), 'rb')),
        #                                           meta=pd.Series(index=_plot_df[col_name].index, dtype='object')).compute()
        # _plot_df[col_ts] = _plot_df[col_ts].map(lambda x: [da.array.ravel(pkl.load(open(Path(x), 'rb')).timeseries)],
        #                                           meta=pd.Series(index=_plot_df[col_name].index, dtype='object')).compute()
        # _plot_df.assign(col_ts=lambda x: [np.ravel(pkl.load(open(x[col_name], 'rb')).timeseries)])
        # col = col.apply(lambda x: Timeseries(filename=x))
        # for idx, row in col.items():
        #     fname = row
        #     with open(fname, "rb") as f:
        #         data = pkl.load(f)
        # slice = _plot_df.xs(idx, drop_level=False)#[
        # pd.IndexSlice[
        #     f"timeseries_{col.name}"
        # ]]
        # ]_plot_df.loc[
        #     idx,
        #     pd.IndexSlice[
        #         f"timeseries_{col.name}"
        #     ],
        # ]
        # z_idx = slice.index.intersection(self.data.zf.index)
        # print(z_idx, self.data.zf.index.names, slice.index.names)
        # sel_id = self.data.zf.loc(z_idx)
        # print(sel_id)
        # _plot_df.loc[
        #             idx,
        #             pd.IndexSlice[
        #                 f"timeseries_{col.name}"
        #             ],
        #         ].map(lambda x: setattr(x, 'zdata', self.zf.loc[z_idx])) #
        # = [np.ravel(data.timeseries)]
        # logger.info(f"Saving {save.name!r}")
        # _plot_df.to_pickle(save)
        # else:
        #  logger.info(f"Reading {save.name!r}")
        # _plot_df = pd.read_pickle(save)

        # if col_len != 1:
        #     # arr1 = dask.array.from_array(_plot_df.iloc[:, 0].values)
        #     # arr2 = dask.array.from_array(_plot_df.iloc[:, 1].values)
        #     arr1 = _plot_df.iloc[:, 0].values
        #     arr2 = _plot_df.iloc[:, 1].values
        #     arr = np.divide(arr1, arr2, where=arr2 != 0)
        #     _plot_df["timeseries"] = arr
        # else:
        #     _plot_df["timeseries"] = _plot_df[f"timeseries_{name}"].values
        #
        # plot_df = _plot_df.copy()
        # del _plot_df

        plot_df = plot_df.loc[..., self.col_sel].copy()
        # setattr(self, f"_plot_df", plot_df)

        # plot_df = self._plot_df['timeseries'].copy()
        # plot_df = plot_df.join(self._plot_zf, how='inner', lsuffix='_v', rsuffix='_z')
        colour_df = self._colour_df

        dfs = {"bin": plot_df, "colour": colour_df}

        # split DataFrames for ions and other atom types
        for key, df in dfs.items():
            # get non-ion atom type names
            not_ions = [a for a in self.data.atoms if a != "ions"]
            # get only ion DataFrame slice
            ion_view = df.xs("ions", level="atoms")
            # remove ions from index
            ion_view.index = ion_view.index.droplevel(["ions"])
            # rename _atoms atom type column to replace generic atoms column
            idx_names = [idstr.strip("_") for idstr in ion_view.index.names]
            ion_view.index.names = idx_names
            # get only non-ion DataFrame slice
            other_view = df.loc[pd.IndexSlice[:, :, :, :, not_ions, :]]
            # remove unnecessary _atoms column from index
            other_view.index = other_view.index.droplevel("_atoms")
            # set dataframes as attributes
            setattr(self, f"_ions_{key}_df", ion_view.copy())
            setattr(self, f"_other_{key}_df", other_view.copy())

    # else:
    #     with open(savename)

    def __repr__(self):
        return self.plot_df.index.__repr__()

    @property
    def plot_df(self):
        select = self.select
        if not hasattr(self, f"_{select}_bin_df"):
            raise Exception(
                f"Plot data for {select} not processed."
            )  # self._process_plot_data(col_sel=self.col_sel)
        logger.info(self.select)

        return getattr(self, f"_{select}_bin_df")

    @property
    def colour_df(self):
        select = self.select
        if not hasattr(self, f"_{select}_colour_df"):
            raise Exception(
                f"Plot data for {select} not processed."
            )  # self._process_plot_data(col_sel=self.col_sel)
        return getattr(self, f"_{select}_colour_df")

    def _get_idx_iter(self, idx, plot_df):
        logger.info(f"idx: {idx}")
        idx_iter = np.array(
            np.meshgrid(
                *[
                    plot_df.index.get_level_values(idxit).unique()
                    for idxit in idx
                ]
            )
        ).T.reshape(-1, len(idx))
        logger.info(f"{idx_iter}")
        return idx_iter

    def _get_attr(self, attr: Literal["clays", "aas", "atoms", "ions"]):
        return pd.unique(self.plot_df.index.get_level_values(attr))

    get_clays = partialmethod(_get_attr, attr="clays")
    get_aas = partialmethod(_get_attr, attr="aas")
    get_ions = partialmethod(_get_attr, attr="ions")
    get_atoms = partialmethod(_get_attr, attr="atoms")
    get_other = partialmethod(_get_attr, attr="other")


# class ModTable(matplotlib.table.Table):
#     def __new__(cls, *args, **kwargs):
#         super().FONTSIZE = 12
#         self = super().__init__(*args, **kwargs)
#         # self.FONTSIZE = 12
#         return self
#


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
        z_edges, v_edges = HistData(edges=h_edges[0]), HistData(
            edges=h_edges[1]
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
        try:
            zfile = Path(zfile)
        except TypeError:
            logger.info(f"No z-density file found.")
            self.has_zdata = False
        else:
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


if __name__ == "__main__":
    p = ArgumentParser(allow_abbrev=False)
    group = p.add_mutually_exclusive_group()
    group.add_argument("-lines", action="store_true", default=False)
    group.add_argument("-bars", action="store_true", default=False)
    group.add_argument("-gbars", action="store_true", default=False)
    p.add_argument("-sel", nargs="+", choices=["i", "o"])
    p.add_argument(
        "-odir", type=Path, required=False, dest="odir", default=False
    )
    p.add_argument("-idir", type=Path, required=True, dest="idir")
    # a_parser.add_argument('-odir')
    p.add_argument("-ignore_sum", action="store_true", default=False)
    p.add_argument("-ymin", type=float, default=None, required=False)
    p.add_argument("-ymax", type=float, default=None, required=False)
    p.add_argument("-xmin", type=float, default=None, required=False)
    p.add_argument("-xmax", type=float, default=None, required=False)
    p.add_argument("-ylabel", type=str, default="y", required=False)
    p.add_argument("-xlabel", type=str, default="x", required=False)
    p.add_argument(
        "-cutoff",
        type=int,
        required=True,
    )
    p.add_argument("-bins", type=float, required=True)
    p.add_argument(
        "-use_abs", action="store_false", dest="use_rel", default=True
    )
    p.add_argument("-namestem", type=str)
    p.add_argument("-analysis", type=str)
    p.add_argument("-atoms", nargs="+", type=str, default=None)
    p.add_argument("-aas", nargs="+", type=str, default=None)
    p.add_argument("-ions", nargs="+", type=str, default=None)
    p.add_argument("-clays", nargs="+", type=str, default=None)
    p.add_argument("-other", nargs="+", type=str, default=False)
    p.add_argument("--grouped", action="store_true", default=False)
    p.add_argument("--table", action="store_true", default=False)
    p.add_argument("--edges", action="store_true", default=False)
    p.add_argument("--add_bulk", action="store_true", default=False)
    p.add_argument("-x")
    p.add_argument("-y")
    p.add_argument("-plsel")
    p.add_argument(
        "-colours", nargs="+", type=str, default=None, dest="colours"
    )
    p.add_argument(
        "--no_atomname", action="store_false", default=True, dest="atomname"
    )
    p.add_argument(
        "-figsize", default=None, nargs=2, dest="figsize", type=float
    )
    p.add_argument(
        "-data2d",
        action="store_true",
        default=False,
        dest="data2d",
        required=False,
    )
    p.add_argument("-zdata", type=Path, required=False, dest="zdir")
    atype_parser = p.add_subparsers(  # 'atom_type arguments',
        dest="atypes"
    )  # 'atom type plots')
    a_parser = atype_parser.add_parser("get_data")
    a_parser.add_argument(
        "-new_bins", type=float, default=None, required=False
    )
    a_parser.add_argument("--overwrite", default=False, action="store_true")
    a_parser.add_argument("-datadir", type=Path, required=True, dest="datadir")
    a_parser.add_argument("--load", type=bool, required=False, dest="load")
    a_parser.add_argument("-save", type=Path, default=False, dest="save")

    args = p.parse_args(sys.argv[1:])

    if args.odir is False:
        plsave = False
        odir = None
    else:
        plsave = True
        odir = args.odir

    if args.data2d is True:
        data = Data2D(
            indir=args.idir,
            zdir=args.zdir,
            namestem=args.namestem,
            cutoff=args.cutoff,
            bins=args.bins,
            ions=args.ions,
            aas=args.aas,
            clays=args.clays,
            load=False,
            odir=odir,
            nameparts=1,
            atoms=args.atoms,
            zstem="zdist",
            zname="zdens",
            analyses=[args.analysis],
        )
        atoms = args.atoms
    else:
        atoms = args.atoms
        if args.atypes is not None:
            data = AtomTypeData2D(
                indir=args.datadir,
                cutoff=args.cutoff,
                bins=args.bins,
                new_bins=args.new_bins,
                odir=args.idir,  # odir="/storage/plots/aadist_full",  # _plots",
                namestem=args.namestem,
                nameparts=1,
                ions=args.ions,
                aas=args.aas,
                clays=args.clays,
                atomnames="atype",
                analysis=args.analysis,
                atoms=atoms,
                save=True,  # args.save,
                overwrite=args.overwrite,
                group_all=True,
                load=args.load,  # '/storage/results.p'
            )
            if args.new_bins is not None:
                args.bins = args.new_bins
            atoms = None

        # d = RawData("../test_data/")
        # data = AtomTypeData2D(
        #     "/storage/results/aadist_full",
        #     cutoff=50,
        #     bins=0.02,
        #     new_bins=0.10,
        #     # odir="/storage/plots/aadist_full",  # _plots",
        #     namestem="aadist",
        #     nameparts=1,
        #     # ions=['Na', 'Ca'],
        #     # aas=['his', 'ala'],
        #     # clays=['NAu-1', ],
        #     atomnames='atype',
        #     analysis="aadens",
        #     atoms=['aas'],
        #     save=True,
        #     odir='/storage/aa_data_split',
        # overwrite=True,
        # group_all=True
        # load=True,#'/storage/results.p'
        # aas=["ctl"],
        # )
        data = Data(
            indir=args.idir,  # '/storage/aa_data_split',
            analysis=args.analysis,
            namestem=args.namestem,
            cutoff=args.cutoff,
            bins=args.bins,
            use_rel_data=args.use_rel,
            ions=args.ions,
            aas=args.aas,
            clays=args.clays,
            group_all=args.grouped,
            atoms=atoms,
            atomname=args.atomname,
            load=False,
        )

        data.ignore_density_sum = args.ignore_sum
    # data = Data(
    #     "/storage/results/zdist_full",
    #     analysis="zdens",
    #     namestem="zdist",
    #     cutoff=50,
    #     bins=0.02,
    #     use_rel_data=args.use_rel,
    #     aas=["ala", "arg"],
    #     atoms=["ions"]
    # group_all=True,
    # atoms=['N', 'NH', 'OT', 'OH', 'CR', 'CA', 'SR']
    # )
    # data.ignore_density_sum = True
    logger.info(f"Start")
    # vdata = Data2D(
    #     indir="/storage/results/rdf_full",
    #     zdir="/storage/results/zdist_full",
    #     cutoff=50,
    #     bins=0.02,
    #     # aas=['gly'],
    #     # clays=['NAu-1'],
    #     # ions=['Na'],
    #     odir="/storage/plots/rdf_full",  # _plots",
    #     namestem="rdf",
    #     other=['OW'],
    #     zstem="zdist",
    #     nameparts=1,
    #     zname="zdens",
    #     # aas=["ctl"],
    #     # load='../data/vdata.pkl'
    #     analyses=['rdens']
    # )
    # vdata.save("../data/rdata.pkl")
    # print(vdata.df)
    # print(vdata.zf)
    # print(data.df)
    # for i in ['Na', 'K', 'Mg', 'Ca', '']:
    # data._get_edges()
    # data.plot(x='clays', y='atoms', select='aas')

    plot_method = {"i": lambda x: x.plot_ions, "o": lambda x: x.plot_other}

    if args.data2d is False:
        if args.gbars is True:
            plot = GaussHistPlot(data, add_missing_bulk=args.add_bulk)
            for s in args.sel:
                plot_method[s](plot)(
                    x=args.x,
                    y=args.y,
                    bars=args.plsel,
                    dpi=200,
                    columnlabel=rf"{args.xlabel}",
                    # r"distance from surface (\AA)",
                    rowlabel=rf"{args.ylabel}",  # 'closest atom type',# r"$\rho_z$ ()",
                    plsave=plsave,
                    odir=args.odir,  # "/storage/plots/aadist_u/",
                    ylim=args.ymax,
                    plot_table=args.table,
                )
        elif args.lines:
            plot = LinePlot(data)
            for s in args.sel:
                plot_method[s](plot)(
                    x=args.x,
                    y=args.y,
                    lines=args.plsel,
                    ylim=args.ymax,
                    dpi=200,
                    columnlabel=rf"{args.xlabel}",  # r"distance from surface (\AA)",
                    rowlabel=rf"{args.ylabel}",  # 'closest atom type',# r"$\rho_z$ ()",
                    plsave=plsave,
                    xlim=args.xmax,
                    odir=args.odir,  # "/storage/plots/aadist_u/",
                    edges=args.edges,
                    figsize=args.figsize,
                    colours=args.colours,
                )
        elif args.bars:
            plot = HistPlot(data)  # , add_missing_bulk=args.add_missing_bulk)
            for s in args.sel:
                plot_method[s](plot)(
                    x=args.x,
                    y=args.y,
                    bars=args.plsel,
                    dpi=200,
                    columnlabel=rf"{args.xlabel}",
                    # r"distance from surface (\AA)",
                    rowlabel=rf"{args.ylabel}",  # 'closest atom type',# r"$\rho_z$ ()",
                    plsave=plsave,
                    odir=args.odir,  # "/storage/plots/aadist_u/",
                    ylim=args.ymax,
                    plot_table=args.table,
                )
    else:
        plot = HistPlot2D(data, col_sel=["rdens"], select="ions")
        plot.plot_ions(
            x="aas", y="clays", bars="atoms", col_sel=["rdens"]
        )  # plot_l.select = 'ions'  # plot.select = 'ions'  # print(plot_l._plot_df, plot_l._colour_df)#, plot_l.colour_df)  # plot_l.plot_ions(  #     y="aas", x="atoms", lines="clays", xlim=10, ylim=0.4  # )  # , plot_table=True)  # plot_l.plot_other(  #     y="aas",  #     x="ions",  #     bars="clays",  #     # ylim=0.1,  #     dpi=200,  #     # edges=True,  #     columnlabel=r"distance from surface (\AA)",  #     rowlabel='closest atom type',# r"$\rho_z$ ()",  #     plsave=True,  #     # xlim=50,  #     odir="/storage/plots/aadist_u/",  # )  # , plot_table=True)  # plot_l = LinePlot(data)  # plot_l.plot_other(  # y="aas",  # x="atoms",  # lines="ions",  # # ylim=0.1,  # dpi=200,  # # edges=True,  # columnlabel=r"distance from surface (\AA)",  # rowlabel='closest atom type',# r"$\rho_z$ ()",  # plsave=True,  # xlim=10,  # # xmax=10,  # edges=True,  # odir="/storage/plots/aadist_xy/",  #     sparse = True  # )  # , plot_table=True)  # plot_l.plot_ions(  #     y="aas",  #     x="atoms",  #     lines="clays",  #     # ylim=0.1,  #     dpi=200,  #     # edges=False,  #     columnlabel=r"distance from surface (\AA)",  #     rowlabel="position density ()",  #     plsave=True,  #     xlim=10,  #     odir="/storage/plots/zdist_ions/",  #     edges=True,  # )  # , plot_table=True)  # plot_l.plot_other(  #     y="aas",  #     x="ions",  #     lines="clays",  #     # ylim=0.02,  #     dpi=200,  #     edges=False,  #     columnlabel=r"distance from surface (\AA)",  #     rowlabel=r"$\rho_z$ ()",  #     xlim=50,  #     plsave=True,  #     odir='/storage/plots/zdist_u/'  # )  # plot_l.plot_other(  #     y="aas",  #     x="ions",  #     lines="atoms",  #     # ylim=0.02,  #     dpi=200,  #     xlim=20,  #     edges=False,  #     columnlabel=r"distance from surface (\AA)",  #     rowlabel="position density ()",  #     plsave=True,  #     odir='/storage/plots/zdist_u/'  # # # )  # plot.plot_ions(  #     y="aas",  #     x="atoms",  #     bars="clays",  #     plot_table=False,  #     plsave=True,  #     odir="/storage/plots/zdist_hist/",  #     # col_sel=["rdens"],  #     # get_new_data=True,  #     # edges=[0, 1],  #     # nbins=200,  # )  # plot.plot_other(#y="aas", x="ions", bars="clays"  #     y="ions",  #     x="clays",  #     bars="atoms",  #     # plot_table=True,  #     odir="/storage/plots/zdist_haa/",  #     rowlabel = 'closest atom type',  # r"$\rho_z$ ()"  # )  # plot_l.plot_ions(y="aas", x="atoms", bars="clays", plot_table=True,  #                plsave=True, odir="/storage/plots/zdist_h/",)  # fig = data.plot_bars(y='ions', x='clays', bars='aas')
