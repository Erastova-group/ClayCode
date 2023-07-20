#!/usr/bin/env python3
import itertools
import json
import pickle as pkl
import re
import sys
from collections import UserString
from functools import cached_property, partialmethod
from pathlib import Path
from typing import Literal, Union, Sequence, List, Optional, Tuple, Any
import itertools as it

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mpc
import numpy as np
import pickle as pkl
from argparse import ArgumentParser
import warnings
from abc import ABC, abstractmethod, abstractproperty

import pandas as pd
from matplotlib import colormaps
from matplotlib.table import Table, table
import colourmap as colourmap

from scipy.signal import (
    find_peaks,
    medfilt,
    wiener,
    lfilter,
    welch,
    qspline1d,
    qspline1d_eval,
)
from scipy.interpolate import splrep, splev
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import os
import logging

from ClayAnalysis.classes import SimDir
from ClayAnalysis.plots import Data, LinePlot, Cutoff, Bins
from ClayAnalysis.utils import change_suffix, open_outfile, get_pd_idx_iter
import numpy as np

logger = logging.getLogger(Path(__file__).stem)

logger.setLevel(logging.DEBUG)


class Peaks:
    def __init__(self, data: Data):
        plot_data = LinePlot(data)
        logger.info('pkks')
        plot_df = plot_data._plot_df
        logger.info(f'{data}')
        self.atoms_df = plot_df.reset_index(["atoms"], drop=True)

        self.idx_names = plot_df.index.droplevel(["x", "atoms", "_atoms"]).names
        idx_values = [
            plot_df.index.get_level_values(idxit).unique() for idxit in self.idx_names
        ]
        self.idx_product = np.array(
            np.meshgrid(*[idx_value for idx_value in idx_values])
        ).T.reshape(-1, len(idx_values))
        self.atom_types = plot_df.index.get_level_values("_atoms").unique()

    def get_peaks(
        self,
        atom_types='all',
            other=None,
        wiener_def=3,
        prominence_def=0.001,
        height_def=0.0005,
        width_def=1,
        wlen_def=15,
        distance_def=2,
        overwrite=False,
        smooting=2,
    interpolation=2):
        if other is None:
            other_str= ''
        else:
            other_str= f'{other} '
        p_dict = {}
        e_dict = {}

        if atom_types == 'all':
            atom_types = self.atom_types
        else:
            if type(atom_types) == str:
                atom_types = [atom_types]
            atom_types = [atom_type for atom_type in atom_types if atom_type in self.atom_types]
        for atom_type in atom_types:
            outname = data._get_edge_fname(atom_type=atom_type,
                                           other=other,
                                           name='pe') #.cwd() / f"peaks_edges/{atom_type}_pe_data.p"
            if outname.is_file() and overwrite is False:
                with open(outname, "rb") as pklfile:
                    pkl_dict = pkl.load(pklfile)
                    e_dict[atom_type] = pkl_dict["edges"]
                    p_dict[atom_type] = pkl_dict["peaks"]
            else:
            #     raw_data = data._get_edge_fname(atom_type=atom_type,
            #                                     name='pe')
            #     if not Path(raw_data).is_file():
            #         data._get_edges()
            #
                accept_query = None
                while accept_query not in ["y", "n"]:
                    accept_query = input(f"Get {atom_type!r} peaks? [y/n]")
                if accept_query == "n":
                    continue
                elif accept_query == "y":
                    df_slice = self.atoms_df.xs(
                        atom_type, level="_atoms", drop_level=False
                    ).copy()
                    if other is not None:
                        df_slice = df_slice.xs(other, level="other", drop_level=False)
                    p_dict[atom_type] = []
                    e_dict[atom_type] = []
                    idx_counter = 0
                    for idx in self.idx_product:
                        try:
                            at_slice = (
                                df_slice.xs(idx, level=self.idx_names, drop_level=False)
                                .copy()
                                .convert_dtypes()
                            )
                            y_vals = at_slice.to_numpy().astype(float)
                            x_vals = (
                                at_slice.index.get_level_values("x")
                                .to_numpy()
                                .astype(float)
                            )
                            if smooting not in [1, None]:
                                logger.info(f"Apply smoothing factor of {smooting}.")
                                spline_len = (len(y_vals) // smooting) * smooting

                                if spline_len == 0:
                                    print(idx)
                                    print(at_slice)
                                print(len(y_vals),
                                    len(x_vals[slice(None, spline_len - 1, smooting)]),
                                    spline_len,
                                    len(y_vals[:spline_len]),
                                )
                                y_spline = splrep(
                                    np.mean(
                                        np.reshape(
                                            x_vals[:spline_len],
                                            ((spline_len) // smooting, smooting),
                                        ),
                                        axis=1,
                                    ),
                                    np.add.reduce(
                                        np.reshape(
                                            y_vals[:spline_len],
                                            ((spline_len) // smooting, smooting),
                                        ),
                                        axis=1,
                                    )
                                )
                                interpolate_len = (len(x_vals) // interpolation) * interpolation
                                x_vals =                                 x_vals = np.mean(
                                        np.reshape(
                                            x_vals[:interpolate_len],
                                            ((interpolate_len) // interpolation, interpolation),
                                        ),
                                        axis=1,
                                    )
                                y_vals = splev(x_vals, y_spline)
                            elif interpolation not in [1, None]:
                                interpolate_len = (len(x_vals) // interpolation) * interpolation
                                x_vals = np.mean(
                                        np.reshape(
                                            x_vals[:interpolate_len],
                                            ((interpolate_len) // interpolation, interpolation),
                                        ),
                                        axis=1,
                                    )
                                y_vals = np.add.reduce(
                                        np.reshape(
                                            y_vals[:interpolate_len],
                                            ((interpolate_len) // interpolation, interpolation),
                                        ),
                                        axis=1,
                                    )
                            accept = False
                            prominence = None
                            height = None
                            width = None
                            wlen = None
                            distance = None
                            wiener_val = None
                            try_kde = False
                            cutoff = False
                            idx_counter += 1
                            logger.info(f"{idx_counter}:")
                            y_vals = y_vals / np.sum(y_vals)
                            while accept is False:
                                x_vals_sel = x_vals
                                y_vals_sel = y_vals
                                print(f"atom type: {atom_type}")
                                if other is None:
                                    print("clay: {}, ion: {}, aa: {}".format(*idx))
                                else:
                                    print("clay: {}, ion: {}, aa: {}, other: {}".format(*idx))
                                plt.close()
                                if wiener_val is None:
                                    print(f"Selected {wiener_val}")
                                    wiener_val = wiener_def
                                    query = False
                                else:
                                    # wiener_val: int = int(input("Enter wiener window: "))
                                    wiener_val = self.process_input(
                                        wiener_val, "wiener window", wiener_val, int
                                    )

                                if query is True:
                                    accept_query = None
                                    while accept_query not in ["y", "n", ""]:
                                        accept_query = input(
                                            f"Use cutoff? [y/n] ({cutoff})"
                                        )
                                    if accept_query == "n":
                                        cutoff = False
                                    elif accept_query == "y":
                                        cutoff = self.process_input(
                                            cutoff, "cutoff", False, float
                                        )

                                    if cutoff is not False:
                                        y_vals_sel = y_vals_sel[x_vals_sel <= cutoff]
                                        x_vals_sel = x_vals[x_vals_sel <= cutoff]

                                    accept_query = None
                                    while accept_query not in ["y", "n", ""]:
                                        accept_query = input(
                                            f"Try KDE? [y/n] ({try_kde})"
                                        )
                                    if accept_query == "n":
                                        try_kde = False
                                    elif accept_query == "y":
                                        try_kde = True

                                    if try_kde is True:
                                        print(f"Selected KDE")
                                        kde_vals = KernelDensity(
                                            kernel="tophat", bandwidth=0.01
                                        ).fit(
                                            np.expand_dims(x_vals_sel, axis=1),
                                            sample_weight=y_vals,
                                        )
                                        y_vals_sel = kde_vals.score_samples(
                                            np.expand_dims(x_vals_sel, axis=1)
                                        )
                                        y_vals_sel -= np.amin(
                                            y_vals_sel,
                                            where=~np.isinf(y_vals_sel),
                                            initial=0,
                                        )
                                        y_vals_sel = np.where(
                                            np.isinf(y_vals_sel), 0, y_vals_sel
                                        )
                                        y_vals_sel = y_vals_sel / np.sum(y_vals_sel)

                                        # if query is True:
                                        plt.plot(
                                            x_vals,
                                            y_vals / np.sum(y_vals),
                                            linestyle="dotted",
                                            label="before kde",
                                        )

                                        # plt.plot(
                                        #     x_vals,
                                        #     y_vals_sel,
                                        #     linestyle="solid",
                                        #     label="kde",
                                        # )

                                print(f"Selected {wiener_val}")
                                if wiener_val > 1:
                                    y_vals_wiener = wiener(y_vals_sel, wiener_val)
                                else:
                                    y_vals_wiener = y_vals_sel
                                y_vals_wiener = y_vals_wiener / np.sum(y_vals_wiener)
                                if query is True:
                                    plt.plot(
                                        x_vals_sel,
                                        y_vals_sel / np.sum(y_vals_sel),
                                        linestyle="dotted",
                                        label="before wiener",
                                    )

                                    plt.plot(
                                        x_vals_sel,
                                        y_vals_wiener,
                                        label="after wiener",
                                        linestyle="solid",
                                    )
                                    plt.legend()
                                    plt.show()
                                    y_vals_sel = y_vals_wiener

                                    accept_query = None
                                    while accept_query not in ["y", "n"]:
                                        accept_query = input("Accept? [y/n]")
                                    if accept_query == "n":
                                        continue
                                    plt.close()
                                    plt.plot(
                                        x_vals_sel,
                                        y_vals_sel,
                                        linestyle="dotted",
                                    )
                                prominence = self.process_input(
                                    prominence, "prominence", prominence_def, float
                                )
                                height = self.process_input(
                                    height, "height", height_def, float
                                )
                                distance = self.process_input(
                                    distance, "distance", distance_def, int
                                )
                                width = self.process_input(
                                    width, "width", width_def, int
                                )
                                wlen = self.process_input(wlen, "wlen", wlen_def, int)
                                peaks, peak_dict = find_peaks(
                                    y_vals_sel,
                                    prominence=prominence,
                                    height=height,
                                    distance=distance,
                                    width=width,
                                    wlen=wlen,
                                )
                                if query is True:
                                    plt.hlines(
                                        height,
                                        np.min(x_vals_sel),
                                        np.max(x_vals_sel),
                                        label='height',
                                        linestyles='dotted',
                                        colors='grey'
                                    )
                                    plt.scatter(
                                        x=x_vals[peaks],
                                        y=y_vals_sel[peaks],
                                        color="red",
                                    )
                                    plt.vlines(
                                        x_vals[peaks],
                                        np.min(y_vals_sel),
                                        np.max(y_vals_sel),
                                    )
                                    logger.info(
                                        f"{atom_type}: Found {len(peaks)} peaks: \n{x_vals_sel[peaks]}"
                                    )
                                    plt.show()
                                    accept_query = None
                                    while accept_query not in ["y", "n"]:
                                        accept_query = input("Accept? [y/n]")
                                    if accept_query == "n":
                                        continue
                                    plt.close()
                                edges = [0]
                                for id in range(0, len(peaks) - 1):
                                    window = np.s_[peaks[id] : peaks[id + 1]]
                                    edge_id = np.argwhere(
                                        y_vals[window] == np.min(y_vals[window])
                                    )[0]
                                    edges.append(x_vals[window.start + edge_id][0])
                                try:
                                    right_base = peak_dict["right_bases"][-1]
                                except IndexError:
                                    right_base = peak_dict["right_base"]
                                if len(peaks) > 1:
                                    final_slice = np.s_[
                                        window.stop : window.stop
                                        + np.min(
                                            [edge_id[0], window.stop - window.start]
                                        )
                                    ]
                                    if y_vals[right_base] <= np.min(
                                        y_vals[final_slice]
                                    ):
                                        right_edge = right_base
                                    else:
                                        right_edge = (
                                            window.stop
                                            + np.argwhere(
                                                y_vals[final_slice]
                                                == np.min(y_vals[final_slice])
                                            )[-1]
                                        )
                                else:
                                    right_edge = right_base
                                try:
                                    right_edge = right_edge[-1]
                                except IndexError:
                                    print(f'{idx} IndexError')
                                    pass
                                edges.append(x_vals[right_edge])
                                edges.append(x_vals[-1])
                                edge_dict = {
                                    "edges": edges,
                                    "cutoff": 20,
                                    "peaks": x_vals[peaks],
                                }
                                plt.plot(
                                    x_vals, y_vals, color="black", label="density"
                                )
                                plt.scatter(
                                    x_vals[peaks],
                                    y_vals[peaks],
                                    color="red",
                                    label="peaks",
                                )
                                plt.vlines(
                                    x_vals[peaks],
                                    np.min(y_vals),
                                    np.max(y_vals),
                                    color="red",
                                )
                                for p in peaks:
                                    plt.annotate(
                                        rf"{x_vals[p]:.2f} \AA",
                                        xy=(x_vals[p], y_vals[p]),
                                        textcoords="offset points",
                                        verticalalignment="bottom",
                                    )
                                plt.vlines(
                                    x_vals[peak_dict["right_bases"]],
                                    np.min(y_vals),
                                    0.3 * np.max(y_vals),
                                    color="blue",
                                    label="right edges",
                                    linestyles="dotted",
                                )
                                plt.vlines(
                                    x_vals[peak_dict["left_bases"]],
                                    np.min(y_vals),
                                    0.3 * np.max(y_vals),
                                    color="green",
                                    label="left edges",
                                    linestyles="dotted",
                                )
                                plt.vlines(
                                    edge_dict["edges"],
                                    np.min(y_vals),
                                    0.5 * np.max(y_vals),
                                    color="orange",
                                    label="edges",
                                )
                                for ei, edge in enumerate(edge_dict["edges"]):
                                    plt.annotate(
                                        rf"{edge:.2f} \AA",
                                        xy=(edge, 0.5 * np.max(y_vals)),
                                        textcoords="offset points",
                                        verticalalignment="bottom",
                                    )
                                plt.legend()
                                plt.show()
                                accept_query = None
                                while accept_query not in ["y", "n", ""]:
                                    accept_query = input("Accept? [y/n]")
                                    if accept_query == "y":
                                        accept = True
                                    elif accept_query == "":
                                        accept = True
                                plt.close()
                                query = True
                            p_dict[atom_type].append(x_vals[peaks])
                            e_dict[atom_type].append(edges)
                        except KeyError:
                            logger.info(f'{idx} KeyError')
                            pass
                        except TypeError:
                            logger.info(f'{idx} TypeError')
                            # except ValueError:
                            #     print("verr")
                            pass
                    n_peaks = 0
                    for p_id, p in enumerate(p_dict[atom_type]):
                        found_peaks = len(p)
                        if found_peaks > n_peaks:
                            n_peaks = found_peaks
                    # with open('peaks.p', 'wb') as ppkl:
                    #     pkl.dump(p_dict, ppkl)
                    p_df = pd.DataFrame(
                        index=np.arange(len(df_slice)), columns=np.arange(n_peaks)
                    )
                    e_df = pd.DataFrame(
                        index=np.arange(len(df_slice)), columns=np.arange(n_peaks + 2)
                    )
                    for p_id, p in enumerate(p_dict[atom_type]):
                        e = e_dict[atom_type][p_id]
                        print(e_df.loc[p_id, : len(p)], e)
                        p_df.iloc[p_id, : len(p)] = p
                        e_df.iloc[p_id, : len(e) - 1] = e[:-1]
                        e_df.iloc[p_id, -1] = e[-1]
                    p_df.dropna(how="all", inplace=True)
                    e_df.dropna(how="all", inplace=True)
                    out_dict = {}
                    out_dict["peaks"] = p_dict[atom_type] = p_df
                    out_dict["edges"] = e_dict[atom_type] = e_df
                    self.save(out_dict, outname=outname, overwrite=overwrite)
            p_vals = p_dict[atom_type]
            e_vals = e_dict[atom_type]
            p_cols = p_vals.columns
            for col in p_cols:
                sep_pks = p_vals[p_vals > np.min(p_vals) + 0.5]
                if p_vals[col].any():
                    p_vals[p_vals.columns[-1] + 1] = np.NaN
                    p_slice = p_vals.loc[:, col:]
                    p_vals.loc[p_slice.index, col:] = p_slice.where(
                        sep_pks[col].isna(), p_slice.shift(1, axis=1, fill_value=np.NaN)
                    )
            plt.close()
            p_vals = p_vals.dropna(axis=1, how="all")
            p_vals = p_vals.mean()
            for x, y in self.get_iter(atom_type=atom_type):
                plt.plot(x, y)
            plt.vlines(p_vals, 0, np.max(y))
            plt.show()
            accept_query = None
            while accept_query not in ["y", "n"]:
                accept_query = input(f"Accept {atom_type} {other_str} peaks? [y/n]")
            while accept_query == "n":
                for pi, p in enumerate(p_vals):
                    accept_peak = None
                    while accept_peak not in ["y", "n"]:
                        plt.vlines(p, 0, np.max(y), colors="red")
                        accept_peak = input(
                            rf"Accept {atom_type} {other_str}peak {pi} ({p:.2f} \AA)? [y/n]"
                        )
                    if accept_peak == "n":
                        p_vals.pop(pi)
                plt.close()
                for x, y in self.get_iter(atom_type=atom_type):
                    plt.plot(x, y)
                plt.vlines(p_vals, 0, np.max(y))
                plt.show()
                accept_query = None
                while accept_query not in ["y", "n"]:
                    accept_query = input(f"Accept {atom_type} {other_str} peaks? [y/n]")
            if accept_query == "y":
                edge_arr = np.zeros(len(p_vals) + 2)
                edge_arr[0] = e_vals[0].mean()
                # if cutoff is None:
                edge_arr[-1] = data.cutoff  # .iloc[:, -1].mean()
                # else:
                #     cutoff =
                for pi, p in enumerate(p_vals):
                    print(pi, p)
                    edgs_sel = np.ravel(e_vals.iloc[:, 1:-1].values)
                    try:
                        edgs_arr = edgs_sel[
                            np.logical_and(p < edgs_sel, edgs_sel < p_vals[pi + 1])
                        ]
                    except KeyError:
                        print("kerr")
                        edgs_arr = edgs_sel[
                            np.logical_and(p < edgs_sel, edgs_sel < edge_arr[-1])
                        ]
                    print(edgs_arr)
                    if edgs_arr.any():
                        edge_arr[pi + 1] = np.mean(edgs_arr)
                e_vals = edge_arr
                plt.close()
                outname = data._get_edge_fname(atom_type=atom_type, name='edges', other=other)
                for x, y in self.get_iter(atom_type=atom_type):
                    plt.plot(x, y)
                plt.vlines(p_vals, 0, np.max(y), label='peaks')
                plt.vlines(e_vals, 0, np.max(y), colors="orange", label='edges')
                for ei, edge in enumerate(e_vals):
                    plt.annotate(
                        rf"{edge:.2f} \AA",
                        xy=(edge, 0.5 * np.max(y)),
                        textcoords="offset points",
                        verticalalignment="bottom",
                    )
                plt.suptitle(f'{atom_type} {other_str}peaks and edges, cutoff: {data.cutoff.num} \AA, bins: {data.bins.num} \AA')
                plt.ylabel('position density ()')
                plt.xlabel(r'$z$-distance (\AA)')
                plt.legend()
                plt.savefig(outname.with_suffix('.png'))
                plt.show()
                accept_query = False
                while accept_query not in ["y", "n"]:
                    accept_query = input(f"Save edges and peaks? [y/n]")
                if accept_query == "y":
                    edge_dict = {
                        "edges": e_vals,
                        "cutoff": np.rint(e_vals[-1]),
                        "peaks": p_vals,
                    }
                    with open(outname, "wb") as edge_file:
                        pkl.dump(edge_dict, edge_file)
                    logger.info(f"Wrote {atom_type} {other_str}edges to {outname}.")

            #         print(at_slice.index.names)
            # plt.legend()
            # plt.show()

            # print(p_dict, edge_dict)

    def get_iter(self, atom_type: str, other=None):
        df_slice = self.atoms_df.xs(atom_type, level="_atoms", drop_level=False).copy()
        if other is not None:
            df_slice = (
                df_slice.xs(other, level="other", drop_level=False)
                .convert_dtypes()
            )
        idx_counter = 0
        for idx in self.idx_product:
            try:
                at_slice = (
                    df_slice.xs(idx, level=self.idx_names, drop_level=False)
                    .copy()
                    .convert_dtypes()
                )
                y_vals = at_slice.to_numpy().astype(float)
                x_vals = at_slice.index.get_level_values("x").to_numpy().astype(float)
                yield x_vals, y_vals
            except KeyError:
                pass

    @staticmethod
    def save(dump_obj, outname, overwrite):
        if not outname.parent.is_dir():
            os.makedirs(outname.parent)
        if not outname.is_file() or overwrite is True:
            with open(outname.resolve(), "wb") as outfile:
                pkl.dump(dump_obj, outfile, fix_imports=True)


    @staticmethod
    def process_input(
        dest: Any, query: str, default: Any, rtype: Optional[type] = None
    ) -> Any:
        assigned = False
        if dest is not None:
            while assigned is False:
                add_str = f"({dest})"
                value = input(f"(Re)set {query}: {add_str}")
                if value != "":
                    if rtype is not None:
                        try:
                            dest = rtype(value)
                            assigned = True
                        except ValueError:
                            pass
                    else:
                        dest = value
                        assigned = True
                else:
                    assigned = True
        else:
            dest = default
        print(f"{query} = {dest}")
        return dest


if __name__ == "__main__":
    data = Data(
        "/storage/results/rdf_full",  # _c",
        cutoff=50,
        bins=0.02,
        odir="../data/peaks/",  # _plots",
        namestem="rdf",
        nameparts=1,
        analysis="rdens",
        other=['OT']
    )
    # print(type(data))
    peaks = Peaks(data=data)
    peaks.get_peaks(
        smooting=6,
        interpolation=2,
        wlen_def=20,
        prominence_def=1E-6,#0.0001,
        height_def=0.001,
        wiener_def=7,
        overwrite=True,
        other='OT'
    )
