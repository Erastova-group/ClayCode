from scipy.signal import find_peaks
from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from ClayAnalysis.plots import Data, HistPlot, LinePlot
from pathlib import Path
import sys
from argparse import ArgumentParser
import logging

logger = logging.getLogger(Path(__file__).stem)

DISTANCE = 2
WIDTH = 1
WLEN = 11
PROMINENCE = 0.02


def get_edge_df(df):
    p_df = df.copy()
    p_df.reset_index(["atoms"], drop=True, inplace=True)
    return p_df


def get_edge_idx_iter(df):
    idx_names = df.index.droplevel(["x", "atoms", "_atoms"]).names
    idx_values = [df.index.get_level_values(idxit).unique() for idxit in idx_names]

    idx_product = np.array(
        np.meshgrid(*[idx_value for idx_value in idx_values])
    ).T.reshape(-1, len(idx_values))
    atom_types = df.index.get_level_values("_atoms").unique()
    return idx_names, idx_product, atom_types


if __name__ == "__main__":
    parser = ArgumentParser(
        add_help=True,
        allow_abbrev=False,
    )
    parser.add_argument(
        "-path", default=False, help="File with analysis data paths.", dest="path"
    )
    parser.add_argument("-o", default=False, help="Output folder paths.", dest="odir")
    parser.add_argument(
        "-cutoff", default=False, help="Peak search cutoff.", dest="cutoff"
    )

    args = sys.argv[1:]

    data = Data(
        args.path,  # _c",
        cutoff=20,
        bins=0.10,
        odir=args.odir,  # _plots",
        namestem="zdist",
        nameparts=1,
        analysis="zdens",
    )
    plot_data = LinePlot(data)
    df = plot_data._plot_df.copy()
    p_df = get_edge_df(df)
    idx_names, idx_iter, atom_types = get_edge_idx_iter(df)

    peak_dict = {}
    edge_dict = {}

    for atom_type in atom_types:
        df_slice = df.xs(atom_type, level="_atoms", drop_level=False).copy()
        for idx in idx_iter:
            try:
                at_slice = (
                    df_slice.xs(idx, level=idx_names, drop_level=False)
                    .copy()
                    .convert_dtypes()
                )
                y_vals = at_slice.to_numpy().astype(np.float64)
                x_vals = (
                    at_slice.index.get_level_values("x").to_numpy().astype(np.float64)
                )
                kde = KernelDensity(kernel="tophat", bandwidth=0.01).fit(
                    X=np.expand_dims(x_vals, axis=1), sample_weight=y_vals
                )
                score = kde.score_samples(np.expand_dims(x_vals, axis=1))
                score -= np.amin(score, where=~np.isinf(score), initial=0)
                score = np.where(np.isinf(score), 0, score)
                #             plt.plot(at_slice.index.get_level_values('x'), score, linestyle='dotted')
                score = np.where(x_vals > args.cutoff, 0, score)

                height = np.mean(score) / np.std(score) * np.max(score)
                #             print(height)
                p = PROMINENCE * np.max(score)
                peaks, peak_dict = find_peaks(
                    score,
                    prominence=p,
                    height=height,
                    distance=DISTANCE,
                    width=WIDTH,
                    wlen=WLEN,
                )
            except KeyError:
                pass
            except ValueError:
                pass
