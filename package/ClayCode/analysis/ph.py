#!/usr/bin/env python3
import logging
import pathlib as pl
import pickle as pkl
import sys
import warnings
from functools import wraps
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from ClayAnalysis import AA

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore")

__all__ = ["get_concentrations", "get_aa_numbers"]

logger = logging.getLogger("pH")

# AA = pl.Path("/storage/aa_test") / "aa.csv"


def exponent_function(f):
    @wraps(f)
    def wrapper(pKa_list: list, *args, **kwargs):
        exp_list = np.arange(-1, len(pKa_list), dtype=np.int_)
        return f(exp_list, pKa_list, *args, **kwargs)

    return wrapper


@exponent_function
def get_concentrations(
    exp_list: List[int], pK_list: List[float], pH: float, c_tot: int
) -> np.float64:
    """
    Determine numbers of amino acid protonation states at specified pH
    """
    pK_list = np.insert(pK_list, 0, pK_list[0])
    c = np.zeros_like(exp_list, dtype=np.float64)
    mask = np.full_like(c, True, dtype=bool)
    mask[1] = False
    exp_array = np.array(exp_list)
    pK_start = np.where(exp_array < 1, exp_array + 1, 2)
    pK_exps = np.where(exp_array < 2, exp_array, 1)
    for idx, e in np.ndenumerate(exp_array):
        idx = idx[0]
        c[idx] = 10.0 ** (
            pK_exps[idx] * np.sum(pH - pK_list[pK_start[idx] : idx + 1])
        )
    c[1] = c_tot / np.sum(c)
    c[mask] *= c[1]
    return np.round(c, 4)


def get_aa_numbers(
    aa: Optional[List[str]],
    pH: Union[int, float, List[Union[int, float]]],
    totmols: int,
    o: Union[str, pl.Path],
    new: bool = False,
) -> Dict[str, List[np.float64]]:
    if not new and pl.Path(o).is_file():
        with open(o, "rb") as infile:
            conc_dict = pkl.load(infile)
    else:
        conc_dict = {}

    pka_df = pd.read_csv(AA / "aa.csv", index_col=0)
    if aa == "all":
        aa = pka_df.index.values

    if "ctl" not in aa:
        # df: index = 'name', columns = ['COOH', 'NH3', 'SIDE']
        pka_df = pka_df.loc[aa]

        if type(pH) == list:
            pH = pH[0]
        else:
            pH = int(pH)

        logger.debug(
            f"Determining amino acid numbers at pH {pH}. "
            f"Total amino acid number in system: {totmols}"
        )

        logger.debug(
            "\t"
            + 5 * " "
            + " | ".join(
                [f"pK{n + 1}" for n in range(len(pka_df.columns) + 1)]
            )
            + " |"
        )
        for aa in pka_df.index:
            aa_sel = pka_df.loc[aa].dropna()
            idxs = aa_sel.values.argsort()
            pka_sorted = aa_sel.values[idxs].astype(np.float64)
            conc_dict[aa] = get_concentrations(pka_sorted, pH, totmols)
            conc_dict[aa] = np.rint(conc_dict[aa])
            aa_list = [
                f"{conc_dict[aa][n]:3.0f}"
                for n, nn in enumerate(conc_dict[aa])
            ]
            # print(conc_dict)
            logger.debug(f"\t{aa.upper()}: " + " | ".join(aa_list) + " |")

    else:
        conc_dict["ctl"] = [0]

    with open(o, "wb") as outfile:
        logger.debug(f"Writing amino acid numbers to insert to {o!r}.")
        pkl.dump(conc_dict, outfile)

    logger.debug("Done!")
    return conc_dict


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        prog="pH",
        description="Get distribution of amino acid \
            protonation states at a specified pH.",
        add_help=True,
        allow_abbrev=False,
    )

    parser.add_argument(
        "-aa",
        type=str,
        nargs="+",
        default=None,
        metavar="aa",
        help="Amino acid(s)",
    )

    parser.add_argument(
        "-pH",
        type=np.float64,
        nargs=1,
        default=7.0,
        metavar="pH_value",
        help="pH for experiment",
    )

    parser.add_argument(
        "-nmols",
        type=np.int_,
        dest="totmols",
        metavar="n_mols",
        help="Total number of amino acid molecules",
    )

    parser.add_argument(
        "-o",
        type=str,
        default="aa_numbers.pkl",
        help="Destination for saving output dictionary",
        metavar="outname",
    )
    parser.add_argument(
        "-new",
        type=str,
        required=False,
        default=False,
        help="Overwrite old output dictionary",
        metavar="new",
    )

    a = parser.parse_args(sys.argv[1:])
    get_aa_numbers(aa=a.aa, pH=a.pH, totmols=a.totmols, o=a.o, new=a.new)
