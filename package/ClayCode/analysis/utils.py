#!/usr/bin/env python3
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess as sp
import sys
from functools import singledispatch
from itertools import chain
from pathlib import Path
from typing import List, Literal, Optional, Union

import MDAnalysis as mda
import numpy as np
import pandas as pd
from ClayCode.core.classes import Dir, File, GROFile, PathType
from ClayCode.core.consts import exec_date, exec_time
from ClayCode.core.utils import convert_num_to_int, get_header
from numpy._typing import NDArray
from tqdm.contrib.logging import logging_redirect_tqdm

logger = logging.getLogger(__name__)

__all__ = [
    "remove_files",
    "change_suffix",
    "convert_num_to_int",
    "get_sequence_element",
    "get_first_item_as_int",
    "check_file_exists",
    "str_to_int",
    "execute_bash_command",
    "get_file_diff",
    "grep_file",
    # "get_logfname",
    "get_search_str",
    "convert_str_list_to_arr",
    "select_file",
    "redirect_tqdm",
]


# path = UC_PATH
# gro_list = UC_PATH.glob(rf'{UC_STEM}[0-9]\{1-2}._gro')
# _uc_data = UCData(UC_FOLDER)
# uc_composition = _uc_data.uc_composition
# uc_charges = _uc_data.uc_charges
# print(uc_composition, uc_charges)
#     from argparse import ArgumentParser
#
#     parser = ArgumentParser(prog="utils.py", add_help=True, allow_abbrev=False)
#
#     parser.add_argument(
#         "-crd",
#         "-crdin",
#         type=str,
#         dest="INFNAME",
#         required=False,
#         metavar="filename",
#         help="Coordinate file path",
#     )
#
#     parser.add_argument(
#         "-ff",
#         required=False,
#         type=str,
#         help="Force field directory",
#         default="/usr/local/gromacs/share/gromacs/top/",
#         metavar="ff directory",
#         dest="FF",
#     )
#
#     conc_args = parser.add_argument_group(
#         title="molecule number arguments:",
#         description="Calculate molecule number from bulk concentration in mol L-1",
#     )
#
#     conc_args.add_argument(
#         "-conc",
#         type=float,
#         dest="CONC",
#         required=False,
#         default=False,
#         metavar="concentration",
#         help="Concentration in mol L-1",
#     )
#
#     conc_args.add_argument(
#         "-savepos",
#         required=False,
#         dest="SAVEPOS",
#         default="positions.dat",
#         metavar="filename",
#         help=' Filename <filename>.dat (default "positions.dat")',
#     )
#
#     center_args = parser.add_argument_group(
#         title="center atoms arguments:",
#         description="Align positions of clay atom and box center.",
#     )
#
#     center_args.add_argument(
#         "--center",
#         required=False,
#         action="store_true",
#         dest="CENTER",
#         default=False,
#         help="Center clay atoms in box.",
#     )
#
#     center_args.add_argument(
#         "-outgro",
#         required=False,
#         default="center._gro",
#         dest="OUTGRO",
#         help='Save centered coordinates to < filename > (default "center._gro"',
#     )
#
#     center_args.add_argument(
#         "-uc",
#         required=False,
#         type=str,
#         help="CLay unit cell selection string",
#         dest="UC",
#         metavar="uc_type",
#         default=None,
#     )
#
#     charges_args = parser.add_argument_group(
#         title="Charge checker",
#         description="Check charges and remove Cl from _gro and top file if < 0 or return n_mols.",
#     )
#
#     charges_args.add_argument(
#         "--neutralise",
#         action="store_true",
#         required=False,
#         default=False,
#         dest="NEUTRAL",
#         help="Check charges and remove Cl from _gro and top file if < 0 or return n_mols.",
#     )
#
#     charges_args.add_argument(
#         "-insertgro",
#         required=False,
#         type=str,
#         dest="INSERT",
#         default=None,
#         metavar="insert coordinates",
#         help="Coordinate file of inserted molecule.",
#     )
#
#     charges_args.add_argument(
#         "-topin",
#         required=False,
#         type=str,
#         dest="ITOP",
#         metavar="filename",
#         help="Input topology filename",
#     )
#
#     charges_args.add_argument(
#         "-topout",
#         required=False,
#         type=str,
#         dest="OTOP",
#         default="insert.top",
#         metavar="filename",
#         help="Output topology filename",
#     )
#
#     charges_args.add_argument(
#         "-n_mols",
#         required=False,
#         type=int,
#         dest="N_MOLS",
#         metavar="n mols",
#         help="Number of inserted molecules",
#         default=None,
#     )
#
#     parser.add_argument(
#         "--add-insert",
#         action="store_true",
#         default=False,
#         required=False,
#         dest="MODTOP",
#     )
#
#     parser.add_argument(
#         "--replace-sol",
#         help="Remove replaced SOL from topology file",
#         required=False,
#         default=0,
#         dest="SOLREPL",
#         action="store_true",
#     )
#
#     parser.add_argument(
#         "--new-dat", action="store_true", default=False, required=False, dest="DAT"
#     )
#
#     p = parser.parse_args(sys.argv[1:])
#
#     n_mols = None
#
#     if p.INFNAME:
#         crdin = mda.Universe(p.INFNAME)
#
#     if p.CONC:
#         n_mols = get_n_mols(conc=p.CONC, crdin=crdin)
#         write_insert_dat(n_mols=n_mols, save=p.SAVEPOS)
#
#     if p.CENTER:
#         center_clay(crdin=crdin, uc_name=p.UC, crdout=p.OUTGRO)
#
#     # if not 'n_mols' in globals():
#     if p.N_MOLS is None and n_mols is None:
#         n_mols = 0
#     elif n_mols is None:
#         n_mols = p.N_MOLS
#
#     if p.MODTOP:
#         add_mols_to_top(
#             topin=p.ITOP, topout=p.OTOP, insert=p.INSERT, n_mols=n_mols, include_dir=p.FF
#         )
#
#     if p.NEUTRAL:
#         remove_charge_Cl(
#             topin=p.ITOP, topout=p.OTOP, insert=p.INSERT, n_mols=n_mols, include_dir=p.FF
#         )
#
#     if p.SOLREPL:
#         remove_replaced_SOL(topin=p.ITOP, topout=p.OTOP, n_mols=n_mols)
#
#     if p.DAT:
#         write_insert_dat(n_mols=n_mols, save=p.SAVEPOS)
#
#     sys.displayhook(n_mols)
#
#     neutralise_system(1, 2, 3, 4, 5, 6, 7)
# #


def remove_files(path, searchstr):
    backupfiles = list(path.glob(rf"{searchstr}"))
    removing = False
    for fname in backupfiles:
        if fname.exists():
            removing = True
            os.remove(fname)
            logger.debug(f"Removing {fname.name}.")
        else:
            logger.debug(f"No backups to remove {fname.name}.")
    return removing


def change_suffix(path: Path, new_suffix: str):
    return path.parent / f'{path.stem}.{new_suffix.strip(".")}'


def get_sequence_element(f):
    def wrapper(seq, id=0):
        try:
            if len(list(seq)) < 2:
                pass
            else:
                logger.debug(1, seq)
        except TypeError:
            logger.debug(2, seq)
            seq = [seq]
        logger.debug(3, seq)
        if type(seq) not in [list, tuple, np.array]:
            raise TypeError(f"Expected sequence, found {type(seq)}")
        if not isinstance(id, int):
            raise TypeError(f"Expected int index, found {type(id)}")
        else:
            result = f(seq[id])
            logger.debug(4, result)
            return result

    return wrapper


@get_sequence_element
@convert_num_to_int
def get_first_item_as_int(seq):
    return seq


def check_file_exists(tempdir: Union[Path, str], file: Union[Path, str]):
    if not Path(file).exists():
        os.rmdir(tempdir)
        raise FileNotFoundError(f"{file!r} does not exist!")


@singledispatch
def str_to_int(str_obj):
    raise TypeError(f"Invalid type for str_obj: {type(str_obj)}")


@str_to_int.register
def _(str_obj: list):
    return list(map(lambda str_item: int(str_item), str_obj))


@str_to_int.register
def _(str_obj: str):
    return int(str_obj)


def execute_bash_command(command, **outputargs):
    output = sp.run(["/bin/bash", "-c", command], **outputargs)
    return output


def get_file_diff(file_1, file_2):
    diff = execute_bash_command(
        f"diff {file_1} {file_2}", capture_output=True, text=True
    )
    return diff.stdout


def grep_file(file, regex: str):
    diff = execute_bash_command(
        f'grep -E "{regex}" {file}', capture_output=True, text=True
    )
    return diff.stdout


# def get_logfname(
#     logname: str,
#     run_name=None,
#     time: Union[Literal[exec_time], Literal[exec_date]] = exec_date,
#     logpath=None,
# ):
#     if logpath is None:
#         logpath = Path().cwd() / "logs"
#         if not logpath.is_dir():
#             os.mkdir(logpath)
#     if run_name is None:
#         run_name = ""
#     else:
#         run_name += "-"
#     return f"{logpath}/{logname}-{run_name}{time}.log"


def get_search_str(match_dict: dict):
    return "|".join(match_dict.keys())


def convert_str_list_to_arr(
    str_list: Union[List[str], List[List[str]]]
) -> np.array:
    array = np.array(list(map(lambda x: x.split(), str_list)), dtype=str)
    arr_strip = np.vectorize(lambda x: x.strip())
    try:
        array = arr_strip(array)
    except:
        logger.debug("Could not convert list to array")
    return array


def copy_final_setup(outpath: Path, tmpdir: Path, rm_tempfiles: bool = True):
    if not outpath.is_dir():
        os.mkdir(outpath)
    gro = select_file(tmpdir, suffix="_gro", how="latest")
    top = select_file(tmpdir, suffix="top", how="latest")
    tpr = select_file(tmpdir, suffix="tpr", how="latest")
    log = select_file(tmpdir, suffix="log", how="latest")
    mdp = select_file(tmpdir, suffix="mdp", how="latest")
    new_files = []
    for file in [gro, top]:
        shutil.move(file, outpath / file.name)
        new_files.append(outpath / file.name)
    for file in [tpr, log, mdp]:
        shutil.move(file, outpath / file.with_stem(f"{file.name}_em").name)
    logger.info(f"Done! Copied files to {outpath.name!r}")
    if rm_tempfiles:
        shutil.rmtree(tmpdir)
    return tuple(new_files)


def select_file(
    path: Union[Path, str],
    searchstr: Optional[str] = None,
    suffix=None,
    how: Literal["latest", "largest"] = "latest",
):
    check_func_dict = {
        "latest": lambda x: x.st_mtime,
        "largest": lambda x: x.st_size,
    }
    check_func = check_func_dict[how]
    logger.debug(f"Getting {how} file:")
    if type(path) != Path:
        path = Path(path)
    if searchstr is None and suffix is None:
        f_iter = path.iterdir()
    else:
        if suffix is None:
            suffix = ""
        if searchstr is None:
            searchstr = "*"
        f_iter = path.glob(rf"{searchstr}[.]*{suffix}")
        backups = path.glob(rf"#{searchstr}[.]*{suffix}.[1-9]*#")
        f_iter = chain(f_iter, backups)
    prev_file_stat = 0
    last_file = None
    for file in f_iter:
        if file.is_dir():
            pass
        else:
            if last_file is None:
                last_file = file
            filestat = os.stat(file)
            last_file_stat = check_func(filestat)
            if last_file_stat > prev_file_stat:
                prev_file_stat = last_file_stat
                last_file = file
    if last_file is None:
        logger.debug(f"No matching files found in {path.resolve()}!")
    else:
        logger.debug(f"{last_file.name} matches")
    return last_file


def get_pd_idx_iter(idx: pd.MultiIndex, name_sel: List[str]):
    idx_names = idx.names
    idx_values = [
        idx.get_level_values(level=name)
        for name in idx_names
        if name in name_sel
    ]
    idx_product = np.array(
        np.meshgrid(*[idx_value for idx_value in idx_values])
    ).T.reshape(-1, len(idx_values))
    # idx_product = np.apply_along_axis(lambda x: '/'.join(x), 1, idx_product)
    return idx_product


def get_u_files(path: Union[str, Path], suffices=["_gro", "top"]):
    files = {}
    path = Path(path)
    largest_files = ["trr"]
    how = "latest"
    for selection in suffices:
        selection = selection.strip(".")
        if selection in largest_files:
            how = "largest"
        files[selection] = select_file(path=path, suffix=selection, how=how)
    return files["_gro"], files["trr"]


# def open_outfile(outpath: Union[Path, str], suffix: str, default: str):
#     if type(outpath) == bool:
#         outpath = Path(f"{default}.json")
#     elif type(outpath) in [str, Path]:
#         outpath = change_suffix(outpath, suffix)
#         if not outpath.parent.is_dir():
#             os.makedirs(outpath.parent)
#     else:
#         raise ValueError(f"Could not interpret {outpath} as path or bool.")
#     return outpath

if __name__ == "__main__":
    print(mda.__version__, "\n", np.__version__)


def _check_methods(C, *methods):
    mro = C.__mro__
    for method in methods:
        for B in mro:
            if method in B.__dict__:
                if B.__dict__[method] is None:
                    return NotImplemented
                break
        else:
            return NotImplemented
    return True


def redirect_tqdm(f):
    def wrapper(*args, **kwargs):
        with logging_redirect_tqdm():
            result = f(*args, **kwargs)
        return result

    return wrapper


def make_1d(arr: NDArray, sel=None, return_dims=False):
    idxs = np.arange(arr.ndim)
    arr_list = []
    if sel == None:
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


def get_paths(
    path: Union[str, PathType] = None,
    infiles: List[Union[PathType, str]] = None,
    inpname: str = None,
    traj_suffix="trr",
):
    logger.info(get_header(f"Getting run files"))
    if path is not None:
        path = Dir(path)
        if infiles is None:
            gro = select_file(path=path, suffix="gro", searchstr=inpname)
            trr = select_file(
                path=path, suffix=traj_suffix, how="largest", searchstr=inpname
            )
        else:
            gro = select_file(
                path=path, suffix="gro", searchstr=infiles[0].strip(".gro")
            )
            trr = select_file(
                path=path,
                suffix=traj_suffix,
                how="largest",
                searchstr=infiles[1].strip(".gro"),
            )
    else:
        gro, trr = infiles
        gro, trr = GROFile(gro), File(trr)
        path = Dir(gro.parent)
    for file_type, file in zip(["coordinate", "trajectory"], [gro, trr]):
        try:
            logger.finfo(
                f"{str(file.resolve())!r}", kwd_str=f"Found {file_type} file: "
            )
        except AttributeError:
            logger.finfo(f"No file {str(file_type)!r} file found.")
            sys.exit(1)
    return gro, trr, path


# def gauss_peak(x, A, mean, sigma, offset=None):
#     # print(type(x), type(A), type(mean), type(x), type(offset))
#     exp_term_1 = np.subtract(x, np.atleast_2d(mean).T)
#     exp_term_2 = np.multiply(sigma, 2)
#     exp_term = np.divide(exp_term_1, np.atleast_2d(exp_term_2).T)
#     exp_term = -np.power(exp_term, 2)
#     exp_term = np.exp(exp_term)
#     # exp = np.divide(- np.power(np.subtract(x, mean), 2), np.power(np.multiply(2, sigma), 2))
#     # B = np.exp(exp_term)
#     gauss_peak = np.multiply(exp_term, np.atleast_2d(A).T)
#     try:
#         gauss_peak = np.sum(gauss_peak, axis=0)
#     except np.AxisError:
#         pass
#     if offset is not None:
#         gauss_peak = np.add(gauss_peak, offset)
#     return gauss_peak


def gauss_peak(x, A, mean, sigma):
    # print(type(x), type(A), type(mean), type(x), type(offset))
    exp_term_1 = np.subtract(x, mean)
    exp_term_2 = np.multiply(sigma, 2)
    exp_term = np.divide(exp_term_1, exp_term_2)
    exp_term = -np.power(exp_term, 2)
    exp_term = np.exp(exp_term)
    # exp = np.divide(- np.power(np.subtract(x, mean), 2), np.power(np.multiply(2, sigma), 2))
    # B = np.exp(exp_term)
    gauss_peak = np.multiply(exp_term, A)
    # except np.AxisError:
    #     pass
    # if offset is not None:
    #     gauss_peak = np.add(gauss_peak, offset)
    return gauss_peak


_n_gauss_peaks = np.vectorize(
    gauss_peak, excluded=["offset"], otypes=[np.float64]
)


def lorentz_peak(x, A, mean, sigma):
    sigma = np.divide(sigma, 2)
    sigma_sq = np.power(sigma, 2)
    div_1 = np.multiply(A, sigma_sq)
    x_diff = np.subtract(x, mean)
    x_diff = np.power(x_diff, 2)
    div_2 = np.add(x_diff, sigma_sq)
    lorentz_peak = np.divide(div_1, div_2)
    return lorentz_peak


def voigt_peak(x, A, mean_g, sigma_g, mean_l, sigma_l, weight):
    gauss_args = (A, mean_g, sigma_g)
    lorentz_args = (A, mean_l, sigma_l)
    gauss_part = gauss_peak(x, *gauss_args)
    lorentz_part = lorentz_peak(x, *lorentz_args)
    gauss_part = np.multiply(gauss_part, weight)
    lorentz_part - np.multiply(lorentz_part, (1 - weight))
    return np.add(gauss_part, lorentz_part)


def peak_offset_decorator(n_args):
    def f_decorator(f):
        def wrapper(x, *args):
            args = np.array(args)
            assert (len(args) - 5) % n_args == 0
            offset = args[-5:]
            args = args[:-5]
            n_peaks = len(args) // n_args
            final_result = []
            args = args.reshape(n_args, n_peaks).T
            # for A, mean, sigma in args:
            for peak_args in args:
                result = f(x, *peak_args)  # A, mean, sigma)
                final_result.append(result)
            try:
                final_result = np.sum(final_result, axis=0)
            except np.AxisError:
                final_result = np.ravel(final_result)
            if np.any(offset) != 0.0:
                noise_start, noise_end, noise_c1, noise_c2, h = offset
                noise = heavyside_func(
                    x, noise_start, noise_end, noise_c1, noise_c2, h
                )
                final_result = np.add(final_result, noise)
            return final_result

        return wrapper

    return f_decorator


def heavyside_func(x, start, stop, c1, c2, h):
    noise_start = np.subtract(x, start)
    noise_end = np.subtract(stop, x)
    noise1 = np.multiply(noise_start, c1)
    noise2 = np.multiply(noise_end, c2)
    noise = np.add(scipy.special.expit(noise1), scipy.special.expit(noise2))
    noise = np.subtract(noise, 1)
    return np.multiply(h, noise)
    # noise = np.multiply(x, k)
    # noise = np.add(noise, d)
    # noise_start = np.sign(start)
    # noise = np.multiply(noise, noise_start)
    # return noise


@peak_offset_decorator(n_args=3)
def n_gauss_peaks(x, *args):
    return gauss_peak(x, *args)  # (A, mean, sigma)


@peak_offset_decorator(n_args=3)
def n_lorentz_peaks(x, *args):
    return lorentz_peak(x, *args)


@peak_offset_decorator(n_args=6)
def n_voigt_peaks(x, *args):
    return voigt_peak(x, *args)
