#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r""":mod:`ClayCode.core.utils` --- Utility functions
===================================================
"""
from __future__ import annotations

import io
import logging
import math
import os
import re
import shutil
import subprocess as sp
import sys
import threading
import time
import warnings
from collections import defaultdict
from functools import partial, singledispatch, wraps
from itertools import chain
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple, Union

import MDAnalysis as mda
import numpy as np
import pandas as pd
import yaml.loader
from caseless_dictionary import CaselessDict
from ClayCode.core.consts import LINE_LENGTH as line_length
from ClayCode.core.consts import TABSIZE, exec_date, exec_time
from numpy.typing import ArrayLike, NDArray

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore")
# sys.stdout.reconfigure(encoding="utf-8")

__all__ = [
    "remove_files",
    "change_suffix",
    "convert_num_to_int",
    "get_sequence_element",
    "get_first_item_as_int",
    "check_file_exists",
    "str_to_int",
    "execute_shell_command",
    "get_file_diff",
    "grep_file",
    "get_logfname",
    "get_search_str",
    "convert_str_list_to_arr",
    "select_file",
    "select_named_file",
    "copy_final_setup",
    "get_subheader",
    "get_header",
    "get_pd_idx_iter",
    "get_u_files",
]

logger = logging.getLogger(__name__)


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


def convert_num_to_int(f):
    def wrapper(number: Union[int, float]):
        if type(number) not in [float, int, np.int_, np.float_]:
            raise TypeError(
                f"Expected float or int type, found {type(number)}!"
            )
        else:
            return f(int(np.round(number, 0)))

    return wrapper


def get_sequence_element(f):
    def wrapper(seq, element_id=0):
        try:
            if len(list(seq)) < 2:
                pass
            else:
                logger.debug(1, seq)
        except TypeError:
            logger.debug(2, seq)
            seq = [seq]
        logger.debug(3, seq)
        # if type(seq) == str:
        #     try:
        #         seq = int(seq)
        #     except ValueError:
        #         raise TypeError(f"Expected numeric values, found {seq}")
        if type(seq) not in [list, tuple, np.array]:
            raise TypeError(f"Expected sequence, found {type(seq)}")
        if not isinstance(element_id, int):
            raise TypeError(f"Expected int index, found {type(element_id)}")
        else:
            result = f(seq[element_id])
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


def execute_shell_command(command):
    try:
        output = sp.run(
            [command], shell=True, capture_output=True, text=True, check=True
        )
    except sp.CalledProcessError:
        shell = sp.run(
            ["echo $SHELL"],
            shell=True,
            text=True,
            check=True,
            capture_output=True,
        ).stdout.strip()
        output = sp.run(
            [shell, "-c", "-i", command],
            capture_output=True,
            text=True,
            check=True,
        )
    return output


def get_file_diff(file_1, file_2):
    diff = execute_shell_command(f"diff {file_1} {file_2}")
    return diff.stdout


def grep_file(file, regex: str) -> str:
    diff = execute_shell_command(f'grep -E "{regex}" {file}')
    return diff.stdout


def get_logfname(
    logname: str,
    run_name=None,
    time: Union[Literal[exec_time], Literal[exec_date]] = exec_date,
    logpath=None,
) -> str:
    if logpath is None:
        logpath = Path().cwd() / "logs"
        if not logpath.is_dir():
            os.mkdir(logpath)
    if run_name is None:
        run_name = ""
    else:
        run_name += "-"
    return f"{logpath}/{logname}-{run_name}{time}.log"


def substitute_kwds(
    string: str, substitutions: dict[str, str], flags: re.RegexFlag
) -> str:
    for key, value in substitutions.items():
        string = re.sub(key, f"{value}", string, flags=flags)
    return string


@singledispatch
def get_search_str(match_obj) -> str:
    raise TypeError(
        f"Could not generate match pattern from {match_obj.__class__.__name__}"
    )


@get_search_str.register(dict)
@get_search_str.register(CaselessDict)
def _(match_obj: dict[str, Any]) -> str:
    return "|".join([str(x) for x in match_obj.keys()])


@get_search_str.register(list)
def _(match_obj: List[str]):
    return "|".join([str(x) for x in match_obj])


def convert_str_list_to_arr(
    str_list: Union[List[str], List[List[str]]]
) -> np.array:
    array = np.array(list(map(lambda x: x.split(), str_list)), dtype=str)
    arr_strip = np.vectorize(lambda x: x.strip())
    try:
        array = arr_strip(array)
    except TypeError:
        logger.debug("Could not convert list to array")
    except IndexError:
        logger.debug("Could not convert list to array")
    return array


def copy_final_setup(outpath: Path, tmpdir: Path, rm_tempfiles: bool = True):
    if not outpath.is_dir():
        os.mkdir(outpath)
    gro = select_file(tmpdir, suffix="gro", how="latest")
    top = select_file(tmpdir, suffix="top", how="latest")
    tpr = select_file(tmpdir, suffix="tpr", how="latest")
    log = select_file(tmpdir, suffix="log", how="latest")
    mdp = select_file(tmpdir, suffix="mdp", how="latest")
    new_files = []
    for file in [gro, top]:
        shutil.move(file, outpath / file.name, copy_function=shutil.copy2)
        new_files.append(outpath / file.name)
    for file in [tpr, log, mdp]:
        shutil.move(
            file,
            outpath / file.with_stem(f"{file.name}_em").name,
            copy_function=shutil.copy2,
        )
    logger.finfo(f"Done! Copied files to {outpath.name!r}")
    if rm_tempfiles:
        shutil.rmtree(tmpdir)
    return tuple(new_files)


class SubprocessProgressBar:
    def __init__(self, label=None, delay=1):
        self.thread = threading.Thread(target=self.run)
        if label is None:
            self.label = ""
        else:
            self.label = f"{label}".expandtabs(TABSIZE)
        self.delay = float(delay)
        self.running = True

    def start(self) -> None:
        self.running = True
        self.t1 = time.time()
        self.thread.start()

    def run(self) -> None:
        while self.running:
            for item in ("-", "\\", "|", "/"):
                time.sleep(self.delay)
                print(
                    f"{self.label}  {item} {time.time() - self.t1:3.2f} s elapsed",
                    end="\r",
                )
                # sys.stdout.flush()

    def stop(self):
        self.running = False
        self.thread.join()
        print(f"{(len(self.label) + 12) * ' '}\r", end="\r")

    @property
    def time(self):
        return time.process_time()

    def run_with_progress(self, f, *args, **kwargs):
        self.start()
        try:
            result = f(*args, **kwargs)
        # except BaseException as e:
        #     return e
        except sp.CalledProcessError as e:
            result = e
        finally:
            self.stop()
        return result


def select_named_file(
    path: Union[Path, str],
    searchstr: Optional[str] = None,
    suffix=None,
    searchlist: List[str] = ["*"],
    how: Literal["latest", "largest"] = "latest",
) -> Union[None, Path]:
    """Select the latest or largest file in `path` directory.
    Must contain `searchstr` in filename and suffix `suffix`.
    Filter results by `searchlist`.
    :param path: path to directory
    :param searchstr: string to search for in filename
    :param suffix: suffix of file to select
    :param searchlist: list of strings to search for in filename
    :param how: how to select file
    :return: selected file"""
    path = Path(path)
    if suffix is None:
        suffix = ""
    if searchstr is None:
        searchstr = ""
    f_iter = list(
        path.glob(rf"*{searchstr.strip('*')}[.]*{suffix.strip('.')}")
    )
    searchlist = list(
        map(
            lambda x: rf'.*{searchstr}{x.strip("*")}[.]*{suffix.strip(".")}',
            searchlist,
        )
    )
    searchstr = "|".join(searchlist)
    pattern = re.compile(rf"{searchstr}", flags=re.DOTALL)
    f_list = [
        path / pattern.search(f.name).group(0)
        for f in f_iter
        if pattern.search(f.name) is not None
    ]
    if len(f_list) == 1:
        match = f_list[0]
    elif len(f_list) == 0:
        match = None
    else:
        logger.error(
            f"Found {len(f_list)} matches: "
            + ", ".join([f.name for f in f_list])
        )
        check_func_dict = {
            "latest": lambda x: x.st_mtime,
            "largest": lambda x: x.st_size,
        }
        check_func = check_func_dict[how]
        prev_file_stat = 0
        last_file = None
        for file in f_list:
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
        match = last_file
        logger.finfo(f"{how} file: {match.name!r}")
    return match


def select_file(
    path: Union[Path, str],
    searchstr: Optional[str] = None,
    suffix=None,
    how: Literal["latest", "largest"] = "latest",
) -> Union[None, Path]:
    """Select latest or largest file in `path` directory.
    Must contain `searchstr` in filename and suffix `suffix`.
    :param path: path to directory
    :param searchstr: string to search for in filename
    :param suffix: suffix of file to select
    :param how: how to select file
    :return: selected file"""
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
        if suffix is None or suffix == "":
            suffix = ""
        else:
            suffix = suffix.strip(".")
            suffix = f".{suffix}"
        if searchstr is None:
            searchstr = "*"
        f_iter = path.glob(rf"{searchstr}{suffix}")
        backups = path.glob(rf"#{searchstr}{suffix}.[1-9]*#")
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


def get_pd_idx_iter(idx: pd.MultiIndex, name_sel: List[str]) -> NDArray:
    """Get product of index levels with names in `name_sel`.
    :param idx: index
    :param name_sel: list of index level names
    :return: array of index values"""
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


def get_u_files(
    path: Union[str, Path], suffices=["gro", "top"]
) -> Tuple[Path, Path]:
    """Get files in with selected suffices and same name stem as `path`.
    :param path: path to file
    :param suffices: suffices of files to select
    :return: tuple of files"""
    files = {}
    path = Path(path)
    largest_files = ["trr"]
    how = "latest"
    for selection in suffices:
        selection = selection.strip(".")
        if selection in largest_files:
            how = "largest"
        files[selection] = select_file(path=path, suffix=selection, how=how)
    return files["gro"], files["trr"]


def _get_header(
    header_str: str, fill: str, n_linechars: int = line_length
) -> str:
    """Get header for printing formatted log headers.
    :param header_str: header string
    :param fill: fill character
    :param n_linechars: number of characters per line
    :return: header string"""
    return (
        f"\n{fill:{fill}>{n_linechars}}\n"
        f"{header_str:^{n_linechars}}\n"
        f"{fill:{fill}>{n_linechars}}\n"
    )


def backup_files(
    new_filename: Union[str, Path],
    old_filename: Optional[Union[str, Path]] = None,
) -> str:
    """Backup files.
    :param new_filename: new filename
    :param old_filename: old filename
    :return: backup string for log"""
    already_exists = list(new_filename.parent.glob(f"{new_filename.name}"))
    already_exists.extend(
        list(new_filename.parent.glob(f"{new_filename.name}.*"))
    )
    backups = []
    backup_str = ""
    if already_exists:
        suffices = [f.suffix.strip(".") for f in already_exists]
        suffices = [
            int(suffix) for suffix in suffices if re.match(r"[0-9]+", suffix)
        ]
        backup_str = f'Backing up old {new_filename.suffix.strip(".")} files.'
        for suffix in reversed(suffices):
            shutil.move(
                f"{new_filename}.{suffix}",
                f"{new_filename}.{suffix + 1}",
                copy_function=shutil.copy2,
            )
            backups.append(f"{new_filename}.{suffix + 1}")
        shutil.copy2(new_filename, f"{new_filename}.1")
        backups.append(f"{new_filename}.1")
    if old_filename:
        shutil.copy2(old_filename, new_filename)
    return backup_str


def _get_info_box(header_str, fill, n_linechars=line_length, n_fillchars=0):
    """Get info box for printing formatted log headers.
    :param header_str: header string
    :param fill: fill character
    :param n_linechars: number of characters per line
    :param n_fillchars: number of fill characters
    :return: info box string"""
    fill_len = n_fillchars // 2
    n_linechars -= 2 * fill_len
    return (
        f"\n{' ':{' '}>{fill_len}}{fill:{fill}>{n_linechars}}\n"
        f"{' ':{' '}>{fill_len}}|{header_str:^{n_linechars-2}}|\n"
        f"{' ':{' '}>{fill_len}}{fill:{fill}>{n_linechars}}\n"
    )


get_header = partial(_get_header, fill="=")

get_subheader = partial(_get_header, fill="-")

get_debugheader = partial(_get_info_box, fill="+")
get_debugheader = partial(get_debugheader, n_fillchars=50)


def open_outfile(outpath: Union[Path, str], suffix: str, default: str):
    """Process output path or set output filename to default and initialise parent directory.
    :param outpath: output path or bool
    :param suffix: suffix of output file
    :param default: default output filename
    :return: output path"""
    if type(outpath) == bool:
        outpath = Path(f"{default}.json")
    elif type(outpath) in [str, Path]:
        outpath = change_suffix(outpath, suffix)
        if not outpath.parent.is_dir():
            os.makedirs(outpath.parent)
    else:
        raise ValueError(f"Could not interpret {outpath} as path or bool.")
    return outpath


def get_file_or_str(f):
    """Get file or string from file or string."""

    @wraps(f)
    def wrapper(file_or_str, *args, **kwargs):
        import json

        import yaml

        read_dict = {".yaml": yaml.safe_load, ".json": json.load}
        # if isinstance(file_or_str, str):
        #
        if not file_or_str:
            file_str = None
        elif isinstance(file_or_str, dict):
            file_str = file_or_str
        else:
            try:
                read_func = read_dict[Path(file_or_str).suffix]
            except KeyError:
                read_func = lambda x: x.read()
            finally:
                if isinstance(file_or_str, str):
                    file_str = read_func(io.StringIO(file_or_str))
                else:
                    with open(file_or_str, "r") as file:
                        file_str = read_func(file)
        # except FileNotFoundError:
        #     file_str = get_file_or_str
        # except OSError:
        #     file_str = get_file_or_str
        # except TypeError:
        #     file_str = get_file_or_str
        # else:
        #     file_str = get_file_or_str
        return f(*args, input_string=file_str, **kwargs)

    return wrapper


def get_arr_bytes(shape: Tuple[int, int], arr_dtype) -> int:
    """Get number of bytes in array.
    :param shape: shape of array
    :param arr_dtype: dtype of array
    :return: number of bytes in array
    """
    return np.prod(shape) * np.empty(shape=(1,), dtype=arr_dtype).itemsize


def get_n_combs(N, k):
    """Get number of combinations of k items from N items.
    :param N: number of items
    :param k: number of items to select
    :return: number of combinations
    """
    return int(
        math.factorial(N - 1) / (math.factorial(k - 1) * math.factorial(N - k))
    )


def property_exists_checker(instance, property_name, prop_type=None):
    """Check if property exists and is of correct type.
    :param instance: instance of class
    :param property_name: name of property to check
    :param prop_type: type of property to check
    :return: None
    """
    prop = getattr(instance, f"_{property_name}")
    if prop is None:
        raise AttributeError(
            f"{instance.__class__.name}.{property_name} not set!"
        )
    if prop_type is not None and type(prop) != prop_type:
        setattr(instance, prop_type(prop))


def parse_yaml_with_duplicate_keys(yaml_source):
    """Parse yaml file and enumerate names of duplicate keys.
    :param yaml_source: yaml file or string
    :return: dictionary with parsed data
    """

    class DuplicateKeyLoader(yaml.loader.Loader):
        pass

    def construct_mapping(loader, node, deep=False):
        """Load data from yaml file and enumerate names of duplicate keys.
        :param loader: yaml loader
        :param node: yaml node
        :param deep: if True, recursively load data
        :return: dictionary with parsed data
        """
        mapping = defaultdict(list)
        remove_keys = []
        for k, v in node.value:
            key = loader.construct_object(k, deep=deep)
            v = loader.construct_object(v, deep=deep)
            mapping[key].append(v)
        for k, v in mapping.copy().items():
            if len(v) == 1:
                mapping[k] = v[0]
            else:
                for i in range(len(v)):
                    mapping[f"{k}_{i+1}"] = v[i]
                remove_keys.append(k)
        for k in remove_keys:
            mapping.pop(k)
        return dict(mapping)

    DuplicateKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )
    return yaml.load(yaml_source, DuplicateKeyLoader)


def parse_yaml(source, enumerate_duplicates=False):
    """Parse yaml file and with option to enumerate names of duplicate keys.
    :param source: yaml file or string
    :param enumerate_duplicates: if True, enumerate names of duplicate keys
    :return: dictionary with parsed data
    """
    if enumerate_duplicates:
        yaml_str = parse_yaml_with_duplicate_keys(source)
    else:
        yaml_str = yaml.safe_load(source)
    return yaml_str
