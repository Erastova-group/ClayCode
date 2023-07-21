#!/usr/bin/env python3
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess as sp
import warnings
from functools import partial, singledispatch, wraps
from itertools import chain
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import MDAnalysis as mda
import numpy as np
import pandas as pd
from ClayCode.core.consts import LINE_LENGTH, exec_date, exec_time

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore")

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
        # if type(seq) == str:
        #     try:
        #         seq = int(seq)
        #     except ValueError:
        #         raise TypeError(f"Expected numeric values, found {seq}")
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


def execute_shell_command(command):
    try:
        output = sp.run(
            [command], shell=True, capture_output=True, text=True, check=True
        )
    except sp.CalledProcessError:
        shell = sp.run(
            ["echo $SHELL"],
            # shell=True,
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


def grep_file(file, regex: str):
    diff = execute_shell_command(f'grep -E "{regex}" {file}')
    return diff.stdout


def get_logfname(
    logname: str,
    run_name=None,
    time: Union[Literal[exec_time], Literal[exec_date]] = exec_date,
    logpath=None,
):
    if logpath is None:
        logpath = Path().cwd() / "logs"
        if not logpath.is_dir():
            os.mkdir(logpath)
    if run_name is None:
        run_name = ""
    else:
        run_name += "-"
    return f"{logpath}/{logname}-{run_name}{time}.log"


def get_search_str(match_dict: dict):
    return "|".join(match_dict.keys())


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
        shutil.move(file, outpath / file.name)
        new_files.append(outpath / file.name)
    for file in [tpr, log, mdp]:
        shutil.move(file, outpath / file.with_stem(f"{file.name}_em").name)
    logger.finfo(f"Done! Copied files to {outpath.name!r}")
    if rm_tempfiles:
        shutil.rmtree(tmpdir)
    return tuple(new_files)


def select_named_file(
    path: Union[Path, str],
    searchstr: Optional[str] = None,
    suffix=None,
    searchlist: List[str] = ["*"],
    how: Literal["latest", "largest"] = "latest",
):
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


def get_u_files(path: Union[str, Path], suffices=["gro", "top"]):
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


def _get_header(header_str, fill, n_linechars=LINE_LENGTH):
    return (
        f"\n{fill:{fill}>{n_linechars}}\n"
        f"{header_str:^{n_linechars}}\n"
        f"{fill:{fill}>{n_linechars}}\n"
    )


def backup_files(new_filename, old_filename=None):
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
                f"{new_filename}.{suffix}", f"{new_filename}.{suffix + 1}"
            )
            backups.append(f"{new_filename}.{suffix + 1}")
        shutil.copy(new_filename, f"{new_filename}.1")
        backups.append(f"{new_filename}.1")
    if old_filename:
        shutil.copy(old_filename, new_filename)
    return backup_str


def _get_info_box(header_str, fill, n_linechars=LINE_LENGTH, n_fillchars=0):
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
    if type(outpath) == bool:
        outpath = Path(f"{default}.json")
    elif type(outpath) in [str, Path]:
        outpath = change_suffix(outpath, suffix)
        if not outpath.parent.is_dir():
            os.makedirs(outpath.parent)
    else:
        raise ValueError(f"Could not interpret {outpath} as path or bool.")
    return outpath


if __name__ == "__main__":
    print(mda.__version__, "\n", np.__version__)


def set_mdp_parameter(
    parameter, value, mdp_str, searchex="[A-Za-z0-9 ._,\-]*?"
):
    new_str = re.sub(
        rf"(?<={parameter})(\s*)(=\s*)\s?({searchex})\s*?((\s?;[a-z0-9 ._,\-])?)(\n)",
        r"\1" + f"= {value} " + r"\4\n",
        mdp_str,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    return new_str


def add_mdp_parameter(parameter, value, mdp_str, searchex="[A-Za-z0-9 ._,]*?"):
    new_str = re.sub(
        rf"(?<={parameter})(\s*)(=\s*)\s?({searchex})\s*?((\s?;[a-z0-9 ._,\-])?)(\n)",
        r"\1= \3" + f" {value} " + r"\4\n",
        mdp_str,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    return new_str


def file_or_str(f):
    @wraps(f)
    def wrapper(file_or_str, *args, **kwargs):
        import json

        import yaml

        read_dict = {".yaml": yaml.safe_load, ".json": json.load}
        if isinstance(file_or_str, str):
            file_str = file_or_str
        else:
            try:
                read_func = read_dict[Path(file_or_str).suffix]
            except KeyError:
                read_func = lambda x: x.read()
            finally:
                with open(file_or_str, "r") as file:
                    file_str = read_func(file)
        # except FileNotFoundError:
        #     file_str = file_or_str
        # except OSError:
        #     file_str = file_or_str
        # except TypeError:
        #     file_str = file_or_str
        # else:
        #     file_str = file_or_str
        return f(*args, input_string=file_str, **kwargs)

    return wrapper


@file_or_str
def set_mdp_freeze_clay(
    uc_names: List[str],
    input_string,
    freeze_dims: List[Union[Literal["Y"], Literal["N"]]] = ["Y", "Y", "Y"],
):
    freezegrpstr = " ".join(uc_names)
    if len(freeze_dims) != 3:
        raise ValueError("Freeze dimensions must have 3 elements")
    freezearray = np.tile(freeze_dims, (len(uc_names)))
    freezedimstr = " ".join(freezearray)
    if not np.isin(freezearray, ["Y", "N"]).all():
        raise ValueError(
            f"Unexpected freeze dimensions value: {freezedimstr!r}"
            "\nAccepted options are: Y and N"
        )
    input_string = set_mdp_parameter("freezegrps", freezegrpstr, input_string)
    input_string = set_mdp_parameter("freezedim", freezedimstr, input_string)
    return input_string


@file_or_str
def mdp_to_yaml(input_string: str) -> Dict[str, str]:
    mdp_options: list = re.findall(
        r"^[a-z0-9\-_]+\s*=.*?(?=[\n;^])",
        input_string,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    # mdp_yaml = {k: v for }
    mdp_yaml: str = "\n".join(mdp_options)
    # mdp_yaml = re.sub("=", ":", mdp_yaml, flags=re.MULTILINE)
    mdp_yaml = re.sub(r"[\t ]+", "", mdp_yaml, flags=re.MULTILINE)
    # mdp_yaml = re.sub(r':', ': !mdp ', flags=re.MULTILINE)
    mdp_yaml = dict(line.split("=") for line in mdp_yaml.splitlines())
    # mdp_yaml = re.sub(r'\n\n+', '\n', mdp_yaml, flags=re.MULTILINE)
    return mdp_yaml
