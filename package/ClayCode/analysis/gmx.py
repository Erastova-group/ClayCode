import os
import re
import subprocess as sp
import warnings
from functools import partial
from pathlib import Path

# from ClayAnalysis.lib import temp_file_wrapper
from ClayAnalysis.utils import execute_bash_command

# from conf.consts import OUTPATH, SYS_FILESTEM
# from cc.paths import MDP

GMX = "gmx_mpi"


def run_gmx_command(commandargs_dict, opt_args_list):
    def func_decorator(func):
        def wrapper(**kwdargs):
            for arg in kwdargs.keys():
                if arg in commandargs_dict.keys():
                    commandargs_dict[arg] = str(kwdargs[arg])
                elif arg in opt_args_list:
                    commandargs_dict[arg] = str(kwdargs[arg])
                    # print(arg)
                else:
                    warnings.warn(
                        f'Invalid argument "-{arg}" passed to "gmx '
                        rf'{re.match("run_gmx_(.*)", func.__name__).group(1)}"'
                    )
            command, outputargs = func()
            # print(commandargs_dict)
            kwd_str = " ".join(
                list(
                    map(
                        lambda key, val: f"-{key} {val}",
                        commandargs_dict.keys(),
                        commandargs_dict.values(),
                    )
                )
            )
            # print(kwd_str)
            output = sp.run(
                ["/bin/bash", "-i", "-c", f"gmx {command} {kwd_str}"],
                **outputargs,
            )
            return (output.stdout, output.stderr)

        return wrapper

    return func_decorator


@run_gmx_command(
    commandargs_dict={"o": "crdout.crdin"},
    opt_args_list=[
        "p",
        "cs",
        "box",
        "scale",
        "maxsol",
        "radius",
        "shell",
        "vel",
        "novel",
        "cp",
    ],
)
def run_gmx_solvate():
    return "solvate", {"capture_output": True, "text": True}


@run_gmx_command(
    commandargs_dict={
        "f": "grompp.mdp",
        "c": "conf.crdin",
        "p": "topol.top",
        "po": "mdout.mdp",
        "pp": "processed.top",
        "o": "topol.tpr",
        "maxwarn": 1,
    },
    opt_args_list=["ndx", "v", "nov", "renum", "norenum", "t"],
)
def run_gmx_grompp():
    return "grompp", {"capture_output": True, "text": True}


@run_gmx_command(
    commandargs_dict={},
    opt_args_list=[
        "s",
        "cpi",
        "table",
        "tablep",
        "tableb",
        "rerun",
        "ei",
        "multidir",
        "awh",
        "membed",
        "mp",
        "mn",
        "o",
        "x",
        "cpo",
        "c",
        "e",
        "g",
        "dhdl",
        "field",
        "tpi",
        "eo",
        "px",
        "pf",
        "ro",
        "ra",
        "rs",
        "rt",
        "mtx",
        "if",
        "swap",
        "deffnm",
        "xvg",
        "dd",
        "ddorder",
        "npme",
        "nt",
        "ntmpi",
        "ntomp",
        "ntomp_pme",
        "pin",
        "pinoffset",
        "pinstride",
        "gpu_id",
        "gputasks",
        "ddcheck",
        "noddcheck",
        "rdd",
        "rcon",
        "v",
        "nov",
        "maxwarn",
    ],
)
def run_gmx_mdrun():
    return "mdrun", {"capture_output": True, "text": True}


@run_gmx_command(
    commandargs_dict={"pname": "Na", "nname": "Cl"},
    opt_args_list=[
        "n",
        "s",
        "p",
        "o",
        "np",
        "pq",
        "nn",
        "nq",
        "rmin",
        "seed",
        "conc",
        "neutral",
        "noneutral",
    ],
)
def run_gmx_genion():
    return "genion", {"capture_output": True, "text": True}


@run_gmx_command(
    commandargs_dict={}, opt_args_list=["f", "s", "n", "oi", "on", "select"]
)
def run_gmx_select():
    return "select", {"capture_output": True, "text": True}


@run_gmx_command(
    commandargs_dict={"try": 9000},
    opt_args_list=[
        "f",
        "ci",
        "ip",
        "n",
        "o",
        "replace",
        "sf",
        "nq",
        "selrpos",
        "box",
        "nmol",
        "seed",
        "radius",
        "scale",
        "dr",
        "rot",
    ],
)
def run_gmx_insert_mols():
    return "insert-molecules", {"capture_output": True, "text": True}


def run_gmx_make_ndx(f: str, o: str):
    """
    Write index for crdin file to ndx.
    :param f: crdin filename
    :type f: str
    :param o: ndx filename
    :type o: str
    """
    _ = execute_bash_command(
        f'echo -e "\n q" | gmx make_ndx -f {f} -o {o}', capture_output=True
    )
    assert Path(o).is_file(), f"No index file {o} was written."


def run_gmx_genion_neutralise(
    s: str,
    p: str,
    o: str,
    n: str,
    pname: str = "Na",
    pq: int = 1,
    nname: str = "Cl",
    nq: int = -1,
):
    """
    Replace SOL with ions to neutralise system charge.
    :param s: input topology tpr filename
    :type s: str
    :param p: output topology top filename
    :type p: str
    :param o: output coordinates crdin file
    :type o: str
    :param n: system index file
    :type n: str
    :param pname: cation name
    :type pname: str
    :param pq: cation charge
    :type pq: int
    :param nname: anion name
    :type nname: int
    :param nq: anion charge
    :type nq: int
    """
    out = execute_bash_command(
        f'echo -e " SOL \n q" | '
        f"gmx genion -s {s} -p {p} -o {o} -n {n} "
        f"-pname {pname} -pq {pq} "
        f"-nname {nname} -nq {nq} "
        f"-rmin 0.2 -neutral",
        capture_output=True,
        text=True,
    )
    err = re.search(r"error", out.stdout)
    assert err is None, f"gmx genion raised an error!"
    return out


@run_gmx_command(
    commandargs_dict={"nsteps": -1},
    opt_args_list=["s", "o", "extend", "until"],
)
def run_gmx_convert_tpr():
    return "convert-tpr", {"capture_output": True, "text": True}
