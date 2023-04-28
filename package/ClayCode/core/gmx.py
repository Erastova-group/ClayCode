import re
import subprocess as sp
import tempfile
import warnings
from functools import cached_property, update_wrapper, wraps
from pathlib import Path
from typing import Tuple

from ClayCode.core.log import logger
from ClayCode.core.utils import execute_bash_command

DEFAULT_GMX = "gmx"


def add_gmx_args(f):
    def wrapper(
        instance, *args, gmx_commands=None, gmx_alias=DEFAULT_GMX, **kwargs
    ):
        update_wrapper(wrapper=wrapper, wrapped=f)
        if gmx_commands is not None:
            assert (
                gmx_commands.__class__.__name__ == "GMXCommands"
            ), "Wrong type: Expected GMXCommands instance!"
        else:
            # logger.info(f'Using GROMACS alias {gmx_alias}')
            gmx_commands = GMXCommands(gmx_alias=gmx_alias)
        result = f(instance, *args, **kwargs)
        instance.gmx_commands = gmx_commands
        return result

    return wrapper


def gmx_command_wrapper(f):
    @wraps(f)
    def wrapper(*args, gmx_commands=None, gmx_alias=DEFAULT_GMX, **kwargs):
        if gmx_commands is not None:
            assert (
                gmx_commands.__class__.__name__ == "GMXCommands"
            ), "Wrong type: Expected GMXCommands instance!"
        else:
            # logger.info(f'Using GROMACS alias {gmx_alias}')
            gmx_commands = GMXCommands(gmx_alias=gmx_alias)
        result = f(*args, gmx_commands=gmx_commands, **kwargs)
        return result

    return wrapper


class GMXCommands:
    def __init__(self, gmx_alias="gmx"):
        self.gmx_alias = gmx_alias
        _ = self.gmx_header
        logger.info(f"\n{self.gmx_info}")

    def run_gmx_command(commandargs_dict, opt_args_list):
        def func_decorator(func):
            def wrapper(self, **kwdargs):
                for arg in kwdargs.keys():
                    if arg in commandargs_dict.keys():
                        commandargs_dict[arg] = str(kwdargs[arg])
                    elif arg in opt_args_list:
                        commandargs_dict[arg] = str(kwdargs[arg])
                    else:
                        warnings.warn(
                            rf'Invalid argument "-{arg}" passed to "gmx '
                            rf'{re.match(r"run_gmx_(.*)", func.__name__).group(1)}"'
                        )
                command, outputargs = func(self)
                kwd_str = " ".join(
                    list(
                        map(
                            lambda key, val: f"-{key} {val}",
                            commandargs_dict.keys(),
                            commandargs_dict.values(),
                        )
                    )
                )
                with tempfile.TemporaryDirectory() as odir:
                    # logger.info(self.gmx_alias)
                    output = sp.run(
                        [
                            "/bin/bash",
                            "-i",
                            "-c",
                            f"cd {odir}; {self.gmx_alias} {command} {kwd_str} -nobackup",
                        ],
                        **outputargs,
                    )
                logger.debug(f"{self.gmx_alias} {command} {kwd_str} -nobackup")
                # logger.info(self.gmx_info)
                out, err = output.stdout, output.stderr
                error = self.search_gmx_error(err)
                if error is None:
                    logger.debug(
                        f"{self.gmx_alias} {command} completed successfully."
                    )
                if command == "mdrun":
                    return (
                        error,
                        err,
                        out,
                    )  # -> gmx process error match, gmx process stderr, gmx process stdout
                else:
                    #    logger.error(f'{self.gmx_alias} {command} raised an error!\n{out}')
                    return (
                        err,
                        out,
                    )  # -> gmx process stderr, gmx process stdout

            return wrapper

        return func_decorator

    def __run_without_args(self) -> Tuple[str, str]:
        output = sp.run(
            [
                "/bin/bash",
                "-i",
                "-c",
                f"{self.gmx_alias}",
            ],
            text=True,
            capture_output=True,
        )
        err, out = output.stderr, output.stdout
        return err, out

    @cached_property
    def gmx_header(self):
        err, out = self.__run_without_args()
        check_str = re.search(
            r":-\) (.+?) \(-:", err, flags=re.MULTILINE | re.DOTALL
        )
        if check_str is not None:
            check_str = check_str.group(1)
        else:
            raise RuntimeError(
                f"{self.gmx_alias} is no valid alias for a GROMACS installation!"
            )
        return check_str

    @cached_property
    def gmx_info(self) -> str:
        output = sp.run(
            [
                "/bin/bash",
                "-i",
                "-c",
                f"{self.gmx_alias}",
            ],
            text=True,
            capture_output=True,
        )
        err, out = output.stderr, output.stdout
        try:
            gmx_version = re.search(
                "GROMACS:\s+(.*?)\n", err, flags=re.MULTILINE | re.DOTALL
            ).group(1)
        except AttributeError:
            gmx_version = re.search(
                "GROMACS - (.*)", self.gmx_header, flags=re.DOTALL
            ).group(1)
        gmx_executable = re.search(
            "Executable:\s+(.*?)\n", err, flags=re.MULTILINE | re.DOTALL
        ).group(1)
        return f"Using GROMACS alias {gmx_version}, in {gmx_executable}"

    @run_gmx_command(
        commandargs_dict={"o": "crdout.gro"},
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
    def run_gmx_solvate(
        self,
    ):
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
    def run_gmx_grompp(self):
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
    def run_gmx_mdrun(self):
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
    def run_gmx_genion(self):
        return "genion", {"capture_output": True, "text": True}

    @run_gmx_command(
        commandargs_dict={},
        opt_args_list=["f", "s", "n", "oi", "on", "select"],
    )
    def run_gmx_select(self):
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
    def run_gmx_insert_mols(self):
        return "insert-molecules", {"capture_output": True, "text": True}

    def run_gmx_make_ndx(self, f: str, o: str):
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

    def run_gmx_genion_conc(
        self, s: str, p: str, o: str, n: str, conc: float, iname: str, iq: int
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
        if iq > 0:
            istr = f"-pname {iname} -pq {iq}"
        else:
            istr = f"-nname {iname} -nq {iq}"
        with tempfile.TemporaryDirectory() as odir:
            output = execute_bash_command(
                f'cd {odir}; echo -e " SOL \n q" | '
                f"{self.gmx_alias} genion -s {s} -p {p} -o {o} -n {n} "
                f"-conc {conc} "
                f"{istr} "
                f"-rmin 0.2 -noneutral -nobackup",
                capture_output=True,
                text=True,
            )
            logger.debug(
                f'echo -e " SOL \n q" | '
                f"{self.gmx_alias} genion -s {s} -p {p} -o {o} -n {n} "
                f"-conc {conc} "
                f"{istr} "
                f"-rmin 0.2 -noneutral -nobackup"
            )
            err, out = output.stderr, output.stdout
            self.search_gmx_error(err)
            # if err is None:
            logger.debug(f"{self.gmx_alias} genion completed successfully.")
            # else:
            #     logger.error(f'{self.gmx_alias} genion raised an error!\n{out}')
        return err, out
        # err = re.search(r"error", out.stdout)
        # assert err is None, f"gmx genion raised an error!"
        # return out

    def run_gmx_genion_neutralise(
        self,
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
        with tempfile.TemporaryDirectory() as odir:
            output = execute_bash_command(
                f'cd {odir}; echo -e " SOL \n q" | '
                f"{self.gmx_alias} genion -s {s} -p {p} -o {o} -n {n} "
                f"-pname {pname} -pq {pq} "
                f"-nname {nname} -nq {nq} "
                f"-rmin 0.2 -neutral -nobackup",
                capture_output=True,
                text=True,
            )
            logger.debug(
                f'echo -e " SOL \n q" | '
                f"{self.gmx_alias} genion -s {s} -p {p} -o {o} -n {n} "
                f"-pname {pname} -pq {pq} "
                f"-nname {nname} -nq {nq} "
                f"-rmin 0.2 -neutral -nobackup"
            )
            err, out = output.stderr, output.stdout
            self.search_gmx_error(err)
            # if err is None:
            logger.debug(f"{self.gmx_alias} genion completed successfully.")
            # else:
            #     logger.error(f'{self.gmx_alias} genion raised an error!\n{out}')
            return err, out  # -> gmx process stderr, gmx process stdout
            # err = re.search(r"error", out.stdout)
            # assert err is None, f"gmx genion raised an error!"
            # return err, out

    def run_gmx_genion_add_n_ions(
        self,
        s: str,
        p: str,
        o: str,
        n: str,
        pname: str = "Na",
        pq: int = 1,
        np: int = 0,
        nname: str = "Cl",
        nq: int = -1,
        nn: int = 0,
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
        with tempfile.TemporaryDirectory() as odir:
            output = execute_bash_command(
                f'cd {odir}; echo -e " SOL \n q" | '
                f"{self.gmx_alias} genion -s {s} -p {p} -o {o} -n {n} "
                f"-pname {pname} -pq {pq} -nn {nn} "
                f"-nname {nname} -nq {nq} -np {np} "
                f"-rmin 0.2 -noneutral -nobackup",
                capture_output=True,
                text=True,
            )
            logger.debug(
                f'echo -e " SOL \n q" | '
                f"{self.gmx_alias} genion -s {s} -p {p} -o {o} -n {n} "
                f"-pname {pname} -pq {pq} -nn {nn} "
                f"-nname {nname} -nq {nq} -np {np} "
                f"-rmin 0.2 -noneutral -nobackup"
            )
            out, err = output.stdout, output.stderr
            self.search_gmx_error(err)
            # if err is None:
            logger.debug(f"{self.gmx_alias} genion completed successfully.")
            # else:
            #     logger.error(f'{self.gmx_alias} genion raised an error!\n{out}')
            return err, out  # -> gmx process stderr, gmx process stdout
        # err = re.search(r"error|invalid", out.stdout)
        # assert err is None, f"gmx genion raised an error!"
        # return out

    @run_gmx_command(
        commandargs_dict={"nsteps": -1},
        opt_args_list=["s", "o", "extend", "until"],
    )
    def run_gmx_convert_tpr(self):
        return "convert-tpr", {"capture_output": True, "text": True}

    @staticmethod
    def search_gmx_error(out: str) -> None:
        check = re.search(
            r":-\) GROMACS", out, flags=re.IGNORECASE | re.MULTILINE
        )
        if check is None:
            raise ValueError(
                f"\n{out}\nNo GROMACS output string used for GROMACS check!"
            )
        err = re.search(
            r"\n?(.?)(error|exception|invalid)(.?\n).*(GROMACS reminds you)?",
            out,
            flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )
        if err is not None:
            if re.search(
                "Too many LINCS warnings",
                out,
                flags=re.MULTILINE | re.IGNORECASE,
            ):
                logger.error("GROMACS terminated due to LINCS warnings!")
            else:
                raise RuntimeError(f"{out}\nGROMACS raised an error!")
            # f'\t{err.group(1)}{err.group(2)}{err.group(3)}'
