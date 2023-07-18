import copy
import logging
import pathlib
import re
import shutil
import subprocess as sp
import sys
import tempfile
import textwrap
import warnings
from functools import cached_property, update_wrapper, wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ClayCode.builder.utils import select_input_option
from ClayCode.core.classes import MDPFile
from ClayCode.core.consts import LINE_LENGTH, MDP, MDP_DEFAULTS
from ClayCode.core.utils import (
    execute_shell_command,
    get_header,
    get_subheader,
    set_mdp_freeze_clay,
    set_mdp_parameter,
)

DEFAULT_GMX = "gmx"

logger = logging.getLogger(__name__)


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
            gmx_commands = GMXCommands(gmx_alias=gmx_alias, *args, **kwargs)
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
            gmx_commands = GMXCommands(gmx_alias=gmx_alias, *args, **kwargs)
        result = f(*args, gmx_commands=gmx_commands, **kwargs)
        return result

    return wrapper


class GMXCommands:
    def __init__(
        self,
        gmx_alias="gmx",
        mdp_template=None,
        mdp_defaults={},
    ):
        self.gmx_alias = gmx_alias
        _ = self.gmx_header
        try:
            self._mdp_template = mdp_template
        except TypeError:
            pass
        self._mdp_template = mdp_template
        self._mdp_defaults = mdp_defaults
        self.logger = logging.getLogger(self.__class__.__name__)
        logger.info(textwrap.fill(f"\n{self.gmx_info}", width=LINE_LENGTH))

    @property
    def mdp_defaults(self):
        if self._mdp_defaults:
            return self._mdp_defaults
        else:
            return MDP_DEFAULTS[int(self.version)]

    @property
    def default_mdp_file(self):
        return MDP / self.version / "mdp_prms.mdp"

    @property
    def mdp_string(self):
        if self.mdp_template is not None:
            with open(self.mdp_template) as mdp_file:
                mdp_str = mdp_file.read()
            return mdp_str

    @property
    def mdp_template(self):
        if self._mdp_template:
            return self._mdp_template
        else:
            return self.default_mdp_file

    @mdp_template.setter
    def mdp_template(self, mdp_filename):
        try:
            self._mdp_template = Path(mdp_filename).with_suffix(".mdp")
            assert (
                self._mdp_template.is_file()
            ), f"MDP file {self._mdp_template.resolve()!r} does not exist"
        except TypeError:
            logger.debug(
                f"Not setting MDP template, mdp_template = {mdp_filename!r}"
            )

    def get_mdp_parameter_file(
        self,
        mdp_file: Optional[str] = None,
        mdp_prms: Optional[Dict[str, str]] = None,
        run_type: Optional[str] = None,
        freeze_dims: Optional[List[str]] = None,
        freeze_grps: Optional[List[str]] = None,
    ):
        # self.logger.info(get_header(f"Getting mdp options\n"))
        mdp_temp_file = True
        if mdp_file:
            file = Path(mdp_file).with_suffix(".mdp")
            if file != self.mdp_template:
                with open(file, "r") as mdp_file:
                    mdp_str = mdp_file.read()
                    mdp_temp_file = None
        if mdp_temp_file:
            mdp_str = self.mdp_string
            mdp_temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mdp",
                prefix=self.mdp_template.stem,
            )
            file = mdp_temp_file = Path(mdp_temp_file.name)
        run_dict = {}
        if (
            not mdp_temp_file
            and not mdp_prms
            and not (freeze_grps and freeze_dims)
        ):
            return
        elif mdp_temp_file and run_type:
            try:
                run_dict.update(MDP_DEFAULTS[run_type])
            except KeyError:
                logger.warning(f"{run_type} is invalid GROMACS run type.")
            except ValueError:
                logger.warning(
                    f"{run_type} is an invalid argument for run_type"
                )
        if mdp_temp_file and not mdp_prms:
            mdp_prms = self.mdp_defaults
        for prm_dict in [run_dict, mdp_prms]:
            try:
                for parameter, value in prm_dict.items():
                    logger.debug(f"{parameter}: {value}")
                    mdp_str = set_mdp_parameter(parameter, value, mdp_str)
            except AttributeError:
                logger.debug("No parameters/run type defined.")
        if isinstance(freeze_dims, list) and isinstance(freeze_grps, list):
            mdp_str = set_mdp_freeze_clay(
                uc_names=freeze_grps,
                file_or_str=mdp_str,
                freeze_dims=freeze_dims,
            )

        with open(file, "w") as mdp_outfile:
            mdp_outfile.write(mdp_str)
        return mdp_temp_file

    def run_gmx_command(
        commandargs_dict: Dict[str, Any], opt_args_list: List[Any]
    ):
        def func_decorator(func: Callable):
            def wrapper(
                self,
                **kwdargs,
            ):
                gmx_args = copy.copy(commandargs_dict)
                for arg in kwdargs.keys():
                    if arg in gmx_args.keys():
                        gmx_args[arg] = str(kwdargs[arg])
                    elif arg in opt_args_list:
                        gmx_args[arg] = str(kwdargs[arg])
                    else:
                        warnings.warn(
                            rf'Invalid argument "-{arg}" passed to "gmx '
                            rf'{re.match(r"run_gmx_(.*)", func.__name__).group(1)}"'
                        )
                temp_file = None
                try:
                    mdp_file = gmx_args["f"]
                    if mdp_file:
                        mdp_file = MDPFile(mdp_file, check=True)
                except KeyError:
                    command, outputargs = func(self)
                except ValueError:
                    command, outputargs = func(self)
                except FileNotFoundError:
                    command, outputargs = func(self)
                else:
                    prm_dict = {}
                    for prm in [
                        "mdp_prms",
                        "run_type",
                        "freeze_dims",
                        "freeze_grps",
                    ]:
                        try:
                            prm_dict[prm] = kwdargs[prm]
                        except KeyError:
                            prm_dict[prm] = None
                    command, outputargs = func(self, mdp_file, **prm_dict)
                    if command == "grompp":
                        if outputargs["temp_file"]:
                            temp_file = outputargs["temp_file"]
                            gmx_args["f"] = temp_file
                    else:
                        logger.warning("Bug: This point should ot be reached!")

                kwd_str = " ".join(
                    list(
                        map(
                            lambda key, val: f"-{key} {val}",
                            gmx_args.keys(),
                            gmx_args.values(),
                        )
                    )
                )
                with tempfile.TemporaryDirectory() as odir:
                    try:
                        output = execute_shell_command(
                            f"cd {odir}; {self.gmx_alias} {command} {kwd_str} -nobackup"
                        )
                        temp_file.unlink()
                    except FileNotFoundError as e:
                        logger.error("This point should not be reached!")
                        sys.exit()
                    except AttributeError:
                        pass
                    except sp.CalledProcessError as e:
                        logger.error(
                            f"GROMACS raised error code {e.returncode}!\n"
                        )
                        print_error = select_input_option(
                            query="Print error message? [y]es/[n]o (default no)\n",
                            instance_or_manual_setup=True,
                            options=["y", "n", ""],
                            result=None,
                            result_map={"y": True, "n": False, "": True},
                        )
                        if print_error:
                            print(e.stderr)
                            if command == "grompp":
                                try:
                                    odir = re.search(
                                        r"-o ([\w/.\-]+) ", kwd_str
                                    ).group(1)
                                    odir = pathlib.Path(odir)
                                    shutil.copy(
                                        temp_file, odir.parent / temp_file.name
                                    )
                                    temp_file.unlink()
                                except AttributeError:
                                    pass
                        sys.exit(e.returncode)
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
                    return (
                        err,
                        out,
                    )  # -> gmx process stderr, gmx process stdout

            return wrapper

        return func_decorator

    def __run_without_args(self) -> Tuple[str, str]:
        output = execute_shell_command(f"{self.gmx_alias}")
        err, out = output.stderr, output.stdout
        return err, out

    @property
    def version(self):
        version = re.search("20\d+", self.gmx_header).group(0)
        if version is None:
            raise RuntimeError(
                f"{self.gmx_header} is not a supported version of GROMACS"
            )
        if int(version) not in MDP_DEFAULTS.keys():
            supported_versions = ", ".join(
                [
                    f"{v}"
                    for v in MDP_DEFAULTS.keys()
                    if str(v).startswith("20")
                ]
            )
            raise ValueError(
                f"GROMACS version {version} is not supported.\n"
                f"ClayCode works with GROMACS {supported_versions}"
            )
        return version

    @cached_property
    def gmx_header(self):
        err, _ = self.__run_without_args()
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
        output = execute_shell_command(f"{self.gmx_alias}")
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
            "f": None,
            "c": "conf.crdin",
            "p": "topol.top",
            "po": "mdout.mdp",
            "pp": "processed.top",
            "o": "topol.tpr",
            "maxwarn": 2,
            "renum": "",
        },
        opt_args_list=["ndx", "v", "nov", "renum", "norenum", "t"],
    )
    def run_gmx_grompp(
        self,
        mdp_file: Optional[str] = None,
        mdp_prms: Optional[Dict[str, str]] = None,
        run_type: Optional[str] = None,
        freeze_dims: Optional[List[str]] = None,
        freeze_grps: Optional[List[str]] = None,
    ):
        temp_file = self.get_mdp_parameter_file(
            mdp_file=mdp_file,
            mdp_prms=mdp_prms,
            run_type=run_type,
            freeze_dims=freeze_dims,
            freeze_grps=freeze_grps,
        )
        return "grompp", {
            "capture_output": True,
            "text": True,
            "temp_file": temp_file,
        }

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
        _ = execute_shell_command(
            f'echo -e "\n q" | {self.gmx_alias} make_ndx -f {f} -o {o}'
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
            output = execute_shell_command(
                f'cd {odir}; echo -e " SOL \n q" | '
                f"{self.gmx_alias} genion -s {s} -p {p} -o {o} -n {n} "
                f"-conc {conc} "
                f"{istr} "
                f"-rmin 0.2 -noneutral -nobackup"
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
            logger.debug(f"{self.gmx_alias} genion completed successfully.")
        return err, out

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
            output = execute_shell_command(
                f'cd {odir}; echo -e " SOL \n q" | '
                f"{self.gmx_alias} genion -s {s} -p {p} -o {o} -n {n} "
                f"-pname {pname} -pq {pq} "
                f"-nname {nname} -nq {nq} "
                f"-rmin 0.2 -neutral -nobackup"
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
            logger.debug(f"{self.gmx_alias} genion completed successfully.")
            return err, out  # -> gmx process stderr, gmx process stdout

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
            output = execute_shell_command(
                f'cd {odir}; echo -e " SOL \n q" | '
                f"{self.gmx_alias} genion -s {s} -p {p} -o {o} -n {n} "
                f"-pname {pname} -pq {pq} -nn {nn} "
                f"-nname {nname} -nq {nq} -np {np} "
                f"-rmin 0.2 -noneutral -nobackup"
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
            logger.debug(f"{self.gmx_alias} genion completed successfully.")
            return err, out  # -> gmx process stderr, gmx process stdout

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
            elif re.search(
                r"The largest distance between excluded atoms is .*?"
                r" forces and energies.",
                out,
                flags=re.MULTILINE | re.DOTALL,
            ):
                logger.error(
                    "GROMACS terminated due to too large distance between atoms!"
                )
            else:
                raise RuntimeError(f"{out}\nGROMACS raised an error!")
