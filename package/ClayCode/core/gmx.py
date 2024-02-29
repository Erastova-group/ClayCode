#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r""":mod:`ClayCode.core.gmx` --- GROMACS commands
=================================================
"""

from __future__ import annotations

import copy
import logging
import pathlib
import re
import shutil
import subprocess as sp
import sys
import tempfile
import warnings
from functools import cached_property, update_wrapper, wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from ClayCode.builder.utils import select_input_option
from ClayCode.core.classes import (
    GROFile,
    MDPFile,
    set_mdp_freeze_groups,
    set_mdp_parameter,
)
from ClayCode.core.consts import ANGSTROM, TABSIZE
from ClayCode.core.utils import SubprocessProgressBar, execute_shell_command
from ClayCode.data.consts import MDP, MDP_DEFAULTS

DEFAULT_GMX = "gmx"

logger = logging.getLogger(__name__)

__all__ = [
    "GMXCommands",
    "gmx_command_wrapper",
    "check_box_lengths",
    "add_gmx_args",
]


def add_gmx_args(f):
    """Add :class:`GMXCommands` to instance if not already present"""

    @wraps(f)
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
    """Pass :class:`GMXCommands` to function if not already specified"""

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
    """Class for running GROMACS commands"""

    def __init__(
        self, gmx_alias="gmx", mdp_template=None, mdp_defaults={}
    ) -> None:
        """Initialise GMXCommands instance.
        :param gmx_alias: GROMACS bash alias, defaults to "gmx"
        :type gmx_alias: str, optional
        :param mdp_template: MDP template file, defaults to None
        :type mdp_template: str, optional
        :param mdp_defaults: MDP defaults, defaults to {}
        :type mdp_defaults: dict, optional
        :return: None
        """
        self.gmx_alias = gmx_alias
        _ = self.gmx_header
        # try:
        #     self._mdp_template = mdp_template
        # except TypeError:
        #     pass
        self._mdp_template = mdp_template
        self._mdp_defaults = mdp_defaults
        self.logger = logging.getLogger(self.__class__.__name__)
        logger.finfo(f"{self.gmx_info}", initial_linebreak=True)
        self._default_mdp_file = None
        self._init_default_mdp_file()

    def _init_default_mdp_file(self):
        if self._default_mdp_file is None:
            _default_mdp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mdp",
                prefix=f"{self.gmx_alias}_{self.version}_mdp_prms",
            )

            default_mdp = MDP_DEFAULTS[int(self.version)]
            default_mdp = list(
                map(
                    lambda prms: f"{prms[0]:<30} = {prms[1]:<30}\n",
                    default_mdp.items(),
                )
            )
            default_mdp = "".join(default_mdp)
            with open(_default_mdp_file.name, "w") as mdp_file:
                mdp_file.write(default_mdp)
            self._default_mdp_file = MDPFile(_default_mdp_file.name)

    @property
    def mdp_defaults(self) -> Dict[str, Any]:
        """Default MDP options for GROMACS version
        :return: MDP defaults
        :rtype: Dict[str, Any]"""
        if self._mdp_defaults:
            return self._mdp_defaults
        else:
            return MDP_DEFAULTS[int(self.version)]

    @property
    def default_mdp_file(self) -> MDPFile:
        """File with default MDP options for GROMACS version
        :return: MDP file
        :rtype: MDPFile"""
        return MDPFile(
            self._default_mdp_file
        )  # MDPFile(MDP / self.version / "mdp_prms.mdp")
        # return MDPFile(MDP / self.version / "mdp_prms.mdp")

    @property
    def mdp_string(self) -> str:
        """String with default MDP options for GROMACS version
        :return: MDP string
        :rtype: str"""
        if self.mdp_template is not None:
            with open(self.mdp_template) as mdp_file:
                mdp_str = mdp_file.read()
            return mdp_str

    @property
    def mdp_template(self) -> MDPFile:
        """Template for MDP options file
        :return: MDP template
        :rtype: MDPFile"""
        if self._mdp_template:
            return MDPFile(self._mdp_template)
        else:
            return MDPFile(self.default_mdp_file)

    @mdp_template.setter
    def mdp_template(self, mdp_filename: str) -> None:
        """Set template for MDP options file
        :param mdp_filename: MDP template filename
        :type mdp_filename: str
        :return: None
        """
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
        freeze_dims: Optional[Union[str, List[str]]] = None,
        freeze_grps: Optional[List[str]] = None,
    ):
        """Write MDP options file.
        `mdp_prms can be all valid MDP options for
        :param mdp_file: MDP options filename, defaults to None
        :type mdp_file: str, optional
        :param mdp_prms: MDP options dictionary, defaults to None
        :type mdp_prms: Dict[str, str], optional
        :param run_type: GROMACS run type, defaults to None
        :type run_type: str, optional
        :param freeze_dims: Dimensions to freeze, defaults to None
        :type freeze_dims: Union[str, List[str]], optional
        :param freeze_grps: Groups to freeze, defaults to None
        :type freeze_grps: List[str], optional
        """
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
                delete=False, suffix=".mdp", prefix=self.mdp_template.stem
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
            mdp_prms = {}
        # Set MDP parameters with user-set parameters taking precedence
        # over internal run-type defaults over internal gmx-version defaults
        for prm_dict in [self.mdp_defaults, run_dict, mdp_prms]:
            try:
                for parameter, value in prm_dict.items():
                    logger.debug(f"{parameter}: {value}")
                    if value != "":
                        mdp_str = set_mdp_parameter(parameter, value, mdp_str)
            except AttributeError:
                logger.debug("No parameters/run type defined.")
        if isinstance(freeze_grps, str):
            freeze_grps = [freeze_grps]
        if isinstance(freeze_dims, list) and isinstance(freeze_grps, list):
            mdp_str = set_mdp_freeze_groups(
                uc_names=freeze_grps,
                file_or_str=mdp_str,
                freeze_dims=freeze_dims,
            )
        mdp_str = re.sub(
            "(^|\n)[^\n]*?\s*?=\s*?\n",
            r"\1",
            mdp_str,
            flags=re.MULTILINE | re.DOTALL,
        )
        with open(file, "w") as mdp_outfile:
            mdp_outfile.write(mdp_str)
        return mdp_temp_file

    def run_gmx_command(
        commandargs_dict: Dict[str, Any], opt_args_list: List[Any]
    ):
        def func_decorator(func: Callable):
            def wrapper(self, debug_mode=False, **kwdargs):
                gmx_args = copy.copy(commandargs_dict)
                for arg, val in kwdargs.items():
                    if val is None:
                        logger.fdebug(
                            debug_mode,
                            'Skipping argument "{arg}" with value "None"',
                        )
                        continue
                    elif arg in gmx_args.keys():
                        gmx_args[arg] = str(val)
                    elif arg in opt_args_list:
                        gmx_args[arg] = str(val)
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
                if command == "grompp":
                    mdp_prms = MDPFile(gmx_args["f"]).to_dict()
                    crd_file = GROFile(gmx_args["c"])
                    box_dims = crd_file.universe.dimensions[:3]
                    check_box_lengths(mdp_prms, box_dims)
                with tempfile.TemporaryDirectory() as odir:
                    try:
                        arrow = "\u279E"
                        label = (
                            f"\t{arrow} Running {self.gmx_alias} {command!r}"
                        )
                        if debug_mode:
                            self.label += f" {kwd_str} -nobackup"
                        progress_bar = SubprocessProgressBar(label=label)
                        output = progress_bar.run_with_progress(
                            execute_shell_command,
                            f"cd {odir}; {self.gmx_alias} {command} {kwd_str} -nobackup",
                        )
                        # else:
                        #     output = execute_shell_command(
                        #         f"cd {odir}; {self.gmx_alias} {command} {kwd_str} -nobackup"
                        #     )
                    except FileNotFoundError as e:
                        logger.ferror("This point should not be reached!")
                        sys.exit(1)
                    except AttributeError:
                        pass
                    except sp.CalledProcessError as e:
                        logger.ferror(
                            f"GROMACS raised error code {e.returncode}!\n"
                        )
                        print_error = select_input_option(
                            query="Print error message? [y]es/[n]o (default yes)\n",
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
                                    shutil.copy2(
                                        temp_file, odir.parent / temp_file.name
                                    )
                                    temp_file.unlink()
                                except AttributeError:
                                    pass
                        # sys.exit(e.returncode)
                        sys.exit(3)
                out, err = output.stdout, output.stderr
                error = self.search_gmx_error(err)
                if error is None:
                    logger.debug(
                        f"{self.gmx_alias} {command} completed successfully."
                    )
                else:
                    pass
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
            "maxwarn": 0,
            "renum": "",
        },
        opt_args_list=["n", "v", "nov", "renum", "norenum", "t"],
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
        return (
            "grompp",
            {"capture_output": True, "text": True, "temp_file": temp_file},
        )

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

    def run_gmx_make_ndx_with_new_sel(
        self, f: str, o: str, sel_str: str, sel_name: Optional[str] = None
    ) -> str:
        """
        Write index for crdin file to ndx.
        :param f: crdin filename
        :type f: str
        :param o: ndx filename
        :type o: str
        """
        output = execute_shell_command(
            f'echo " {sel_str} \n q" | {self.gmx_alias} make_ndx -f {f} -o {o} -nobackup'
        )
        outfile = Path(o)
        if not outfile.is_file():
            logger.ferror(f"No index file {o!r} was written.")
            sys.exit(3)
        out, err = output.stdout, output.stderr
        # group_outp_error = re.search(
        #     r"\n>.*\n\s*\d+\s+(.*)\s*:\s+(\d+)\s+atoms\s*\n.*>",
        #     out,
        #     flags=re.MULTILINE | re.DOTALL,
        # )
        no_residue = re.search(
            "\n>\s*?\n.*?\nFound 0 atoms with (.*?)\s*?\n",
            out,
            flags=re.MULTILINE | re.DOTALL,
        )
        if no_residue is not None:
            logger.ferror(
                f"Invalid group selector: {sel_str}.\n No atoms with {no_residue.group(1)} were found."
            )
            sys.exit(3)
        syntax_error = re.search(
            "\n>\s*?\n.*?\nSyntax error: (.*?)\n.*?\n>",
            out,
            flags=re.MULTILINE | re.DOTALL,
        )
        if syntax_error is not None:
            logger.ferror(
                f"Invalid group selector: {sel_str}.\n {syntax_error.group(1)}"
            )
            sys.exit(3)
        group_outp = re.search(
            r"\n>\s*?\n.*?(\d+)\s*?(atoms)?\s*?\n+?>",
            out,
            flags=re.MULTILINE | re.DOTALL,
        )
        if group_outp is None:
            logger.ferror(
                f"Invalid group selector: {sel_str}.\n No group was created."
            )
            sys.exit(3)
        else:
            group_name = re.sub("\s+", "_", sel_str)
            group_name = re.sub("[rati]_?", r"", group_name)
            group_name = re.sub("!_", "!", group_name)
            # group_name = group_outp.group(1)
            group_n_atoms = int(group_outp.group(1))
        if sel_name is not None:
            with open(o, "r") as ndx_file:
                ndx_str = ndx_file.read()
            ndx_str = re.sub(
                f"\[ {re.escape(group_name)} \]",
                f"[ {sel_name} ]",
                ndx_str,
                flags=re.MULTILINE | re.DOTALL,
            )
            with open(o, "w") as ndx_file:
                ndx_file.write(ndx_str)
            group_name = sel_name
        logger.finfo(
            f"Group {group_name!r} with {group_n_atoms} atoms was created.",
            indent="\t",
        )
        return out

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
                logger.ferror("GROMACS terminated due to LINCS warnings!")
            elif re.search(
                r"The largest distance between excluded atoms is .*?"
                r" forces and energies.",
                out,
                flags=re.MULTILINE | re.DOTALL,
            ):
                logger.ferror(
                    "GROMACS terminated due to too large distance between atoms!"
                )
            else:
                raise RuntimeError(f"{out}\nGROMACS raised an error!")


def check_box_lengths(mdp_prms, box_dims):
    cutoffs = []
    for cutoff in ["rlist", "rvdw", "rcoulomb"]:
        try:
            cutoffs.append(float(mdp_prms[cutoff]))
        except KeyError:
            pass
        except ValueError:
            pass
    max_cutoff = np.max(cutoffs) * 10  # in A
    min_box_length = np.min(box_dims)
    if max_cutoff >= (0.5 * min_box_length):
        logger.finfo(
            f"Shortest box vector ({min_box_length:.1f} {ANGSTROM}) needs to be at least twice as long as "
            f"the selected GROMACS cutoff ({max_cutoff:.1f} {ANGSTROM}).\n\n"
            f"Aborting model construction."
        )
        sys.exit(4)
