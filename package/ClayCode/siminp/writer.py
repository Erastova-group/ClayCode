from __future__ import annotations

import logging
import re
import shutil
import tempfile
from abc import ABC, abstractmethod
from collections import UserDict, UserString
from functools import cached_property, wraps
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Literal,
    NewType,
    Optional,
    Type,
    TypeVar,
    Union,
    get_origin,
)

import numpy as np
from caseless_dictionary import CaselessDict
from ClayCode.core.cctypes import AnyDir, PathOrStr
from ClayCode.core.classes import (
    Dir,
    File,
    ForceField,
    GROFile,
    MDPFile,
    dict_to_mdp,
    get_mdp_data_dict_function_decorator,
    get_mdp_data_dict_method_decorator,
    init_path,
    mdp_to_dict,
    set_mdp_freeze_groups,
    set_mdp_parameter,
)
from ClayCode.core.consts import (
    FF,
    MDP,
    MDP_DEFAULTS,
    USER_MDP,
    exec_date,
    exec_time,
)
from ClayCode.core.lib import generate_restraints, select_clay
from ClayCode.core.utils import (
    get_file_or_str,
    get_search_str,
    property_exists_checker,
)
from ClayCode.siminp.types import (
    DefaultGMXRunType,
    GMXRunType,
    NondefaultGMXRun,
)
from MDAnalysis import AtomGroup

logger = logging.getLogger(__name__)


def add_kwargs(
    instance: NondefaultGMXRun, n_args: Optional[int] = None, **kwargs
) -> None:
    assert instance.__class__.__name__ in [
        "EMRun",
        "EQRun",
        "DSpaceRun",
        "EQRunFixed",
        "EQRunRestrained",
    ], f"Wrong instance type: {instance.__class__.__name__}, expected {NondefaultGMXRun}"
    if n_args is None:
        n_args = len(instance._options)
    found_args = 0
    for k, v in kwargs.items():
        if k.lower() in instance._options:
            setattr(instance, k.lower(), v)
            found_args += 1
    if not n_args == found_args:
        raise ValueError(
            f"Expected {n_args} arguments for initialising {instance.__class__.__name__!r}, found {found_args}"
        )


class GMXRun(UserDict, ABC):
    _default_mdp_key = ""
    _options = ["nsteps"]

    def __init__(
        self,
        run_id: int,
        name: str,
        igro: Optional[PathOrStr] = None,
        itop: Optional[PathOrStr] = None,
        deffnm: Optional[str] = None,
        odir: Optional[PathOrStr] = None,
    ):
        self._top = itop
        self._gro = None
        if run_id == 1 and igro is None:
            raise ValueError(f"First run requires GRO file specification")
        elif igro is not None:
            self.gro = igro
            if itop is None:
                self.top = self.gro.top
        self._mdp = None
        self.deffnm = deffnm
        self._name = name
        self.run_id = run_id
        if odir is None:
            odir = self.name
        else:
            odir = Dir(odir)
            odir = odir / self.name
        self.odir = Dir(odir)
        if deffnm is None:
            self.deffnm = name
        self.outname = self.odir / self.deffnm

    @property
    def gro(self):
        return self._gro

    @gro.setter
    def gro(self, gro: GROFile):
        self._gro = GROFile(gro)

    @property
    def top(self):
        return self._top

    @top.setter
    def top(self, top: GROFile):
        self._top = GROFile(top)

    @property
    def odir(self):
        return self._odir

    @odir.setter
    def odir(self, odir):
        if odir is not None:
            self._odir = Dir(odir) / self.name

    @property
    def mdp(self) -> Union[None, MDPFile]:
        if self._mdp is not None:
            return self._mdp
        else:
            logger.error(f"No MDP file specified")

    @mdp.setter
    def mdp(self, mdp: MDPFile):
        assert (
            type(mdp) == MDPFile
        ), f"Unexpected type {type(mdp)} given for MDP options file"
        self._mdp = mdp

    def get_run_command(self, gmx_alias="gmx"):
        if self.cpt is not None:
            cpt_str_1 = f" -t {self.cpt}"
            cpt_str_2 = f" -cpi {self.cpt}"
        else:
            cpt_str_1 = ""
            cpt_str_2 = ""

        return (
            f"\n# {self.run_id}: {self._name}:\n"
            f'{gmx_alias} grompp -f {self.mdp} -c {self.gro} -p {self.top} -o {self.outname.with_suffix(".tpr")} -pp {self.outname.with_suffix(".top")}{cpt_str_1}\n'
            f"{gmx_alias} mdrun -s {self.outname.with_suffix('.tpr')} -v -deffnm {self.outname}{cpt_str_2}\n"
        )

    def add_nsteps(self):
        self.mdp.add({"nsteps": self.nsteps}, replace=True)

    def add_restraints(self):
        outfile = self.top.with_name(f"{self.top.stem}_posres" + ".itp")
        for restraint, fcs in self.restraints.items():
            if restraint.lower() == "clay":
                restraint_atoms = select_clay(
                    universe=self.gro.universe
                ).residues.atoms
            else:
                restraint_atoms = self.gro.universe.select_atoms(
                    f"residue {restraint}"
                )
            generate_restraints(
                restraint_atoms, outfile, fcs, add_to_file=True
            )
            with open(self.top, "r+") as top_file:
                top_str = top_file.read()
                top_str = re.sub(
                    r"(^\s*\[ system \])",
                    rf'\n; Include position restraints\n#include "{outfile.name}"\n\1',
                    top_str,
                    flags=re.MULTILINE | re.DOTALL,
                )
                top_file.write(top_str)

    def add_freeze(self):
        replace = True
        if isinstance(self.freeze, str):
            freeze_list = [self.freeze]
        else:
            freeze_list = self.freeze
        for freeze_group in freeze_list:
            if freeze_group == "clay":
                freeze_atoms = select_clay(universe=self.gro.universe)
            else:
                freeze_atoms = self.gro.universe.select_atoms(
                    f"residue {freeze_group}"
                )
            self.mdp.add(
                {
                    "freezegrps": np.unique(
                        freeze_atoms.residues.resnames
                    ).tolist(),
                    "freezedims": ["Y", "Y", "Y"],
                },
                replace=replace,
            )
            replace = False

    def process_mdp(self):
        for item in self._options:
            process_function = getattr(self, f"add_{item.lower()}")
            process_function()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

    @property
    def name(self):
        return f"{self._run_id_str}{self._name}"

    @property
    def _run_id_str(self):
        if self.run_id:
            return_str = f"{self.run_id}_"
        else:
            return_str = ""
        return return_str

    def copy_inputfiles(self):
        for filetype in ["gro", "top"]:
            file = getattr(self, filetype)
            shutil.copy(file, self.odir)
            setattr(self, filetype, self.odir / file)


class _GMXRunType(ABC):
    _options = []
    _gmx_version_mdp = {}
    _run_type_mdp = {}
    _ensemble_mdp = {"NVT": None, "NpT": None}

    def __new__(cls, *args, **kwargs):
        _default_mdp_prms = MDP_DEFAULTS
        for k, v in MDP_DEFAULTS.items():
            if re.fullmatch("[0-9]+", f"{k}"):
                cls._gmx_version_mdp[k] = v
            elif re.fullmatch(get_search_str(cls._ensemble_mdp), f"{k}"):
                cls._ensemble_mdp[k] = v
            else:
                cls._run_type_mdp[k] = v
        return cls(*args, **kwargs)

    def __init__(
        self,
        name: str,
        run_id: int,
        options: Dict[str, Any] = None,
        mdp_prms: Optional[Union[MDPFile, File, str]] = None,
        gmx_alias: Optional[str] = None,
        gmx_version: Optional[str] = None,
        outpath: Optional[Union[str, Dir, Path]] = None,
    ):
        self.id = run_id
        self.name = name
        self.run_type = None
        self.ensemble = None
        self.get_specs()
        self._mdp_prms = None
        self._gmx_alias = gmx_alias
        self._gmx_version = None
        self._gmx_template = None
        self._outpath = outpath
        self._mdp_template = MDPFile(
            tempfile.NamedTemporaryFile(
                suffix=".mdp", dir=self.outpath, delete=False
            ).name,
            gmx_version=gmx_version,
        )
        self.get_run_type_prms()
        self.gmx_version = gmx_version
        self._mdp_prms = self._mdp_template
        self.mdp_prms: MDPFile = mdp_prms
        if options is None:
            options = {}
        for k, v in options.items():
            if k.lower() in self.__class__._options:
                self.__setattr__(k.lower(), v)
            else:
                logger.error(f"Unrecognised option {k.lower()!r}")
        self.process()

    def get_specs(self):
        try:
            self.ensemble = re.search(
                get_search_str(self._ensemble_mdp, self.name)
            ).group(0)
        except AttributeError:
            pass

    def __repr__(self):
        return f"{self.__class__.__name__}({self.id}: {self.name!r})"

    def __str__(self):
        return f"{self.__class__.__name__}({self.id}: {self.name!r})"

    def get_run_type_prms(self):
        try:
            mdp_defaults = MDP_DEFAULTS[self._default_mdp_key]
        except KeyError:
            mdp_defaults = {}
        self.mdp_prms.add(mdp_defaults, replace=True)

    @property
    def gmx_version(self) -> Union[None, int]:
        return self._gmx_version

    @gmx_version.setter
    def gmx_version(self, gmx_version: Union[None, int]):
        if gmx_version is not None:
            if self._gmx_version != gmx_version:
                try:
                    mdp_defaults = MDP_DEFAULTS[gmx_version]
                except KeyError:
                    logger.error(
                        f"No known parameters for GROMACS version {gmx_version}"
                    )

                else:
                    self._mdp_template = MDPFile(
                        self._mdp_template, gmx_version=gmx_version
                    )
                    self._mdp_template.add(mdp_defaults, replace=False)
                    self._gmx_version = gmx_version

    @property
    def mdp_prms(self) -> MDPFile:
        tmp_mdp = self._mdp_prms
        if tmp_mdp is None:
            tmp_mdp = self._mdp_template
            if self.gmx_version is not None:
                tmp_mdp.add(MDP_DEFAULTS[self.gmx_version], replace=False)
        else:
            tmp_mdp = self._mdp_prms
        return tmp_mdp

    @mdp_prms.setter
    def mdp_prms(
        self, prms_or_file: Union[str, MDPFile, Dict[str, Any]]
    ) -> None:
        if prms_or_file is not None:
            try:
                prms_file = MDPFile(prms_or_file, gmx_version=self.gmx_version)
                self._mdp_prms = prms_file
            except:
                if self._mdp_prms is not None:
                    self._mdp_prms.add(prms_or_file, replace=True)
                else:
                    self._mdp_template.add(prms_or_file, replace=True)

    def write_new_mdp(self, new_name: Union[str, MDPFile]) -> None:
        shutil.copy2(self.mdp_prms, new_name)
        new_mdp = MDPFile(new_name, gmx_version=self.mdp_prms.gmx_version)
        new_mdp.string = dict_to_mdp(new_mdp.parameters)
        self.mdp_prms = new_mdp

    @property
    def outpath(self):
        return self._outpath

    @outpath.setter
    def outpath(self, outpath):
        try:
            self._outpath = Dir(outpath)
            init_path(self._outpath)
        except:
            logger.error(f"Could not initialise outpath with {outpath!r}")

    @abstractmethod
    def process(self):
        pass


class EMRun(GMXRun):
    _options = ["freeze"]
    _default_mdp_key = "EM"

    def __init__(
        self,
        run_id: int,
        name: str,
        igro=None,
        itop=None,
        deffnm=None,
        odir=None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            run_id=run_id,
            igro=igro,
            itop=itop,
            deffnm=deffnm,
            odir=odir,
        )
        add_kwargs(self, **kwargs)


class EQRunFixed(GMXRun):
    _options = ["nsteps", "freeze"]
    _default_mdp_key = "EQ"

    def __init__(
        self,
        run_id: int,
        name: str,
        igro=None,
        itop=None,
        deffnm=None,
        odir=None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            run_id=run_id,
            igro=igro,
            itop=itop,
            deffnm=deffnm,
            odir=odir,
        )
        add_kwargs(self, **kwargs)

    # def process_mdp(self):
    #     fix_list = []
    #     if isinstance(self.fix, str):
    #         fix_list.append(self.fix)
    #     else:
    #         fix_list = self.fix
    #     for fixed_group in fix_list:
    #         if fixed_group == 'clay':
    #             fixed_atoms = select_clay(universe=self.gro.universe)
    #         else:
    #             fixed_atoms = self.gro.universe.select_atoms(f'residue {fixed_group}')
    #     self.mdp.add(
    #         {
    #             "freezegrps": np.unique(fixed_atoms.residues.resnames).tolist(),
    #             "freezedims": ["Y", "Y", "Y"],
    #         },
    #         replace=True,
    #     )
    #     self.mdp.add({'nsteps': self.nsteps}, replace=True)


class EQRunRestrained(GMXRun):
    _options = ["nsteps", "restraints"]
    _default_mdp_key = "EQ"

    def __init__(
        self,
        run_id: int,
        name: str,
        igro=None,
        itop=None,
        deffnm=None,
        odir=None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            run_id=run_id,
            igro=igro,
            itop=itop,
            deffnm=deffnm,
            odir=odir,
        )
        add_kwargs(self, **kwargs)


class EQRun(GMXRun):
    _options = ["nsteps"]
    _default_mdp_key = "EQ"

    def __init__(
        self,
        run_id: int,
        name: str,
        igro=None,
        itop=None,
        deffnm=None,
        odir=None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            run_id=run_id,
            igro=igro,
            itop=itop,
            deffnm=deffnm,
            odir=odir,
        )
        add_kwargs(self, **kwargs)


class DSpaceRun(GMXRun):
    _options = [
        "d_space",
        "remove_steps",
        "sheet_wat",
        "uc_wat",
        "percent_wat",
    ]
    _default_mdp_key = "D-SPACE"

    def __init__(
        self,
        run_id: int,
        name: str,
        igro=None,
        itop=None,
        deffnm=None,
        odir=None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            run_id=run_id,
            igro=igro,
            itop=itop,
            deffnm=deffnm,
            odir=odir,
        )
        add_kwargs(self, n_args=3, **kwargs)


class GMXRunFactory:
    _run_types = (
        ("EM", EMRun),
        ("EQ_(NVT|NpT)", EQRun),
        ("EQ_(NVT|NpT)_F", EQRunFixed),
        ("EQ_(NVT|NpT)_R", EQRunRestrained),
        ("D-SPACE", DSpaceRun),
    )
    _default = GMXRun

    @classmethod
    def init_subclass(cls, name: str, run_id: int, **kwargs) -> GMXRunType:
        _cls = cls._default
        for k, v in cls._run_types:
            name_match = re.match(f"{k}", name, flags=re.IGNORECASE)
            if name_match:
                _cls = v
        return _cls(name=name, run_id=run_id, **kwargs)


class MDPWriter:
    def __init__(
        self,
        template=None,
        run_type: Union[
            Literal["EM"], Literal["D_SPACE"], Literal["EQ"], Literal["P"]
        ] = None,
    ):
        template = File(template)
        if template.is_file() and not template.is_relative_to(MDP):
            template_file = USER_MDP / template
            shutil.copy(template, template_file)
        else:
            template_file = MDP / template
        self.template_file = MDPFile(template_file)
        if not template_file.is_file():
            if template.is_file():
                ...

        with open(self.template_file, "r") as template_file:
            mdp_str = template_file.read()
        self._string = self.__string = mdp_str

    def freeze_groups(
        self,
        resnames: List[str],
        freeze_dims: List[Union[Literal["Y"], Literal["N"]]] = ["Y", "Y", "Y"],
    ):
        self.string = set_mdp_freeze_groups(
            uc_names=resnames, em_template=self.string, freeze_dims=freeze_dims
        )

    @property
    def string(self):
        return self._string

    @string.setter
    def string(self, string: str):
        self._string = string

    def modify_mdp_str(f):
        @wraps(f)
        def wrapper(self, **run_prms):
            mdp_str = self.string
            for parameter, value in run_prms:
                mdp_str = f(parameter=parameter, value=value, mdp_str=mdp_str)
            self.string = mdp_str

        return wrapper

    def reset(self):
        self._string = self.__string

    @modify_mdp_str
    def set(self, parameter: str, value: str, mdp_str):
        return set_mdp_parameter(
            parameter=parameter, value=value, mdp_str=mdp_str
        )

    @modify_mdp_str
    def add(self, parameter: str, value: str, mdp_str):
        return set_mdp_parameter(
            parameter=parameter, value=value, mdp_str=mdp_str
        )


class MDPRunGenerator:
    """Class for generating a series of GROMACS run inputs from a template MDP file."""

    _options = CaselessDict({})
    _gmx_version_mdp = {}
    __run_type_mdp = {}
    _ensemble_mdp = {"NVT": None, "NpT": None}
    _default_mdp_options = CaselessDict({})

    def __repr__(self):
        run_str = ", ".join([f"{x.name!r}" for x in self._runs])
        if run_str != "":
            run_str = f"{run_str} - "
        return f"{self.__class__.__name__}({run_str}GROMACS: {self._gmx_alias} {self.gmx_version})"

    def __str__(self):
        return self.__repr__()

    @classmethod
    def init_mdp_prms(
        cls,
        mdp_prms: Optional[Union[MDPFile, File, str, Dict[str, Any]]] = None,
        run_options: Optional[Union[MDPFile, str]] = None,
    ):
        """Initialise the MDP option for the run.
        :param mdp_prms: Default MDP parameters for all run types
        :param run_options: Run type MDP options
        """
        _default_mdp_prms = MDP_DEFAULTS
        if mdp_prms is not None:
            _default_mdp_prms = cls.update_run_type_mdp(
                mdp_prms, _default_mdp_prms
            )
        if run_options is not None:
            _default_mdp_prms = cls.update_run_type_mdp(
                run_options, _default_mdp_prms
            )
        for k, v in _default_mdp_prms.items():
            if re.fullmatch("[0-9]+", f"{k}"):
                cls._gmx_version_mdp[k] = v
            elif re.fullmatch(get_search_str(cls._ensemble_mdp), f"{k}"):
                cls._ensemble_mdp[k] = v
            else:
                cls.__run_type_mdp[k] = v
        cls._default_mdp_prms = _default_mdp_prms

    def __init__(
        self,
        gmx_alias: str = "gmx",
        gmx_version: Union[int, str] = 0,
        mdp_prms: Optional[Union[MDPFile, File, str, Dict[str, Any]]] = None,
        run_options: Optional[Union[MDPFile, str]] = None,
    ):
        """Initialise the run generator.
        :param gmx_alias: Alias for GROMACS executable
        :param gmx_version: GROMACS version
        :param mdp_prms: Default MDP parameters for all run types
        :param run_options: Run type MDP options
        """
        self.init_mdp_prms(mdp_prms, run_options)
        self._mdp_prms = None
        self._gmx_alias = gmx_alias
        self._gmx_version = None
        self._mdp_template = MDPFile(
            tempfile.NamedTemporaryFile(suffix=".mdp", delete=False).name,
            gmx_version=gmx_version,
        )
        self.gmx_version = gmx_version
        self._mdp_template = self.update_run_type_mdp(
            self._mdp_template, self._gmx_version_mdp[self.gmx_version]
        )
        self._mdp_prms = self._mdp_template
        self._run_type_mdp = self.__run_type_mdp.copy()
        # self._add_user_specs(run_options)
        self._runs = []

    @staticmethod
    @get_mdp_data_dict_function_decorator
    def update_run_type_mdp(other, data):
        """Update the run type MDP options with the given MDP options.
        :param other: MDP options to add
        :param data: MDP options to update
        :return: Updated MDP options
        """
        for k, v in other.items():
            if type(v) not in [dict, CaselessDict]:
                for kd, vd in data.items():
                    if re.fullmatch("[0-9]+", f"{kd}"):
                        # sub_search_str = get_search_str(vd)
                        if k in vd:
                            data[kd][k] = v
        for k, v in other.items():
            if type(v) in [dict, CaselessDict]:
                if k not in data:
                    data[k] = CaselessDict({})
                for ki, vi in v.items():
                    data[k][ki] = vi
        return data

    @cached_property
    def run_factory(self):
        return GMXRunFactory()

    def add_run(
        self,
        run_id,
        name,
        igro=None,
        odir=None,
        deffnm=None,
        itop=None,
        **kwargs,
    ):
        """Add a run to the run sequence.
        :param run_id: Run ID
        :param name: Run name
        :param igro: Input GRO file
        :param odir: Output directory
        :param deffnm: Default GROMACS mdrun output file stem
        :param itop: Input topology file
        :param mdp_options: MDP options
        :param kwargs: Other run options
        """
        # run_factory: GMXRunType = GMXRunFactory()
        new_run = self.run_factory.init_subclass(
            name=name,
            igro=igro,
            itop=itop,
            deffnm=deffnm,
            odir=odir,
            run_id=run_id,
            **kwargs,
        )
        while len(self._runs) < run_id:
            self._runs.append(None)
        self._runs[run_id - 1] = new_run

    def write_runs(
        self,
        odir: Union[Dir, str],
        run_script_name=None,
        run_script_template=None,
        **kwargs,
    ):
        """Write the run script for the run sequence.
        :param odir: Output directory
        :param run_script_name: Name of the run script
        :param run_script_template: Template for the run script
        """
        odir = Dir(odir)
        init_path(odir)
        if run_script_name is None:
            run_script_name = "run.sh"
        if run_script_template is None:
            with open(odir / run_script_name, "w") as scriptfile:
                scriptfile.write("#!/bin/bash\n")
        else:
            shutil.copy2(run_script_template, odir / run_script_name)
        init_data_copied = False
        with open(odir / run_script_name, "a") as scriptfile:
            for run in self._runs:
                run.odir = odir
                init_path(run.odir)
                if not init_data_copied:
                    run.copy_inputfiles()
                    run.cpt = None
                    init_top = run.top
                    init_gro = run.gro
                else:
                    new_top = run.odir / init_top.name
                    shutil.copy2(init_top, new_top)
                    run.top = new_top
                    run.cpt = prev_cpt
                    run.gro = init_gro
                run.mdp: MDPFile = self.write_mdp_options(
                    run.name, outpath=run.odir
                )
                run.process_mdp()
                if init_data_copied:
                    run.gro = prev_outname.with_suffix(".gro")
                else:
                    init_data_copied = True
                if not re.fullmatch(
                    f"([A-Za-z0-9\-_]*_)*EM(_[A-Za-z0-9\-_]*)*", run.name
                ):
                    prev_cpt = prev_outname.with_suffix(".cpt")
                else:
                    prev_cpt = None
                scriptfile.write(run.get_run_command(self._gmx_alias))
                prev_outname = run.outname
        print("done")

    def write_mdp_options(self, run_name: str, outpath: Dir) -> MDPFile:
        """Write the MDP options for the given run.
        :param run_name: Name of the run
        :param outpath: Output directory
        :return: MDP options file
        """
        mdpfile = outpath / f"{run_name}.mdp"
        with open(mdpfile, "w+") as file:
            file.write(
                f"; {run_name!r} run parameters, written on {exec_date} at {exec_time}"
            )
        mdpfile = MDPFile(mdpfile)
        mdp_prms = self._mdp_prms
        mdpfile.add(mdp_prms)
        for specifier in (
            self._gmx_version_mdp,
            self._ensemble_mdp,
            self._run_type_mdp,
        ):
            search_str = get_search_str(specifier)
            match = re.fullmatch(
                f"([A-Za-z0-9\-_]*_)*({search_str})(_[A-Za-z0-9\-_]*)*",
                run_name,
            )
            if match:
                mdpfile.add(specifier[match.group(2)], replace=True)
        return mdpfile

    @property
    def gmx_version(self) -> Union[None, int]:
        """GROMACS version.
        :return: GROMACS version"""
        return self._gmx_version

    @gmx_version.setter
    def gmx_version(self, gmx_version: Union[None, int]):
        """Set the GROMACS version.
        :param gmx_version: GROMACS version"""
        if gmx_version is not None:
            if self._gmx_version != gmx_version:
                try:
                    mdp_defaults = self._gmx_version_mdp[gmx_version]
                except KeyError:
                    logger.error(
                        f"No known parameters for GROMACS version {gmx_version}"
                    )
                else:
                    self._mdp_template = MDPFile(
                        self._mdp_template, gmx_version=gmx_version
                    )
                    self._mdp_template.add(mdp_defaults, replace=False)
                    self._gmx_version = gmx_version

    @property
    def mdp_prms(self) -> MDPFile:
        """MDP parameters."""
        tmp_mdp = self._mdp_prms
        if tmp_mdp is None:
            tmp_mdp = self._mdp_template
            # if self.gmx_version is not None:
            #     tmp_mdp.add(MDP_DEFAULTS[self.gmx_version], replace=False)
        else:
            tmp_mdp = self._mdp_prms
        return tmp_mdp

    @mdp_prms.setter
    def mdp_prms(
        self, prms_or_file: Union[str, MDPFile, Dict[str, Any]]
    ) -> None:
        """Set the MDP parameters.
        :param prms_or_file: MDP parameters"""
        if prms_or_file is not None:
            try:
                prms_file = MDPFile(prms_or_file, gmx_version=self.gmx_version)
                self._mdp_prms = prms_file
            except:
                if self._mdp_prms is not None:
                    self._mdp_prms.add(prms_or_file, replace=True)
                else:
                    self._mdp_template.add(prms_or_file, replace=True)

    # def write_new_mdp(self, new_name: Union[str, MDPFile]) -> None:
    #     shutil.copy2(self.mdp_prms, new_name)
    #     new_mdp = MDPFile(new_name, gmx_version=self.mdp_prms.gmx_version)
    #     new_mdp.string = dict_to_mdp(new_mdp.parameters)
    #     self.mdp_prms = new_mdp
