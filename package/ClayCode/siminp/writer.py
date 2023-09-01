import logging
import re
import shutil
import tempfile
from abc import ABC, abstractmethod
from collections import UserDict, UserString
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from ClayCode.core.classes import (
    Dir,
    File,
    MDPFile,
    dict_to_mdp,
    init_path,
    mdp_to_dict,
    set_mdp_freeze_groups,
    set_mdp_parameter,
)
from ClayCode.core.consts import MDP, MDP_DEFAULTS, USER_MDP
from ClayCode.core.utils import (
    get_file_or_str,
    get_search_str,
    property_exists_checker,
)

logger = logging.getLogger(__name__)


class _GMXRunType(ABC):
    _options = []
    _default_mdp_keys = "X"
    _ensembles = ["NVT", "NpT"]

    def __init__(
        self,
        name: str,
        run_id: int,
        options: Dict[str, Any],
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
        for k, v in options.items():
            if k.lower() in self.__class__._options:
                self.__setattr__(k.lower(), v)
            else:
                logger.error(f"Unrecognised option {k.lower()!r}")
        self.process()

    def get_specs(self):
        try:
            self.ensemble = re.search(
                get_search_str(self._ensembles, self.name)
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


class EMRun(_GMXRunType):
    _options = ["freeze_clay"]
    _default_mdp_key = "EM"

    def __init__(
        self,
        name: str,
        run_id: int,
        options: Dict[str, Any],
        mdp_prms: Optional[Union[MDPFile, File, str]] = None,
        gmx_alias: Optional[str] = None,
        gmx_version: Optional[str] = None,
    ):
        super().__init__(
            name, run_id, options, mdp_prms, gmx_alias, gmx_version
        )
        assert len(options) == len(
            self.__class__._options
        ), f"Missing {len(self.__class__._options) - len(options)} run options!"

    def process(self):
        if self.freeze_clay:
            self.mdp_prms.freeze


class EQRunFixed(_GMXRunType):
    _options = ["nsteps"]
    _default_mdp_key = "EQ"

    def __init__(
        self,
        name: str,
        run_id: int,
        options: Dict[str, Any],
        mdp_prms: Optional[Union[MDPFile, File, str]],
        gmx_alias: Optional[str] = None,
        gmx_version: Optional[str] = None,
    ):
        super().__init__(
            name, run_id, options, mdp_prms, gmx_alias, gmx_version
        )


class EQRunRestrained(_GMXRunType):
    _options = ["nsteps", "restraint_k"]
    _default_mdp_key = "EQ"

    def __init__(
        self,
        name: str,
        run_id: int,
        options: Dict[str, Any],
        mdp_prms: Optional[Union[MDPFile, File, str]],
        gmx_alias: Optional[str] = None,
        gmx_version: Optional[str] = None,
    ):
        super().__init__(
            name, run_id, options, mdp_prms, gmx_alias, gmx_version
        )


class EQRun(_GMXRunType):
    _options = ["nsteps"]
    _default_mdp_key = "EQ"

    def __init__(
        self,
        name: str,
        run_id: int,
        options: Dict[str, Any],
        mdp_prms: Optional[Union[MDPFile, File, str]],
        gmx_alias: Optional[str] = None,
        gmx_version: Optional[str] = None,
    ):
        super().__init__(
            name, run_id, options, mdp_prms, gmx_alias, gmx_version
        )


class DSpaceRun(_GMXRunType):
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
        name: str,
        run_id: int,
        options: Dict[str, Any],
        mdp_prms: Optional[Union[MDPFile, File, str]],
        gmx_alias: Optional[str] = None,
        gmx_version: Optional[str] = None,
    ):
        super().__init__(
            name, run_id, options, mdp_prms, gmx_alias, gmx_version
        )


class OtherRun(_GMXRunType):
    _options = "nsteps"
    _default_mdp_key = "P"

    def __init__(
        self,
        name: str,
        run_id: int,
        options: Dict[str, Any],
        mdp_prms: Optional[Union[MDPFile, File, str]],
        gmx_alias: Optional[str] = None,
        gmx_version: Optional[str] = None,
    ):
        super().__init__(
            name, run_id, options, mdp_prms, gmx_alias, gmx_version
        )


class GMXRunFactory:
    _run_types = {
        "EM": EMRun,
        "EQ_(NVT|NpT)_F": EQRunFixed,
        "EQ_(NVT|NpT)_R": EQRunRestrained,
        "EQ_(NVT|NpT)": EQRun,
        "D-SPACE": DSpaceRun,
    }
    _default = OtherRun

    @classmethod
    def init_subclass(
        cls,
        name: str,
        run_id: int,
        options: Dict[str, Any],
        mdp_prms: Optional[Union[MDPFile, File, str]] = None,
        gmx_alias: Optional[str] = None,
        gmx_version: Optional[str] = None,
    ):
        _cls = cls._default
        for k, v in cls._run_types.items():
            name_match = re.match(f"{k}", name, flags=re.IGNORECASE)
            if name_match:
                _cls = v
        return _cls(name, run_id, options, mdp_prms, gmx_alias, gmx_version)


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
