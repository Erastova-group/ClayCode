#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
""":mod:`ClayCode.core.classes` --- ClayCode classes
===================================================
"""
from __future__ import annotations

import logging
import os
import re
import tempfile
from collections import UserList
from collections.abc import Sequence
from copy import copy as _copy
from copy import deepcopy
from functools import (
    cached_property,
    partialmethod,
    singledispatch,
    update_wrapper,
    wraps,
)
from io import StringIO
from pathlib import Path as _Path
from pathlib import PosixPath as _PosixPath
from typing import (
    Any,
    AnyStr,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    NewType,
    NoReturn,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)

import numpy as np
import pandas as pd
import yaml
from caseless_dictionary import CaselessDict
from ClayCode.core.cctypes import (
    FileNameMatchSelector,
    PathOrStr,
    PathType,
    StrNum,
    StrNumOrListDictOf,
    StrOrListDictOf,
    StrOrListOf,
)
from ClayCode.core.utils import (
    get_file_or_str,
    get_search_str,
    select_named_file,
)
from ClayCode.data.consts import FF, ITP_KWDS
from ClayCode.data.consts import KWD_DICT as _KWD_DICT
from ClayCode.data.consts import MDP_DEFAULTS
from MDAnalysis import AtomGroup, ResidueGroup, Universe
from pandas.errors import EmptyDataError
from parmed import Atom, Residue

# logging.getLogger("numexpr").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
# -----------------------------------------------------------------------------
# class decorators
# -----------------------------------------------------------------------------


def add_method(cls):
    """Add new method to existing class"""

    # @update_wrapper(cls)
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)

        setattr(cls, func.__name__, wrapper)

    return decorator


def add_property(cls):
    """Add new property to existing class"""

    @update_wrapper(cls)
    def decorator(func):
        @update_wrapper(func)
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)

        setattr(cls, func.__name__, wrapper)

    return decorator


# -----------------------------------------------------------------------------
# string functions and classes
# -----------------------------------------------------------------------------


def split_fname(fname: str) -> List[Union[str, None]]:
    """Split a filename into stem and suffix.
    'file_name.dat' -> ['file_name', 'dat']
    'file.name.dat' -> ['file.name', 'dat']
    'file_name' -> ['file_name']
    """
    name_split = fname.split(sep=".", maxsplit=1)
    if len(name_split) > 2:
        name_split = [".".join(name_split[:-1]), name_split[-1]]
    elif len(name_split) == 1:
        name_split.append(None)
    return name_split


def match_str(
    searchstr: str,
    pattern: Union[StrNum, Sequence[StrNum], Tuple[str], Mapping[str, Any]],
) -> Union[None, str]:
    """Match a string agains a string of list of search patterns.
    Return matching string or None if no match is found."""
    check = None
    if type(pattern) == str:
        match_pattern = pattern
    else:
        match_pattern = get_match_pattern(pattern)
    if type(pattern) in (str, int, float):
        if searchstr == match_pattern:
            check = searchstr
        else:
            check = None
        return check
    elif type(pattern) == list:
        try:
            check = re.fullmatch(match_pattern, searchstr).group(0)
        except AttributeError:
            check = None
    return check


@singledispatch
def get_match_pattern(pattern: StrOrListOf):
    """Generate search pattern from list, str, dict, int or float"""
    if hasattr(pattern, "parent"):
        return str(pattern.resolve())
    else:
        raise TypeError(f"Unexpected type {type(pattern)!r}!")


@get_match_pattern.register(list)
@get_match_pattern.register(set)
@get_match_pattern.register(tuple)
def _(pattern) -> str:
    """Get match pattern from list items.
    ['a', 'b'] -> 'a|b'
    """
    pattern_list = pattern
    pattern_list = [get_match_pattern(item) for item in pattern_list]
    return "|".join(pattern_list)


@get_match_pattern.register
def _(pattern: str) -> str:
    """Get str match pattern.
    'ab' -> 'ab'
    """
    return pattern


@get_match_pattern.register(float)
@get_match_pattern.register(int)
def _(pattern) -> str:
    """Get match pattern from float or int
    12 -> '12'
    """
    return f"{pattern}"


@get_match_pattern.register
def _(pattern: dict) -> str:
    """Get match pattern from dict keys.
    {'a': 'x', 'b': 'y' -> 'a|b'}
    """
    pattern_dict = pattern
    pattern_list = [get_match_pattern(item) for item in pattern_dict.keys()]
    return "|".join(pattern_list)


# Kwd = NewType("Kwd", str)

# MDPFile decorators


def key_match_decorator(parameter_dict: Dict[str, Any]):
    def f_decorator(f):
        @wraps(f)
        def wrapper(self, other, *args, **kwargs):
            assert isinstance(
                self, MDPFile
            ), f"Expected MDPFile, found {self.__class__.__name__}"
            search_str = get_search_str(getattr(self, parameter_dict))
            return f(self, other, search_str, *args, **kwargs)

        return wrapper

    return f_decorator


def get_mdp_data_dict_function_decorator(f):
    @wraps(f)
    def wrapper(other: Union[Dict[str, str], MDPFile, str], *args, **kwargs):
        if type(other) == MDPFile:
            other_data = other.parameters
        elif isinstance(other, str):
            try:
                other_data = MDPFile(other).parameters
            except:
                other_data = mdp_to_dict(other)
        elif isinstance(other, dict):
            other_data = other
        else:
            raise TypeError(
                f"Unexpected type {other.__class__.__name__!r} for {f.__name__.strip('__')!r}"
            )
        return f(other_data, *args, **kwargs)

    return wrapper


def get_mdp_data_dict_method_decorator(f):
    @wraps(f)
    def wrapper(
        self, other: Union[Dict[str, str], MDPFile, str], *args, **kwargs
    ):
        assert isinstance(
            self, MDPFile
        ), f"Expected MDPFile, found {self.__class__.__name__}"
        if type(other) == type(self):
            other_data = other.parameters
        elif isinstance(other, str):
            try:
                other_data = MDPFile(other).parameters
            except:
                other_data = mdp_to_dict(other)
        elif isinstance(other, dict):
            other_data = other
        else:
            raise TypeError(
                f"Unexpected type {other.__class__.__name__!r} for {f.__name__.strip('__')!r}"
            )
        return f(self, other_data, *args, **kwargs)

    return wrapper


class Kwd(str):
    """str container for '.itp' and '.top' file parameters"""

    def match(
        self, pattern: StrOrListDictOf, mode: Literal["full", "parts"] = "full"
    ) -> Union[Kwd, None]:
        """Match keywords against a search pattern and return match"""
        if mode == "full":
            check = match_str(self, pattern)
        else:
            if mode == "parts":
                check = re.match(
                    rf"[a-zA-Z-_\d.]*{pattern}[a-zA-Z-_\d.]*", self
                )
            else:
                raise ValueError(
                    f'{mode!r} is not a valid value for {"mode"!r}'
                )
        if check is not None:
            check = self
        return check


# KwdList = NewType("KwdList", UserList)


class KwdList(UserList):
    """List of Kwd objects"""

    def __init__(self, data):
        self.data = []
        for item in data:
            if item not in self.data:
                self.data.append(Kwd(item))

    def __add__(self, other: Kwd):
        return self.append(self, other)

    def __sub__(self, other: Kwd):
        return self.remove(other)

    def match(
        self,
        pattern: str,
        mode: Literal["full", "parts"] = "full",
        keep_dims: bool = False,
    ) -> List[Union[Kwd, None]]:
        """Match keywords against a search pattern and return matching item or none"""
        match = [obj.match(pattern, mode) for obj in self.data]
        if not keep_dims:
            match = [obj for obj in match if obj is not None]
        return match

    def sort_kwds(self, sort_map: List[str]) -> KwdList:
        """Sort list elements to match order in sort_map"""
        sort_map.reverse()
        for kwds in sort_map:
            result_id = self.filter([kwds], mode="full", keep_dims=True)
            idx = np.argwhere(np.array(result_id) != None)
            if len(idx) == 0:
                pass
            else:
                self.data.insert(0, self.data.pop(int(idx)))
        return self

    def extend(self, other: Iterable[str]) -> NoReturn:
        """Add new item to list"""
        for item in other:
            if item not in self.data:
                self.data.append(item)

    def append(self, item: str) -> NoReturn:
        """Add new item to list"""
        if item not in self.data:
            self.data.append(item)

    def filter(
        self,
        pattern: str,
        mode: Literal["full", "parts"] = "full",
        keep_dims: bool = False,
    ) -> List[Union[Kwd, None]]:
        """Match keywords against a search pattern and return matching items"""
        pattern = get_match_pattern(pattern)
        match_list = [obj.match(pattern, mode) for obj in self.data]
        if not keep_dims:
            match_list = [obj for obj in match_list if obj is not None]
        return match_list

    def concat(self, join="|") -> str:
        """Concatenate items to string"""
        return f"{join}".join(self.data)


# -----------------------------------------------------------------------------
# file object classes
# -----------------------------------------------------------------------------


# Adaptation _PathParents class from Lib/pathlib.py module
class _PathParents(Sequence):
    """This object provides sequence-like access to the logical ancestors
    of a path.  Don't try to construct it yourself."""

    __slots__ = ("_pathcls", "_drv", "_root", "_parts")

    def __init__(self, path):
        # We don't store the instance to avoid reference cycles
        self._pathcls = DirFactory
        self._drv = path._drv
        self._root = path._root
        self._parts = path._parts

    def __len__(self):
        if self._drv or self._root:
            return len(self._parts) - 1
        else:
            return len(self._parts)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return tuple(self[i] for i in range(*idx.indices(len(self))))

        if idx >= len(self) or idx < -len(self):
            raise IndexError(idx)
        return self._pathcls._from_parsed_parts(
            self._drv, self._root, self._parts[: -idx - 1]
        )

    def __repr__(self):
        return "<{}.parents>".format(self._pathcls.__name__)


# BasicPath = NewType("BasicPath", _Path)


# Extension of Path with PosixPath class from Lib/pathlib.py module
# (imported as _Path and _PosixPath)
class BasicPath(_Path):
    """Container for custom pathlib.Path objects"""

    _flavour = _PosixPath._flavour
    __slots__ = ()

    def _match_deorator(method):
        """Match name against seacrch pattern object"""

        @wraps(method)
        def wrapper(
            self: BasicPath,
            pattern: StrNumOrListDictOf,
            mode: FileNameMatchSelector = "full",
        ) -> Union[BasicPath, None]:
            pattern = get_match_pattern(pattern)
            if mode == "full":
                check = match_str(self.name, pattern)
            else:
                if mode == "stem":
                    check = match_str(self.name_split[0], pattern)
                elif mode in ["ext", "suffix"]:
                    check = match_str(self.name_split[1], pattern)
                elif mode == "parts":
                    check = re.match(
                        rf"[a-zA-Z-_\d.]*{pattern}[a-zA-Z-_\d.]*", self.name
                    )
                else:
                    raise ValueError(
                        f'{mode!r} is not a valid value for {"mode"!r}'
                    )
            if check is not None:
                check = method(self)
            return check

        return wrapper

    def __new__(cls, *args, **kwargs):
        _cls = cls.check_os()
        self = _cls._from_parts(args)
        return self

    def __init__(self, *args, check: bool = False):
        if check:
            self.check()

    def check(self) -> NoReturn:
        """Check if file exists"""
        if not self.exists():
            raise FileNotFoundError(r"{self.name} does not exist")
        else:
            logger.debug(f"{self.name} exists")

    @property
    def name_split(self) -> Tuple[str, str]:
        return self.stem, self.suffix

    @classmethod
    def check_os(cls) -> Union[BasicPath, NotImplementedError]:
        return (
            cls
            if os.name == "posix"
            else NotImplementedError(
                "ClayCode does not support Windows systems."
            )
        )

    @_match_deorator
    def match_name(self) -> Union[str, None]:
        return self.name

    @_match_deorator
    def filter(self) -> Union[BasicPath, None]:
        return self


class File(BasicPath):
    """pathlib.Path subclass for file objects"""

    _suffix = "*"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def gro(self):
        return self._get_filetype(".gro")

    @property
    def top(self):
        return self._get_filetype(".top")

    def _get_filetype(self, suffix: str):
        suffix = suffix.strip(".")
        if suffix != self.suffix.strip("."):
            return PathFactory(self.with_suffix(f".{suffix}"))
        else:
            return self

    def check(self):
        if not self.is_file():
            raise FileNotFoundError(f"{self.name} is not a file.")
        if self._suffix != "*" and self.suffix != self._suffix:
            raise ValueError(f"Expected {self._suffix}, found {self.suffix}")
        else:
            logger.debug("Correct file extension.")

    @property
    def parent(self) -> Dir:
        """The logical parent of the path."""
        drv = self._drv
        root = self._root
        parts = self._parts
        if len(parts) == 1 and (drv or root):
            return DirFactory(self)
        return DirFactory(self._from_parsed_parts(drv, root, parts[:-1]))

    @property
    def parents(self) -> Dir:
        return _PathParents(self)


# -------------------------------------------------------------------------------------------------
# ITP/TOP parameter classes
# -------------------------------------------------------------------------------------------------


class ParametersBase:
    """Base class for GROMACS topology parameter collections."""

    kwd_list = []

    def _arithmetic_type_check(method: Callable) -> Union[Callable, None]:
        @wraps(method)
        def wrapper(self, other):
            if other.__class__ == self.__class__:
                return method(self, other)
            elif hasattr(other, "collection"):
                if self.__class__ == other.collection:
                    return method(self, other)
            else:
                raise TypeError(
                    "Only Parameters of the same class can be combined.\n"
                    f"Found {self.__class__.__name__!r} and {other.__class__.__name__!r}"
                )

        return wrapper

    def __init__(self, *data):
        self.data = {}
        self.__kwds = KwdList([])
        for prm in data:
            if prm.kwd != "None":
                self._add_prm(prm)
        self.kwd_list = prm.kwd_list

    def _add_prm(self, prm):
        if prm.kwd != "None":
            setattr(self, prm.kwd, prm)
            self.kwds.append(prm.kwd)

    @property
    def kwds(self):
        return self.__kwds.sort_kwds(self.kwd_list)

    def __repr__(self):
        return f"{self.__class__.__name__}({list(self.kwds)!r})"

    @_arithmetic_type_check
    def __add__(self, other):
        if self.__class__ == other.__class__:
            for prm in other.kwds:
                if prm.kwd in self.kwds:
                    new = self.prm + other.prm
                else:
                    self._add_prm(prm)
        elif hasattr(other, "collection"):
            if other.collection == self.__class__:
                self._add_prm(other)
        else:
            raise TypeError(
                f"Expected instance of {self.__class__.name!r} or single parameter, "
                f"found {other.__class__.__name__!r}"
            )
        return self

    def __mul__(self, other: int):
        assert isinstance(
            other, int
        ), f"Multiplicator must be type {int.__name__!r}"
        for i in range(other):
            yield self

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            if self.kwds == other.kwds:
                for kwd in self.kwds:
                    if (
                        self.__getattribute__(kwd).full_df.sort_values()
                        == other.__getattribute__(kwd).full_df.sort_values()
                    ):
                        return True
        return False


class SystemParameters(ParametersBase):
    """GROMACS topology parameter collection class for system"""

    pass


class MoleculeParameters(ParametersBase):
    """GROMACS topology parameter collection class for molecules"""

    @property
    def name(self):
        if "moleculetype" in self.kwds:
            name = self.moleculetype
        else:
            name = None
        return name


class Parameters(ParametersBase):
    """General GROMACS topology parameter collection class"""

    pass


class ParametersFactory:
    """Factory class for GROMACS topology parameter collection classes"""

    def __new__(cls, data, *args, **kwargs):
        if type(data) == dict:
            data_list = []
            for key in data:
                if key is None:
                    data_list.append(
                        ParameterFactory({key: data[key]}, *args, **kwargs)
                    )
        data_list = data
        _cls = data_list[0].collection
        self = _cls(data_list)
        return self


class ParameterBase:
    """Base class for GROMACS topology parameters"""

    kwd_list = []
    suffix = ".itp"
    collection = ParametersBase

    def _arithmetic_type_check(method):
        @wraps(method)
        def wrapper(self, other):
            if other.__class__ != self.__class__:
                if other.__class__ == self.__class__.collection:
                    return other + self
                else:
                    raise TypeError(
                        "Only Parameters of the same class can be combined.\n"
                        f"Found {self.__class__.__name__!r} and {other.__class__.__name__!r}"
                    )
            else:
                return method(self, other)

        return wrapper

    def __init__(self, kwd, data, path=None):
        self._string: str = data
        self._kwd: Kwd = Kwd(kwd)
        self._df: pd.DataFrame = pd.DataFrame()
        self.init_df()
        self.update_df()
        self._path: ITPFile = path

    def __str__(self):
        return f"{self.__class__.__name__}({self.kwd!r})\n\n{self._df}\n"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.kwd!r})\n{self._df}"

    # def __getitem__(self, df_loc: Optional = None):
    #     if df_loc == None:
    #         df_loc = ":,:"
    #     return eval(method'self.full_df.loc[{df_loc}]')

    @_arithmetic_type_check
    def __add__(self, other):
        if other.__class__ == self.collection:
            new = other + self
        elif other.kwd == self.kwd:
            new_str = self._string + f"\n{other._string}"
            new = self.__class__(self.kwd, new_str, self._path)
        else:
            new = self.collection(self, other)
        return new

    @_arithmetic_type_check
    def __sub__(self, other):
        if other.__class__ == self.collection:
            new = other - self
        elif other.kwd == self.kwd:
            new_str = re.search(
                f"(.*){other.string}(.*)",
                self._string,
                flags=re.MULTILINE | re.DOTALL,
            )
            new_str = "".join([new_str.group(1), new_str.group(2)])
            new = self.__class__(self.kwd, new_str, self._path)
        else:
            new = self._string
        return new

    def __mul__(self, other: int):
        assert isinstance(
            other, int
        ), f"Multiplicator must be type {int.__name__!r}"
        for i in range(other):
            yield self

    def update_df(self) -> None:
        try:
            df = pd.read_csv(
                StringIO(self._string), sep="\s+", comment=";", header=None
            )
            column_names = list(_KWD_DICT[self.suffix][self.kwd].keys())
            if len(column_names) > len(df.columns):
                max_len = len(df.columns)
            else:
                max_len = len(column_names)
            df = df.iloc[:, :max_len]
            df.columns = column_names[:max_len]
            self._df = pd.concat([self._df, df])
            self._df.drop_duplicates(inplace=True)
        except EmptyDataError:
            pass

    @property
    def df(self) -> pd.DataFrame:
        return self._df.dropna(axis=1)

    def init_df(self) -> None:
        try:
            df = pd.DataFrame(
                columns=list(_KWD_DICT[self.suffix][self._kwd].keys())
            )
            df = df.astype(dtype=_KWD_DICT[self.suffix][self.kwd])
            self._df = df
        except KeyError:
            pass

    @property
    def ff(self):
        if self._path is not None and self._path.parent.suffix == ".ff":
            return self._path.parent
        else:
            return None

    def itp(self):
        if self._path is not None and self.suffix == ".itp":
            return self._path.name
        else:
            return None

    @property
    def kwd(self):
        return self._kwd

    @property
    def string(self):
        return self._string

    @cached_property
    def ptype(self):
        return self.__class__


class Parameter(ParameterBase):
    """Class for general GROMACS topology parameters"""

    kwd_list = [
        "defaults",
        "atomtypes",
        "bondtypes",
        "pairtypes",
        "angletypes",
        "dihedraltypes",
        "constrainttypes",
    ]
    collection = Parameters

    def __init__(self, kwd, data, path=None):
        if data is not None:
            data = re.sub(
                r"^\s*?#\s*?[a-zA-Z0-9_\-().,\s]+?\n",
                "",
                data,
                flags=re.MULTILINE,
            )
        self._string: str = data
        self._kwd: Kwd = Kwd(kwd)
        self._df: pd.DataFrame = pd.DataFrame()
        self.init_df()
        self.update_df()
        self._path: ITPFile = path


class MoleculeParameter(ParameterBase):
    """Class for GROMACS molecule parameters"""

    kwd_list = [
        "moleculetype",
        "atoms",
        "bonds",
        "pairs",
        "pairs_nb",
        "angles",
        "dihedrals",
        "exclusions",
        "constraints",
        "settles",
    ]
    collection = MoleculeParameters

    # def __init__(self, name: Optional[str]=None, **kwargs):
    #     self._data = {}
    #     if name is not None:
    #         self._data[name] = {**kwargs}

    # def append(self, name, _data):
    #     self._data[name] = _data


class SystemParameter(ParameterBase):
    """Class for GROMACS system parameters"""

    kwd_list = ["system", "molecules"]
    collection = SystemParameters

    # def __repr__(self):
    #     return method'{self.__class__.__name__}({self.name})'


class ParameterFactory:
    """Factory class for GROMACS topology parameters"""

    _prm_types = [Parameter, MoleculeParameter, SystemParameter]
    default = ParameterBase

    def __new__(cls, kwd, data, path):
        for prm_type in cls._prm_types:
            if kwd in prm_type.kwd_list:
                return prm_type(kwd, data, path)
        return cls.default(kwd, data)


# class ParameterCollectionFactory(ParameterFactory):
#     default = ParametersBase
#
#     def __new__(cls, _data):
#         for prm_type in cls._prm_types:
#             if kwd in prm_type.kwd_list:
#                 return prm_type.collection(kwd, _data)
#         return cls.default(kwd, _data)


# class ITPString:
#     def __init__(self, string):
#         self.string = self.process_string(string)
#         self.definitions = []
#
#     @staticmethod
#     def process_string(string):
#         """Remove comments from raw string."""
#         return re.sub(r"\s*;[a-zA-Z\d\s_.-]*?", "", self.string, flags=re.MULTILINE)
#
#     def __get_def_sections(self, ifstr=''):
#         defs = re.findall(rf'\s*#if{ifstr}def\s*(.+?)\n(.*?)#endif', x.read_text(), flags=re.MULTILINE | re.DOTALL)
#         n_defs = len(defs)
#         if n_defs > 0:
#             for d in defs:
#                 self.definitions.append(Definition(d))
#                 name, text = d
#                 text = text.split('#else')


# get_defs = partialmethod(__get_def_sections, ifstr='')
# get_ndfs = partialmethod(__get_def_sections, ifstr='n')


class Definition:
    def __init__(self, name: str, ifdef=None, ifndef=None):
        self.name = name
        self._defs = [ifdef, ifndef]

    @property
    def ifdef(self):
        return self._defs[0]

    @property
    def ifndef(self):
        return self._defs[0]

    @ifdef.setter
    def ifdef(self, prm):
        if self.ifdef is None:
            self._defs[0] = prm
        else:
            self._defs[0] += prm

    @ifndef.setter
    def ifndef(self, prm):
        if self.ifdef is None:
            self._defs[1] = prm
        else:
            self._defs[1] += prm

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.name!r}: ({self.ifdef.__class__.__name__}, "
            f"{self.ifndef.__class__.__name__}))"
        )

    def __eq__(self, other):
        if self.name == other.name:
            return True

    @property
    def string(self):
        return (
            f"\n#ifdef {self.name}\n"
            f"{self.ifdef}\n"
            f"#else\n{self.ifndef}\n#endif\n"
        )

    @property
    def default_str(self):
        return "\n".join([d.string for d in self.ifndefs])


class DefinitionList(UserList):
    def __init__(self, data: List[Definition]):
        self.data = []
        for d in data:
            self.data.append(d)

    @property
    def string(self):
        return "\n".join([d.string for d in self.data])

    @property
    def defaults(self):
        return [d.ifdef for d in self.data]


class Molecule:
    def __init__(self, name):
        self.name = name
        self._prms = MoleculeParameter(None, None)
        self._definitions = []

    @property
    def kwds(self):
        return self._prms.kwds

    def add_prm(self, prm):
        self._prms += prm

    def add_definition(self, definition: Definition):
        self._definitions.append(definition)

    @property
    def defaults(self):
        return self._prms + self._definitions.ifndef

    @property
    def definitions(self):
        return [d for d in self._definitions.name]

    def __iter__(self):
        yield self._prms


# TODO: add FE parameters to index
# TODO: add Molecule class (name it differently bc of mda)

# PRM_INFO_DICT = {
#     "n_atoms": cast(
#         Callable[[Universe], Dict[str, int]],
#         lambda u: dict(
#             [(r, u.select_atoms(method"moltype {r}").n_atoms) for r in u.atoms.moltypes]
#         ),
#     ),
#     "charges": cast(
#         Callable[[Universe], Dict[str, float]],
#         lambda u: dict(zip(u.atoms.moltypes, np.round(u.atoms.residues.charges, 4))),
#     ),
# }

# class UnitCell:
#     def __init__(
#         self, name: Union[str, BasicPath],
#             include_dir: Optional[Union[str, BasicPath]] = None
#     ):
#         name = File(name)
#         if name.suffix != '':
#             name = name.stem
#         self.name = name
#         self.gro = GROFile(name.with_suffix('.gro'))
#         self.itp = ITPFile(name.with_suffix('.itp'))
#         self.idx = self.name[-2:]
#         assert self.gro.suffix == ".gro", method'Expected ".gro" file, found {crds.suffix!r}'
#         self.u = MDAnalysis.Universe(str(self.gro))
#         self._ff = None
#         if include_dir is not None:
#             self.ff = include_dir
#
#     def get_uc_prms(self, prm_str: str,
#     include_dir: Optional[Union[ForceField, str]],
#     write=False,
#     force_update=False,
# ) -> dict:
#         dict_func = PRM_INFO_DICT[prm_str]
#         residue_itp = self.itp
#         prop_file = self.itp.parent / method".saved/{residue_itp.stem}_{prm_str}.p"
#
#         if (force_update is True) or (not prop_file.is_file()):
#             atom_u = Universe(
#                 str(residue_itp),
#                 topology_format="ITP",
#                 include_dir=str(include_dir),
#                 infer_system=True,
#             )
#             prop_dict = dict_func(atom_u)
#             if write is True:
#                 with open(prop_file, "wb") as prop_file:
#                     pkl.dump(prop_dict, prop_file)
#         else:
#             with open(prop_file, "rb") as prop_file:
#                 prop_dict = pkl.read(prop_file)
#         return prop_dict
#         if ff is not None:
#             ff = ForceField(ff)
#             assert ff.suffix == ".ff", method'Expected ".ff" directory, found {ff.suffix!r}'
#         else:
#             ff = self.ff
#         print(ff.itp_filelist)
#         nbfile = ff.itp_filelist.filter("ffnonbonded", mode="stem")[0]
#         return nbfile
#         # print(type(nbfile))
#         # ff_u = MDAnalysis.Universe(str(nbfile), topology_format='ITP',
#         #     include_dir=str(include_dir), infer_system=True)
#         # print(ff_u)
#
#     def __repr__(self):
#         return method"{self.__class__.__name__}({self.gro.stem!r})"
#
#     @property
#     def ff(self):
#         if self._ff is not None:
#             return self._ff
#         else:
#             print("No force field specified.")
#
#     @ff.setter
#     def ff(self, ff: Optional[Union[str, BasicPath]] = None, include = 'all', exclude = None):
#         if type(ff) != ForceField:
#             ff = ForceField(ff, include=include, exclude=exclude)
#         self._ff = ff


class ITPFile(File):
    """Container for .itp file contents.
    Attributes:
        `ff`: force field name
        `kwds`: keywords
        `string`: file contents as string without comments
        `definitions`: ifdef, ifndef options
        `parameters`: force field parameters (atomtypes, bondtypes, angletypes, dihedraltypes)
        `molecules`: name, atoms, bonds, angles, dihedrals
        `system`: system parameters"""

    _suffix = ".itp"
    _kwd_dict = _KWD_DICT[_suffix]
    _prm_types = [Parameter, Parameters]
    _mol_types = [MoleculeParameter, MoleculeParameters]  # , Molecules]
    _sys_types = [SystemParameter, SystemParameters]  # , Systems]
    _kwd_pattern = re.compile(
        r"(?<=\[ )[a-zA-Z_-]+(?= \])", flags=re.MULTILINE | re.DOTALL
    )
    collections_map = {"moleculetype": "moleculetypes"}

    def __init__(self, *args, **kwargs):
        # check if file exists
        super(ITPFile, self).__init__(*args, **kwargs)
        if self.is_file():
            self._string = self.process_string(self.read_text())
            self.__string = self.read_text()
        else:
            self.touch()
        if self.parent.suffix == ".ff":
            self.ff = self.parent.name
        self.__reset()

    @staticmethod
    def process_string(string):
        """Remove comments from raw string."""
        new_str = re.sub(
            r"\s*;[~<>?|`@$£%&*!{}()#^/\[\]\\a-zA-Z\d\s_.+=\"\',-]*?\n",
            "\n",
            string,
            flags=re.MULTILINE,
        )
        new_str = re.sub(r"\n+", "\n", new_str, flags=re.MULTILINE)
        return new_str

    @property
    def string(self):
        return self._string

    @string.setter
    def string(self, string):
        self.__string = string
        self._string = self.process_string(string)
        self.__reset()

    def __reset(self):
        self._prm_str_dict = {}
        self.__kwds = []
        self._definitions = {}
        self._prms = [Parameter(None, None), []]
        self._mols = [MoleculeParameter(None, None), []]
        self._sys = SystemParameter(None, None)

    def _split_str(
        self, section: Literal["system", "molecules", "parameters"]
    ):
        """Split file string into 'parameters', 'moleculetype', 'system' sections"""
        if section == "parameters":
            prms_str = self.string.split(sep="[ moleculetype")[0]
            return prms_str
        else:
            if section in self.kwds:
                if section == "system":
                    sys_str = re.search(
                        r"\[ system \]([#\[\]a-zA-Z\d\s_.+-]*)",
                        self.string,
                        flags=re.MULTILINE | re.DOTALL,
                    ).group(0)
                    return sys_str
                elif section == "moleculetype":
                    mol_str = re.search(
                        r"\[ moleculetype \][#\[\]a-zA-Z\d\s_.+-]+(?<=\[ system \])*",
                        self.string,
                        flags=re.MULTILINE | re.DOTALL,
                    ).group(0)
                    return mol_str
            else:
                return ""

    def get_section(self, prm: Literal["parameters", "molecules", "system"]):
        try:
            return self._prm_str_dict[prm]
        except KeyError:
            self._get_str_sections(prm)
        try:
            return self._prm_str_dict[prm]
        except KeyError:
            return ""

    def _get_str_sections(self, section):
        """Dictionary of string sections for 'parameter', 'moleculetype', 'system'"""
        if section not in self._prm_str_dict.keys():
            self._prm_str_dict[section] = self._split_str(section)

    def __get_def_sections(self, ifstr: Literal["", "n"] = ""):
        for section in ["parameter", "moleculetype", "system"]:
            defs = re.findall(
                rf"\s*#if{ifstr}def\s*([\[\]a-zA-Z\d\s_.+-]+?)\n(.*?)#endif",
                self.get_section(section),
                flags=re.MULTILINE | re.DOTALL,
            )
            n_defs = len(defs)
            definition_list = DefinitionList([])
            if n_defs > 0:
                for d in defs:
                    name, text = d
                    # self._definitions.append(name)
                    textsplit = text.split("#else")
                    print(textsplit)
                    prm_list = []
                    for split in textsplit:
                        prms = self.__parse_str(string=split)
                        prm_list.append(prms)
                    if ifstr == "n":
                        prm_list.reverse()
                    definition_list.append(Definition(name, *prm_list))
            self._definitions[section] = definition_list

    def _get_definitions(self):
        self.get_defs()
        self.get_ndefs()

    get_defs = partialmethod(__get_def_sections, ifstr="")
    get_ndefs = partialmethod(__get_def_sections, ifstr="n")

    @property
    def definitions(self):
        if len(self._definitions) != 3:
            self._get_definitions()
        return list(self._definitions.values())

    @property
    def kwds(self):
        if len(self.__kwds) == 0:
            self.kwds = None
        return self.__kwds.sort_kwds(sort_map=[*self._kwd_dict.keys()])

    @kwds.setter
    def kwds(self, kwd_list: List[str] = None):
        if kwd_list is None:
            kwd_list = []
        self.__kwds.extend(kwd_list)
        kwds = re.findall(self._kwd_pattern, self.read_text())
        self.__kwds.extend(kwds)
        self.__kwds = KwdList(self.__kwds)
        self.__kwds.sort_kwds(list(self._kwd_dict.keys()))

    # def parse(self):
    #     sections = self.read_text().split(sep='[ ')
    #     kwds = self.kwds.concat()
    #     section_pattern = re.compile(rf"\s*({kwds}) \]\s*\n(.*)", flags=re.MULTILINE | re.DOTALL)
    #     for section in sections:
    #         match = re.search(section_pattern, section)
    #         if match is None:
    #             pass
    #         else:
    #             kwd = match.group(1)
    #             string = match.group(2)
    #             definition = re.split('#if[n]def', string)
    #
    #             print(kwd)
    #             prm = ParameterFactory(kwd, match.group(2))
    #             self.assign_parameter(prm)

    # @staticmethod
    # def __parse_str(string):
    #     prms = None
    #     sections = string.split(sep='[ ')
    #     kwds = get_match_pattern(_KWD_DICT['.itp'])
    #     section_pattern = re.compile(rf"\s*({kwds}) \]\s*\n([a-zA-Z\d#_.]*)", flags=re.MULTILINE | re.DOTALL)
    #     for section in sections:
    #         match = re.search(section_pattern, section)
    #         if match is None:
    #             pass
    #         else:
    #             kwd = match.group(1)
    #             prm = ParameterFactory(kwd, match.group(2))
    #             if prms == None:
    #                 prms = prm
    #             else:
    #                 prms += prm
    #     return prms

    def __getitem__(self, item: str):
        if item in self.kwds:
            return self.__parse_str(self.string, item)

    def __setitem__(self, key: str, value: Union[pd.DataFrame, str]):
        assert type(value) in [
            str,
            pd.DataFrame,
        ], f"Unexpected datatype ({type(value)}) for {key} value"
        if not isinstance(value, str):
            new_str = value.reset_index().to_string(
                index=False, float_format=lambda x: "{:.6f}".format(x)
            )
            new_str = f"; {new_str}\n\n"
        else:
            kwd_list = ITP_KWDS[key]
            max_kwd_len = max([len(kwd) for kwd in kwd_list])
            header = "\t".join([f"{kwd:{max_kwd_len}}" for kwd in kwd_list])
            new_str = f"; {header}\n{value}\n\n"
        if self.string != "":
            repl_str = re.sub(
                rf"({key} ]\s*\n)[;\sa-zA-Z\d#_.\-\n()^%$'\[\]]*",
                f"\1{new_str}",
                self.raw_string,
                flags=re.MULTILINE | re.DOTALL,
            )
            if re.search(key, repl_str, flags=re.MULTILINE) is None:
                repl_str = f"{self.raw_string}\n[ {key} ]\n{new_str}"
        else:
            repl_str = f"[ {key} ]\n{new_str}"
        repl_str = re.sub(r"\n\{2,}", "\n" * 2, repl_str, flags=re.MULTILINE)
        self.string = repl_str

    def __contains__(self, item: str):
        return item in self.kwds

    def get_parameter(self, prm_name: str):
        if prm_name in self.kwds:
            return self.__parse_str(self.string, prm_name)

    # @staticmethod
    def __parse_str(self, string, kwds: Optional[str] = None):
        prms = None
        kwd_dict = _KWD_DICT[".itp"]
        sections = string.split(sep="[ ")
        if kwds is None:
            kwds = get_match_pattern(kwd_dict)
        else:
            if isinstance(kwds, list):
                kwds = KwdList(kwds)
            elif isinstance(kwds, str):
                kwds = Kwd(kwds)
            kwds = get_match_pattern(kwds)
        section_pattern = re.compile(
            rf"\s*({kwds}) ]\s*\n([\sa-zA-Z\d#_.-]*)",
            flags=re.MULTILINE | re.DOTALL,
        )
        for section in sections:
            match = re.search(section_pattern, section)
            if match is None:
                pass
            else:
                kwd = match.group(1)
                prm = ParameterFactory(kwd, match.group(2), self.resolve())
                if prms is None:
                    prms = prm
                else:
                    prms += prm
        return prms

    @staticmethod
    def _get_def(definition: (Tuple[str])):
        name, text = definition
        text = text.split("#else")

    def _add_attrs(self):
        if len(self.data) == 0:
            self.data = None
        # kwd_sel = self.kwds.filter_list(['atomtypes', 'moleculetype', 'molecules'])
        kwd_sel = self.kwds
        for kwd in kwd_sel:
            print("setting", kwd)
            self.__setattr__(kwd, self.get_df(kwd))

    # TODO: make Parameters UserDict (or not)

    def assign_parameter(self, parameter, definition_id: Literal[0, 1] = 0):
        if parameter.__class__ in self._prm_types:
            self._prms[definition_id] = parameter
            # if hasattr(self, parameter.kwd):
            #     pass
            # else:
            #     self.__setattr__(parameter.kwd, parameter.kwd)
        elif parameter.__class__ in self._mol_types:
            self._mols[definition_id] += parameter
        elif parameter.__class__ in self._sys_types:
            self._sys[definition_id] += parameter

    def get_df(self, section):
        ...

    def write(self):
        with open(self, "w") as outfile:
            outfile.write(self.__string)

    @property
    def raw_string(self):
        return self.__string


class GROFile(File):
    """Container for .gro file contents."""

    _suffix = ".gro"

    def __init__(self, *args, **kwargs):
        # check if file exists
        super(GROFile, self).__init__(*args, **kwargs)
        self.__topology = None
        self.__universe = None
        self.description = None

    @property
    def string(self):
        with open(self, "r") as file:
            string = file.read()
        return string

    @property
    def df(self):
        df = pd.read_csv(
            str(self.resolve()),
            index_col=[0],
            skiprows=2,
            header=None,
            sep="\s+",
            names=["at-type", "atom-id", "x", "y", "z"],
            nrows=self.n_atoms,
        )
        return df

    @property
    def universe(self):
        # if self.__universe is None:
        #     self.__get_universe()
        # self.__get_universe()
        if self.__universe is not None:
            return self.__universe
        elif self.is_file():
            u = Universe(str(self.resolve()))
            # pos = u.atoms.positions
            # if re.search("interlayer", self.name) and (
            #     "SOL" in u.residues.resnames or "iSL" in u.residues.resnames
            # ):
            #     for residue in u.residues:
            #         if residue.resname in ["SOL", "iSL"]:
            #             residue.atoms.guess_bonds()
            #             assert len(residue.atoms.bonds) == 2
            #     sol = u.select_atoms("resname iSL SOL")
            #     sol.positions = sol.unwrap(compound="residues")
            return u

    def reset_universe(self):
        self.__universe = None

    # def __get_universe(self):
    #     return Universe(str(self.resolve()))

    @universe.setter
    def universe(
        self, universe: Union[AtomGroup, ResidueGroup, Atom, Residue, Universe]
    ):
        if universe.__class__.__name__ in ["AtomGroup", "Atom"]:
            with tempfile.NamedTemporaryFile(suffix=".gro") as grofile:
                universe.write(grofile.name)
                universe = Universe(grofile.name)
        elif universe.__class__.__name__ in ["ResidueGroup", "Residue"]:
            with tempfile.NamedTemporaryFile(suffix=".gro") as grofile:
                universe.atoms.write(grofile.name)
                universe = Universe(grofile.name)
        elif universe.__class__.__name__ == "Universe":
            pass
        else:
            raise TypeError(
                f"Unexpected type {universe.__class__.__name__} for universe!"
            )
        self.__universe = universe

    def write(self, topology=None):
        try:
            self.universe.atoms.write(str(self.resolve()))
            if self.description is None:
                self.description = self.stem
            with open(self, "r") as grofile:
                gro_str = grofile.read()
            gro_str = re.sub(
                "Written by MDAnalysis",
                f"{self.description}",
                gro_str,
                flags=re.MULTILINE | re.IGNORECASE,
            )
            with open(self, "w") as grofile:
                grofile.write(gro_str)
        except IndexError:
            logger.debug("Not writing empty Universe")
        if topology is not None:
            self.__topology = topology
        if self.__topology is not None:
            logger.debug(f"Writing topology {self.top}")
            topology.reset_molecules()
            topology.add_molecules(self.universe.atoms)
            topology.write(self.top)
        self.__universe = None

    @property
    def n_atoms(self):
        return int(
            self.string.splitlines()[1]
        )  # get_system_n_atoms(crds=self.universe, write=False)

    @property
    def dimensions(self):
        return self.string.splitlines()[self.n_atoms + 1]

    @property
    def top(self):
        return TOPFile(self.with_suffix(".top"), check=False)


class MDPFile(File):
    _suffix = ".mdp"

    def __init__(self, *args, gmx_version=0, **kwargs):
        File.__init__(self, *args, **kwargs)
        parameters = self.to_dict()
        self._string = None
        self._allowed_prms = {}
        self.gmx_version = gmx_version
        if gmx_version is not None:
            try:
                if gmx_version == 0:
                    self._allowed_prms = []
                    [
                        self._allowed_prms.extend(list(MDP_DEFAULTS[x].keys()))
                        for x in MDP_DEFAULTS.keys()
                        if re.fullmatch("[0-9]+", f"{x}")
                    ]
                    self._allowed_prms = np.unique(
                        [self._allowed_prms]
                    ).tolist()
                else:
                    self._allowed_prms = MDP_DEFAULTS[gmx_version]
            except KeyError:
                logger.info(
                    f"No parameter information found for GROMACS {gmx_version}"
                )
            else:
                self.check_keys(parameters)
        self._parameters = parameters

    @key_match_decorator("_allowed_prms")
    @get_mdp_data_dict_method_decorator
    def check_keys(self, other, search_str):
        for k in other.keys():
            if not re.search(
                k, search_str, flags=re.IGNORECASE | re.MULTILINE
            ):
                logger.error(f"Invalid parameter {k} for {self.gmx_version}")
                # print(f'{k}: ""')
                # # raise KeyError(f"Invalid parameter {k} for {self.gmx_version}")

    @property
    def parameters(self):
        if self._parameters is None:
            self._parameters = CaselessDict(self.to_dict())
        return {k: v for k, v in self._parameters.items() if v != ""}

    @parameters.setter
    @get_mdp_data_dict_method_decorator
    def parameters(self, parameters):
        self.add(parameters, replace=True)

    def to_dict(self):
        return mdp_to_dict(self)

    @key_match_decorator("_allowed_prms")
    @get_mdp_data_dict_method_decorator
    def add(
        self, other: Union[Dict[str, str], MDPFile], search_str, replace=True
    ):
        new_string = self.string
        freezegrps = None
        freezedims = ["Y", "Y", "Y"]
        if self.gmx_version:
            self.check_keys(other)
        for k, v in other.items():
            if k.lower() == "freezegrps":
                freezegrps = v
            elif k.lower() == "freezedim":
                freezedims = v
            elif re.search(k, search_str, flags=re.IGNORECASE | re.MULTILINE):
                if not re.search(
                    f"{k}\s*=", new_string, flags=re.IGNORECASE | re.MULTILINE
                ):
                    new_string = add_new_mdp_parameter(k, v, new_string)
                else:
                    if replace:
                        new_string = set_mdp_parameter(k, v, new_string)
                    else:
                        new_string = add_mdp_parameter(k, v, new_string)
            else:
                logger.finfo(
                    f'Invalid parameter {k} for GROMACS version {self.gmx_version if self.gmx_version else "unknown"}'
                )
        if freezegrps is not None and freezegrps != [] and freezegrps != "":
            new_string = set_mdp_freeze_groups(
                file_or_str=new_string,
                uc_names=freezegrps,
                freeze_dims=freezedims,
                replace=replace,
            )
        self.string = new_string

    # @key_match_decorator("parameters")
    # @get_mdp_data_dict_method_decorator
    # def __add__(
    #     self, other: Union[Dict[str, str], MDPFile], search_str, replace=True
    # ):
    #     new_string = self.string
    #     freezegrps = None
    #     freezedims = ["Y Y Y"]
    #     for k, v in other.items():
    #         if k.lower() == "freezegrps":
    #             freezegrps = v
    #         elif k.lower() == "freezedims":
    #             freezedims = v
    #         if re.match(search_str, k, flags=re.IGNORECASE):
    #             if replace:
    #                 new_string = set_mdp_parameter(k, v, new_string)
    #         else:
    #             new_string = add_mdp_parameter(k, v, new_string)
    #     if freezegrps:
    #         set_mdp_freeze_groups(freezegrps, new_string, freezedims)
    #     self.string = new_string
    #
    # def set_mdp_freeze_groups(self, freeze_groups, freezedims):
    #     self.string = set_mdp_freeze_groups(
    #         freeze_groups, self.string, freezedims
    #     )

    @key_match_decorator("parameters")
    def remove(self, other: List[str], search_str: str):
        new_string = self.string
        other = list(map(lambda x: x.lower(), other))
        for prm in other:
            if prm in self.parameters.keys():
                new_string = set_mdp_parameter(prm, "")
        self.string = new_string

    @property
    def string(self):
        if self._string is None:
            self._string = self.read_text()
        return self._string

    @string.setter
    def string(self, string):
        if string != self._string:
            self._string = None
            self.write_prms(text=string, all=True)
            self._parameters = None

    def write_prms(self, text: str, all: bool = False):
        if all:
            self.write_text(text)
        else:
            text = re.sub(
                r'^.+?=["\'\s]+(\n|\Z)', "", text, flags=re.MULTILINE
            )
            self.write_text(text)


class TOPFile(ITPFile):
    _suffix = ".top"

    def __init__(self, *args, **kwargs):
        super(TOPFile, self).__init__(*args, **kwargs)
        self.__coordinates = None

    @property
    def gro(self) -> GROFile:
        return GROFile(self.with_suffix(".gro"))


class YAMLFile(File):
    _suffix = ".yaml"

    def __init__(self, *args, **kwargs):
        super(YAMLFile, self).__init__(*args, **kwargs)
        self._data = None

    @property
    def data(self):
        if self._data is None:
            self._data = self._read_data()
        return self._data

    # @data.setter
    # def data(self, data):
    #     self._data = data

    def _read_data(self) -> Dict[Any, Any]:
        try:
            with open(self, "r") as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"{str(self.name)!r} does not exist.")
            return None

    def _write_data(self):
        if self._data:
            with open(self, "w") as file:
                yaml.dump(self._data, file)
            self._data = None

    @staticmethod
    @get_file_or_str
    def get_data(input_string: dict):
        # @get_file_or_str
        # def get_dict(input_string: PathOrStr):
        #     return input_string
        return input_string  # get_file_or_str(lambda x: input_string)

    @data.setter
    def data(self, data: Union[str, dict, PathType]):
        data = self.get_data(data)
        if data:
            self._data = data
            self._write_data()


class CSVFile(File):
    _suffix = ".csv"

    def __init__(self, *args, **kwargs):
        super(CSVFile, self).__init__(*args, **kwargs)
        self._data = None

    @property
    def data(self) -> Union[pd.DataFrame, None]:
        if self._data is None:
            self._data = self._read_data()
        return self._data

    def _read_data(
        self, sep=",", header: Optional[int] = 0, dtype: str = object
    ) -> Union[pd.DataFrame, None]:
        try:
            with open(self, "r") as file:
                return pd.read_csv(file, sep=sep, header=header, dtype=dtype)
        except FileNotFoundError:
            logger.warning(f"{str(self.name)!r} does not exist.")
            return None

    @data.setter
    def data(self, data: pd.DataFrame):
        if data is not None:
            if len(data) != 0:
                self._data = data
                self._write_data()

    def __add__(self, other):
        data = self.data
        if data and other:
            other.columns = data.columns
            other.index.names = data.index.names
            data = data.append(other)
        self._data = data
        self._write_data()

    def _write_data(self, index=False):
        if self._data is not None:
            if len(self._data) != 0:
                self._data.to_csv(self, sep=",", index=index, header=True)
                logger.info(f"Writing {self.name!r}")
                self._data = None


class Dir(BasicPath):
    _suffices = "*"
    _suffix = "*"

    def check(self):
        if not self.is_dir():
            raise FileNotFoundError(f"{self.name} is not a directory.")
        if self._suffix != "*" and self.suffix != self._suffix:
            raise ValueError(
                f"Expected {self._suffix!r}, found {self.suffix!r}"
            )

    def _get_filelist(self, ext: StrOrListOf = _suffices) -> FileList:
        assert type(ext) in [
            str,
            list,
        ], f"'ext': Expected 'str' or 'list' instance, found {ext.__class__.__name__!r}"
        if ext == "*":
            filelist = [file for file in self.iterdir()]
        else:
            format_ext = lambda x: "." + x.strip(".") if x != "" else x
            ext_match_list = lambda x: list(self.glob(f"*{x}"))
            if type(ext) == str:
                ext = format_ext(ext)
                filelist = ext_match_list(ext)
            elif type(ext) == list:
                ext = list(map(lambda x: format_ext(x), ext))
                filelist = []
                for e in ext:
                    filelist.append(ext_match_list(e))
        filelist = PathListFactory(filelist)
        return filelist
        # if hasattr(self, '_order'):
        #     order = {}
        #     for file in filelist:
        #         if file.stem in self._order.keys():
        #             order[file] =
        #     filelist = sorted(filelist, key=lambda file: self._order[file])
        # else:
        #     filelist = sorted(filelist)
        # return filelist

    def with_suffix(self, suffix):
        """Return a new path with the file suffix changed.  If the path
        has no suffix, add given suffix.  If the given suffix is an empty
        string, remove the suffix from the path.
        """
        f = self._flavour
        if f.sep in suffix or f.altsep and f.altsep in suffix:
            raise ValueError("Invalid suffix %r" % (suffix,))
        if suffix and not suffix.startswith(".") or suffix == ".":
            raise ValueError("Invalid suffix %r" % (suffix))
        name = self.name
        if not name:
            raise ValueError("%r has an empty name" % (self,))
        old_suffix = self.suffix
        if not old_suffix:
            name = name + suffix
        else:
            name = name[: -len(old_suffix)] + suffix
        return FileFactory(
            self._from_parsed_parts(
                self._drv, self._root, self._parts[:-1] + [name]
            )
        )

    @property
    def dirlist(self) -> DirList:
        return DirList(self)

    @property
    def filelist(self) -> FileList:
        return PathListFactory(self)

    @property
    def itp_filelist(self) -> ITPList:
        return ITPList(self)

    @property
    def gro_filelist(self) -> GROList:
        return GROList(self)

    def __copy__(self) -> Dir:
        return _copy(self)


class FFDir(Dir):
    _suffix = ".ff"
    _suffices = ".itp"
    pass

    def check(self):
        if not self.is_dir():
            raise FileNotFoundError(f"{self.name} is not a directory.")
        assert (
            self.suffix == ".ff"
        ), f"Expected {self._suffix!r}, found {self.suffix!r}"

    def _get_ff_parameter(self, prm_name: str):
        assert (
            prm_name in _KWD_DICT[".itp"]
        ), f"{prm_name!r} is not a recognised parameter."
        # full_df = pd.DataFrame(columns=  )
        for itp in self.itp_filelist:
            if prm_name in itp.kwds:
                # split_str = itp.get_parameter("atomtypes")
                split_str = itp[prm_name]
        return split_str

    # def atomtypes(self):
    #     for itp in self.itp_filelist:
    #         if 'atomtypes' in itp.kwds:
    #             split_str = itp.string.split('[ ')
    #             for split in split_str:
    #                 if re.match('')


class BasicPathList(UserList):
    def __init__(self, path: PathType, ext="*", check=False, order=None):
        if not isinstance(path, list):
            self.path = DirFactory(path)
            # if order is not None:
            #     self._order = order
            files = self.path._get_filelist(ext=ext)
            self._data = []
            for file in files:
                self._data.append(PathFactory(file, check=check))
        else:
            self._data = path
            path = []
            for file in self._data:
                if file.parent not in path:
                    path.append(file.parent)
                if len(path) == 1:
                    self.path = DirFactory(path[0])
                else:
                    self.path = None
        self.data = self._data

    def _reset_paths(method):
        @wraps(method)
        def wrapper(self, *args, pre_reset=True, post_reset=True, **kwargs):
            if pre_reset is True:
                self.data = self._data
            result = deepcopy(method(self, *args, **kwargs))
            if post_reset is True:
                self.data = self._data
            return result

        return wrapper

    @property
    def names(self) -> List[str]:
        return self.extract_fnames()

    @property
    def stems(self) -> List[str]:
        return self.extract_fstems()

    @property
    def suffices(self) -> List[str]:
        return self.extract_fsuffices()

    @_reset_paths
    def _extract_parts(
        self, part="name", pre_reset=True, post_reset=True
    ) -> List[Tuple[str, str]]:
        data_list = self.data
        data_list = [getattr(obj, part) for obj in data_list]
        self.data = data_list
        return self

    @_reset_paths
    def filter(
        self,
        pattern: Union[List[str], str],
        mode: Literal["full", "parts", "name", "stem"] = "parts",
        keep_dims: bool = False,
        pre_reset=False,
        post_reset=False,
    ):
        match = [obj.filter(pattern, mode=mode) for obj in self.data]
        if not keep_dims:
            match = [obj for obj in match if obj is not None]
        self.data = match
        return self

    extract_fstems = partialmethod(_extract_parts, part="stem")
    extract_fnames = partialmethod(_extract_parts, part="name")
    extract_fsuffices = partialmethod(_extract_parts, part="suffix")

    def __sub__(self, other):
        remove_list = deepcopy(self)
        remove_list.filter(other, pre_reset=False, post_reset=False)
        subtracted_data = [
            item for item in self.data if item not in remove_list
        ]
        return remove_list.filter(subtracted_data)

    def __copy__(self):
        return _copy(self)


class FileList(BasicPathList):
    _ext = "*"

    def __init__(self, path: Union[BasicPath, Dir], check=False, order=None):
        if not isinstance(path, list):
            self.path = DirFactory(path)
            # if order is not None:
            #     self._order = order
            files = self.path._get_filelist(ext=self.__class__._ext)
            self._data = []
            for file in files:
                self._data.append(PathFactory(file, check=check))
        else:
            self._data = path
            path = []
            for file in self._data:
                if file.parent not in path:
                    path.append(file.parent)
                if len(path) == 1:
                    self.path = DirFactory(path[0])
                else:
                    self.path = None
        # self.data = self._data
        # self.path = DirFactory(path)
        # files = self.path._get_filelist(ext=self.__class__._ext)
        # self._data = []
        # for file in files:
        #     self._data.append(FileFactory(file, check=check))
        self.data = self._data


class PathList(FileList):
    pass


class ITPList(FileList):
    _ext = ".itp"
    _order = {"forcefield": 0, "ffnonbonded": 1, "ffbonded": 2}
    pass

    def __init__(self, path, check=False):
        super().__init__(path, check)
        order = self.__class__._order
        for key in self.stems:
            if key not in order.keys():
                if re.search(r".*nonbonded", key) is not None:
                    order[key] = 4
                elif re.search(r".*bonded", key) is not None:
                    order[key] = 5
                else:
                    order[key] = 6

        # super().__init__(path, check, order)
        self._data = self.data = sorted(
            self._data, key=lambda itp: order[itp.stem]
        )
        self.order = [order[key.stem] for key in self._data]

    def __copy__(self):
        return _copy(self)


class GROList(FileList):
    _ext = ".gro"
    pass

    def __copy__(self):
        return _copy(self)


class MDPList(FileList):
    _ext = ".mdp"
    pass


class TOPList(ITPList):
    _ext = ".top"
    pass


class DirList(BasicPathList):
    _ext = ""

    def __init__(self, path: Union[BasicPath, Dir], check=False):
        self.path = DirFactory(path)
        files = self.path._get_filelist(ext=self.__class__._ext)
        self._data = []
        for file in files:
            if file.is_dir():
                self._data.append(DirFactory(file, check=check))
        self.data = self._data


class FFList(DirList):
    _ext = ".ff"
    _prm_info_dict = {
        "n_atoms": cast(
            Callable[[Universe], Dict[str, int]],
            lambda u: dict(
                [
                    (r, u.select_atoms(f"moltype {r}").n_atoms)
                    for r in u.atoms.moltypes
                ]
            ),
        ),
        "charges": cast(
            Callable[[Universe], Dict[str, float]],
            lambda u: dict(
                zip(u.atoms.moltypes, np.round(u.atoms.residues.charges, 4))
            ),
        ),
    }
    pass


class ForceField:
    _order = {
        "forcefield": 0,
        "ffnonbonded": 1,
        "ffbonded": 2,
        r"*ffnonbonded": 3,
        ".*ffbonded": 4,
        r".*?((non)?bonded)": 5,
    }

    def __init__(
        self, path: Union[BasicPath, Dir], include="all", exclude=None
    ):
        self._init_path(path)
        self.include(include)
        self.exclude(exclude)

    def _init_path(self, path):
        if type(path) == ForceField:
            self.__dict__ = path.__dict__
        else:
            if type(path) == FFDir:
                self.path = path
            else:
                self.path = FFDir((FF / path).with_suffix(".ff"))
            self.name: str = self.path.name
            self._itp_list = self.itp_filelist = ITPList(self.path)

    def include(self, itp_names):
        self.itp_filelist = self._itp_list
        if itp_names == "all":
            pass
        else:
            self.itp_filelist = self.itp_filelist.filter(
                itp_names, mode="stem", keep_dims=False
            )
        return self.itp_filelist

    def exclude(self, itp_names):
        if itp_names is None:
            pass
        else:
            exclude_list = self.itp_filelist
            exclude_list = exclude_list.filter(
                itp_names, mode="stem", keep_dims=False
            )
            self.itp_filelist = [
                item for item in self.itp_filelist if item not in exclude_list
            ]
            print(type(self.itp_filelist))
        return self.itp_filelist

    def __repr__(self):
        return (
            f"{self.path.name}: ["
            + ", ".join([itp.stem for itp in self.itp_filelist])
            + "]"
        )

    def __getitem__(self, item):
        prm = self.path._get_ff_parameter(item)
        return prm

    def __iter__(self):
        return self.name, self.itp_filelist


# -----------------------------------------------------------------------------
# Path object factories
# -----------------------------------------------------------------------------


class _PathFactoryBase(BasicPath):
    _file_types = {}
    _default = BasicPath

    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls, *args, **kwargs)
        _cls = cls._get_path_subclass(self)
        return _cls(*args, **kwargs)

    @classmethod
    def _get_path_subclass(cls, obj):
        if obj.suffix in cls._file_types:
            _cls = cls._file_types[obj.suffix]
        else:
            _cls = cls._default
        return _cls


class DirFactory(_PathFactoryBase):
    _file_types = {".ff": FFDir}
    _default = Dir


class FileFactory(_PathFactoryBase):
    _file_types = {
        ".itp": ITPFile,
        ".gro": GROFile,
        ".top": TOPFile,
        ".mdp": MDPFile,
    }
    _default = File


class PathFactory(BasicPath):
    _default = BasicPath

    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls._default, *args, **kwargs)
        if self.is_dir():
            self = DirFactory(*args, **kwargs)
        elif self.is_file() or self.suffix != "":
            self = FileFactory(*args, **kwargs)
        else:
            pass
        return self


class PathListFactory(BasicPathList):
    _file_types = {
        ".itp": ITPList,
        ".gro": GROList,
        ".mdp": MDPList,
        ".top": TOPList,
        ".ff": FFList,
    }
    _default = BasicPathList

    def __new__(cls, dir, ext="*", check=False):
        if ext == "*":
            path_list = cls._default(dir, ext, check)
            suffix = np.unique(path_list.suffices)
            if len(suffix) != 1:
                return path_list
            else:
                suffix = suffix[0]
        else:
            suffix = ext
        if suffix in cls._file_types.keys():
            return cls._file_types[suffix](dir, check)
        else:
            return cls._default(dir, suffix, check)


class SimDir(Dir):
    _suffix = ""
    _suffices = [".gro", ".top", ".trr", ".log", ".tpr"]
    _trr_sel = "largest"
    _subfolder = False

    @property
    def idx(self) -> Tuple[str, str, str]:
        """
        Extracts run information from path and returns as pd.MultiIndex
        :return: clay, ion, aa
        :rtype: Tuple(str, str, str)
        """
        idsl = pd.IndexSlice
        if self.subfolder is False:
            idsl = idsl[[*str(self.resolve()).split("/")[-3:]]]
        else:
            idsl = idsl[[*str(self.resolve()).split("/")[-4:-1]]]
        return idsl

    @cached_property
    def gro(self) -> _PosixPath:
        suffix = "gro"
        searchlists = [["_n", "_06", "_7"], ["_neutral", "_setup"]]
        for searchlist in searchlists:
            f = select_named_file(
                self.resolve(),
                searchlist=searchlist,  # ["_n", "_06", "_neutral", "_setup", "_7"],
                suffix=suffix,
                how="latest",
            )
            if f is None:
                continue
            else:
                break
        try:
            logger.finfo(kwd_str=f"{suffix!r}: ", message=f"{f.name!r}")
        except AttributeError:
            logger.finfo(kwd_str=f"{suffix!r}: ", message="No file found")
        return f
        # return select_named_file(
        #     path=self.resolve(), suffix="gro", searchlist=FILE_SEARCHSTR_LIST
        # )

    @cached_property
    def tpr(self) -> _PosixPath:
        suffix = "tpr"
        searchlists = [["_n", "_06", "_7"], ["_em", "_neutral", "_setup"]]
        for searchlist in searchlists:
            f = select_named_file(
                self.resolve(),
                searchlist=searchlist,  # ["_n", "_06", "_neutral", "_setup", "_7"],
                suffix=suffix,
                how="latest",
            )
            if f is None:
                continue
                # method = select_named_file(self.resolve(),
                #                       searchlist=[''])
            else:
                break
        try:
            logger.debug(f"{suffix!r}: {f.name!r}")
        except AttributeError:
            logger.debug(f"{suffix!r}: No file found")
        return f
        # return select_named_file(
        #     path=self.resolve(), suffix="gro", searchlist=FILE_SEARCHSTR_LIST
        # )

    @cached_property
    def top(self) -> _PosixPath:
        suffix = "top"
        searchlists = [["_n", "_06", "_neutral", "_setup", "_7"], ["em"]]
        for searchlist in searchlists:
            f = select_named_file(
                self.resolve(),
                searchlist=searchlist,  # ["_n", "_06", "_neutral", "_setup", "_7"],
                suffix=suffix,
                how="latest",
            )
            if f is None:
                continue
            else:
                break
        try:
            logger.finfo(f"{suffix!r}: {f.name!r}")
        except AttributeError:
            logger.finfo(f"{suffix!r}: No file found")
        return f
        # return select_named_file(
        #     path=self.resolve(), suffix="top", searchlist=FILE_SEARCHSTR_LIST
        # )

    @cached_property
    def trr(self) -> _PosixPath:
        suffix = "trr"
        f = select_named_file(
            self.resolve(),
            searchlist=["_n", "_06", "_7"],
            suffix=suffix,
            how="largest",
        )
        try:
            logger.debug(f"{suffix!r}: {f.name!r}")
        except AttributeError:
            logger.debug(f"{suffix!r}: No file found")
        return f
        # return select_named_file(
        #     path=self.resolve(), suffix="trr", searchlist=FILE_SEARCHSTR_LIST
        # )

    @cached_property
    def log(self) -> _PosixPath:
        suffix = "log"
        f = select_named_file(
            self.resolve(),
            searchlist=["_n", "_06", "_neutral", "_setup", "_7"],
            suffix=suffix,
            how="latest",
        )
        try:
            logger.debug(f"{suffix!r}: {f.name!r}")
        except AttributeError:
            logger.debug(f"{suffix!r}: No file found")
        return f

    @property
    def pathdict(self):
        file_dict = dict(
            [
                (k, getattr(self, k))
                for k in ["gro", "top", "trr", "log", "tpr"]
            ]
        )
        return file_dict

    @property
    def missing(self):
        suffices = [
            path_type
            for path_type, path in self.pathdict.items()
            if path is None
        ]
        return suffices

    @property
    def suffices(self):
        suffices = [
            path_type
            for path_type, path in self.pathdict.items()
            if path is not None
        ]
        return suffices

    @property
    def trr_sel(self) -> Literal["largest", "latest"]:
        return self._trr_sel

    @trr_sel.setter
    def trr_sel(self, trr_sel: Literal["largest", "latest"]):
        self._trr_sel = trr_sel

    @property
    def subfolder(self) -> bool:
        return self._subfolder

    @subfolder.setter
    def subfolder(self, subfolder: bool):
        self._subfolder = subfolder

    @property
    def base_dir(self):
        if self.subfolder is True:
            base = SimDir(self.parent, check=False)
        else:
            base = self
        return base


def init_path(path, exist_ok=True):
    if not Dir(path).is_dir():
        os.makedirs(path, exist_ok=exist_ok)
        logger.finfo(f"Creating new directory {str(path)!r}")


def set_mdp_parameter(
    parameter, value, mdp_str, searchex="[A-Za-z0-9 ._,\-]*?"
):
    value = mdp_value_formatter(value)
    new_str = re.sub(
        rf"(?<={parameter})(\s*)(=\s*)\s?({searchex})\s*?((\s?;[a-z0-9 ._,\-])?)(\n)",
        r"\1" + f"= {value} " + r"\4\n",
        mdp_str,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    return new_str


def add_mdp_parameter(
    parameter, value, mdp_str, searchex="[A-Za-z0-9 ._,]*?"
) -> str:
    value = mdp_value_formatter(value)
    new_str = re.sub(
        rf"(?<={parameter})(\s*)(=\s*)\s?({searchex})\s*?((\s?;[a-z0-9 ._,\-])?)(\n)",
        r"\1= \3" + f" {value} " + r"\4\n",
        mdp_str,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    return new_str


def add_new_mdp_parameter(
    parameter, value, mdp_str, searchex="[A-Za-z0-9 ._,]*?"
) -> str:
    value = mdp_value_formatter(value)
    mdp_str = (
        re.sub(r"(.*)\n?\Z", r"\1", mdp_str, flags=re.MULTILINE)
        + f"\n{parameter:<25} = {value}"
    )
    return mdp_str


@get_file_or_str
def set_mdp_freeze_groups(
    uc_names: List[str],
    input_string,
    freeze_dims: Union[
        Literal["Y"], Literal["N"], List[Union[Literal["Y"], Literal["N"]]]
    ] = ["Y", "Y", "Y"],
    replace=True,
) -> str:
    if isinstance(freeze_dims, str):
        if freeze_dims not in ["Y", "N"]:
            raise ValueError(
                f"Unexpected freeze dimensions value: {freeze_dims!r}"
                "\nAccepted options are: Y and N"
            )
        else:
            freeze_dims = [freeze_dims for _ in range(3)]
    if not (
        len(freeze_dims) % 3 == 0 and len(freeze_dims) // 3 == len(uc_names)
    ):
        raise ValueError(
            "Freeze dimensions must have either 1 or 3 elements "
            "in total or 3 elements per group"
        )
    freezegrpstr = " ".join(uc_names)
    freezearray = np.tile(freeze_dims, (len(uc_names)))
    freezedimstr = " ".join(freezearray)
    if not np.isin(freezearray, ["Y", "N"]).all():
        raise ValueError
    for freeze_str, freeze_value in zip(
        ["freezegrps", "freezedim"], [freezegrpstr, freezedimstr]
    ):
        if re.search(
            freeze_str, input_string, flags=re.IGNORECASE | re.MULTILINE
        ):
            if replace:
                input_string = set_mdp_parameter(
                    freeze_str, freeze_value, input_string
                )
            else:
                input_string = add_mdp_parameter(
                    freeze_str, freeze_value, input_string
                )
        else:
            input_string = add_new_mdp_parameter(
                freeze_str, freeze_value, input_string
            )

    return input_string


@get_file_or_str
def mdp_to_dict(input_string: str) -> Dict[str, str]:
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
    mdp_yaml = CaselessDict(line.split("=") for line in mdp_yaml.splitlines())
    # mdp_yaml = re.sub(r'\n\n+', '\n', mdp_yaml, flags=re.MULTILINE)
    return mdp_yaml


def mdp_value_formatter(
    value: Union[List[Union[str, int, float]], str, int, float]
) -> str:
    if isinstance(value, list):
        value = list(map(lambda v: str(v).lower(), value))
        value = " ".join(value)
    else:
        value = str(value)
    return value


def dict_to_mdp(
    mdp_dict: Dict[str, Union[List[Union[str, int, float]], str, int, float]]
) -> str:
    prm_list = [
        f"{str(k):<25} = {mdp_value_formatter(v)}" for k, v in mdp_dict.items()
    ]
    prm_str = "\n".join(prm_list)
    return prm_str
