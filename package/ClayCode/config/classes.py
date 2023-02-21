from __future__ import annotations
import os
import re
from collections import UserList #, UserString, UserDict
from collections.abc import Sequence
from functools import (
    partialmethod,
    singledispatch,
    wraps,
    cached_property,
)
from io import StringIO
from pathlib import Path as _Path, PosixPath as _PosixPath
from typing import (
    Union,
    List,
    Literal,
    AnyStr,
    Iterable,
    Optional,
    Tuple,
    Dict,
    Any,
    NewType,
    NoReturn, cast, Callable #, Type,
)

# import MDAnalysis
# import MDAnalysis.units
import numpy as np
import pandas as pd
from MDAnalysis import Universe
from pandas.errors import EmptyDataError

from ClayCode import UCS, FF
from ClayCode.config._consts import KWD_DICT as _KWD_DICT
from ClayCode.config.utils import select_named_file  # select_file,

import logging

logger = logging.getLogger(_Path(__file__).stem)
logger.setLevel(logging.DEBUG)

# -----------------------------------------------------------------------------
# class decorators
# -----------------------------------------------------------------------------


def add_method(cls):
    """Add new method to existing class"""

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)

        setattr(cls, func.__name__, wrapper)

    return decorator


def add_property(cls):
    """Add new property to existing class"""

    def decorator(func):
        @wraps(func)
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


def match_str(searchstr: str, pattern: str) -> Union[None, str]:
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
def get_match_pattern(pattern: Union[AnyStr, List[AnyStr]]):
    """Generate search pattern from list, str, dict, int or float"""
    raise TypeError(f"Unexpected type {type(pattern)!r}!")


@get_match_pattern.register
def _(pattern: list) -> str:
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


class Kwd(str):
    """str container for '.itp' and '.top' file parameters"""

    def match(
        self,
        pattern: Union[str, List[str], Dict[str, Any]],
        mode: Literal["full", "parts"] = "full",
    ) -> Union[Kwd, None]:
        """Match keywords against a search pattern and return match"""
        if mode == "full":
            check = match_str(self, pattern)
        else:
            if mode == "parts":
                check = re.match(rf"[a-zA-Z-_\d.]*{pattern}[a-zA-Z-_\d.]*", self)
            else:
                raise ValueError(f'{mode!r} is not a valid value for {"mode"!r}')
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

        def wrapper(
            self,
            pattern: Union[str, List[str], int, float, Dict[str, Any]],
            mode: Literal["full", "stem", "ext", "suffix", "parts"] = "full",
        ) -> Union[BasicPath, None]:
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
                    raise ValueError(f'{mode!r} is not a valid value for {"mode"!r}')
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
            logger.info(f"{self.name} exists")

    @property
    def name_split(self) -> Tuple[str, str]:
        return self.stem, self.suffix

    @classmethod
    def check_os(cls):
        return (
            cls
            if os.name == "posix"
            else NotImplementedError("ClayCode does not support Windows systems.")
        )

    @_match_deorator
    def match_name(self) -> str:
        return self.name

    @_match_deorator
    def filter(self):
        return self


Dir = NewType("Dir", _Path)


class File(BasicPath):
    _suffix = "*"

    def check(self):
        if not self.is_file():
            raise FileNotFoundError(f"{self.name} is not a file.")
        if self._suffix != "*":
            assert (
                self.suffix == self._suffix
            ), f"Expected {self._suffix}, found {self.suffix}"
        else:
            logger.info("Correct file extension.")

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


# ----------------------------------------------------------------------------------------------------------------------
# ITP/TOP parameter classes
# ----------------------------------------------------------------------------------------------------------------------


class ParametersBase:
    kwd_list = []

    def _arithmetic_type_check(method):
        def wrapper(self, other):
            if other.__class__ == self.__class__:
                return method(self, other)
            elif hasattr(other, "collection"):
                if self.__class__ == other.collection:
                    return method(self, other)
            raise TypeError(
                "Only Parameters of the same class can be combined.\n"
                f"Found {self.__class__.__name__!r} and {other.__class__.__name__!r}"
            )
            return result

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
        assert isinstance(other, int), f"Multiplicator must be type {int.__name__!r}"
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
    pass


class MoleculeParameters(ParametersBase):
    @property
    def name(self):
        if "moleculetype" in self.kwds:
            name = self.moleculetype
        else:
            name = None
        return name


class Parameters(ParametersBase):
    pass


class ParametersFactory:
    def __new__(cls, data, *args, **kwargs):
        if type(data) == dict:
            data_list = []
            for key in data:
                if key != None:
                    data_list.append(
                        ParameterFactory({key: data[key]}, *args, **kwargs)
                    )
        data_list = data
        _cls = data_list[0].collection
        self = _cls(data_list)
        return self


class ParameterBase:
    kwd_list = []
    suffix = ".itp"
    collection = ParametersBase

    def _arithmetic_type_check(method):
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
        self.__string: str = data
        self.__kwd: Kwd = Kwd(kwd)
        self._df: pd.DataFrame = pd.DataFrame()
        self.init_df()
        self.update_df()
        self.__path: ITPFile = path

    def __str__(self):
        return f"{self.__class__.__name__}({self.kwd!r})\n\n{self._df}\n"
        # return self.__string

    def __repr__(self):
        return f"{self.__class__.__name__}({self.kwd!r})\n{self._df}"

    # def __getitem__(self, df_loc: Optional = None):
    #     if df_loc == None:
    #         df_loc = ":,:"
    #     return eval(f'self.full_df.loc[{df_loc}]')

    @_arithmetic_type_check
    def __add__(self, other):
        if other.__class__ == self.collection:
            new = other + self
        elif other.kwd == self.kwd:
            new_str = self.__string + f"\n{other.__string}"
            new = self.__class__(self.kwd, new_str, self.__path)
        else:
            new = self.collection(self, other)
        return new

    @_arithmetic_type_check
    def __sub__(self, other):
        if other.__class__ == self.collection:
            new = other - self
        elif other.kwd == self.kwd:
            new_str = re.search(
                f"(.*){other.string}(.*)", self.__string, flags=re.MULTILINE | re.DOTALL
            )
            new_str = "".join([new_str.group(1), new_str.group(2)])
            new = self.__class__(self.kwd, new_str, self.__path)
        else:
            new = self.__string
        return new

    def __mul__(self, other: int):
        assert isinstance(other, int), f"Multiplicator must be type {int.__name__!r}"
        for i in range(other):
            yield self

    def update_df(self) -> None:
        try:
            df = pd.read_csv(
                StringIO(self.__string), sep="\s+", comment=";", header=None
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
            df = pd.DataFrame(columns=list(_KWD_DICT[self.suffix][self.__kwd].keys()))
            df = df.astype(dtype=_KWD_DICT[self.suffix][self.kwd])
            self._df = df
        except KeyError:
            pass

    @property
    def ff(self):
        if self.__path is not None and self.__path.parent.suffix == ".ff":
            return self.__path.parent
        else:
            return None

    def itp(self):
        if self.__path is not None and self.suffix == '.itp':
            return self.__path.name
        else:
            return None

    @property
    def kwd(self):
        return self.__kwd

    @property
    def string(self):
        return self.__string

    @cached_property
    def ptype(self):
        return self.__class__


class Parameter(ParameterBase):
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


class MoleculeParameter(ParameterBase):
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
    kwd_list = ["system", "molecules"]
    collection = SystemParameters

    # def __repr__(self):
    #     return f'{self.__class__.__name__}({self.name})'


class ParameterFactory:
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

Definition = NewType("Definition", Any)


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
#             [(r, u.select_atoms(f"moltype {r}").n_atoms) for r in u.atoms.moltypes]
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
#         assert self.gro.suffix == ".gro", f'Expected ".gro" file, found {crds.suffix!r}'
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
#         prop_file = self.itp.parent / f".saved/{residue_itp.stem}_{prm_str}.p"
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
#             assert ff.suffix == ".ff", f'Expected ".ff" directory, found {ff.suffix!r}'
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
#         return f"{self.__class__.__name__}({self.gro.stem!r})"
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
    """.itp container.
    Attributes:
        `ff`: force field name
        `kwds`: keywords
        `string`: file contents as string without comments
        `definitions`: ifdef, ifndef options
        `parameters`: force field parameters (atomtypes, bondtypes, angletypes, dihedraltypes)
        `molecules`: name, atoms, bonds, angles, dihedrals
        `system`: system parameters"""

    suffix = ".itp"
    _kwd_dict = _KWD_DICT[suffix]
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
        self.string = self.process_string(self.read_text())
        self._prm_str_dict = {}
        if self.parent.suffix == ".ff":
            self.ff = self.parent.name
        self.__kwds = []
        self._definitions = {}
        self._prms = [Parameter(None, None), []]
        self._mols = [MoleculeParameter(None, None), []]
        self._sys = SystemParameter(None, None)

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

    def _split_str(self, section: Literal["system", "molecules", "parameters"]):
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

    def get_parameter(self, prm_name: str):
        if prm_name in self.kwds:
            return self.__parse_str(self.string, prm_name)

    # @staticmethod
    def __parse_str(self, string, kwds: Optional[str] = None):
        prms = None
        kwd_dict = _KWD_DICT[".itp"]
        sections = string.split(sep="[ ")
        if kwds == None:
            kwds = get_match_pattern(kwd_dict)
        else:
            if type(kwds) == list:
                kwds = KwdList(kwds)
            elif type(kwds) == str:
                kwds = Kwd(kwds)
            kwds = get_match_pattern(kwds)
        section_pattern = re.compile(
            rf"\s*({kwds}) ]\s*\n([\sa-zA-Z\d#_.-]*)", flags=re.MULTILINE | re.DOTALL
        )
        for section in sections:
            match = re.search(section_pattern, section)
            if match is None:
                pass
            else:
                kwd = match.group(1)
                prm = ParameterFactory(kwd, match.group(2), self.resolve())
                if prms == None:
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


class GROFile(File):
    pass


class MDPFile(File):
    pass


class TOPFile(ITPFile):
    pass


class Dir(BasicPath):
    _suffices = "*"
    _suffix = "*"

    def check(self):
        if not self.is_dir():
            raise FileNotFoundError(f"{self.name} is not a directory.")
        if self._suffix != "*":
            assert (
                self.suffix == self._suffix
            ), f"Expected {self._suffix!r}, found {self.suffix!r}"

    def _get_filelist(self, ext: Union[str, list] = _suffices):
        assert type(ext) in [
            str,
            list,
        ], f"'ext': Expected 'str' or 'list' instance, found {ext.__class__.__name__!r}"
        if ext == "*":
            filelist = [file for file in self.iterdir()]
        else:
            format_ext = lambda x: "." + x.strip(".")
            ext_match_list = lambda x: list(self.glob(f"*{x}"))
            if type(ext) == str:
                ext = format_ext(ext)
                filelist = ext_match_list(ext)
            elif type(ext) == list:
                ext = list(map(lambda x: format_ext(x), ext))
                filelist = []
                for e in ext:
                    filelist.append(ext_match_list(e))
        return sorted(filelist)

    @property
    def filelist(self):
        return PathListFactory(self)

    @property
    def itp_filelist(self):
        return ITPList(self)


class FFDir(Dir):
    _suffix = ".ff"
    _suffices = ".itp"
    pass

    def check(self):
        if not self.is_dir():
            raise FileNotFoundError(f"{self.name} is not a directory.")
        assert self.suffix == ".ff", f"Expected {self._suffix!r}, found {self.suffix!r}"

    def _get_ff_parameter(self, prm_name: str):
        assert (
            prm_name in _KWD_DICT[".itp"]
        ), f"{prm_name!r} is not a recognised parameter."
        # full_df = pd.DataFrame(columns=  )
        for itp in self.itp_filelist:
            if "atomtypes" in itp.kwds:
                # split_str = itp.get_parameter("atomtypes")
                split_str = itp['atomtypes']
        return split_str

    # def atomtypes(self):
    #     for itp in self.itp_filelist:
    #         if 'atomtypes' in itp.kwds:
    #             split_str = itp.string.split('[ ')
    #             for split in split_str:
    #                 if re.match('')


class BasicPathList(UserList):
    def __init__(self, path: Union[BasicPath, Dir], ext="*", check=False):
        self.path = DirFactory(path)
        files = self.path._get_filelist(ext=ext)
        self._data = []
        for file in files:
            self._data.append(PathFactory(file, check=check))
        self.data = self._data

    def _reset_paths(f):
        def wrapper(self, *args, pre_reset=True, post_reset=True, **kwargs):
            if pre_reset is True:
                self.data = self._data
            result = f(self, *args, **kwargs)
            return result
            if post_reset is True:
                self.data = self._data
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
    def _extract_parts(self, part="name",
                       pre_reset=True,
                       post_reset=True) -> List[Tuple[str, str]]:
        data_list = self._data
        data_list = [getattr(obj, part) for obj in data_list]
        self.data = data_list
        return self

    @_reset_paths
    def filter(
        self,
        pattern: Union[List[str], str],
        mode: Literal["full", "parts"] = "parts",
        keep_dims: bool = False,
            pre_reset=False,
            post_reset=False
    ):
        match = [obj.filter(pattern, mode=mode) for obj in self.data]
        if not keep_dims:
            match = [obj for obj in match if obj is not None]
        self.data = match
        return self

    extract_fstems = partialmethod(_extract_parts, part="stem")
    extract_fnames = partialmethod(_extract_parts, part="name")
    extract_fsuffices = partialmethod(_extract_parts, part="suffix")


class FileList(BasicPathList):
    _ext = "*"

    def __init__(self, path: Union[BasicPath, Dir], check=False):
        self.path = DirFactory(path)
        files = self.path._get_filelist(ext=self.__class__._ext)
        self._data = []
        for file in files:
            self._data.append(FileFactory(file, check=check))
        self.data = self._data



class PathList(FileList):
    pass


class ITPList(FileList):
    _ext = ".itp"
    pass

    def __copy__(self):
        return self


class GROList(FileList):
    _ext = ".gro"
    pass


class MDPList(FileList):
    _ext = ".mdp"
    pass


class TOPList(FileList):
    _ext = ".top"
    pass


class DirList(BasicPathList):
    _ext = "*"

    def __init__(self, path: Union[BasicPath, Dir], check=False):
        self.path = DirFactory(path)
        files = self.path._get_filelist(ext=self.__class__._ext)
        self._data = []
        for file in files:
            self.data.append(DirFactory(file, check=check))
        self.data = self._data


class FFList(DirList):
    _ext = ".ff"
    _prm_info_dict = {
        "n_atoms": cast(
            Callable[[Universe], Dict[str, int]],
            lambda u: dict(
                [(r, u.select_atoms(f"moltype {r}").n_atoms) for r in u.atoms.moltypes]
            ),
        ),
        "charges": cast(
            Callable[[Universe], Dict[str, float]],
            lambda u: dict(zip(u.atoms.moltypes, np.round(u.atoms.residues.charges, 4))),
        ),
    }
    pass


class ForceField:

    def __init__(self, path: Union[BasicPath, Dir], include="all", exclude=None):
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
                self.path = FFDir((FF / path).with_suffix('.ff'))
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
            exclude_list = exclude_list.filter(itp_names, mode="stem", keep_dims=False)
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
    _file_types = {".itp": ITPFile, ".gro": GROFile, ".top": TOPFile, ".mdp": MDPFile}
    _default = File


class PathFactory(BasicPath):
    _default = BasicPath

    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls._default, *args, **kwargs)
        if self.is_dir():
            self = DirFactory(*args, **kwargs)
        elif self.is_file():
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
            logger.info("subfolder")
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
            logger.info(f"{suffix!r}: {f.name!r}")
        except AttributeError:
            logger.info(f"{suffix!r}: No file found")
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
            logger.info(f"{f}")
            if f is None:
                continue
                # f = select_named_file(self.resolve(),
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
            logger.info(f"{suffix!r}: {f.name!r}")
        except AttributeError:
            logger.info(f"{suffix!r}: No file found")
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
        # return select_named_file(
        #     path=self.resolve(), suffix="log", searchlist=FILE_SEARCHSTR_LIST
        # )

    # @cached_property
    # def suffices(self):
    #     return self._suffices

    # @cached_property
    # def __pathlist = [self.gro, self.top, self.log, self.trr]

    @property
    def pathdict(self):
        file_dict = dict(
            [(k, getattr(self, k)) for k in ["gro", "top", "trr", "log", "tpr"]]
        )
        return file_dict

    @property
    def missing(self):
        suffices = [
            path_type for path_type, path in self.pathdict.items() if path is None
        ]
        return suffices

    @property
    def suffices(self):
        suffices = [
            path_type for path_type, path in self.pathdict.items() if path is not None
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

class UnitCell(ITPFile):
    @property
    def idx(self):
        return self.stem[2:]

    @property
    def clay_type(self):
        return self.parent.name

    @property
    def uc_stem(self):
        return self.stem[:2]

    @property
    def atoms(self):
        return self.data


class UCData(Dir):
    _suffices = ['.gro', '.itp']

    def __init__(self, path, uc_stem=None, ff=None):
        if uc_stem is None:
            self.uc_stem = self.name[-2:]
        else:
            self.uc_stem = uc_stem
        self.ff = ForceField(ff)
        id_cols = list(map(lambda x: str(x[-2:]), self.available))
        self.atomtypes = self.ff['atomtypes'].df
        idx = self.atomtypes.iloc[:, 0]
        cols = [*id_cols, 'charge', 'sheet']
        self.full_df = pd.DataFrame(index=idx,
                                    columns=cols)
        self.full_df['charge'].update(self.atomtypes.set_index('at-type')['charge'])
        self.get_df_sheet_annotations()
        self.full_df['sheet'].fillna('X', inplace=True)
        self.full_df.fillna(0, inplace=True)
        self.uc_list = [UnitCell(itp) for itp in self.itp_filelist]
        self.read_ucs()
        self.occ = self.full_df.groupby('sheet').sum()
        self.full_df.set_index('sheet', append=True, inplace=True)
        self.full_df.sort_index(inplace=True, level=1, sort_remaining=True)
        self.full_df.index = self.full_df.index.reorder_levels(['sheet', 'at-type'])
        charge = self.full_df[id_cols].copy()
        charge = charge.apply(lambda x: x * self.full_df['charge'], raw=True)
        self.total_charge = charge[id_cols].sum().round(2).convert_dtypes()
        self.total_charge.name = 'charge'
        # self.total_charge.index = pd.MultiIndex.from_tuples([('C', 'tot')], names=['sheet', 'at-type'])
        self.df = self.full_df.reset_index('at-type').filter(regex=r'^(?![X].*)', axis=0)
        self.df = self.df.reset_index().set_index(['sheet', 'at-type'])
        # self.uc_df = pd.concat(self.df[id_cols], self.total_charge)
        # self.full_df = pd.concat(self.full_df)
        ...

    @property
    def uc_composition(self):
        return self.full_df.reindex(self.atomtypes, fill_value=0).filter(regex=r'^(?![oOhH].*)', axis=0)

    @property
    def idxs(self):
        return self.full_df.columns

    def check(self):
        if not self.is_dir():
            raise FileNotFoundError(f"{self.name} is not a directory.")

    def read_ucs(self):
        for uc in self.uc_list:
            atoms = uc['atoms'].df
            self.full_df[f'{uc.idx}'].update(atoms.value_counts('at-type'))
        # self.full_df.loc[pd.IndexSlice[:], pd.IndexSlice[f'{uc.idx}', 'charge']].update(atoms.groupby('at-type').mean()['charge'])

    def get_df_sheet_annotations(self):
        old_index = self.full_df.index
        regex_dict = {'T': r'[a-z]+t',
                      'O': r'[a-z]*[a-gi-z][o2]',
                      'C': 'charge'
                      }
        index_extension_list = []
        for key in regex_dict.keys():
            for element in old_index:
                match = re.fullmatch(regex_dict[key], element)
                if match is not None:
                    index_extension_list.append((key, match.group(0)))
        new_index = pd.MultiIndex.from_tuples(index_extension_list)
        new_index = new_index.to_frame().set_index(1)
        self.full_df['sheet'].update(new_index[0])

    
    @property
    def available(self):
        return self.itp_filelist.extract_fstems()

    def __str__(self):
        return

# class ForceField(FileParser):
#     def __init__(self, include='all', exclude=None):
#         super().__init__(relpath=pp.FF_PATH, fname='')
#         # self.__available = self._get_filelist()
#         self.__atomtypes = pd.DataFrame()
#         self.__moleculetypes = pd.DataFrame()
#         self.__atomtypes = pd.DataFrame(columns=[*super().kwd_dict['atomtypes']]
#                                         )
#         self.__atomtypes = self.__atomtypes.astype(dtype=self._get_dtypes('atomtypes',
#                                                                          [*super().kwd_dict['atomtypes']]
#                                                                          )
#                                                    )
#         self.__moleculetypes = pd.DataFrame(columns=[*super().kwd_dict['atoms']]
#                                             )
#         self.__moleculetypes = self.__moleculetypes.astype(dtype=self._get_dtypes('atoms',
#                                                                          [*super().kwd_dict['atoms']]
#                                                                          )
#                                                    )
#         self.__ff_dict = {'atomtypes': self.__atomtypes,
#                           'atoms': self.__moleculetypes
#                           }
#         self.__ff_selection = {}
#         self.select_ff(include=include, exclude=exclude)
#         self._read_ff()
#         self.get_atomtypes()
#         self.get_moleculetypes()
#
#     @property
#     def available(self):
#         return self.extract_fnames(self._get_filelist(path=self.path.parent, ext='.ff'))
#
#     @staticmethod
#     def print_ff_not_found(ff, searchstr='include'):
#         print(f'"{ff}" was listed in "{searchstr}" but was not '
#               'found in force field selection.\n')
#
#     def __include_ff(self, sel_dict, include='all'):
#         # Include all available ff
#         if include == 'all':
#             print(f'Including all force fields in "{self.path.name}".')
#             include = self.available
#         # include all ".itp" files for 1 selected ff
#         elif isinstance(include, (str, os.PathLike)):
#             print(f'Including {include}.')
#             include = self.match_ff_pathname(include)
#             sel_dict[include.stem] = 'all'
#         # Include all available ".itp" files for a lift of available ff
#         if isinstance(include, (list, np.ndarray)) and len(include) == 1:
#             ff_sel = self.match_ff_pathname(include[0])
#             sel_dict[ff_sel.stem] = 'all'
#         elif isinstance(include, (list, np.ndarray)) and len(include) > 1:
#             print('Including list:')
#             for ff_sel in include:
#                 print(f'{ff_sel}')
#                 ff_sel = self.match_ff_pathname(ff_sel)
#                 sel_dict[ff_sel.stem] = 'all'
#         # Include specific ".itp" files fora selection of ff
#         elif isinstance(include, dict):
#             print('Including dict:')
#             # sel_dict = {}
#             for ff_sel in include.keys():
#                 print(f'{ff_sel}')
#                 avail_str = ''.join(np.unique(dt.extract_fname(self.available,
#                                                                stem=True
#                                                                )
#                                               )
#                                     )
#                 print(f'available ff: {avail_str}')
#                 sel_dict = include
#                 try:
#                     sel_dict = self.match_dict_keys(sel_dict,
#                                                     ff_sel,
#                                                     avail_str
#                                                     )
#                 except AttributeError:
#                     self.print_ff_not_found(ff_sel)
#                     break
#             print(sel_dict)
#
#         # check itp files of selected ff
#         for ff_sel in sel_dict.keys():
#             print(f'Checking itp files: {sel_dict[ff_sel]}')
#             ff_sel = dt.extract_fname(ff_sel, stem=True)
#             print(self.path.parent)
#             avail_fnames = dt.extract_fname(self._get_filelist(path=self.path.parent / f'{ff_sel}.ff',
#                                                                ext='itp'
#                                                                )
#                                             )
#             avail_stems = dt.extract_fname(self._get_filelist(path=self.path.parent / f'{ff_sel}.ff',
#                                                               ext='itp'),
#                                            stem=True
#                                            )
#             print(f'All available itp files: {avail_fnames}')
#             if isinstance(sel_dict[ff_sel], (str, pl.Path)):
#                 sel_dict[ff_sel] = [sel_dict[ff_sel]]
#             else:
#                 sel_dict[ff_sel] = list(sel_dict[ff_sel])
#             print(len(sel_dict[ff_sel]), type(sel_dict[ff_sel]),[*sel_dict[ff_sel]])
#             if sel_dict[ff_sel][0] == 'all':
#                 print(f'list of len 1: {sel_dict[ff_sel]}')
#                 sel_dict[ff_sel] =  avail_stems
#             sel_dict[ff_sel] = self.match_fname_list(sel_dict[ff_sel],
#                                                      avail_fnames,
#                                                      ext='.itp'
#                                                      )
#         print(sel_dict)
#         return sel_dict
#
#     def select_ff(self, include='all', exclude=None):
#         # Initialise dictionary for ff selection
#         old_sel = self.__ff_selection
#         sel_dict = {}
#         incl_dict = {}
#         excl_dict = {}
#         incl_dict = self.__include_ff(incl_dict, include)
#         if exclude != None:
#             excl_dict = self.__include_ff(excl_dict, exclude)
#         print(incl_dict, excl_dict)
#         for ff_sel in incl_dict.keys():
#             if ff_sel not in excl_dict.keys():
#                 sel_dict[ff_sel] = incl_dict[ff_sel]
#             else:
#                 sel_dict[ff_sel] = list(filter(lambda incl: incl not in excl_dict[ff_sel],
#                                                incl_dict[ff_sel]
#                                                )
#                                         )
#         sel_dict = dict(filter(lambda ff_item: len(ff_item[1]) != 0,
#                                sel_dict.items()
#                                )
#                         )
#         self.__ff_selection = sel_dict
#         if old_sel != self.__ff_selection:
#             self._read_ff()
#         #return sel_dict
#
#     @property
#     def ff_selection(self) -> dict:
#         print(self.__ff_selection)
#         if len(self.__ff_selection.keys()) != 0:
#             return self.__ff_selection
#         else:
#             raise KeyError('No force field was selected.')
#
#     def _read_ff(self):
#         print(self.ff_selection)
#         try:
#             ff_sel_dict = self.ff_selection
#             print(ff_sel_dict.keys())
#         except AttributeError:
#             print('No force field was selected.')
#         for ff_sel in ff_sel_dict.keys():
#             for itp_num, itp in enumerate(ff_sel_dict[ff_sel]):
#                 try:
#                     print(self.path.parent / f'{ff_sel}.ff',
#                                           f'{itp}.itp')
#                     itp_file = FileParser(self.path.parent / f'{ff_sel}.ff',
#                                           f'{itp}.itp'
#                                           )
#                     print(f'reading {itp_file.name}')
#                 except IOError:
#                     print(f'could not open {itp}')
#                 itp_file.parse_itp(self.__ff_dict)
#             print(self.__ff_dict)
#         return self.__ff_dict
#
#     def get_atomtypes(self, ff_sel='all'):
#         if ff_sel == 'all':
#             ff_sel = self.ff_selection.keys()
#         if isinstance(ff_sel, (str, pl.Path)):
#             ff_sel = [dt.extract_fname(ff_sel, stem=True)]
#         elif isinstance(ff_sel, (np.ndarray, t.Sequence)):
#             ff_sel = dt.extract_fname(ff_sel, stem=True)
#         elif isinstance(ff_sel, dict):
#             ff_sel = dt.extract_fname(ff_sel.keys(), stem=True)
#         print(ff_sel, self.__ff_dict['atomtypes'].loc[self.__ff_dict['atomtypes']['ff'].isin(ff_sel)])
#         ids = pd.IndexSlice
#         if len(self.__ff_dict['atomtypes']) != 0:
#             self.__atomtypes = self.__ff_dict['atomtypes'].loc[self.__ff_dict['atomtypes']['ff'].isin(ff_sel),
#                                                                ['ff', 'itp', 'at_type', 'mass', 'charge', 'sigma', 'epsilon']
#                                                                ].drop_duplicates(keep='first')
#             return self.__atomtypes
#
#     def get_moleculetypes(self, ff_sel='all'):
#         if ff_sel == 'all':
#             ff_sel = self.ff_selection.keys()
#         if isinstance(ff_sel, (str, pl.Path)):
#             ff_sel = [dt.extract_fname(ff_sel, stem=True)]
#         elif isinstance(ff_sel, (np.ndarray, t.Sequence)):
#             ff_sel = dt.extract_fname(ff_sel, stem=True)
#         elif isinstance(ff_sel, dict):
#             ff_sel = dt.extract_fname(ff_sel.keys(), stem=True)
#         print(ff_sel, self.__ff_dict['atoms'].loc[self.__ff_dict['atoms']['ff'].isin(ff_sel)])
#         # ids = pd.IndexSlice
#         if len(self.__ff_dict['atoms']) != 0:
#             self.__moleculetypes = self.__ff_dict['atoms'].loc[self.__ff_dict['atoms']['ff'].isin(ff_sel),
#                                                                ['ff', 'itp', 'res_name', 'at_type', 'at_name', 'mass', 'charge']
#                                                                ].drop_duplicates(keep='first')
#             # self.__moleculetypes = self.__moleculetypes.loc[self.__moleculetypes.is_unique()]
#             print(type(self.__moleculetypes), self.__moleculetypes.size, self.__moleculetypes)
#         return self.__moleculetypes
#
#     @property
#     def atomtypes(self):
#         if len(self.__atomtypes) > 0:
#             return self.__atomtypes.set_index(['ff', 'itp'])['at_type'].drop_duplicates(keep='first')
#
#
#     @property
#     def moleculetypes(self):
#         if len(self.__moleculetypes) > 0:
#             return self.__moleculetypes.set_index(['ff', 'itp'])['res_name'].drop_duplicates(keep='first')
#
#     @property
#     def atomic_charges(self):
#         if len(self.__atomtypes) > 0:
#             return self.__atomtypes.set_index(['ff', 'itp', 'at_type'])['charge']
#
#     @property
#     def atomic_masses(self):
#         if len(self.__atomtypes) > 0:
#             return self.__atomtypes.set_index(['ff', 'itp', 'at_type'])['mass']
#
#     @property
#     def sigma_epsilon(self):
#         if len(self.__atomtypes) > 0:
#             return self.__atomtypes.set_index(['ff', 'itp', 'at_type'])['sigma', 'epsilon']
#
#     @property
#     def __molecule_sums(self):
#         if len(self.__moleculetypes) > 0 and len(self.__atomtypes) > 0:
#             mols = self.__moleculetypes.set_index(['ff', 'at_type'])
#             atoms = self.__atomtypes.set_index(['ff', 'at_type'])
#             mols.update(atoms)
#             mols.reset_index(inplace=True)
#             mols.dropna(inplace=True)
#             mols = mols.loc[:, ['ff','res_name', 'mass', 'charge']]
#             mols_grouper = mols.groupby(['ff', 'res_name'])
#             return mols_grouper.sum()
#             # mol_masses = pd.merge(self.__moleculetypes, self.__atomtypes.loc[:, ['ff', 'itp']], how='outer', on=['ff', 'itp'])#, 'at_type'])#.groupby(['ff', 'itp', 'res_name'])
#             # return mol_masses
#
#     @property
#     def molecule_masses(self):
#         if len(self.__molecule_sums) > 0:
#             return self.__molecule_sums['mass']
#
#     @property
#     def molecule_charges(self):
#         if len(self.__molecule_sums) > 0:
#             return self.__molecule_sums['charge']
#
#
# class UCData(FileParser):
#     #exclude_dict = {'O': []}
#     # print(clay_atom_types)
#     def __init__(self, clay_type, ff='ClayFF_Fe'):
#         super().__init__(fname=clay_type, relpath=pp.UC_PATH.parent)
#         print(f'path: {self.full_path} name {self.name}')
#         self.__clay_selection = {}
#         self.select_clay_type(clay_type)
#         print(f'clay selection {self.__clay_selection}')
#         self.__moleculetypes = pd.DataFrame()
#         self.__moleculetypes = pd.DataFrame(columns=[*super().kwd_dict['atoms']]
#                                             )
#         self.__moleculetypes = self.__moleculetypes.astype(dtype=self._get_dtypes('atoms',
#                                                                                   [*super().kwd_dict['atoms']]
#                                                                                   )
#                                                            )
#         self.__clay_dict = {'atoms': self.__moleculetypes
#                            }
#         #self.select_ff(include=include, exclude=exclude)
#         self._read_ucs()
#         self.__ff = ff
#         # self.get_atomtypes()
#         # self.get_moleculetypes()
#         self.__atomtypes = ForceField(include='ClayFF_Fe').atomtypes
#         self.get_uc_composition()
#         self.get_uc_charges()
#
#
#     @property
#     def clay_type_selection(self) -> dict:
#         print(self.__clay_selection)
#         if len(self.__clay_selection.keys()) != 0:
#             return self.__clay_selection
#         else:
#             raise KeyError('No clay type was selected.')
#
#     def select_clay_type(self, clay_type):
#         print(self.path.is_dir())
#         print(self.path)
#         if (self.path / clay_type).is_dir():
#             self.name = clay_type
#             print(f'name {self.name}')
#             self.__clay_selection[self.name] = self.available
#             print(self.__clay_selection)
#         else:
#             print(f'{self.path} is not dir')
#
#     def read_ucs(self):
#         pass
#
#     @property
#     def available(self):
#         print(self._get_filelist(path=self.path))
#         return self.extract_fnames(self._get_filelist(path=self.full_path, ext='itp'))
#
#     def _read_ucs(self):
#         print(self.clay_type_selection)
#         try:
#             ff_sel_dict = self.clay_type_selection
#             print(ff_sel_dict.keys())
#         except AttributeError:
#             print('No force field was selected.')
#         for ff_sel in ff_sel_dict.keys():
#             print(ff_sel)
#             for itp_num, itp in enumerate(ff_sel_dict[ff_sel]):
#                 print(ff_sel_dict[ff_sel], self.path, ff_sel, itp)
#                 # print(+'\n'+self.full_path / f'{ff_sel}',
#                 #                           f'{itp}')
#                 try:
#                     itp_file = FileParser(self.path / f'{ff_sel}',
#                                           f'{itp}'
#                                           )
#                     print(f'reading {itp_file.full_path}')
#                 except IOError:
#                     print(self.path)
#                     print(f'could not open {itp}')
#                 print(self.__clay_dict)
#                 itp_file.parse_itp(self.__clay_dict)
#             print(self.__clay_dict)
#         return self.__clay_dict
#
#     # def parse_itp(self, ff_dict: dict, kwd_dict=kwd_dict):
#
#     def get_uc_charges(self):
#         print(self.__clay_dict['atoms'].groupby('res_name').sum())
#         self.__uc_charges =  self.__clay_dict['atoms'].groupby('res_name').sum()['charge'].round(0)\
#             .astype(np.int32)#.apply(lambda count: count / 2)
#
#     @property
#     def uc_charges(self):
#         return self.__uc_charges.round(2)
#
#     @staticmethod
#     def extract_uc_numbers(uc_list):
#         try:
#             uc_list = list(uc_list)
#             print(uc_list)
#             match =  list(map(lambda uc_name: int(re.search(rf'{c.UC_STEM}([0-9]*)', uc_name).group(1)), uc_list))
#             print(match)
#             return match
#         except TypeError:
#             print('Unit cell names must be in a list of format "[a-zA-A]+\d+".')
#         except AttributeError:
#             print(match)
#             print('Unit cell names must be in a list.')
#
#
#     def get_uc_composition(self):
#         #print(pd.crosstab(self.__clay_dict['atoms']['at_type'], self.__clay_dict['atoms']['res_name'], dropna=False))
#         self.__uc_composition = self.__clay_dict['atoms'].loc[:,['res_name', 'at_type', 'id']].reset_index()\
#             .pivot_table(index='at_type',
#                          columns='res_name',
#                          aggfunc='size',
#                          fill_value=0).astype(np.int32)#.apply(lambda count: count / 2)
#         #col_names = self.extract_uc_numbers(self.__uc_composition.columns)
#         #col_names = list(map(lambda uc_name: int(re.search('[a-zA-Z]+(\d+)', uc_name).group(1)), col_names))
#         #self.__uc_composition.columns = col_names
#         #self.__uc_composition = self.__clay_dict['atoms'].groupby(['res_name', 'at_type']).count()['id'].reset_index()
#         #self.__uc_composition['id'].name = 'uc_composition'
#         # self.__uc_composition['atom_type_count'] = \
#         #self.__uc_composition = self.__uc_composition.pivot(index='at_type', columns='res_name').fillna(0)#groupby(['res_name', 'at_type']).count())
#         # self.__uc_composition.
#             #.groupby(['res_name', 'at_type']).count()
#
#     @property
#     def uc_composition(self):
#         return self.__uc_composition.reindex(self.atomtypes, fill_value=0).filter(regex=r'^(?![oOhH].*)', axis=0)
#
#     @property
#     def atomtypes(self):
#         return self.__atomtypes.values
#
#     @property
#     def uc_df(self):
#         print(self.uc_composition)
#         print(self.uc_charges)
#         uc_df = self.uc_composition.append(self.uc_charges)
#         uc_df.columns = self.extract_uc_numbers(uc_df.columns)
#         uc_df = uc_df.reset_index().set_index('at_type')
#         uc_df.index.name = 'element'
#         return uc_df
#
#     def transform_uc_df_index(self):
#         old_index = self.uc_df.index
#         regex_dict = {'T': r'[a-z]+t',
#                       'O': r'[a-z]+[o2]',
#                       'C': 'charge'
#                       }
#         index_extension_list = []
#         for key in regex_dict.keys():
#             for element in old_index:
#                 match = re.match(regex_dict[key], element)
#                 if match != None:
#                     index_extension_list.append((key, match.group(0)))
#         new_index = pd.MultiIndex.from_tuples(index_extension_list)
#         print(new_index)
#         self.uc_df.reindex(new_index)
#         print(self.uc_df)
#         extra_index = []

ucs = UCData(UCS / 'D21', ff='ClayFF_Fe')
print(ucs.available)
print(ucs.uc_composition)
...
