import os
import re
from collections import UserList, UserString, UserDict
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
    NoReturn,
)

import MDAnalysis
import MDAnalysis.units
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from ClayAnalysis import KWD_DICT as _KWD_DICT, FILE_SEARCHSTR_LIST
from ClayAnalysis.utils import select_file, select_named_file

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
            check = re.search(rf"{match_pattern}", searchstr).group(0)
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


Kwd = NewType("Kwd", str)


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


KwdList = NewType("KwdList", UserList)


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


BasicPath = NewType("BasicPath", _Path)


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
            # print(pattern)
            if mode == "full":
                check = match_str(self.name, pattern)
            else:
                if mode == "stem":
                    check = match_str(self.name_split[0], pattern)
                elif mode in ["ext", "suffix"]:
                    check = match_str(self.name_split[1], pattern)
                elif mode == "parts":
                    # print(self.name)
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
        # print(drv, root, parts)
        if len(parts) == 1 and (drv or root):
            return Dir(self)
        return Dir(self._from_parsed_parts(drv, root, parts[:-1]))

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
        # print(self.kwd_list)

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
                        self.__getattribute__(kwd).df.sort_values()
                        == other.__getattribute__(kwd).df.sort_values()
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
        self.__string = data
        self.__kwd = Kwd(kwd)
        self._df = pd.DataFrame()
        self.init_df()
        self.update_df()
        self.__path = path

    def __str__(self):
        return self.__string

    def __repr__(self):
        return f"{self._df}\n{self.__class__.__name__}({self.kwd!r})"

    # def __getitem__(self, df_loc: Optional = None):
    #     if df_loc == None:
    #         df_loc = ":,:"
    #     return eval(f'self.df.loc[{df_loc}]')

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

    def update_df(self):
        try:
            df = pd.read_csv(
                StringIO(self.__string), sep="\s+", comment=";", header=None
            )
            column_names = list(_KWD_DICT[self.suffix][self.kwd].keys())
            # print(len(column_names), len(df.columns))
            if len(column_names) > len(df.columns):
                max_len = len(df.columns)
            else:
                max_len = len(column_names)
            df = df.iloc[:, :max_len]
            df.columns = column_names[:max_len]
            # print(max_len, len(column_names))
            self._df = pd.concat([self._df, df])
            # print(self._df)
            self._df.drop_duplicates(inplace=True)
        except EmptyDataError:
            pass

    @property
    def df(self):
        return self._df.dropna(axis=1)

    def init_df(self):
        try:
            df = pd.DataFrame(columns=list(_KWD_DICT[self.suffix][self.__kwd].keys()))
            df = df.astype(dtype=_KWD_DICT[self.suffix][self.kwd])
            self._df = df
        except KeyError:
            pass

    @property
    def ff(self):
        if self.__path != None and self.__path.parent == ".ff":
            return self.__path.parent
        else:
            return None

    def itp(self):
        if self.__path != None:
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
    #     self.data = {}
    #     if name is not None:
    #         self.data[name] = {**kwargs}

    # def append(self, name, data):
    #     self.data[name] = data


class SystemParameter(ParameterBase):
    kwd_list = ["system", "molecules"]
    collection = SystemParameters

    # def __repr__(self):
    #     return f'{self.__class__.__name__}({self.name})'


class ParameterFactory:
    _prm_types = [Parameter, MoleculeParameter, SystemParameter]
    default = ParameterBase

    def __new__(cls, kwd, data):
        for prm_type in cls._prm_types:
            if kwd in prm_type.kwd_list:
                return prm_type(kwd, data)
        return cls.default(kwd, data)


# class ParameterCollectionFactory(ParameterFactory):
#     default = ParametersBase
#
#     def __new__(cls, data):
#         for prm_type in cls._prm_types:
#             if kwd in prm_type.kwd_list:
#                 return prm_type.collection(kwd, data)
#         return cls.default(kwd, data)


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
class UnitCell:
    def __init__(
        self, crds: Union[str, BasicPath], ff: Optional[Union[str, BasicPath]] = None
    ):
        crds = File(crds)
        assert crds.suffix == ".gro", f'Expected ".gro" file, found {crds.suffix!r}'
        self.gro = crds
        self.u = MDAnalysis.Universe(str(self.gro))
        self._ff = None
        if ff != None:
            self.ff = ff

    def get_ff_prms(self, ff: Optional[Union[str, BasicPath]] = None):
        if ff != None:
            ff = FFDir(ff)
            assert ff.suffix == ".ff", f'Expected ".ff" directory, found {ff.suffix!r}'
            logger.info(f"FF directory: {ff.name}")
        else:
            ff = self.ff
        nbfile = ff.itp_filelist.filter("ffnonbonded", mode="stem")[0]
        return nbfile
        # print(type(nbfile))
        # ff_u = MDAnalysis.Universe(str(nbfile), topology_format='ITP',
        #     include_dir=str(ff), infer_system=True)
        # print(ff_u)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.gro.stem!r})"

    @property
    def ff(self):
        if self._ff != None:
            return self._ff
        else:
            logger.error("No force field specified.")

    @ff.setter
    def ff(self, ff: Optional[Union[str, BasicPath]] = None):
        ff = FFDir(ff, check=True)
        self._ff = ff


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
                    # print(textsplit)
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

    def get_parameter(self, prm_name: str):
        if prm_name in self.kwds:
            return self.__parse_str(self.string, prm_name)

    @staticmethod
    def __parse_str(string, kwds: Optional[str] = None):
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
                prm = ParameterFactory(kwd, match.group(2))
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
        # print(kwd_sel)
        kwd_sel = self.kwds
        for kwd in kwd_sel:
            logger.debug("setting", kwd)
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

    def _get_ff_parameter(self, prm_name: str):
        assert (
            prm_name in _KWD_DICT[".itp"]
        ), f"{prm_name!r} is not a recognised parameter."
        # df = pd.DataFrame(columns=  )
        for itp in self.itp_filelist:
            if "atomtypes" in itp.kwds:
                split_str = itp.get_parameter("atomtypes")
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
        self.data = []
        for file in files:
            self.data.append(PathFactory(file, check=check))

    @property
    def names(self):
        return self.extract_fnames()

    @property
    def stems(self):
        return self.extract_fstems()

    @property
    def suffices(self):
        return self.extract_fsuffices()

    def _extract_parts(self, part="name"):
        data_list = self.data
        data_list = [getattr(obj, part) for obj in data_list]
        return data_list

    def filter(
        self,
        pattern: Union[List[str], str],
        mode: Literal["full", "parts"] = "parts",
        keep_dims: bool = False,
    ) -> List[Union[Kwd, None]]:
        match = [obj.filter(pattern, mode=mode) for obj in self.data]
        if not keep_dims:
            match = [obj for obj in match if obj != None]
        return match

    extract_fstems = partialmethod(_extract_parts, part="stem")
    extract_fnames = partialmethod(_extract_parts, part="name")
    extract_fsuffices = partialmethod(_extract_parts, part="suffix")


class FileList(BasicPathList):
    _ext = "*"

    def __init__(self, path: Union[BasicPath, Dir], check=False):
        self.path = Dir(path)
        files = self.path._get_filelist(ext=self.__class__._ext)
        self.data = []
        for file in files:
            self.data.append(FileFactory(file, check=check))


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
        self.data = []
        for file in files:
            self.data.append(DirFactory(file, check=check))


class FFList(DirList):
    _ext = ".ff"
    pass


class ForceField:
    def __init__(
        self, path: Union[BasicPath, Dir], check=False, include="all", exclude=None
    ):
        self.path = FFDir(path)
        self.name = self.path.name
        self._itp_list = self.itp_filelist = ITPList(self.path)
        # print(type(self._itp_list), type(self.itp_filelist))
        self.include(include)
        self.exclude(exclude)

    def include(self, itp_names):
        if itp_names == "all":
            pass
        else:
            self.itp_filelist.filter(itp_names, mode="stem", keep_dims=False)
        return self.itp_filelist

    def exclude(self, itp_names):
        if itp_names == None:
            pass
        else:
            exclude_list = self.itp_filelist
            exclude_list = exclude_list.filter(itp_names, mode="stem", keep_dims=False)
            self.itp_filelist = [
                item for item in self.itp_filelist if item not in exclude_list
            ]
        return self.itp_filelist

    def __repr__(self):
        return (
            f"{self.path.name}: ["
            + ", ".join([itp.stem for itp in self.itp_filelist])
            + "]"
        )


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
                # print(2)
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
