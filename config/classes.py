import re
from abc import ABC, abstractmethod
from pathlib import Path as _Path, PosixPath as _PosixPath
import os
import attr
from typing import (
    Union,
    List,
    Dict,
    TypeVar,
    NoReturn,
    ClassVar,
    Optional,
    Literal,
    NewType,
    Generic,
    Type,
    NamedTuple,
    Iterable,
)


from io import StringIO
from numba import jit
from pandas.errors import EmptyDataError
from pandocfilters import OrderedList

from config.lib import convert_to_list

# from config.defaults import _FF

from typing_extensions import TypeAlias, ParamSpec, ParamSpecKwargs, ParamSpecArgs
from collections import UserList, UserDict, OrderedDict
from functools import (
    partialmethod,
    singledispatchmethod,
    singledispatch,
    wraps,
    partial,
    cached_property,
)
import numpy as np
import pandas as pd
from collections.abc import Sequence, Collection, MutableSequence, MutableMapping
import itertools as it
import ast

ITP_KWD = {
    "defaults": ["nbfunc", "comb-rule", "gen-pairs", "fudgeLJ", "fudgeQQ"],
    "atomtypes": [
        "at-type",
        "at-number",
        "mass",
        "charge",
        "ptype",
        "sigma",
        "epsilon",
    ],
    "bondtypes": ["ai", "aj", "b0", "kb"],
    "pairtypes": ["ai", "aj", "V", "W"],
    "angletypes": ["ai", "aj", "ak", "theta0", "ktheta"],
    "dihedraltypes": ["ai", "aj", "ak", "al", "phi0", "phitheta"],
    "constrainttypes": ["ai", "aj", "b0"],
    "nonbond_params": ["ai", "aj", "V", "W"],
    "moleculetype": ["res-name", "n-excl"],
    "atoms": [
        "id",
        "at-type",
        "res-number",
        "res-name",
        "at-name",
        "charge-nr",
        "charge",
        "mass",
    ],
    "bonds": ["ai", "aj", "funct", "b0", "kb"],
    "pairs": ["ai", "aj", "funct", "theta0", "ktheta"],
    "angles": ["ai", "aj", "ak"],
    "dihedrals": ["ai", "aj", "ak", "al"],
    "system": ["sys_name"],
    "molecules": ["res-name", "mol-number"],
    "settles": ["at-type", "func", "doh", "dhh"],
    "exclusions": ["ai", "aj", "ak"],
    "nonbond_params": [""]
}
ITP_DTYPE = {
    "at-type": "str",
    "at-number": "int32",
    "ptype": "str",
    "sigma": "float64",
    "epsilon": "float64",
    "id": "int32",
    "res-number": "int32",
    "res-name": "str",
    "at-name": "str",
    "charge-nr": "int32",
    "charge": "float64",
    "mass": "float64",
    "FF": "str",
    "itp": "str",
    "ai": "int16",
    "aj": "int16",
    "ak": "int16",
    "al": "int16",
    "k0": "float64",
    "b0": "float64",
    "kb": "float64",
    "ktheta": "float64",
    "phi0": "float64",
    "phitheta": "float64",
    "V": "str",
    "W": "str",
    "nbfunc": "int16",
    "func": "int16",
    "comb-rule": "int16",
    "gen-pairs": "str",
    "fudgeLJ": "float32",
    "fudgeQQ": "float32",
    "n-excl": "int16",
    "doh": "float32",
    "dhh": "float32",
    "funct": "int16"
}

File = NewType("File", _PosixPath)
Dir = NewType("Dir", _PosixPath)
PosixFile = NewType("File", _PosixPath)
PosixDir = NewType("Dir", _PosixPath)
ITPFile = NewType("ITPFile", _PosixPath)
GROFile = NewType("GROFile", _PosixPath)
TOPFile = NewType("ITPFile", _PosixPath)
MDPFile = NewType("MDPFile", _PosixPath)
PathList = NewType("PathList", List[PosixFile])
ITPList = NewType("ITPList", PathList)
GROList = NewType("GROList", PathList)
TOPList = NewType("TOPList", PathList)
PathLike: TypeAlias = Union[
    File, Dir, PosixFile, PosixDir, _PosixPath, _Path, ITPList, ITPFile, GROFile, MDPFile, TOPFile
]
# Definition = NewType('Definition', Definition)
# Parameter = NewType('Parameter', Parameter)


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
        print(idx)
        if isinstance(idx, slice):
            return tuple(self[i] for i in range(*idx.indices(len(self))))

        if idx >= len(self) or idx < -len(self):
            raise IndexError(idx)
        return self._pathcls._from_parsed_parts(
            self._drv, self._root, self._parts[: -idx - 1]
        )

    def __repr__(self):
        return "<{}.parents>".format(self._pathcls.__name__)

# 
# # class decorators
# 
# 
# def add_method(cls):
#     def decorator(func):
#         @wraps(func)
#         def wrapper(self, *args, **kwargs):
#             return func(self, *args, **kwargs)
# 
#         setattr(cls, func.__name__, wrapper)
# 
#     return decorator
# 
# 
# def add_property(cls):
#     def decorator(func):
#         @wraps(func)
#         def wrapper(self, *args, **kwargs):
#             return func(self, *args, **kwargs)
# 
#         setattr(cls, func.__name__, wrapper)
# 
#     return decorator


# data object functions

class PathBase(ABC):
    def check_os(subclass: Type[_Path]) -> Union[NotImplementedError, Type[_Path]]:
        """
        OS check for pathlib Path initialisation
        :param subclass: Path object
        :type subclass: _Path
        :return: Path object or NotImplementedError
        :rtype: _PosixPath
        """
        return (
            subclass
            if os.name == "posix"
            else NotImplementedError("ClayCode does not support Windows systems.")
        )

    def filter_name(
            self, pattern: str, mode: Literal["full", "stem", "ext", "parts"] = "full"
    ) -> Union[str, None]:
        if mode == "full":
            match = match_str(self.name, pattern)
        else:
            name_split = split_fname(self.name)
            if mode == "stem":
                match = match_str(name_split[0], pattern)
            elif mode == "ext":
                match = match_str(name_split[1], pattern)
            elif mode == "parts":
                pattern = re.escape(pattern)
                match = re.match(rf"[a-zA-Z-_\d.]*{pattern}[a-zA-Z-_\d.]*", self.name)
            else:
                raise ValueError(f'{mode!r} is not a valid value for {"mode"!r}')
        if match is not None:
            match = self
        return match

    def match_name(
            self, pattern: str, mode: Literal["full", "stem", "ext", "parts"] = "parts"
    ) -> Union[str, None]:
        if mode == "full":
            match = match_str(self.name, pattern)
        else:
            name_split = split_fname(self.name)
            if mode == "stem":
                match = match_str(name_split[0], pattern)
            elif mode == "ext":
                match = match_str(name_split[1], pattern)
            elif mode == "parts":
                pattern = re.escape(pattern)
                try:
                    match = re.match(
                        rf"[a-zA-Z-_\d.]*{pattern}[a-zA-Z-_\d.]*", self.name
                    ).group(0)
                except AttributeError:
                    match = None
            if match is not None:
                match = self.name
        return match

    def match_namelist(
            self,
            pattern_list: List[str],
            mode: Literal["full", "stem", "ext", "parts"] = "parts",
    ) -> Union[str, None]:
        if mode == "full":
            match = match_list(self.name, pattern_list)
        else:
            name_split = split_fname(self.name)
            if mode == "stem":
                match = match_list(name_split[0], pattern_list)
            elif mode == "ext":
                match = match_list(name_split[1], pattern_list)
            elif mode == "parts":
                # pattern = re.escape(pattern_list)
                pattern = "[a-zA-Z-_\d.]*|[a-zA-Z-_\d.]*".join(pattern_list)
                try:
                    match = re.match(
                        rf"[a-zA-Z-_\d.]*{pattern}[a-zA-Z-_\d.]*", self.name
                    ).group(0)
                except AttributeError:
                    match = None
                except:
                    print("except")
            if match is not None:
                match = self.name
        return match

    def filter_namelist(
            self,
            pattern_list: List[str],
            mode: Literal["full", "stem", "ext", "parts"] = "parts",
    ) -> Union[str, None]:
        if mode == "full":
            match = match_list(self.name, pattern_list)
        else:
            name_split = split_fname(self.name)
            if mode == "stem":
                match = match_list(name_split[0], pattern_list)
            elif mode == "ext":
                match = match_list(name_split[1], pattern_list)
            elif mode == "parts":
                # pattern = re.escape(pattern_list)
                pattern = "[a-zA-Z-_\d.]*|[a-zA-Z-_\d.]*".join(pattern_list)
                match = re.match(rf"[a-zA-Z-_\d.]*{pattern}[a-zA-Z-_\d.]*", self.name)
        if match is not None:
            match = self
        return match

PathBase.register(_Path)

def split_fname(filename: str) -> List[Union[str, None]]:
    name_split = filename.split(sep=".", maxsplit=1)
    if len(name_split) > 2:
        name_split = [".".join(name_split[:-1]), name_split[-1]]
    elif len(name_split) == 1:
        name_split.append(None)
    return name_split


def match_str(string: str, pattern: str) -> Union[str, None]:
    # pattern = re.escape(pattern)
    # print(re.match(pattern, string))
    # if re.match(pattern, string) is not None:
    if string == pattern:
        match = string
    else:
        match = None
    return match


def match_list(string: str, pattern_list: List[str]) -> Union[str, None]:
    pattern = "|".join(pattern_list)
    try:
        match = re.match(rf"{pattern}", string).group(0)
    except AttributeError:
        match = None
    return match


# @add_method(_PosixPath)
# def filter_name(
#     self, pattern: str, mode: Literal["full", "stem", "ext", "parts"] = "full"
# ) -> Union[str, None]:
#     if mode == "full":
#         match = match_str(self.name, pattern)
#     else:
#         name_split = split_fname(self.name)
#         if mode == "stem":
#             match = match_str(name_split[0], pattern)
#         elif mode == "ext":
#             match = match_str(name_split[1], pattern)
#         elif mode == "parts":
#             pattern = re.escape(pattern)
#             match = re.match(rf"[a-zA-Z-_\d.]*{pattern}[a-zA-Z-_\d.]*", self.name)
#         else:
#             raise ValueError(f'{mode!r} is not a valid value for {"mode"!r}')
#     if match is not None:
#         match = self
#     return match


# @add_method(_PosixPath)



# @add_method(_PosixPath)



# @add_method(_PosixPath)


def check_path_glob(
    path_list: list,
    length: Union[int, None],
    check_func: ["is_dir", "is_file", "exists"] = "exists",
) -> NoReturn:
    if length is None or len(path_list) == length:
        if eval(f"path_list[0].{check_func}()"):
            pass
        else:
            raise TypeError(
                f"{path_list[0].name} did not pass type check with {check_func}"
            )
    elif len(path_list) == 0:
        raise IOError(f"No file or directory found.")
    elif len(path_list) > length or len(path_list) < length:
        raise ValueError(
            "Wrong length of file list!\n"
            f"Found {len(path_list)} files:.\n"
            "\n".join(path_list)
        )
    else:
        print("other")


class GROFile(File, _Path):
    def __new__(cls, *args, **kwargs):
        file = super(_Path, File).__new__(cls.check_os(PosixFile), *args, **kwargs)
        return super(File, file).__new__(cls.check_os(GROPosixFile), *args, **kwargs)

class GROPosixFile(PosixFile, GROFile):
    pass

class FFDir(Dir, _Path):
    def __new__(cls, *args, **kwargs):
        file = super(_Path, File).__new__(cls.check_os(PosixDir), *args, **kwargs)
        return super(File, file).__new__(cls.check_os(FFPosixDir), *args, **kwargs)

class FFPosixDir(PosixDir, FFDir):
    pass

class TOPFile(File, _Path):
    def __new__(cls, *args, **kwargs):
        file = super(_Path, File).__new__(cls.check_os(PosixFile), *args, **kwargs)
        return super(File, file).__new__(cls.check_os(TOPPosixFile), *args, **kwargs)

class TOPPosixFile(PosixFile, TOPFile):
    pass

class MDPFile(File, _Path):
    def __new__(cls, *args, **kwargs):
        file = super(_Path, File).__new__(cls.check_os(PosixFile), *args, **kwargs)
        return super(File, file).__new__(cls.check_os(MDPPosixFile), *args, **kwargs)

class TOPPosixFile(PosixFile, MDPFile):
    pass

class File(_Path):
    def __new__(cls, *args, **kwargs):
        new = super().__new__(cls.check_os(PosixFile), *args, **kwargs)
        return new

    def __init__(self, *args, ext="*", check=False):
        if check:
            path_glob = list(self.parent.glob(f"{self.name}{ext}"))
            check_path_glob(path_glob, length=1, check_func="is_file")

    @property
    def parent(self):
        """The logical parent of the path."""
        drv = self._drv
        root = self._root
        parts = self._parts
        # print(drv, root, parts)
        if len(parts) == 1 and (drv or root):
            return Dir(self)
        return Dir(self._from_parsed_parts(drv, root, parts[:-1]))

    @property
    def parents(self):
        return _PathParents(self)

    @cached_property
    def string(self):
        with open(self, "r") as file:
            return file.read()



class PosixFile(_PosixPath, File):
    pass

def invert_str_dict(dict_obj):
    new_dict = {}
    for k, v in dict_obj.items():
        for j in v.keys():
            if j not in new_dict.keys():
                new_dict[j] = {k: dict_obj[k][j]}
                new_dict[j][k] = dict_obj[k][j]
    return new_dict

def get_list_property(obj, prop, list_type=list):
    prop_list = list_type([])
    for item in obj:
        prop_list.append(getattr(item, prop))
    return prop_list

class Dir(_Path):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls.check_os(PosixDir), *args, **kwargs)

    def __init__(self, *args, create=False):
        path_glob = list(self.resolve().parent.glob(f"{self.resolve().name}"))
        if create and not self.is_dir():
            print(f"Creating directory {self.resolve()}")
            os.mkdir(self.resolve())
        else:
            check_path_glob(path_glob, length=1, check_func="is_dir")


    def _get_filelist(self, ext="*"):
        filelist = list(self.glob(rf"*[.]{ext}"))
        if len(filelist) != 0:
            return filelist

    _get_itp_filelist = partialmethod(_get_filelist, "itp")
    _get_gro_filelist = partialmethod(_get_filelist, "gro")
    _get_top_filelist = partialmethod(_get_filelist, "top")
    _get_mdp_filelist = partialmethod(_get_filelist, "mdp")

    def get_filelist(self, ext="*"):
        return PathList(self, ext)

    @property
    def filelist(self) -> PathList:
        return PathList(self, ext="*")

    @property
    def itp_filelist(self) -> ITPList:
        return ITPList(self)

    @property
    def gro_filelist(self) -> GROList:
        return PathList(self, ext="gro")

    @property
    def top_filelist(self) -> TOPList:
        return PathList(self, ext="top")


class PosixDir(_PosixPath, Dir):
    pass


class PathList(UserList):
    def __init__(self, data: Union[PathLike, List[PathLike]], ext="*", check=False):
        if isinstance(data, (list, tuple)):
            self.path = None
            self.data = data
        else:
            self.path = Dir(data)
            self.data = self.path._get_filelist(ext=ext)
            for idx, obj in enumerate(self.data):
                try:
                    self.data[idx] = File(obj, check=check)
                except:
                    self.data[idx] = Dir(obj, create=not check)

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

    def match_name(
        self,
        pattern: str,
        mode: Literal["full", "stem", "ext", "parts"] = "full",
        keep_dims: bool = False,
    ) -> List[Union[str, None]]:
        match = [obj.match_name(pattern, mode) for obj in self.data]
        if not keep_dims:
            match = [obj for obj in match if obj is not None]
        return match

    def match_namelist(
        self,
        pattern_list: List[str],
        mode: Literal["full", "stem", "ext", "parts"] = "full",
        keep_dims: bool = False,
    ) -> List[Union[str, None]]:
        match = [obj.match_namelist(pattern_list, mode) for obj in self.data]
        if not keep_dims:
            match = [obj for obj in match if obj is not None]
        return match

    def filter_name(
        self,
        pattern: str,
        mode: Literal["full", "stem", "ext", "parts"] = "full",
        keep_dims=False,
    ) -> List[Union[str, None]]:
        match = [obj.filter_name(pattern, mode) for obj in self.data]
        if not keep_dims:
            match = [obj for obj in match if obj is not None]
        return match

    def filter_namelist(
        self,
        pattern_list: List[str],
        mode: Literal["full", "stem", "ext", "parts"] = "full",
        keep_dims=False,
    ) -> List[Union[str, None]]:
        match = [obj.filter_namelist(pattern_list, mode) for obj in self.data]
        if not keep_dims:
            match = [obj for obj in match if obj is not None]
        return match

    extract_fstems = partialmethod(_extract_parts, part="stem")
    extract_fnames = partialmethod(_extract_parts, part="name")
    extract_fsuffices = partialmethod(_extract_parts, part="suffix")


class ITPList(PathList):
    def __init__(
        self,
        data: Union[List[Union[str, PathLike]], str, PathLike],
        include: Union[Literal["all", str, List[str]]] = "all",
        exclude: Optional[Union[str, str, List[str]]] = None,
    ):
        super().__init__(data, ext="itp", check=True)
        for idx, itp in enumerate(self.data):
            self.data[idx] = ITPFile(itp)
        self._data = self.data
        self.__attr_list = []
        self._dict = {}
        self._defs = {}
        self.__defs = {}
        self.__alts = {}
        self.include(include)
        self.exclude(exclude)


    @property
    def itp_data(self):
        if len(self._dict) == 0:
            self.itp_data = None
        return self._dict

    @itp_data.setter
    def itp_data(self, itp_dict: Optional):
        if itp_dict is None:
            self.get_data()
        else:
            self._dict = itp_dict

    @property
    def itp_defs(self):
        if len(self._dict) == 0:
            self.itp_data = None
        return self.__defs

    @itp_data.setter
    def itp_defs(self, itp_dict: Optional):
        if itp_dict is None:
            self.get_data()
        else:
            self.__defs = itp_dict

    @property
    def itp_ndefs(self):
        if len(self._dict) == 0:
            self.itp_data = None
        return self.__alts

    @itp_data.setter
    def itp_ndefs(self, itp_dict: Optional):
        if itp_dict is None:
            self.get_data()
        else:
            self.__alts = itp_dict

    @property
    def definitions(self):
        return invert_str_dict(self.__defs)

    @property
    def nondefinitions(self):
        return invert_str_dict(self.__alts)

    @property
    def string(self):
        str = ''
        for kwd in self.kwds:
            try:
                str += f'[ {kwd} ]\n{self.itp_data[kwd]}\n'
            except KeyError:
                print(f'Skipping {kwd}')
        for d in self.definitions:
            str += f'#ifdef {d}\n'
            for k in self.definitions[d]:
                if len(self.nondefinitions[d][k]) != 0:
                    str += f'[ {k} ]\n{self.definitions[d][k]}\n'
            str += '#else\n'
            for k in self.nondefinitions[d]:
                if len(self.nondefinitions[d][k]) != 0:
                    str += f'[ {k} ]\n{self.nondefinitions[d][k]}\n'
            str += '#endif'
        return str


#TODO add #define bonds or other to data
    def get_data(self):
        for itp in self.data:
            print(itp.name, itp.kwds)
            if 'moleculetype' in itp.kwds:
                pass
            else:
                self._dict = merge_dicts(self._dict, itp.data)
                print(self._dict, itp.data)
                self.__defs = merge_dicts(self.__defs, itp.definitions)
                self.__alts = merge_dicts(self.__alts, itp.nondefinitions)
                for kwd in self._dict:
                    print(kwd)
                    self._dict[kwd] = self.delete_duplicate_lines(self._dict[kwd])
                    try:
                        for d in self.__defs[kwd]:
                            print(d)
                            self.__defs[kwd][d] = self.delete_duplicate_lines(self.__defs[kwd][d])
                            self.__alts[kwd][d] = self.delete_duplicate_lines(self.__alts[kwd][d])
                    except KeyError:
                        # print('not key')
                        continue
                    except TypeError:
                        # print('type')
                        continue



    @staticmethod
    def delete_duplicate_lines(check_str):
        checked = []
        check_list = check_str.splitlines()
        for line in check_list:
            if line not in checked:
                checked.append(line)
        return '\n'.join(checked)


    def include(self, include):
        if include == "all":
            self.data = self.data
        elif type(include) == str:
            self.data = self.filter_name(itp_name(include))
        elif type(include) == list:
            include = [itp_name(itp) for itp in include]
            self.data = self.filter_namelist(include)
        else:
            raise TypeError(f"selection must be list or str, got {type(include)!r}")
        self.data.sort()
        self._update_attrs()

    def exclude(self, exclude):
        # print('exclude')
        if exclude is None or len(exclude) == 0:
            return
        if type(exclude) == str:
            # print('str')
            exclude = self.filter_name(
                itp_name(exclude),
            )
            print(exclude)
        elif type(exclude) in [ITPList, list]:
            try:
                exclude = exclude.names
            except AttributeError:
                # print('error')
                exclude = [itp_name(itp) for itp in exclude]
            exclude = self.filter_namelist(exclude)
        else:
            print(exclude, type(exclude))
            raise TypeError(
                f'{"selection"!r} must be {"list"!r} or {"str"!r} type, found{type(exclude)!r}'
            )
        for itp in exclude:
            self.data.remove(itp) # = list(it.dropwhile(lambda itp: itp in exclude, self.data))
        # print(exclude, self.data)
        self.data.sort()
        self._update_attrs()

    def reset_selection(self):
        self.data = self._data

    def sort(self):
        sort_map = ["ffbonded.itp", "ffnonbonded.itp", "forcefield.itp"]
        for file in sort_map:
            result_id = self.filter_namelist([file], mode="full", keep_dims=True)
            idx = np.argwhere(np.array(result_id) != None)
            if len(idx) == 0:
                pass
            else:
                self.data.insert(0, self.data.pop(int(idx)))

    @property
    def kwds(self):
        kwd_list = KwdList(data=[])
        for file in self.data:
            kwd_list.extend(file.kwds)
        kwd_list.sort_kwds(sort_map=list(ITP_KWD.keys()))
        return kwd_list

    def sort_kwds(self):
        return

    def _update_attrs(self):
        for item in self.data:
            if item.stem not in self.__attr_list:
                self.__setattr__(item.stem, item)
                self.__attr_list.append(item.stem)
        for item in self.__attr_list:
            if item not in self.stems:
                self.__delattr__(item)
                self.__attr_list.remove(item)



def _get_name(name_part, ext):
    return f"{split_fname(name_part)[0]}.{ext}"

ff_name = partial(_get_name, ext="ff")
itp_name = partial(_get_name, ext="itp")


def raise_type_error(obj, expected):
    raise TypeError(f"Expected {expected!r} type, found {type(obj)!r}")


def merge_dicts(dict_1: Dict[str, str], dict_2: Dict[str, str], join_str: str = '\n'):
    new_dict = {**dict_1, **dict_2}
    for key, value in new_dict.items():
        if key in dict_1 and dict_2:
            new_dict[key] = f'{dict_1[key]}{join_str}{value}\n'
    return new_dict

class Kwd(str):
    def filter_list(
        self,
        pattern_list: List[str],
        mode: Literal["full", "parts"] = "full",
    ) -> Union[str, None]:
        if mode == "full":
            match = match_list(self, pattern_list)
        elif mode == "parts":
            pattern = "[a-zA-Z-_\d.]*|[a-zA-Z-_\d.]*".join(pattern_list)
            match = re.match(rf"[a-zA-Z-_\d.]*{pattern}[a-zA-Z-_\d.]*", self)
        else:
            raise ValueError(f'{mode!r} is not a valid value for {"mode"!r}')
        if match is not None:
            match = self
        return match

    def filter_str(
        self,
        pattern: str,
        mode: Literal["full", "parts"] = "full",
    ) -> Union[str, None]:
        if mode == "full":
            match = match_str(self, pattern)
        elif mode == "parts":
            pattern = re.escape(pattern)
            match = re.match(rf"[a-zA-Z-_\d.]*{pattern}[a-zA-Z-_\d.]*", self)
        else:
            raise ValueError(f'{mode!r} is not a valid value for {"mode"!r}')
        if match is not None:
            match = self
        return match


class KwdList(UserList):
    def __init__(self, data):
        self.data = []
        for item in data:
            if item not in self.data:
                self.data.append(Kwd(item))

    def match_str(
        self,
        pattern: str,
        mode: Literal["full", "parts"] = "full",
        keep_dims: bool = False,
    ) -> List[Union[str, None]]:
        match = [obj.match_str(pattern, mode) for obj in self.data]
        if not keep_dims:
            match = [obj for obj in match if obj is not None]
        return match

    def match_list(
        self,
        pattern_list: List[str],
        mode: Literal["full", "parts"] = "full",
        keep_dims: bool = False,
    ) -> List[Union[str, None]]:
        match = [obj.match_list(pattern_list, mode) for obj in self.data]
        if not keep_dims:
            match = [obj for obj in match if obj is not None]
        return match

    def sort_kwds(self, sort_map: List[str]):
        # sort_map = ITP_KWD.keys()
        sort_map.reverse()
        for kwds in sort_map:
            # print(kwds)
            result_id = self.filter_list([kwds], mode="full", keep_dims=True)
            idx = np.argwhere(np.array(result_id) != None)
            if len(idx) == 0:
                pass
            else:
                # print("sorting", kwds, idx, idx.shape, len(idx), self.data)
                self.data.insert(0, self.data.pop(int(idx)))
                # print(self.data)
        return self

    def extend(self, other: Iterable[str]) -> None:
        for item in other:
            if item not in self.data:
                self.data.append(item)

    def filter_str(
        self,
        pattern: str,
        mode: Literal["full", "parts"] = "full",
        keep_dims=False,
    ) -> List[Union[str, None]]:
        match = [obj.filter_str(pattern, mode) for obj in self.data]
        if not keep_dims:
            match = [obj for obj in match if obj is not None]
        return match

    def filter_list(
        self,
        pattern_list: List[str],
        mode: Literal["full", "parts"] = "full",
        keep_dims=False,
    ) -> List[Union[str, None]]:
        match = [obj.filter_list(pattern_list, mode) for obj in self.data]
        if not keep_dims:
            match = [obj for obj in match if obj is not None]
        return match

    def concat(self, join='|'):
        return f'{join}'.join(self.data)


class Molecules(UserDict):
    kwd_list = ['moleculetype', 'atoms',
                  'bonds', 'pairs',
                  'pairs_nb', 'angles',
                  'dihedrals', 'exclusions',
                  'constraints', 'settles']
    def __init__(self, name: Optional[str]=None, **kwargs):
        self.data = {}
        if name is not None:
            self.data[name] = {**kwargs}

    def append(self, name, data):
        self.data[name] = data


# class ParamFactory:
#     def __init__(self) #, kwd: str, data: str):
#         self._params = []
# #
#     def get_param_type(self, key, param):
#         self._params[key] = param
#
# #     def create(self, key, **kwargs):
# #         param = self._params.get(key)
# #         if not param:
# #             raise ValueError(key)
# #         return param(**kwargs)
#
#
class ParameterBase(UserDict):
    kwd_list = []
    def __init__(self, kwd, data='', **kwargs):
        print('init')
        super().__init__({kwd: data}, **kwargs)
        self.df = pd.DataFrame.from_dict({kwd: data.split()})


    @cached_property
    @classmethod
    def dtype_dict(cls):
        dtypes = {}
        for k, v in ITP_DTYPE.items():
            if k in cls.kwd_list:
                dtypes[k] = v
        return dict

    @cached_property
    @classmethod
    def kwd_dict(cls):
        kwds = {}
        for k, v in ITP_KWD.items():
            if k in cls.kwd_list:
                kwds[k] = v
        return dict


    @property
    def kwd(self):
        return Kwd(self.data.keys())

    def __getitem__(self, key, df_sel_method=lambda df: df.loc[:]):
        if key in self.kwds:
            return self.data[key].df.df_sel_method

    @cached_property
    def ptype(self):
        return self.__class__





    # def __init__(self, kwd:str, data: str, **kwargs):
    #     self._cls = self._get_prm_type(kwd)
    #     self._cls.get_data(data)
    # @abstractmethod
    # pass


class Definition(UserDict):
    pass

class Parameter(ParameterBase):
    kwd_list = ['defaults', 'atomtypes',
                      'bondtypes', 'pairtypes',
                      'angletypes', 'dihedraltypes',
                      'constrainttypes']

    # def __init__(self, kwd, data='', **kwargs):
    #     super().__init__({kwd: data})
    #     self.df = pd.DataFrame()
    #     # pass
#
# # class MoleculeList(UserList):
# #     def __init__(self, names: List[str]):
#
class System(UserList):
    name_kwd = 'system'
    kwd_list = ['system', 'molecules']


    # def __init__(self, kwd, data='', **kwargs):
    #     super().__init__(kwd, data, **kwargs)
    #     pass

class Molecule(ParameterBase):
    name_kwd = 'moleculetype'
    kwd_list = ['moleculetype', 'atoms', 'bonds', 'pairs', 'angles', 'dihedrals', 'settles', 'exclusions']

    # def __init__(self, kwd, data='', **kwargs):
    #     print(super().__class__)
    #     super().__init__({kwd: data})
    #     self.string = 'a'
    #     print(self.__bases__)

class ParameterListBase(UserList):
    item_type = None

    def __init__(self, data: List[item_type]=[]):
        self.data = []
        for item in data:
            if item.kwd not in self.kwds:
                self.data.append(item)

    @property
    def kwds(self):
       return KwdList(self.data)


class Molecules:
    item_type = Molecule
    pass

class Parameters:
    item_type = Parameter
    pass

#
#     @property
#     @abstractmethod
#     def kwds(self):
#         pass
#
#     @staticmethod
#     def _get_prm_type(kwd):
#         if kwd in Parameter.kwds:
#             return Parameter(kwd)
#         elif kwd in Molecule.kwds:
#             return Molecule(kwd)
#         elif kwd in Definition.kwds:
#             return Definition(kwd)
#
#
#
#
# class SubParameter:
#     def __init__(self, name: str, data: str):
#         self.name


class TopologyParameterFactory: #Factory(ABC):
    _prm_types = {Parameter, Molecule, System}

    def __new__(cls, kwd, data='', **kwargs):
        print('new', kwd, **kwargs)
        _cls = cls.get_type(kwd)
        print(_cls)
        self = _cls(kwd, data, **kwargs)
        return self

    @classmethod
    def get_type(cls, kwd):
        for prm_type in cls._prm_types:
            if kwd in prm_type.kwd_list:
                print(kwd, 'found')
                break
            else:
                prm_type = None
        if prm_type is None:
            raise KeyError(f'{kwd!r} is not a valid itp parameter.')
        return prm_type





class TopologyParameterListFactory:
    _prm_types = TopologyParameterFactory._prm_types

    def __new__(cls, kwd, *args, **kwargs):
        cls = cls._get_prm_type(kwd)
        self = cls(kwd, *args, **kwargs)
        return self

    @classmethod
    def _get_prm_type(cls, kwd):
        for prm_type in cls._prm_types:
            if kwd in prm_type.kwd_list:
                return prm_type








#

#     __slots__ = ['name', 'ifval', 'elseval', 'ifdef', 'ifndef']
#
#     def __init__(self, name, ifval='', elseval=''):
#         self.name = name
#         self.ifval = ifval
#         self.elseval = elseval
#         self.data = {self.name: [self.ifval, self.elseval]}
#
#
# class Parameter:
#     def __init__(self, name: str, data, definitions: List[Definition] = []
#                  ):
#         self.name = name
#         self.data = data
#         for d in definitions:
#             assert type(d) == Definition
#         self.definitions = definitions
#
#     @property
#     def ifdef(self):
#         return self.ifval, self.elseval
#
#     @property
#     def ifndef(self):
#         return self.elseval, self.ifval
#
#     @ifdef.setter
#     def ifdef(self, ifval, elseval):
#         self.ifval = ifval
#         self.elseval = elseval
#
#     @ifndef.setter
#     def ifndef(self, ifval, elseval):
#         self.ifval = elseval
#         self.elseval = ifval

class ITPFile(File, _Path):
    def __new__(cls, *args, **kwargs):
        # print("new")
        file = super(_Path, File).__new__(cls.check_os(PosixFile), *args, **kwargs)
        return super(File, file).__new__(cls.check_os(ITPPosixFile), *args, **kwargs)

    def __init__(self, *args, ext="*", check=False):
        # print("init", type(self))
        super().__init__(self, *args, ext="itp")
        print(1)
        self.ff = self.parent.name
        print(1)
        self._prms = Parameters()
        print(1)
        self._mols = Molecules()
        print(1)
        self._sys = System()
        print(1)
        self.__kwds = []
        # self._prms = {}
        # self._dict = {}
        # self._defs = {}
        # self.__defs = {}
        # self.__alts = {}
        # self._mols = {}
        # self._sys = {}
        # self._add_attrs()
        # self.molecules = MoleculeList()
        # self.__atomtypes = pd.DataFrame(columns=[*self.__class__.kwd_dict["atomtypes"]])
        # self.__atomtypes = self.__atomtypes.astype(
        #     dtype=self._get_dtypes("atomtypes", [*self.__class__.kwd_dict["atomtypes"]])
        # )
        # self.__moleculetypes = pd.DataFrame(columns=[*self.__class__.kwd_dict["atoms"]])
        # self.__moleculetypes = self.__moleculetypes.astype(
        #     dtype=self._get_dtypes("atoms", [*self.__class__.kwd_dict["atoms"]])
        # )
        # print(self.__atomtypes)

    @classmethod
    def _get_dtypes(cls, key: str, kwd_list: list) -> dict:
        dtypes = list(map(lambda kwd: cls.dtype_dict[kwd], cls.kwd_dict[key]))
        return dict(zip(kwd_list, dtypes))

    # @classmethod
    # def __convert_df_dict_dtypes(cls, df_dict: dfDict) -> None:
    #     for key in df_dict.keys():
    #         #     try:
    #         df_dict[key] = df_dict[key].astype(cls._get_dtypes(key, cls.kwd_dict[key]))

    # def parse(self):
    #     with open(self, 'r') as itp_file:
    #         itp_str = itp_file.read()
    #     file_iter = iter(file.splitlines())


    @property
    def kwds(self):
        if len(self.__kwds) == 0:
            self.kwds = None
        return self.__kwds

    @kwds.setter
    def kwds(self, kwd_list: List[str] = None):
        # print("setting kwds")
        if kwd_list is None:
            kwd_list = []
        self.__kwds.extend(kwd_list)
        kwds = re.findall(
            r"(?<=\[ )[a-zA-Z_]+(?= \])", self.string, flags=re.MULTILINE | re.DOTALL
        )
        self.__kwds.extend(kwds)
        # for kwd in kwds:
        #     if kwd not in self.__kwds and kwd != "defaults":
        #         self.__kwds.append(kwd)
        self.__kwds = KwdList(self.__kwds)
        self.__kwds.sort_kwds(list(ITP_KWD.keys()))

    @property
    def data(self):
        if len(self._dict) == 0:
            self.data = None
        return self._dict

    @data.setter
    def data(self, data_dict):
        if data_dict is not None:
            self._dict = data_dict
        self.parse_itp()

    @cached_property
    def definitions(self):
        if len(self._dict) == 0:
            self.data = None
        return self.__defs

    @property
    def nondefinitions(self):
        if len(self._dict) == 0:
            self.data = None
        return self.__alts



    # def __select_data_dict(self, define: Literal[0,1,2], kwd: str, definition: Optional[str] =None):
    #     if define == 0:
    #         return self._dict[kwd]
    #     elif define == 1:
    #         return self.__defs[kwd][definition]
    #     elif define == 2:
    #         return self.__alts[kwd][definition]
    #     else:
    #         raise ValueError(f'{"define"!r} must have values 0, 1 or 2')


    def parse_itp(self):
        prm_dict = {Parameter: self._prms,#{0: self._dict[0], 1: self.__defs[1], 2: self.__alts[2]},
                 Molecule: self._mols,#{0: self._mols, 1: self._mols.defs, 2: self._mols.alts},
                 System: self._sys}#{0: self._sys}}
        sections = self.string.split(sep="[ ")
        kwds = self.kwds.concat()
        # kwds = '|'.join(self.kwds)
        define = 0
        definition = None
        prm_dict = prm_dict[Parameter]
        for section in sections:
            # print(section[:20], self.kwds)
            match = re.search(
                rf"\s*({kwds}) \]\s*\n(.*)", section, flags=re.MULTILINE | re.DOTALL
            )
            # print(match)
            if match is None:
                pass
                # print("no match", section[:10])
            else:
                kwd = match.group(1)
                print(kwd)
                prm = TopologyParameterFactory(kwd)
                print(prm)
        #         if kwd in Params.param_list:
        #             prm_dict = dicts['prms']
        #         elif kwd in Molecules.param_list:
        #             prm_dict = dicts['mols']
        #         elif kwd in System.param_list:
        #             prm_dict = dicts['sys']
        #         else:
        #             raise KeyError(f'Unknown parameter {kwd!r}')
        #
        #         if kwd not in prm_dict.keys():
        #             prm_dict[kwd] = TopologyParameterFactory(name=kwd)
        #
        #         # print("kwd", kwd, define, definition)
        #         # if kwd not in self._dict.keys():
        #         #     print("new kwd", kwd)
        #         #     self._dict[kwd] = {}
        #         # if None not in self._dict[kwd].keys():
        #         #     self._dict[kwd][None] = "" #f'; {"    ".join(ITP_KWD[kwd])}'
        #         # if kwd not in self.__defs.keys():
        #         #     print("set defs t")
        #         #     self.__defs[kwd] = {}
        #         # if kwd not in self.__alts.keys():
        #         #     print("set alts t")
        #         #     self.__alts[kwd] = {}
        #         match = match.group(2)
        #         if define in [1, 2]:
        #             if definition not in self.__alts[kwd].keys():
        #                 print("set alts t", kwd)
        #                 print(definition)
        #                 self.__alts[kwd][definition] = ""  # f'#else\n'
        #         if definition not in self.__defs[kwd].keys():
        #             print("set defs t", definition)
        #             self.__defs[kwd][definition] = ""  # f'#ifdef {definition}\n'
        #         for line in match.splitlines():
        #             if re.match(r"^;", line.strip()) is not None:
        #                 pass
        #             else:
        #                 if re.search(r"^\s*[A-Za-z\d\s]+\s\d{1,3}\s*", line.strip()) is not None:
        #                     line_match = re.search(
        #                         r"^\s*[A-Za-z\d\s]+\s\d{1,3}\s*", line.strip()
        #                     ).group(0)
        #                     if kwd == 'moleculetype':
        #                         print(line_match, kwd)
        #                     # print(re.search(r"^[A-Za-z\d\s]+\s\d{1,3}\s", line.strip()))
        #                     if (
        #                         re.search(
        #                             rf"{line_match}",
        #                             dicts[define][kwd][definition],
        #                             flags=re.MULTILINE | re.DOTALL,
        #                         )
        #                         is None
        #                     ):
        #                         dicts[define][kwd][definition] += "\n" + line.strip()
        #                     else:
        #                         pass
        #                 elif re.match(r"[#]\s*ifdef\s+[a-zA-Z_\d]+", line) is not None:
        #                     definition = re.match(r"[#]\s*ifdef\s+([a-zA-Z_\d]+)", line).group(
        #                         1
        #                     )
        #                     print("ifdef")
        #                     define = 1
        #                 elif re.match(r"[#]\s*ifndef\s+[a-zA-Z_\d]+", line) is not None:
        #                     print("ifndef")
        #                     definition = re.match(
        #                         r"[#]\s*ifndef\s+([a-zA-Z_\d]+)", line
        #                     ).group(1)
        #                     define = 2
        #                 elif re.match(r"[#]\s*else", line) is not None:
        #                     print("else", define)
        #                     if define == 1:
        #                         define = 2
        #                     elif define == 2:
        #                         define = 1
        #                 elif re.match(r"[#]\s*endif", line) is not None:
        #                     define = 0
        #                     definition = None
        #                 else:
        #                     print(line)
        #
        #
        # for kwd in self._dict:
        #     self._dict[kwd] = self._dict[kwd][None]
        # remove_list = []
        # for kwd in self.__defs:
        #     print(kwd)
        #     for d, dd in zip(self.__defs[kwd], self.__alts[kwd]):
        #         print(d, dd)
        #         if d is None:
        #             if dd is None:
        #                 remove_list.append(kwd)
        # for item in remove_list:
        #     # print(item)
        #     self.__defs.pop(item)
        #     self.__alts.pop(item)
    # @jit(nopython=False, )

    # def parse_itp(self):
    #     dicts = {'prms': {},#{0: self._dict[0], 1: self.__defs[1], 2: self.__alts[2]},
    #              'mols': {},#{0: self._mols, 1: self._mols.defs, 2: self._mols.alts},
    #              'sys': {}}#{0: self._sys}}
    #     sections = self.string.split(sep="[ ")
    #     kwds = self.kwds.concat()
    #     # kwds = '|'.join(self.kwds)
    #     define = 0
    #     definition = None
    #     prm_dict = dicts['prms']
    #     for section in sections:
    #         print(section[:20], self.kwds)
    #         match = re.search(
    #             rf"\s*({kwds}) \]\s*\n(.*)", section, flags=re.MULTILINE | re.DOTALL
    #         )
    #         print(match)
    #         if match is None:
    #             pass
    #             # print("no match", section[:10])
    #         else:
    #             kwd = match.group(1)
    #             print("kwd", kwd, define, definition)
    #             if kwd not in self._dict.keys():
    #                 print("new kwd", kwd)
    #                 self._dict[kwd] = {}
    #             if None not in self._dict[kwd].keys():
    #                 self._dict[kwd][None] = "" #f'; {"    ".join(ITP_KWD[kwd])}'
    #             if kwd not in self.__defs.keys():
    #                 print("set defs t")
    #                 self.__defs[kwd] = {}
    #             if kwd not in self.__alts.keys():
    #                 print("set alts t")
    #                 self.__alts[kwd] = {}
    #             match = match.group(2)
    #             if define in [1, 2]:
    #                 if definition not in self.__alts[kwd].keys():
    #                     print("set alts t", kwd)
    #                     print(definition)
    #                     self.__alts[kwd][definition] = ""  # f'#else\n'
    #             if definition not in self.__defs[kwd].keys():
    #                 print("set defs t", definition)
    #                 self.__defs[kwd][definition] = ""  # f'#ifdef {definition}\n'
    #             for line in match.splitlines():
    #                 if re.match(r"^;", line.strip()) is not None:
    #                     pass
    #                 else:
    #                     if re.search(r"^\s*[A-Za-z\d\s]+\s\d{1,3}\s*", line.strip()) is not None:
    #                         line_match = re.search(
    #                             r"^\s*[A-Za-z\d\s]+\s\d{1,3}\s*", line.strip()
    #                         ).group(0)
    #                         if kwd == 'moleculetype':
    #                             print(line_match, kwd)
    #                         # print(re.search(r"^[A-Za-z\d\s]+\s\d{1,3}\s", line.strip()))
    #                         if (
    #                             re.search(
    #                                 rf"{line_match}",
    #                                 dicts[define][kwd][definition],
    #                                 flags=re.MULTILINE | re.DOTALL,
    #                             )
    #                             is None
    #                         ):
    #                             dicts[define][kwd][definition] += "\n" + line.strip()
    #                         else:
    #                             pass
    #                     elif re.match(r"[#]\s*ifdef\s+[a-zA-Z_\d]+", line) is not None:
    #                         definition = re.match(r"[#]\s*ifdef\s+([a-zA-Z_\d]+)", line).group(
    #                             1
    #                         )
    #                         print("ifdef")
    #                         define = 1
    #                     elif re.match(r"[#]\s*ifndef\s+[a-zA-Z_\d]+", line) is not None:
    #                         print("ifndef")
    #                         definition = re.match(
    #                             r"[#]\s*ifndef\s+([a-zA-Z_\d]+)", line
    #                         ).group(1)
    #                         define = 2
    #                     elif re.match(r"[#]\s*else", line) is not None:
    #                         print("else", define)
    #                         if define == 1:
    #                             define = 2
    #                         elif define == 2:
    #                             define = 1
    #                     elif re.match(r"[#]\s*endif", line) is not None:
    #                         define = 0
    #                         definition = None
    #                     else:
    #                         print(line)
    #
    #
    #     for kwd in self._dict:
    #         self._dict[kwd] = self._dict[kwd][None]
    #     remove_list = []
    #     for kwd in self.__defs:
    #         print(kwd)
    #         for d, dd in zip(self.__defs[kwd], self.__alts[kwd]):
    #             print(d, dd)
    #             if d is None:
    #                 if dd is None:
    #                     remove_list.append(kwd)
    #     for item in remove_list:
    #         # print(item)
    #         self.__defs.pop(item)
    #         self.__alts.pop(item)

    @cached_property
    def sorted_str(self):
        str = ""
        for kwd in self.data:
            str += f"\n[ {kwd} ]\n; {'    '.join(ITP_KWD[kwd])}{self.data[kwd]}\n"
            if kwd in self.__defs.keys():
                for definition in self.__defs[kwd]:
                    str += f"\n#ifdef {definition}\n"
                    if self.__defs[kwd][definition] != "":
                        print(self.__defs[kwd][definition])
                        str += f"[ {kwd} ]\n; {'    '.join(ITP_KWD[kwd])}{self.__defs[kwd][definition]}\n"
                    if definition in self.__alts[kwd].keys():
                        print(self.__alts[kwd][definition])
                        str += f"#else\n"
                        if self.__alts[kwd][definition] != "":
                            str += f"[ {kwd} ]\n; {'    '.join(ITP_KWD[kwd])}\n{self.__alts[kwd][definition]}\n"
                    str += "#endif\n"
        return str

    def _add_attrs(self):
        if len(self.data) == 0:
            self.data = None
        # kwd_sel = self.kwds.filter_list(['atomtypes', 'moleculetype', 'molecules'])
        # print(kwd_sel)
        kwd_sel = self.kwds
        for kwd in kwd_sel:
            print('setting', kwd)
            self.__setattr__(kwd, self.get_df(kwd))


    def get_df(self, kwd):
        try:
            df = pd.read_csv(StringIO(self.data[kwd]+'\n'+'\n'.join([*self.__alts[kwd].values()])),
                             sep='\s+',
                                          comment=';', header=None)
        except EmptyDataError:
            df = pd.DataFrame()
        print(df, len(df.columns))
        df.columns = [*self.kwd_dict[kwd][:len(df.columns)]]

        print(df)
        df = df.astype(self._get_dtypes(kwd, [*self.kwd_dict[kwd]][:len(df.columns)]))
        return df
    #     # print(kwd)
    #
    # @staticmethod
    # def __append_line(line, line_match, kwd, data_dict):
    #     if (
    #             re.search(
    #                 rf"{line_match}",
    #                 data_dict[kwd],
    #                 flags=re.MULTILINE | re.DOTALL,
    #             )
    #             is None
    #     ):
    #         data_dict[kwd] += "\n" + line
    #         # print(line_match)
    #     else:
    #         # print('exists')
    #         pass


class ITPPosixFile(PosixFile, ITPFile):
    pass


class ClayType(UserDict):
    sheet_dict = {"T": ["at", "st", "fet"], "O": ["ao", "fe2", "feo", "mgo", "lio"]}
    sheet_dict = {"T": r"[a-z]+t", "O": r"[a-z]+[o2]", "C": "charge"}

    def __init__(self, path):
        self.path = Dir(path)
        self.data = {}
        self.itp_files = ITPList(self.path)
        self.gro_files = GROList(self.path)
        # for uc in ITPList.stems:
        #     uc_name = re.match(r'([A-Za-z\d]+?)([\d]{2}).*]')
        #     print(uc_name)

    @property
    def name(self):
        return self.path.name


class UnitCell(UserDict):
    pass


class FF(UserList):
    def __init__(
        self,
        path: Dir,
        itp_list: ITPList,
        assign: Optional[Union["clay", "ions", "protein", "solvent", "other"]] = None,
    ):
        self.path = Dir(path)
        self.name = self.path.name
        self.stem = self.path.stem
        self.data = itp_list
        self.__type = 'other'
        # self.selection = ForceField.selection
        # self.available = ForceField.available

    @property
    def assign(self):
        return self.__type

    @assign.setter
    def assign(self, value):
        if value in ["clay", "ions", "protein", "solvent"]:
            self.__type = value
        else:
            self.__type = 'other'

    @cached_property
    def available(self):
        return self.path.itp_filelist

    @property
    def selection(self):
        return self.data.names

    @property
    def itp_names(self):
        return self.data.names

    @property
    def itp_files(self):
        return self.data

    def string(self):
        return self.data.string



class ForceField(UserDict):
    def __init__(
        self,
        include: Union[
            Literal["all", str, List[str], Dict[str, Union[str, List[str]]]]
        ] = "all",
        exclude: Optional[
            Union[str, List[str], Dict[str, Union[str, List[str]]]]
        ] = None,
        path=None,
    ):
        self.data = {}
        self.path = Dir(path, create=False)
        # self.__available = {}
        # for ff in self.path.get_filelist(ext='ff'):
        #     self.__available[Dir(ff)] = ITPList(data=ff)
        self.__selection = []
        self.__attr_list = []
        self.select_ff(include, exclude)
        self.__dict = {}
        self._kwd_dict = {}

    @cached_property
    def available(self):
        avail_dict = {}
        for ff in self.path.get_filelist(ext="ff"):
            avail_dict[Dir(ff)] = ITPList(data=ff)
        return avail_dict

    @property
    def kwds(self):
        for ff in self.__selection:
            if ff.name not in self._kwd_dict:
                self._kwd_dict[ff.name] = []
            for itp in ff.itp_files:
                print(itp.name, self._kwd_dict[ff.name])
                self._kwd_dict[ff.name].extend(itp.kwds)
            self._kwd_dict[ff.name] = KwdList(np.unique(self._kwd_dict[ff.name]))
        return self._kwd_dict

    def parse(self):
        print(self.__selection)
        for ff in self.__selection:
            print(ff.path)
            for itp in ff.itp_files.kwds:
                self.__dict[itp] = ff.itp_files
            #     for kwd in itp.kwds:
            #         print(kwd)
            #         if kwd not in self.__dict:
            #             self.__dict[kwd] = pd.DataFrame(columns=[*ITP_KWD[kwd]])
            #         self.__dict[kwd] = pd.concat(self.__dict[kwd], pd.DataFrame.from_dict(itp.data[kwd]))

    @property
    def selection(self):
        selection = {}
        if self.data != {}:
            for key in self.data:
                selection[key.name] = self.data[key].names
            return selection
            print(selection)
        else:
            raise KeyError("No force field was selected.")

    def read(self):
        if self.__selection != {}:
            for ff in self.__selection:
                pass

    def ff_path(self, ff):
        return self.path / ff_name(ff)

    # @staticmethod
    # def check_sel(path, sel):
    #     if type(sel) in (list, np.array):
    #         print(type(sel))
    #         if len(sel) == 1:
    #             sel = sel[0]
    #         else:
    #             checked = []
    #             for file in sel:
    #                 print(path, file)
    #                 check = list(path.parent.glob(rf'{file}*'))
    #                 print(check)
    #             if np.any(checked) is None:
    #                 not_found = np.array(checked is None)
    #                 not_found = sel[~not_found]
    #                 print(not_found)
    #     if type(sel) in (str, PosixDir):
    #         print(type(sel), sel, path)
    #         checked = PathList(path).filter_name(sel, mode='parts')[0]
    #         print('checked', checked)
    #         if checked is None:
    #             raise FileNotFoundError(f'{sel!r} could not be found')
    #     return checked

    # @cached_property
    # def available(self):
    #     return self.__available

    def _get_selection(self, results, selection):
        if selection is None:
            return results
        elif isinstance(selection, (list, np.ndarray)):
            if len(selection) == 1:
                selection = selection[0]
            else:
                for ff in selection:
                    if not isinstance(ff, str):
                        raise_type_error(ff, str)
                    ff = self.ff_path(ff)
                    results[ff] = ITPList(ff)
                return results
        if isinstance(selection, str):
            if selection == "all":
                results = self.available
            elif selection == None:
                return results
            else:
                selection = self.ff_path(selection)
                results[selection] = ITPList(selection)
        elif isinstance(selection, dict):
            for ff in selection:
                if not isinstance(ff, str):
                    raise_type_error(ff, str)
                itp_files = selection[ff]
                ff_path = self.ff_path(ff)
                results[ff_path] = ITPList(ff_path, itp_files)
        else:
            raise_type_error(selection, str)
        return results

    def select_ff(self, include="all", exclude=None):
        if self.data == {}:
            include_dict = {}
        else:
            include_dict = self.data
        exclude_dict = {}
        include_dict = self._get_selection(include_dict, include)
        if exclude is not None:
            exclude_dict = self._get_selection(exclude_dict, exclude)
        ff_sel = []
        for ff in include_dict:
            # print(ff)
            self.data[ff] = include_dict[ff]
            print(self.data[ff])
            try:
                # print(exclude_dict)
                self.data[ff].exclude(exclude_dict[ff])
            except KeyError:
                pass
            if len(self.data[ff]) == 0 or self.data[ff] == None:
                del self.data[ff]
            else:
                ff_sel.append(FF(ff, self.data[ff]))
        self.__selection = ff_sel
        self._update_attrs()
        print(ff_sel)

    @property
    def list(self):
        return get_list_property(self.__selection, 'stem', list)

    @property
    def name_list(self):
        return get_list_property(self.__selection, 'name', list)

    @property
    def path_list(self):
        return get_list_property(self.__selection, 'path', PathList)

    def _update_attrs(self):
        for item in self.__attr_list:
            self.__delattr__(item)
        for item in self.__selection:
            self.__setattr__(item.stem, item)
            self.__attr_list.append(item.stem)






class ClayComposition:
    def __init__(
        self, csv_fname: Union[str, PathLike], sysname: str, uc_type: str, n_cells
    ):
        pass

    # def _select_ff(self, results: Dict[PosixFile, Union[List[PosixFile], PosixFile]],
    #                               selection: Union[
    #         str, os.PathLike,
    #         None,
    #         List[Union[str, os.PathLike]],
    #         Dict[Union[str, os.PathLike], Union[str, os.PathLike]],
    #     ]):
    #     # Include all available FF
    #     if isinstance(selection, (str, os.PathLike)):
    #         # print("Including str")
    #         if selection is None:
    #             # print("None")
    #             return sel_dict
    #         if selection == "all":
    #             print(f'Including all force fields in "{self.path.name}".')
    #             selection = self.available
    #         # selection all ".itp" files for 1 selected FF
    #         else:  # if isinstance(selection, (str, os.PathLike)):
    #             print(f"Including {selection}.")
    #             selection = self.match_ff_pathname(selection)
    #             sel_dict[selection.stem] = "all"
    #         # Include all available ".itp" files for a lift of available FF
    #     elif isinstance(selection, (list, np.ndarray)):
    #         if len(selection) == 1:
    #             ff_incl = self.match_ff_pathname(selection[0])
    #             sel_dict[ff_incl.stem] = "all"
    #         elif len(selection) > 1:
    #             print("Including list:")
    #             for ff_incl in selection:
    #                 print(f"{ff_incl}")
    #                 ff_incl = self.match_ff_pathname(ff_incl)
    #                 sel_dict[ff_incl.stem] = "all"
    #         else:
    #             raise ValueError("Force field selection list cannot be empty")
    #     elif isinstance(selection, dict):
    #         print("Including dict:\n" f"{selection, type(selection)}")
    #         # sel_dict = {}
    #         for ff_incl in selection.keys():
    #             print(f"{ff_incl} available: {self.available}")
    #             avail_str = "".join(
    #                 np.unique(extract_fname(self.available, stem=True))
    #             )
    #             # print(f"available FF: {avail_str}")
    #             sel_dict = selection
    #             try:
    #                 # print(sel_dict, ff_incl, avail_str)
    #                 sel_dict = self.match_dict_keys(sel_dict, ff_incl, avail_str)
    #             except AttributeError:
    #                 self.print_ff_not_found(ff_incl)
    #                 break
    #     elif selection == None:
    #         pass
    #     else:
    #         raise TypeError(
    #             "Force field argument must be supplied as str, list or dict"
    #         )
    #     if len(sel_dict) != 0:
    #         # check itp files of selected FF
    #         for ff_incl in sel_dict.keys():
    #             print(f"Checking itp files: {sel_dict[ff_incl]}")
    #             ff_incl = extract_fname(ff_incl, stem=True)
    #             # print(self.data)
    #             avail_fnames = extract_fname(
    #                 self._get_filelist(path=self.path / f"{ff_incl}.ff", ext="itp")
    #             )
    #             # print(f"available fnames: {avail_fnames}")
    #             avail_stems = extract_fname(
    #                 self._get_filelist(path=self.path / f"{ff_incl}.ff", ext="itp"),
    #                 stem=True,
    #             )
    #             # print(f'All available itp files: {avail_fnames}')
    #             if isinstance(sel_dict[ff_incl], (str, pl.Path)):
    #                 sel_dict[ff_incl] = [sel_dict[ff_incl]]
    #             else:
    #                 sel_dict[ff_incl] = list(sel_dict[ff_incl])
    #             # print(len(sel_dict[ff_incl]), type(sel_dict[ff_incl]), [*sel_dict[ff_incl]])
    #             if sel_dict[ff_incl][0] == "all":
    #                 # print(f'list of len 1: {sel_dict[ff_incl]}')
    #                 sel_dict[ff_incl] = avail_stems
    #             sel_dict[ff_incl] = self.match_fname_list(
    #                 sel_dict[ff_incl], avail_fnames, ext=".itp"
    #             )
    #     return sel_dict

class PathFactoryBase(_Path):
    _file_types = {}
    _default = _Path
    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls.check_os(PosixFile), *args, **kwargs)
        _cls = cls._get_path_subclass(self)
        return _cls(self.resolve(), *args, **kwargs)

    @classmethod
    def _get_path_subclass(cls, obj):
        if obj.suffix in cls._file_types:
            _cls = cls._file_types[obj.suffix]
        else:
            _cls = cls._default
        return _cls

class DirFactory(PathFactoryBase, ABC):
    _file_types = {'.ff': FFDir}
    _default = Dir


class FileFactory(PathFactoryBase, ABC):
    _file_types = {'.itp': ITPFile, '.gro': GROFile, '.top': TOPFile, '.mdp': MDPFile}
    _default = File


class PathFactory(_Path):
    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls.check_os(PosixFile), *args, **kwargs)
        if self.is_dir():
            print('is dir')
            self = DirFactory(self.resolve(), *args, **kwargs)
        elif self.is_file():
            self = FileFactory(self.resolve(), *args, **kwargs)
        else:
            self = _Path(self.resolve(), *args, **kwargs)
        return self
