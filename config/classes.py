import re
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
    Generic, Type,
)

from itertools import filterfalse

from numba import jit

from config.lib import convert_to_list
from attrs import validators, define


from typing_extensions import TypeAlias, ParamSpec, ParamSpecKwargs, ParamSpecArgs
from collections import UserList
from functools import partialmethod, singledispatchmethod, singledispatch, wraps, partial
import numpy as np
import pandas as pd
from collections.abc import Sequence
import itertools as it
import ast

File = NewType('File', _PosixPath)
Dir = NewType('Dir', _PosixPath)
PosixFile = NewType('File', _PosixPath)
PosixDir = NewType('Dir', _PosixPath)
FileList = NewType('FileList', List[PosixFile])
ITPList = NewType('ITPList', FileList)
GROList = NewType('GROList', FileList)
TOPList = NewType('TOPList', FileList)
PathLike: TypeAlias = Union[File, Dir, PosixFile, PosixDir, _PosixPath, _Path]


# class decorators

def add_method(cls):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)
        setattr(cls, func.__name__, wrapper)
    return decorator


def add_property(cls):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)
        setattr(cls, func.__name__, wrapper)
    return decorator

# path object functions

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

# @add_method(_PosixPath)
# def match_name(
#         self, full=False, stem=False, ext=False, parts=False
# ) -> Union[str, None]:
#     """
#     Match filenames by full name, stem or suffix.
#     :param self: pathlib file object
#     :type self: pl._PosixPath
#     :param full: full name
#     :type full: Union[Literal[False], str]
#     :param stem: filename stem
#     :type stem: Union[Literal[False], str]
#     :param ext: filename suffix
#     :type ext: Union[Literal[False], str]
#     :param parts:
#     :type parts:
#     :return:
#     :rtype:
#     """
#     if full:
#         full = re.escape(full)
#         name_split = re.split(".", full, maxsplit=1)
#         fname = [".".join(name_split[:-1]), name_split[-1]]
#         # print(stem, ext)
#     else:
#         fname = [stem, ext]
#         for part in range(2):
#             if fname[part]:
#                 fname[part] = re.escape(fname[part])
#             else:
#                 fname[part] = "[a-zA-Z-_0-9]*"
#         # stem, ext = map(lambda fstr: re.escape(fstr), (stem, ext))
#         # print(fname, stem, ext)
#     if parts:
#         # print("parts")
#         parts = "[a-zA-Z-_0-9]*"
#     else:
#         parts = ""
#     # print(fname, parts, rf"({fname[0]}){parts}[.]{fname[1]}")
#     try:
#         name_match = re.search(rf"({fname[0]}){parts}[.]*{fname[1]}", self.name).group(
#             0
#         )
#         # print(name_match, "match")
#         return name_match
#     except AttributeError:
#         return


def split_fname(filename: str) -> List[Union[str, None]]:
    name_split = filename.split(sep='.', maxsplit=1)
    print(name_split)
    if len(name_split) > 2:
        name_split = [".".join(name_split[:-1]), name_split[-1]]
    elif len(name_split) == 1:
        name_split.append(None)
    return name_split


def match_str(string: str, pattern: str) -> Union[str, None]:
    if string == pattern:
        match = string
    else:
        match = None
    return match


def match_list(string: str, pattern_list: List[str]) -> Union[str, None]:
    pattern = '|'.join(pattern_list)
    try:
        match = re.match(rf'{pattern}', string).group(0)
    except AttributeError:
        match = None
    return match

@add_method(_PosixPath)
def match_pattern(self,
               pattern: str,
               mode: Literal['full', 'stem', 'ext', 'parts'] = 'parts') -> Union[str, None]:
    if mode == 'full':
        match = match_str(self.name, pattern)
    else:
        name_split = split_fname(self.name)
        if mode == 'stem':
            match = match_str(name_split[0], pattern)
        elif mode == 'ext':
            match = match_str(name_split[1], pattern)
        elif mode == 'parts':
            pattern = re.escape(pattern)
            try:
                match = re.match(rf'[a-zA-Z-_\d.]*{pattern}[a-zA-Z-_\d.]*', self.name).group(0)
            except AttributeError:
                match = None
        if match is not None:
            match = self.name
    return match


@add_method(_PosixPath)
def match_namelist(self,
                   pattern_list: List[str],
               mode: Literal['full', 'stem', 'ext', 'parts'] = 'parts') -> Union[str, None]:
    if mode == 'full':
        match = match_list(self.name, pattern_list)
    else:
        name_split = split_fname(self.name)
        if mode == 'stem':
            match = match_list(name_split[0], pattern_list)
        elif mode == 'ext':
            match = match_list(name_split[1], pattern_list)
        elif mode == 'parts':
            # pattern = re.escape(pattern_list)
            pattern = '[a-zA-Z-_\d.]*|[a-zA-Z-_\d.]*'.join(pattern_list)
            try:
                match = re.match(rf'[a-zA-Z-_\d.]*{pattern}[a-zA-Z-_\d.]*', self.name).group(0)
            except AttributeError:
                match = None
        if match is not None:
            match = self.name
    return match


# def match_list_pattern(f):
#     @wraps(f)


#
# @match_pattern
# @add_method(_PosixPath)
# def match_name(self,
#                pattern: str,
#                mode: Literal['full', 'stem', 'ext', 'parts'] = 'parts') -> Union[str, None]:
#     return pattern
#
#
# @match_pattern
# @add_method(_PosixPath)
# def match_namelist(self,
#                    pattern: List[str],
#                    mode: Literal['full', 'stem', 'ext', 'parts'] = 'parts') -> Union[str, None]:
#     return '|'.join(pattern)

# @add_method(FileList)
# def _match_list_pattern(self, match_f, pattern, mode='full', keep_dims=False):
#     match = [obj.f(pattern, mode) for obj in self.data]
#     if not keep_dims:
#         match = [obj for obj in match if obj is not None]
#     return match
#
# match_names = singledispatchmethod(_match_list_pattern, match_f=match_name)


# @add_method(_PosixPath)
# def match_name(self,
#                pattern: str,
#                mode: Literal['full', 'stem', 'ext', 'parts'] = 'parts') -> Union[str, None]:
#     if mode == 'full':
#         match = match_str(self.name, pattern)
#     else:
#         name_split = split_fname(self.name)
#         if mode == 'stem':
#             match = match_str(name_split[0], pattern)
#         elif mode == 'ext':
#             match = match_str(name_split[1], pattern)
#         elif mode == 'parts':
#             pattern = re.escape(pattern)
#             try:
#                 match = re.match(rf'[a-zA-Z-_\d.]*{pattern}[a-zA-Z-_\d.]*', self.name).group(0)
#             except AttributeError:
#                 match = None
#         if match is not None:
#             match = self.name
#     return match


def check_path_glob(
        path_list: list,
        length: Union[int, None],
        check_func: ["is_dir", "is_file", "exists"] = "exists"
) -> NoReturn:

    # print(path_list)
    if length is None or len(path_list) == length:
        if eval(f"path_list[0].{check_func}()"):
            # print(f"{path_list[0]}.{check_func}")
            pass
        else:
            raise TypeError(
                f"{path_list[0].name} did not pass type check with {check_func}"
            )
    elif len(path_list) == 0:
        # print(path_list)
        raise IOError(f"No file or directory found.")
    elif len(path_list) > length or len(path_list) < length:
        raise ValueError(
            "Wrong length of file list!\n"
            f"Found {len(path_list)} files:.\n"
            "\n".join(path_list)
        )
    else:
        print("other")


class File(_Path):
    def __new__(cls, *args, **kwargs):
        return super().__new__(check_os(PosixFile), *args, **kwargs)

    def __init__(self, *args, ext="*", check=False):
        # print(self.is_file())
        if check:
            # print(f"Checking {self.name}")
            path_glob = list(self.parent.glob(f"{self.name}{ext}"))
            check_path_glob(path_glob, length=1, check_func="is_file")


class PosixFile(_PosixPath, File):
    pass


class Dir(_Path):
    def __new__(cls, *args, **kwargs):
        return super().__new__(check_os(PosixDir), *args, **kwargs)

    def __init__(self, *args, create=False):
        # print(f"Checking {self.resolve()}")
        # print(self.resolve().exists())
        path_glob = list(self.resolve().parent.glob(f"{self.resolve().name}"))
        if create and not self.is_dir():
            print(f"Creating directory {self.resolve()}")
            os.mkdir(self.resolve())
        else:
            check_path_glob(path_glob, length=1, check_func="is_dir")
            # print(self, type(self))
        # print(self.name)

    def _get_filelist(self, ext="*"):
        filelist = list(self.glob(rf"*[.]{ext}"))
        if len(filelist) != 0:
            return filelist

    _get_itp_filelist = partialmethod(_get_filelist, "itp")
    _get_gro_filelist = partialmethod(_get_filelist, "gro")
    _get_top_filelist = partialmethod(_get_filelist, "top")
    _get_mdp_filelist = partialmethod(_get_filelist, "mdp")

    def get_filelist(self, ext='*'):
        return FileList(self, ext)

    @property
    def filelist(self) -> FileList:
        return FileList(self, ext='*')

    @property
    def itp_filelist(self) -> ITPList:
        return FileList(self, ext='itp')

    @property
    def gro_filelist(self) -> GROList:
        return FileList(self, ext='gro')

    @property
    def top_filelist(self) -> TOPList:
        return FileList(self, ext='top')

    # path = attr.ib(type=Union[str, _Path],
    #                converter=Path,
    #                validator=validate_path)


class PosixDir(_PosixPath, Dir):
    pass

#
# def _apply_path_method(method):
#     def wrapper(instance: FileList, *args,
#                 # part: Literal['name', 'stem', 'suffix'] = 'name',
#                 **kwargs):
#         data_list = instance.data
#         data_list = [method(obj, *args, **kwargs) for obj in data_list]
#         # data_list = [method(eval(f'obj.{part}', *args, **kwargs) for obj in data_list]
#         return data_list
#     return wrapper


class FileList(UserList):
    def __init__(self, path, ext="*"):
        self.path = Dir(path)
        self.data = self.path._get_filelist(ext=ext)
        for idx, obj in enumerate(self.data):
            try:
                self.data[idx] = File(obj)
            except:
                self.data[idx] = Dir[obj]

    @property
    def names(self):
        return self.extract_fnames()

    @property
    def stems(self):
        return self.extract_fstems()

    @property
    def suffices(self):
        return self.extract_fsuffices()

    @staticmethod
    def _match_fname(
            fstem: str, full_namestr: str, ext: str = "", extended_fname: bool = False
    ):
        fstem, ext = map(lambda x: re.escape(x), (fstem, ext))
        # print(fstem, ext)
        if extended_fname:
            extend = "[a-zA-Z-_0-9]*"
        else:
            extend = ""
        try:
            # print('matching')
            # print(f'{fstem} {full_namestr}, {ext}')
            fname_match = re.search(rf"({fstem}){extend}[.]*{ext}", full_namestr).group(
                1
            )
            # print(fname_match, "match")
            return fname_match
        except AttributeError:
            return

    def filter(
            self,
            check_list: List[str],
            ext: str = "",
            extended_fname: bool = False,
            stem=False,
    ) -> List[str]:
        if stem:
            filter_list = self.stems
        else:
            filter_list = self.names
        # print(filter_list, check_list)
        # print(list_obj)
        # print(check_list)
        check_list = convert_to_list(check_list)
        # print(check_list)
        if len(check_list) > 1:
            check_str = "|".join(check_list)
        else:
            check_str = check_list[0]
        match_list = []
        # print(len(list_obj))
        if len(filter_list) > 1:
            # print('long')
            for item in filter_list:
                # print(item, check_str, ext)
                match_list.append(
                    self._match_fname(fstem=item, full_namestr=check_str, ext=ext)
                )
        else:
            # print(list_obj[0])
            match_list.append(
                self._match_fname(filter_list[0], check_str, ext, extended_fname)
            )
        match_list = sorted(
            list(filter(lambda match_item: match_item is not None, match_list)),
            key=match_list.index,
        )
        return match_list

    # def filter(self, check_obj, ext=False, stem=False):
    #     if stem:
    #         filter_list = self.stems
    #     else:
    #         filter_list = self.names
    #     check_str = '|'.join(UserList(check_obj))
    #     match_list = []
    #     for item in filter_list:
    #         match_list.append()
    #     search_obj = UserList(search_obj)
    #     data_list = self.names
    #     sel_list = []
    #     for o, obj in data_list:
    #         data_list[o] = obj.match_name()

    # @singledispatchmethod
    def _extract_parts(self, part="name"):
        data_list = self.data
        data_list = [getattr(obj, part) for obj in data_list]
        return data_list

    def match_name(self, pattern: str, mode='full', keep_dims=False):
        match = [obj.match_name(pattern, mode) for obj in self.data]
        if not keep_dims:
            match = [obj for obj in match if obj is not None]
        return match

    def match_namelist(self, pattern_list: List[str], mode='full', keep_dims=False):
        match = [obj.match_namelist(pattern_list, mode) for obj in self.data]
        if not keep_dims:
            match = [obj for obj in match if obj is not None]
        return match

    extract_fstems = partialmethod(_extract_parts, part="stem")
    extract_fnames = partialmethod(_extract_parts, part="name")
    extract_fsuffices = partialmethod(_extract_parts, part="suffix")



# a = FileList('/storage/')
# print(a.filter(['fftw'], extended_fname=True, stem=True), type(a))
# b=a.match_name()
# print('match', b)

class ITPList(FileList):
    def __init__(self, path, name=None, include="all", exclude=None):
        super().__init__(path, ext='itp')
        # self.path = Dir(path)
        # self.data = sorted(self.path.itp_filelist)
        self.selection = None
        self.include(include)
        self.exclude(exclude)

    def include(self, include):
        if include == 'all':
            self.selection = self.names
        else:
            include = include.convert_to_list()
            self.selection = self.filter(check_list=include,
                                         ext='itp',
                                         extended_fname=False,
                                         stem=True
                                         )

    def exclude(self, exclude):
        if exclude is None:
            pass
        else:
            exclude = exclude.convert_to_list()
            exclude = self.filter(check_list=exclude,
                                  ext='itp',
                                  extended_fname=False,
                                  stem=True
                                  )
            self.selection = [it.dropwhile(lambda itp: itp in exclude),
                              self.selection]
