import re
from functools import partialmethod, wraps
from pathlib import Path as _Path, PosixPath as _PosixPath
import os
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
)

from typing_extensions import TypeAlias, ParamSpec, ParamSpecKwargs, ParamSpecArgs
from collections import UserList, Counter
from dataclasses import dataclass
import itertools as it
import ast
from config.lib import convert_to_list

PosixDir = NewType("PosixDir", _PosixPath)
PosixFile = NewType("PosixFile", _PosixPath)

Dir = NewType("Dir", Union[_PosixPath, PosixDir])
File = NewType("Folder", Union[_PosixPath, PosixFile])
Class = ParamSpec("Class")
Subclass = ParamSpec("Subclass")
PathLike = NewType("PathLike", Union[os.PathLike, str])
PathLike_or_False: TypeAlias = Union[PathLike, Literal[False]]
File_or_Dir: TypeAlias = Union[File, Dir]


# ===================================
# Generic class methods
# ================================


def add_method(cls: Class) -> Class:
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)

        setattr(cls, func.__name__, wrapper)

    return decorator


# ================
# extensions for pathlib Path
# =================


@add_method(_PosixPath)
def match_name(
    self,
    full: PathLike_or_False = False,
    stem: PathLike_or_False = False,
    ext: PathLike_or_False = False,
    parts: bool = False,
) -> Union[str, None]:
    if full:
        # full = re.escape(full)
        print(full)
        name_split = full.split(".", maxsplit=1)
        print(name_split)
        if len(name_split) == 1:
            name_split.append('[a-z]*')
        fname = [".".join(name_split[:-1]), name_split[-1]]
        print(fname)
        print(f'full: {full}, stem: {stem}, ext: {ext}')
    elif stem or ext:
        if not ext:
            ext = '[a-z]*'
        if stem:

        fname = [stem, ext]
        for part in range(2):
            if fname[part]:
                fname[part] = re.escape(fname[part])
            else:
                fname[part] = "[a-zA-Z-_0-9]*"
        print(f'fname: {fname}, stem: {stem}, ext: {ext}')
    if parts:
        print("parts")
        parts = "[a-zA-Z-_0-9]*"
    else:
        parts = ""
    print(full, fname, parts, rf"{parts}({fname[0]}){parts}\.*{fname[1]}")
    try:
        name_match = re.search(rf"{parts}({fname[0]}){parts}\.*{fname[1]}", self.name).group(
            0
        )
        print(name_match, "match")
        return name_match
    except AttributeError:
        return None


def check_path_glob(
    path_list: List[PathLike],
    length: Union[int, None],
    check_func: ["is_dir", "is_file", "exists"] = "exists",
) -> NoReturn:

    print(path_list)
    if length is None or len(path_list) == length:
        print(path_list[0], type(path_list[0]))
        if eval(f"path_list[0].{check_func}()"):
            print(f"{path_list[0]}.{check_func}() == True")
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


def check_os(
    subclass: Union[PosixFile, PosixDir]
) -> Union[Union[PosixFile, PosixDir], NotImplementedError]:
    return (
        subclass
        if os.name == "posix"
        else NotImplementedError("ClayCode does not support Windows systems.")
    )


class Dir(_Path):
    def __new__(cls, *args, **kwargs) -> Union[Dir, NoReturn]:
        return super().__new__(check_os(PosixDir), *args, **kwargs)

    def __init__(self, *args, create: bool = False) -> NoReturn:
        # print(f"Checking {self.resolve()}")
        print(self.resolve().exists())
        path_glob = list(self.parent.glob(f"{self.resolve().name}"))
        check_path_glob(path_glob, length=1, check_func="is_dir")
        # print(self, type(self))
        if create:
            print(f"Creating directory {self.resolve()}")
            os.mkdir(self.resolve())
        # print(self.name)

    def get_filelist(
        self, ext: str = "*"
    ) -> Union[List[File_or_Dir], FileNotFoundError]:
        filelist = list(self.glob(rf"*[.]{ext}"))
        if len(filelist) != 0:
            return filelist
        else:
            raise FileNotFoundError(f"No files found in {self.name}.")

    get_itp_filelist = partialmethod(get_filelist, "itp")
    get_gro_filelist = partialmethod(get_filelist, "gro")
    get_top_filelist = partialmethod(get_filelist, "top")
    get_top_filelist = partialmethod(get_filelist, "top")

    @property
    def filelist(self) -> Union[List[File_or_Dir], FileNotFoundError]:
        return self.get_filelist()

    @property
    def itp_filelist(self) -> Union[List[File_or_Dir], FileNotFoundError]:
        return self.get_itp_filelist()

    @property
    def gro_filelist(self) -> Union[List[File_or_Dir], FileNotFoundError]:
        return self.get_gro_filelist()

    @property
    def top_filelist(self) -> Union[List[File_or_Dir], FileNotFoundError]:
        return self.get_top_filelist()


class PosixDir(_PosixPath, Dir):
    pass


class File(_Path):
    def __new__(cls, *args, **kwargs) -> Union[File, NoReturn]:
        return super().__new__(check_os(PosixFile), *args, **kwargs)

    def __init__(self, *args, ext: str = "*", check: bool = False) -> NoReturn:
        # print(self.is_file())
        if check:
            # print(f"Checking {self.name}")
            path_glob = list(self.parent.glob(f"{self.name}{ext}"))
            check_path_glob(path_glob, length=1, check_func="is_file")


class PosixFile(_PosixPath, File):
    pass


class FileList(UserList):
    def __init__(self, path: PathLike, ext: str = "*") -> NoReturn:
        self.path = Dir(path)
        self.data = self.path.get_filelist(ext=ext)

    @property
    def names(self) -> List[str]:
        return self.extract_fnames()

    @property
    def stems(self) -> List[str]:
        return self.extract_fstems()

    @property
    def suffices(self) -> list[str]:
        return self.extract_fsuffices()

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
        print(filter_list, check_list)
        # print(list_obj)
        print(check_list)
        check_list = convert_to_list(check_list)
        print(check_list)
        if len(check_list) > 1:
            check_str = "|".join(check_list)
        else:
            check_str = check_list[0]
        match_list = []
        # print(len(list_obj))
        if len(filter_list) > 1:
            # print('long')
            for item in filter_list:
                print(item, check_str, ext)
                match_list.append(
                    _match_fname(
                        name_pattern=item,
                        full_name=check_str,
                        ext=ext,
                        match_parts=extended_fname,
                    )
                )
        else:
            # print(list_obj[0])
            match_list.append(
                _match_fname(filter_list[0], check_str, ext, extended_fname)
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
    def _extract_parts(
        self, part: Literal["name", "stem", "suffix"] = "name"
    ) -> List[str]:
        data_list = self.data
        data_list = [getattr(obj, part) for obj in data_list]
        return data_list

    extract_fstems = partialmethod(_extract_parts, part="stem")
    extract_fnames = partialmethod(_extract_parts, part="name")
    extract_fsuffices = partialmethod(_extract_parts, part="suffix")


def _match_fname(
    name_pattern: str, full_name: str, ext: Optional[str] = None, match_parts: bool = False, stem: bool=False
) -> Union[PathLike, NoReturn]:
    if ext is None:
            ext='*'
    name_pattern, ext = map(lambda x: re.escape(x), (name_pattern, ext))
    print(name_pattern, ext)
    ext = ext.strip('.')
    print(ext)
    if match_parts:
        extend = "[a-zA-Z-_0-9]*"
    else:
        extend = ""
    try:
        # print('matching')
        print(f"{name_pattern} {full_name}, {ext}")
        fname_match = re.search(rf"({extend}{name_pattern}){extend}[.]*{ext}", full_name)#.group(
            # 0
        # )
        print(fname_match.group(0), fname_match.group(1))
        # if stem:
        #     fname_match = fname_match.removesuffix()
        print(fname_match, "match")
        return fname_match
    except AttributeError:
        pass

# print(_match_fname('bonded', 'ffnonbonded.itp', ext='itp', match_parts=True, stem=False))

f = File("../data/clay_units/D21/D211.itp")
type(f)
match = f.match_name(ext='itp')
print(f'\n\n{match}')
d = Dir("../data/FF/ClayFF_Fe.FF", create=False)
# print(d.match_name(stem='ClayFF_Fe'))
d_match = d.match_name('ClayFF', parts=True)
print('\n\n', d_match)
# fl = PathList(data=d)
# print(fl)
# new_list = fl.filter(check_list="bonded", ext="itp", extended_fname=True)
# print(new_list)
