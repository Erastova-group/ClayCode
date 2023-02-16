











# @add_method(_PosixPath)
# def validate_path(self)

# a = _PosixPath('config/params.py')
# print(a.match_name(stem='consts'))









    # def __
    #
    # def __
    #     print(self)
    #     print(self.itp_filelist)
    #     setattr(self, 'paths', self._get_itp_filelist())
    #     self.paths = 'a'

    # print(self.paths)


# a = PathList('../data/FF/ClayFF_Fe.FF')
# print('a', a.__dict__)
# b = a.filter(['ffnonbonded'], stem=True)
# print(b)




# itp = ITPList(data='../data/FF/ClayFF_Fe.FF', selection='ffnonbonded')
# print(itp.names, itp.selection)
import os
from typing import Union
from typing_extensions import TypeAlias
from pathlib import Path as _Path, PosixPath as _PosixPath

from attr import  validators
from attrs import define
import attr

from config.classes import File, Dir


def validate_path(instance, attribute, value):
    try:
        path = _PosixPath(value)
    except TypeError:
        raise TypeError(f"{attribute} must be str or path like type.")
    # if convert:
    #     setattr(instance, attribute, data)
    # if not type(value) in [str, PosixFile, PosixDir, _PosixPath, _Path, os.PathLike]:
    #     raise TypeError(f"{attribute} must be str or data like type.")

def make_path(instance, attribute, value):
    path = _PosixPath(value)
    if path.is_file():
        setattr(instance, attribute, File(value))
    elif path.is_dir():
        setattr(instance, attribute, Dir(value))
    else:
        setattr(instance, attribute, _PosixPath)

# def convert_to_list(list_or_str):
#     if type(list_or_str) == list:
#         return list_or_str
#     else:
#         return [list_or_str]


@define
class FF:
    type = attr.ib(type=str, validator=validators.matches_re("clay|ions"))
    name = attr.ib(type=Union[str, _Path], validator=validate_path, converter=make_path)


def process_ff(instance):
    temp_ff = instance.ff
    for ff_type in ["clay", "ions"]:
        new_ff = {"selection": [], "exclude": []}
        try:
            ff = instance.ff[ff_type]
            print(ff_type)
            if "selection" in ff.keys():
                pass
        except KeyError:
            pass
FileType: TypeAlias = Union[File, Dir, os.PathLike]


# class DirList(UserList):
#     def __init__(self, *elements, create=False):
#         super().__init__(elements)
#         setattr(self, "data", [Dir(item, create=create) for item in self.data])
#
#
# class PathList(UserList):
#     def __init__(self, *elements, ext="*", check=False):
#         super().__init__(elements)
#         # print("data:", self.data)
#         setattr(self, "data", [File(item, ext=ext, check=check) for item in self.data])
#

# class PathList(UserList):
#     def __init__(self, *elements, create=False):
#         setattr(self, 'data', UserList(map(lambda file: File(path, create=create), UserList(elements))))

# a = DirList("../data", create=False)
# print(a)
# b = PathList("lib.py", "errors.py", check=True)
# print(b)


# @define(kw_only=True)
# class ForceField:
#     type = attr.ib(type=str, validator=validators.matches_re('clay|ions'))
#     name = attr.ib(type=Union[str, _Path])
#     data = attr.ib(
#         type=Union[str, _Path, List[Union[str, _Path]]], converter=convert_to_list
#     )
#     _include = attr.ib(type=Union[Dict[str, str], str, List[str], "all"], default="all")
#     _exclude = attr.ib(
#         type=Union[Dict[str, str], str, List[str], None],
#         default=None,
#         converter=convert_to_list,
#     )
#     _dtype_dict = attr.ib(init=False)
#
#     def __attrs_post_init__(self):
#         setattr(self, '_dtype_dict', {
#         "at_type": "str",
#         "at_number": "int32",
#         "ptype": "str",
#         "sigma": "float64",
#         "epsilon": "float64",
#         "id": "int32",
#         "res_number": "int32",
#         "res_name": "str",
#         "at_name": "str",
#         "charge_nr": "int32",
#         "charge": "float64",
#         "mass": "float64",
#         "FF": "str",
#         "itp": "str",
#     })
#         print(self.data, self.name, self._include)
#         ff_list = []
#         for ff_path in self.data:
#             ff_path = _Path(ff_path)
#             print(ff_path.exists())
#             ff_list.extend(list(ff_path.glob(f"{self.name}[.f]*")))
#         print(len(ff_list))
#         if len(ff_list) == 1:
#             itp_list = ITPList(data=ff_list[0])
#             if len(itp_list.paths) != 0:
#                 self.data = ff_path
#                 self.itp_list = itp_list
#             print(itp_list.names)
#         if len(ff_list) == 0:
#             raise NotADirectoryError(f"No force field with name {self.name} was found.")
#         if self._include != "all":
#             self.itp_list = self.itp_list.names
#             # ff_path = list(ff_path.glob(f'{ff_path}/{self.name}[.f]*'))
#             # print(results(ff_path))
#             # if ff_path / self.name.is_dir():
#             #     setattr(self, 'data', ff_path / self.name)
#
# FF = ForceField(type='clay', data='../data/FF', name='ClayFF_Fe')
# a = File('params.py', check=True)
# print(a.resolve())
# b = Dir('.')

# class ForceField:
#     def __init__(self, ff_path, sel_dict, type='clay'):
#         self.data = Dir(ff_path)
#         self.available = PathList(self.data, ext='FF')
#         self.type = type
#         if 'selection' in sel_dict.keys():
#             try:
#                 self.selection = self.data
#                 print(self.available['selection'].stems, self.available)
#             except:
#                 pass
#             # print(self.selection)
#         print(self.available.names, sel_dict['selection'])
#         selection = [sel_dict.keys()]
#
#
# FF = ForceField(ff_path='/home/hannahpollak/claycode/data/FF', sel_dict={'selection': 'ClayFF_Fe'})
#
