import os
import re
import warnings
from pathlib import Path, PosixPath as _PosixPath
from functools import wraps, partial, singledispatch
import yaml

import numpy as np
import MDAnalysis as mda

from collections.abc import Sequence

# from config.types import File, Dir
from config.classes import ForceField, UCData

warnings.filterwarnings("ignore", category=DeprecationWarning)
from attr import validators, converters, define, Factory
import attr

# from config.types import PosixDir, PosixFile
from typing import Union
from os import PathLike

# __all__ = ['get_vars', 'get_params', 'get_paths', '']

class AttrChecks:
    def check_build(self):
        pass

    def check_siminp(self):
        pass


#
# def extract_fname(obj, stem=False, check=False):
#     if isinstance(obj, str):
#         try:
#             obj = File(obj, check=check)
#         except:
#             try:
#                 obj = Dir(obj)
#             except:
#                 obj = _PosixPath(obj)
#         if stem == True:
#             obj = pl.Path(obj).stem
#         else:
#             obj = pl.Path(obj).name
#     elif isinstance(obj, (Sequence, np.ndarray)):
#         obj = np.asarray(obj)
#         for i, item in enumerate(obj):
#             if stem == True:
#                 obj[i] = pl.Path(item).stem
#             else:
#                 obj[i] = pl.Path(item).name
#     else:
#         print(type(obj))
#         raise TypeError("Wrong dtype")
#     return obj
#
#
# extract_fstem = partial(extract_fname, stem=True)

def read_yaml(file):
    with open(file, 'r') as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
    return yaml_data

def get_vars_decorator(func):
    def wrapper(module):
        return {
            func(key): value
            for key, value in module.__dict__.items()
            if not key.startswith("__")
        }

    return wrapper


@get_vars_decorator
def get_vars(key):
    return key


@get_vars_decorator
def get_params(key):
    return str.lower(key)


@get_vars_decorator
def get_paths(key):
    return key


# class OutpathMethods:
def init_outpath(instance):
    """
    Create output directory if not existent.
    :param instance: :class: `Params` instance
    :type instance: Params
    :rtype: NoReturn
    """
    for path in [instance.outpath.parent, instance.outpath]:
        try:
            os.mkdir(path)
        except FileExistsError:
            print(
                f"WARNING! {path} already exists.\n"
                "Existing files will be overwritten"
            )


def check_outpath(instance):
    """
    Checks if output directory exists and assigns ```outpath``` to :class: `Params` instance.
    :param instance: :class: `Params` instance
    :type instance: Params
    :rtype: NoReturn
    """
    if not Path(instance.outpath).is_dir():
        raise NotADirectoryError(f"{instance.outpath} is not a directory.")
    instance.__setattr__("outpath", instance.outpath / instance.sysname)


def get_outdir(instance, outname):
    setattr(instance, "outpath", instance.outpath / outname)
    print(instance.outpath)


def check_bool_false(instance, attribute, value):
    if type(value) == bool and value != False:
        raise ValueError(f"Invalid argument for {attribute.name}!")


def check_build(instance, attribute, value):
    if type(value) == str and value != "new" and value != "load":
        raise ValueError(
            "Invalid option for builder!\n"
            'Specify "new" to build new clay model or "load" to load existing files.'
        )
    elif type(value) == dict:
        if list(value.keys())[0] != "load":
            raise ValueError(f"{value} is invalid option for builder!")
        elif value["load"][0] not in ["np", "gro"]:
            print(value["load"][0])
            raise ValueError(
                "Invalid option for builder!\n"
                'Use "np" or "gro" to specify which file type should be loaded.'
            )
        elif np.array(
                list(
                    map(
                        lambda fname: Path(fname).suffix not in ["", ".gro", ".npy"],
                        value["load"][1:],
                    )
                )
        ).any():
            print(
                np.array(list(map(lambda fname: Path(fname).suffix, value["load"][1:])))
            )
            raise ValueError("Invalid option for builder!\n" "Wrong file extension.")
    else:
        pass


def assign_prms_decorator(func):
    def wrapper(instance):
        for prm_key in instance.prm_dict:
            try:
                print(f'Setting {func(prm_key)} to {instance.prm_dict[prm_key]}')
                print(instance, func(prm_key), instance.prm_dict[prm_key])
                setattr(instance, func(prm_key), instance.prm_dict[prm_key])
                print('set')
            except AttributeError:
                print('\nerror\n')
                continue

    return wrapper


# def assign_consts(instance):
#     for prm_key in instance.prm_dict:
#         try:
#             setattr(instance, str.lower(prm_key), instance.prm_dict[prm_key])
#             print(f"Setting {prm_key!r} to {instance.prm_dict[prm_key]!r}.")
#         except AttributeError:
#             continue


@assign_prms_decorator
def assign_params(prm_key):
    print(prm_key)
    return str.lower(prm_key)


@assign_prms_decorator
def assign_paths(prm_key):
    return f"_{str.lower(prm_key)}"


@assign_prms_decorator
def assign_prms(prm_key):
    return prm_key


#     for prm_key in instance.prm_dict:
#         try:
#             setattr(instance, str.lower(prm_key), instance.prm_dict[prm_key])
#             print(f"Setting {prm_key!r} to {instance.prm_dict[prm_key]!r}.")
#         except AttributeError:
#             continue
#
# def assign_paths(instance):
#     paths_list = ['_data_path',
#                   '_ff_dir',
#                   '_uc_dir',
#                   '_mdp_path',
#                   '_mdp_dict',
#                   ]


def print_files(file_list):
    print("The following  files are available:")
    for filename in file_list:
        print(f"{filename!r}\n")


def find_files(instance, attribute):
    if not hasattr(instance, attribute):
        print("no attr")
        print(f"Searching {attribute!r} files:")
        path = instance.outpath.parent
        file_list = sorted(path.glob(f"**/*.{attribute}"))
        if len(file_list) == 1:
            value = file_list[0]
            setattr(instance, attribute, str(value))
            print(f"Setting {attribute!r} to {value!r}")
        elif len(file_list) > 1:
            print_files(file_list=file_list)
            raise AttributeError("Please specify which file should be used")
        else:
            raise FileNotFoundError(
                f"No {attribute!r} files were found in {instance.outpath.parent!r}"
            )
    else:
        print("attr")


def file_exists(instance, attribute, value):
    if not Path(value).is_file():
        raise FileNotFoundError(f"Selected {attribute!r} file {value!r} does not exist")


def dir_exists(instance, attribute, value):
    if not Path(value).is_dir():
        raise NotADirectoryError(
            f"Selected {attribute!r} directory {value!r} does not exist"
        )


def get_uc_numbers(instance):
    if hasattr(instance, "uc_wat"):
        try:
            u = mda.Universe(str(instance.gro))
        except AttributeError:
            print('No {"gro"!r} file specified.')
        except IOError:
            print(f"Could not read {instance.gro!r}")


def set_il_solv(instance):
    if (hasattr(instance, "il_solv") and instance.il_solv != False) or not hasattr(
            instance, "il_solv"
    ):
        attr_list = []
        for attribute in ["_il_waters", "_uc_waters", "_spacing_waters"]:
            if hasattr(instance, attribute):
                attr_list.append(attribute)
                print(attr_list)
        if len(attr_list) != 1:
            raise AttributeError(
                "One specification for interlayer solvation allowed."
                "Found {len(attr_list)}: "
                " ".join(attr_list)
            )
        else:
            attribute = attr_list[0]
            match = re.match(r"[_]([a-z]*)", attribute).group(1)
            setattr(instance, "il_solv", {match: getattr(instance, attribute)})
            delattr(instance, attribute)
    else:
        raise AttributeError("No interlayer solvation specified.")


# @define
# class FF:
#     type = attr.ib(type=str, validator=validators.matches_re("clay|ions"))
#     name = attr.ib(type=Union[str, Path], validator=validate_path, converter=Path)
#
class ForceFieldSelection:
    def __init__(self):
        self.clay_ff = None
        self.ions_ff = None


#
# from package.builder.base import ForceField, UCData


@singledispatch
def convert_to_list(object):
    raise TypeError(f'Function not defined for type {type(object)!r}!')
    pass


@convert_to_list.register(str)
def _(object):
    return [object]


@convert_to_list.register(tuple)
@convert_to_list.register(np.ndarray)
def _(object):
    return list(object)

@convert_to_list.register(list)
def _(object):
    return object


# @convert_to_list.register(dict)
# def _(object):
#     dict_lists = []
#     for k, v in object.items():
#
#     if len(object.keys()) == 1:
#         keys = list(object.keys())
#     else:
#         keys = list(*object.keys())
#     if len(object.values()) == 1:
#         values = list(object.values())
#     else:
#         values = []
#
#     return list(*object.keys()), list(*object.values())

# a = 'abc'
# b = {'ab': ['cd', 'aa']}
#
# print([b], list(b))
#
# print(convert_to_list(a), convert_to_list(b))

def process_ff(instance):
    temp_ff = instance.ff
    print(temp_ff)
    ff_sel = ForceFieldSelection()
    for ff_type in ["clay", "ions"]:
        new_ff = {"selection": 'all', "exclude": None}
        try:
            print(instance.ff)
            ff = instance.ff[ff_type]
            print(ff_type)
            try:
                new_ff['exclude'] = ff['exclude']
            except KeyError:
                pass
            try:
                new_ff['selection'] = ff['selection']
            except KeyError:
                new_ff['selection'] = ff
            print(ff, instance.ff_dir)
            setattr(ff_sel, ff_type, ForceField(instance.ff_dir, include=new_ff['selection'],
                                                exclude=new_ff['exclude']))

            print(f'\n\nsetting {ff_type}_ff')
            setattr(instance, f'{ff_type}_ff', ForceField(instance.ff_dir, include=new_ff['selection'],
                                                          exclude=new_ff['exclude']))
            print(eval(f'instance.{ff_type}_ff'))
        except KeyError:
            print('except')


def process_ucs(instance):
    print(instance.clay_ff.atomtypes)
    uc = UCData(path=instance.clay_units_dir,
                       clay_type=instance.clay_type,
                                           atomtypes_df=instance.clay_ff.atomtypes)
    print(uc.uc_charges, uc.atomtypes)
    setattr(instance, 'clay_units', UCData(path=instance.clay_units_dir,
                                           clay_type=instance.clay_type,
                                           atomtypes_df=instance.clay_ff.atomtypes))
    setattr(instance, 'uc_charges', uc.uc_charges)
    print(instance.clay_units)

