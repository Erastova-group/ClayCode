"""Global default variables for ClayCode."""
import shutil
from functools import partial
from typing import Literal, Union

import yaml
from pkg_resources import resource_filename

from .classes import Dir, File, PathList

#
# DATA = ''
#
# ClayFF = Dir(resource_filename(__name__, "../data/FF/ClayFF_Fe.ff"))
# AmberIons = Dir(resource_filename(__name__, "../data/FF/Ions.ff"))
# charmm27 = Dir(resource_filename(__name__, "../data/FF/charmm27.ff"))
# charmm36 = Dir(resource_filename(__name__, "../data/FF/charmm36.ff"))
#
# _FF = [ClayFF, AmberIons, charmm27, charmm36]
#
#
# D11 = Dir(resource_filename(__name__, "../data/clay_units/D11"))
# D21 = Dir(resource_filename(__name__, "../data/clay_units/D21"))
# T11 = Dir(resource_filename(__name__, "../data/clay_units/T11"))
# T21 = Dir(resource_filename(__name__, "../data/clay_units/D21"))
#
# _CLAY_UNITS = [D11, D21, T11, T21]
#
# MOLECULES = Dir(resource_filename(__name__, "../data/molecules"))
#
# AA = Dir(resource_filename(__name__, "../data/molecules/amino_acids"))
#
# CC_DIR = Dir(resource_filename(__name__, ".."))
# BUILD_DIR = Dir(resource_filename(__name__, "../package/build"))
# SIMINP_DIR = Dir(resource_filename(__name__, "../package/siminp"))
#
# DATA_DIR = Dir(resource_filename(__name__, "../data"))
# FF_DIR = Dir(resource_filename(__name__, "../data/FF"))
# CLAY_UNITS_DIR = Dir(resource_filename(__name__, "../data/clay_units"))
# MDP_DIR = Dir(resource_filename(__name__, "../data/mdp"))
#
# USER_DATA_DIR = Dir(resource_filename(__name__, "../data/user"))
# USER_FF_DIR = Dir(resource_filename(__name__, "../data/user/FF"), create=True)
# USER_CLAY_UNITS_DIR = Dir(
#     resource_filename(__name__, "../data/user/clay_units"), create=True
# )
# USER_MDP_DIR = Dir(resource_filename(__name__, "../data/user/mdp"), create=True)
#
# WDIR = Dir.cwd()
#
# _MDP_DICT = {mdp.stem: File(mdp) for mdp in list(MDP_DIR.glob("*"))}
#
#
# def get_user_data():
#     for dir in globals()["USER_FF_DIR"], globals()["USER_CLAY_UNITS_DIR"]:
#         entries = list(dir.glob(r"*"))
#         if len(entries) != 0:
#             for entry in entries:
#                 print(f"Adding {entry.name!r} to {dir.name!r}")
#                 print(entry.name)
#                 if entry.name not in globals():
#                     globals()[entry.name] = Dir[entry]
#                     globals()[f"_{dir.name}"].append(Dir(entry))
#         else:
#             print(f"No user data to add to {dir.name!r}.")
#     data = list(USER_MDP_DIR.glob(".mdp"))
#     for mdp in data:
#         globals()["MDP_DICT"][mdp.stem] = File(mdp)
#
#
# get_user_data()
#
#
# def add_new_data(path, kind=Literal[Union["FF", "clay_units", "mdp"]]):
#     kind = f"_{kind}"
#     path = Dir(path)
#     if kind == "FF":
#         if path.suffix != ".ff":
#             print(path.suffix)
#             raise ValueError(f'Force field path must end in {".ff"!r}!')
#         else:
#             name = path.name
#             dest = globals()
#     elif kind == "clay_units":
#         clay_itp = sorted(path.itp_filelist.stems)
#         clay_gro = sorted(path.gro_filelist.stems)
#         if not clay_itp == clay_gro:
#             raise ValueError(f'Names of {".itp"!r} and {".gro"!r} files must match!')
#         else:
#             name = path.name
#             dest = globals()
#     elif kind == "mdp":
#         if not File(path).match_name(ext=".mdp"):
#             raise ValueError("Wrong file format!")
#         else:
#             name = path.stem
#             dest = MDP_DIR
#     else:
#         raise ValueError(f'kind must be in {["FF", "CLAY_UNITS", "mdp"]}')
#     dest_path = eval(f"USER_{kind.upper()}_DIR/path.name")
#     if dest_path.name in PathList(dest[f"{kind.upper()}_DIR"]).names:
#         raise ValueError(f"{name} is already registered!")
#     else:
#         print(
#             f"Adding {name!r} (from {str(path)!r}) to {str(dest_path.parent / dest_path.name)!r}"
#         )
#         dest[path.name] = dest_path.resolve()
#         shutil.copytree(path, dest_path, dirs_exist_ok=True)
#         globals()[kind.upper()].append(globals()[path.name])
#         print(f"Updating internal data:")
#         get_user_data()
#
#
# add_ff = partial(add_new_data, kind="FF")
# add_uc = partial(add_new_data, kind="clay_units")
# add_mdp = partial(add_new_data, kind="mdp")
