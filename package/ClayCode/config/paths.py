# """Global data variables for clay model construction."""
# import os
# import shutil
# from functools import partial
# from typing import Literal, Union
#
# from pkg_resources import resource_filename
#
# from config.classes import Dir, File
#
#
#
# ClayFF = Dir(resource_filename(__name__, "../data/FF/ClayFF_Fe.ff"))
# AmberIons = Dir(resource_filename(__name__, "../data/FF/AmberIons.ff"))
# charmm27 = Dir(resource_filename(__name__, "../data/FF/charmm27.ff"))
# charmm36 = Dir(resource_filename(__name__, "../data/FF/charmm36.ff"))
#
# FF = [ClayFF, AmberIons, charmm27, charmm36]
#
#
# D11 = Dir(resource_filename(__name__, "../data/clay_units/D11"))
# D21 = Dir(resource_filename(__name__, "../data/clay_units/D21"))
# T11 = Dir(resource_filename(__name__, "../data/clay_units/T11"))
# T21 = Dir(resource_filename(__name__, "../data/clay_units/D21"))
#
# CLAY_UNITS = [D11, D21, T11, T21]
#
# MOLECUES = Dir(resource_filename(__name__, "../data/molecules"))
#
# AA = Dir(resource_filename(__name__, "../data/molecules/amino_acids"))
#
# CC_DIR = Dir(resource_filename(__name__, ".."))
# BUILD_DIR = Dir(resource_filename(__name__, "../package/builder"))
# SIMINP_DIR = Dir(resource_filename(__name__, "../package/siminp"))
#
# DATA_DIR = Dir(resource_filename(__name__, "../data"))
# FF_DIR = Dir(resource_filename(__name__, "../data/FF"))
# CLAY_UNITS_DIR = Dir(resource_filename(__name__, "../data/clay_units"))
# MDP_DIR = Dir(resource_filename(__name__, "../data/mdp"))
#
# USER_DATA_DIR = Dir(resource_filename(__name__, "../data/user"), create=True)
# USER_FF_DIR = Dir(resource_filename(__name__, "../data/user/FF"), create=True)
# USER_CLAY_UNITS_DIR = Dir(resource_filename(__name__, "../data/user/clay_units"), create=True)
# USER_MDP_DIR = Dir(resource_filename(__name__, "../data/user/mdp"), create=True)
#
# WDIR = Dir.cwd()
#
# MDP_DICT = {mdp.stem: File(mdp) for mdp in list(MDP_DIR.glob("*"))}
# print(MDP_DICT, "\n", type(MDP_DICT["em"]))
#
#
#
# # def add_user_data(data, type: ['FF', 'clay_units'] = None):
# #     path_dict = {'FF': [USER_FF_DIR, '.ff', FF],
# #                  'clay_units': [USER_UC_DIR, '', UCS]}
# #     data = Dir(data, create=False)
# #     if type in path_dict.keys():
# #         print([path_dict[type][2]])
# #         if data.stem in path_dict[type][2]:
# #             raise KeyError('{data.stem} has the same name as internal data. Please rename the folder.')
# #         print(f'Adding {data.name} to {path_dict[type][0]}.')
# #         new_dir = path_dict[type][0] / f'{data.stem}{path_dict[type][1]}'
# #         try:
# #             shutil.copytree(data, new_dir)
# #         except FileExistsError:
# #             print(f'{data.name} is has already been added, please remove existing data before adding again.')
# #         globals()[str.upper(new_dir.stem)] = Dir(path_dict[type][0] / f'{data.stem}{path_dict[type][1]}', create=False)
# #     else:
# #         raise ValueError(f'Data type must be one of '
# #                          ', '.join(path_dict.keys()))
#
#
# # def add_ff(ff_path):
# #     ff_path = Dir(ff_path, create=False)
# #     shutil.copy(ff_path, USER_FF_DIR)
# #     # globals().
# #         new_ff = USER_FF_DIR / str.upper(ff_name)
# #         if new_ff.is_dir():
# #             globals()[str.upper(ff_name)] = new_ff
#
#
# def get_user_data():
#     for path in USER_FF_DIR, USER_CLAY_UNITS_DIR:
#         print(path)
#         entries = list(path.glob(r'*'))
#         print(entries)
#         if len(entries) != 0:
#             print(not 0)
#             for entry in entries:
#                 print(entry.name)
#                 if entry.name not in globals():
#                     globals()[entry.name] = Dir[entry]
#                     globals()[path.name].append(Dir(entry))
#     data = list(USER_MDP_DIR.glob('.mdp'))
#     for mdp in data:
#         globals()['MDP_DICT'][mdp.stem] = File(mdp)
#
#
# # add_user_data('/home/hannahpollak/AA_LDH/ClayFF.ff', 'FF')
#
# # add_ff('newff')
#
#
# # SRCDIR = Path(__file__).parents[1]
#
#
# # # stem of single layer .gro and .top files gives numbered .gro and .top names
# # # (e.g. NON_7x5_1.gro)
# # SHEET_FILESTEM = f'{CLAY_TYPE}_{X_DIM}x{Y_DIM}_'
#
# # output files location
# # OUTPATH = SRCDIR / 'output'
#
# # make Path self of uc data
# # DATA_PATH = SRCDIR / 'data'
#
# # FF files location
# # FF_PATH = DATA_PATH / 'FF'
#
# # UC_PATH = DATA_PATH / 'clay_units'
#
# # Layer composition save
# # UC_ARRAYS = DATA_PATH / 'uc_comp.npy'
#
# # MDP_PATH = DATA_PATH / 'mdp'
#
#
# # print(MDP_DICT)
#
# # UC_PATH = DATA_PATH / CLAY_TYPE
#
#
#
# def add_new_data(data, kind=Literal[Union['FF', 'clay_units', 'mdp']]):
#     data = Dir(data)
#     if kind == 'FF':
#         if data.suffix != '.ff':
#             print(data.suffix)
#             raise ValueError(f'Force field data must end in {".ff"!r}!')
#         else:
#             name = data.name
#             dest = globals()
#     elif kind == 'clay_units':
#         clay_itp = sorted(data.itp_filelist.stems)
#         clay_gro = sorted(data.gro_filelist.stems)
#         if not clay_itp == clay_gro:
#             raise ValueError(f'Names of {".itp"!r} and {".gro"!r} files must match!')
#         else:
#             name = data.name
#             dest = globals()
#     elif kind == 'mdp':
#         if not File(data).match(ext='.mdp'):
#             raise ValueError('Wrong file format!')
#         else:
#             name = data.stem
#             dest = MDP_DIR
#     dest_path = eval(f'USER_{kind.upper()}_DIR/data.name')
#     dest[data.name] = dest_path.resolve()
#     shutil.copytree(data, dest_path, dirs_exist_ok=True)
#     if kind in ['FF', 'clay_units']:
#         if globals()[data.name] not in globals()[kind.upper()]:
#             globals()[kind.upper()].append(globals()[data.name])
#         else:
#             raise ValueError(f'{name} is already registered!')
#
# add_ff = partial(add_new_data, kind='FF')
# add_uc = partial(add_new_data, kind='clay_units')
# add_mdp = partial(add_new_data, kind='mdp')
#
# # print(CLAY_UNITS[-1], FF[-1])
# # get_user_data()
# # print(CLAY_UNITS[-1], FF[-1])
# #
# # # add_ff('/home/hannahpollak/test.ff')
# # # add_uc('../data/clay_units/D21')
# # print(CLAY_UNITS, '\n', FF, '\n',  MDP_DICT)
#
