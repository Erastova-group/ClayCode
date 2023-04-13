from functools import (
    cached_property, wraps,
)
from pathlib import Path

from ClayCode.core.classes import ITPList


# from conf.consts import (
#     UC_PATH,
#     UC_STEM,
#     UC_FOLDER,
#     X_CELLS,
#     MDP_DICT,
#     NA,
#     X_EL,
#     ISL,
#     D_SPACING,
#     FF_PATH,
#     ALT_FF_PATH,
#     FF_SEL_DICT,
#     SOLVATE,
#     SYSNAME,
#     Y_CELLS,
#     N_SHEETS,
#     OUTPATH,
#     FF_SEL,
#     EXP_DF,
#     X_EL,
#     WATERS,
#     ISL_SPEC,
#     SHEET_N_CELLS,
# )
# from ratios import CHARGE
# from parsing import UCData, FileParser, ForceField
# from ratios import UC_INDEX_LIST, UC_NUMBERS_LIST
# from readwrite import get_filename
# import readwrite as rw
# from dtypes import PathType, extract_fname, extract_fstem
# from utils import run_gmx_solvate, run_gmx_grompp, run_gmx_genion, run_gmx_select
# from typing import NoReturn, Any, AnyStr, Union, List, Dict
# from builder import InterlayerSolvent, Ions

# import top


class TopologyConstructorBase:
    # _ff_path_dict = {"local": FF_PATH, "other": ALT_FF_PATH}

    def __init__(self, uc_data, ff):  # , alt_ff_path_prefix=None):
        # if alt_ff_path_prefix is None:
        #     self._ff: ForceField = uc_data.ff #TopologyConstructorBase._ff_path_dict[ff_path_key]
        # else:
        self.ff = ff
        self.uc_data = uc_data
        self.__mol_str = ''

    @cached_property
    def _ff_head(self):
        ff_head_str = "; selection FF params for clay, water and ions\n"
        # for ff_sel in ['clay', 'ions']:
        ff_itps = ITPList([*self.ff['clay'].itp_filelist,
              *self.ff['ions'].itp_filelist])
        for itp_file in ff_itps:
            ff_head_str += f'#include "{itp_file}"\n'
        return ff_head_str + "\n"

    @cached_property
    def _uc_head(self):
        uc_head_str = "; include clay unit cell topology\n"
        uc_head_str += "".join(
            list(
                map(
                    lambda uc_itp: f"""#include "{uc_itp}"\n""",
                    sorted(self.uc_data.group_itp_filelist),
                )
            )
        )
        return uc_head_str + "\n"

    @cached_property
    def _molecule_head(self):
        mol_head_str = "[ system ]\n" \
                       f" {self.uc_data.name}\n" \
                       "[ molecules ]\n" \
                       "; Compound        nmols\n"
        return mol_head_str

    @cached_property
    def header(self):
        return self._ff_head + self._uc_head + self._molecule_head

    def write_function_output(func):
        @wraps(func)
        def wrapper(self, fname=None, *args, **kwargs):
            if fname is None:
                try:
                    fname = self.filename
                    print(f"No filename selected, writing topology to {self.filename}")
                except AttributeError:
                    raise AttributeError("No filename assigned.")
            outstr = self.header + func(self, *args, **kwargs)
            try:
                with open(fname, "w") as outfile:
                    outfile.write(outstr)
                    print(f"Writing to {fname}.")
            except IOError:
                raise IOError(f"Could not open {fname} for writing.")

        return wrapper

    def add_molecules(self, universe):
        res_groups = universe.residues
        for resnr, residue in enumerate(res_groups):
            if resnr == 0:
                n_residues = 1
            else:
                if residue.resname == prev_res.resname:
                    n_residues += 1
                else:
                    self.__mol_str += f"{prev_res.resname}\t{n_residues}\n"
                    n_residues = 1
            prev_res = residue
        self.__mol_str += f"{prev_res.resname}\t{n_residues}\n"

    def reset_molecules(self):
        self.__mol_str = ''

    @property
    def mol_str(self):
        return self.__mol_str

    def write(self, fname):
        fname = Path(fname).with_suffix('.top')
        with open(fname, 'w') as topfile:
            topfile.write(self.header + self.mol_str)

# class InterlayerTopology(TopologyConstructorBase):
#     n_wat = InterlayerSolvent().n_wat
#     ions_dict = Ions().get_ion_mols()
#
#     def __init__(self):
#         super().__init__(ff_path_key="local")
#
#     @classmethod
#     def get_isl_solv_str(cls):
#         return f"SOL    {cls.n_wat}\n"
#
#     @classmethod
#     def get_il_ions_str(cls):
#         ion_str = "".join(
#             map(
#                 lambda y: f"{y}    {cls.ions_dict[y]}\n",
#                 sorted(cls.ions_dict.keys(), reverse=True),
#             )
#         )
#         return ion_str
#
#     @classmethod
#     def get_top_str(cls):
#         return cls.get_isl_solv_str() + cls.get_il_ions_str()
#
#     @cached_property
#     def filename(self):
#         return get_filename("spc", ext="top")
#
#     @TopologyConstructorBase.write_function_output
#     def write_ions_top(self):
#         return self.get_top_str()
#
#     @TopologyConstructorBase.write_function_output
#     def write_spc_top(self):
#         return self.get_isl_solv_str()
#
#     @TopologyConstructorBase.write_function_output
#     def write_empty_top(self):
#         return ""


# class SheetTopology(InterlayerTopology):
#     solv_fname_dict = {False: "", True: "solv"}
#
#     def __init__(self, sheetnum: [int, str], solvate_il=ISL):
#         super().__init__()
#         self.__sheet_dict = SheetCoords().sheet_dict
#         self.__solv_str = self.__class__.solv_fname_dict[solvate_il]
#         self.__sheetnum = 0
#         self.sheetnum = sheetnum
#
#     @property
#     def sheet_array(self):
#         return self.__sheet_dict[self.sheetnum]
#
#     @property
#     def sheetnum(self):
#         return self.__sheetnum
#
#     @sheetnum.setter
#     def sheetnum(self, sheetnum):
#         assert isinstance(sheetnum, (int, str))
#         if int(sheetnum) in list(map(lambda n: int(n), self.__sheet_dict.keys())):
#             self.__sheetnum = int(sheetnum)
#         else:
#             raise IndexError(f"There is no sheet number {sheetnum}.")
#
#     @property
#     def uc_comb_str(self):
#         """Generate sheet description in box topology file."""
#         uc_comb_str = "\n".join(self.sheet_array) + "\n"
#         uc_comb_str = uc_comb_str.replace("\n", "   1\n")
#         return uc_comb_str
#
#     @cached_property
#     def IL_str(self):
#         return super().get_top_str()
#
#     @property
#     def filename(self):
#         return get_filename(self.__solv_str, str(self.sheetnum), ext="top")
#
#     def get_top_str(self):
#         sheet_top_str += self.uc_comb_str
#         if ISL:
#             sheet_top_str += self.IL_str
#         return sheet_top_str
#
#     @TopologyConstructorBase.write_function_output
#     def write_top(self):
#         return self.get_top_str()


# class BoxTopology(SheetTopology):
#     fname_prm_list = ["solv", "ions", "ext"]
#     n_sheets = N_SHEETS
#
#     def __init__(self, fname_prms=[], ff_path_key="local"):
#         super().__init__(sheetnum=0)
#         self.__solv_str = self.solv_fname_dict[ISL]
#         self.__fname_prm_list = [self.__solv_str]
#
#     @property
#     def fname_prm_list(self):
#         return self.__fname_prm_list
#
#     @singledispatchmethod
#     def get_fname_prm_list(self, prm):
#         if prm in BoxTopology.fname_prm_list and prm not in self.__fname_prm_list:
#             self.__fname_prm_list.append(prm)
#         return self.__fname_prm_list
#
#     @get_fname_prm_list.register(list)
#     def _(self, obj):
#         return list(map(lambda item: self.get_fname_prm_list(item), obj))
#
#     @get_fname_prm_list.register(str)
#     def _(self, obj):
#         return self.get_fname_prm_list(obj)
#
#     @property
#     def filename(self):
#         return get_filename(*self.fname_prm_list, ext="top")
#
#     def get_top_str(self):
#         topstr = ""
#         for sheet in range(N_SHEETS):
#             self.sheetnum = sheet
#             if SOLVATE and self.sheetnum == (N_SHEETS - 1):
#                 topstr += self.uc_comb_str
#             else:
#                 topstr += self.uc_comb_str + self.IL_str
#         return topstr
