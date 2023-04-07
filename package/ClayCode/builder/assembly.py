import shutil
import re
from functools import cached_property
from typing import Optional, List
from pathlib import Path

import numpy as np
import pandas as pd

from ClayCode.builder.consts import GRO_FMT
from ClayCode.core.classes import FileFactory
from ClayCode.core.lib import add_resnum


class Sheet:
    ...

class Box:
    ...

class InterlayerSolvent:
    def __init__(self, x_):
        ...

class Builder:
    def __init__(self, build_args):
        self.args = build_args
        sheet = Sheet(self.args._uc_data,
                      uc_ids=self.args.sheet_uc_weights.index.values,
                      uc_numbers=self.args.sheet_uc_weights.values,
                      x_cells=self.args.x_cells,
                      y_cells=self.args.y_cells,
                      fstem=self.args.filestem,
                      outpath=self.args.outpath)
        for sheet_id in range(self.args.n_sheets):
            sheet.n_sheet = sheet_id
            sheet.write_gro()

    def get_filename(self, *solv_ion_args, suffix=None, sheetnum: Optional[int]=None, tcb_spec=None):
        if sheetnum is not None:
            sheetnum = f'_{int(sheetnum)}'
        else:
            sheetnum = ''
        if tcb_spec is not None:
            if tcb_spec in ['T', 'C', 'B']:
                tcb_spec = f'_{tcb_spec}'
            else:
                raise ValueError(f'{tcb_spec} was given for "tcb". Accepted '
                                 '"tcb_spec" values are "T", "C", "B".')
        else: tcb_spec = ''
        solv_ion_list = ['solv', 'ions']
        arg_list = [s for s in solv_ion_list if s in solv_ion_args]
        other_args = set(solv_ion_args) - set(arg_list)
        for a in sorted(other_args):
            arg_list.append(a)
        fstem = f'{self.args.filestem}_{sheetnum}_{tcb_spec}'
        fstem = '_'.join([fstem, *arg_list])
        return (self.args.outpath / f'{fstem}.{suffix}')

class Sheet:
    def __init__(self,
                 uc_data,
                 uc_ids: List[int],
                 uc_numbers: List[int],
                 x_cells: int,
                 y_cells: int,
                 fstem: str,
                 outpath: Path,
                 n_sheet: int = None):
        self.uc_data = uc_data
        self.uc_ids = uc_ids
        self.uc_numbers = uc_numbers
        self.dimensions = self.uc_data.dimensions[:3] * [x_cells, y_cells, 1]
        # self.dimensions[:3] *=
        self.fstem = fstem
        self.outpath = outpath
        self.__n_sheet = None
        self.__random = None



    def get_filename(self, suffix):
        return FileFactory(self.outpath / f'{self.fstem}_{self.n_sheet}{suffix}')

    @property
    def n_sheet(self):
        if self.__n_sheet is not None:
            return self.__n_sheet
        else:
            raise AttributeError(f'No sheet number set!')

    @n_sheet.setter
    def n_sheet(self, n_sheet):
        self.__n_sheet = n_sheet
        self.__random = np.random.default_rng(n_sheet)
        # self.__random = np.random.Generator(n_sheet)

    @property
    def random_generator(self):
        if self.__random is not None:
            return self.__random
        else:
            raise AttributeError(f'No sheet number set!')


    # @property
    # def __uc_array(self):
    #     return np.repeat(self.uc_ids, self.uc_numbers)

    @property
    def uc_array(self):
        uc_array = np.repeat(self.uc_ids, self.uc_numbers)
        self.random_generator.shuffle(uc_array)#self.__uc_array)
        return uc_array

    def write_gro(self):
        filename: Path = self.get_filename(suffix='.gro')
        if filename.is_file():
            print(f'\n{filename.parent}/{filename.name} already exists, creating backup.')
            self.backup(filename)
        gro_df = self.uc_data.gro_df
        sheet_df = pd.concat([gro_df.filter(regex=f'[A-Z0-9]+{uc_id}', axis=0) for uc_id in self.uc_array])
        n_atoms = [self.uc_data.n_atoms[uc_id] for uc_id in self.uc_array]
        # sheet_df.loc[:, ('x', 'y')] += self.uc_dimensions[:2] * self.xy_mask
        sheet_df.reset_index(['atom-id'], inplace=True)
        sheet_df['atom-id'] = np.arange(1, len(sheet_df) + 1)
        sheet_df = sheet_df.loc[:, ['at-type', 'atom-id', 'x', 'y', 'z']]
        sheet_n_atoms = len(sheet_df)
        with open(filename, 'w') as grofile:
            grofile.write(f'{self.fstem} sheet {self.n_sheet}\n{sheet_n_atoms}\n')
            for idx, entry in sheet_df.reset_index().iterrows():
                line = entry.to_list()
                grofile.write(GRO_FMT.format(*re.split(r'(\d+)', line[0], maxsplit=1)[1:], *line[1:]))
            grofile.write(f'{self.format_dimensions(self.dimensions)}\n')
        add_resnum(crdin=filename, crdout=filename)
        gro_universe = filename.universe
    @staticmethod
    def format_dimensions(dimensions):
        return ''.join([f'{dimension:12.4f}' for dimension in dimensions])
    @cached_property
    def uc_dimensions(self):
        return self.uc_data.dimensions

        # uc_array = np.repeat(self.tcb_namelist, uc_numbers_list)
        #     # tcb_str = tcbSpec(sheet)
        #     # tcb_namelist = TCBNameList(uc_names_list, tcb_str)
        #     # uc_array = np.repeat(tcb_namelist, uc_numbers_list)
        #     np.random.shuffle(uc_array)
        #     print(uc_array)
        #     sheet_uc_grofiles_dict[sheet] = uc_array
        #     print(sheet_uc_grofiles_dict[sheet])
        # np.save(sheetfile, [*sheet_uc_grofiles_dict.items()])
        # return sheet_uc_grofiles_dict


    @cached_property
    def xy_mask(self):
        xy_mask = pd.DataFrame({
            'x': np.tile(np.arange(0, self.x_cells).repeat(self.uc_n_atoms), self.y_cells),
            'y': np.arange(0, self.y_cells).repeat(self.uc_n_atoms * self.x_cells)})
        xy_box = uc_dim[:2]

# TODO: add n_atoms and uc data to match data

    def backup(self, filename: Path):
        sheets_backup: Path = filename.with_suffix(f'{filename.suffix}.1')
        backups = filename.parent.glob(f'*.{filename.suffix}.*')
        for backup in reversed(list(backups)):
            n_backup = int(backup.suffices[-1].strip('.'))
            new_backup = backup.with_suffix(f'{filename.suffix}.{n_backup + 1}')
            shutil.move(backup, new_backup)
        shutil.move(filename, sheets_backup)
