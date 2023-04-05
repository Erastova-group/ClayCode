import shutil
from typing import Optional, List, Dict
from pathlib import Path

import numpy as np


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
                 uc_ids: List[int],
                 uc_numbers: List[int],
                 x_cells: int,
                 y_cells: int,
                 fstem: str,
                 outpath: Path,
                 n_sheet: int = None):
        self.uc_ids = uc_ids
        self.uc_numbers = uc_numbers
        self.uc_dim = self.dimesions = match_data.uc_dim
        self.dimensions[:3] *= [x_cells, y_cells, 1]
        self.fstem = fstem
        self.outpath = outpath
        self.__n_sheet = None
        self.__random = None

    def get_filename(self, suffix):
        return self.outpath / f'{self.fstem}_{self.n_sheet}{suffix}'

    @property
    def n_sheet(self):
        if self.__n_sheet is not None:
            return self.__n_sheet
        else:
            raise AttributeError(f'No sheet number set!')

    @n_sheet.setter
    def n_sheet(self, n_sheet):
        self.__n_sheet = n_sheet
        bit_generator = np.random.BitGenerator(n_sheet)
        self.__random = np.random.Generator(bit_generator)

    @property
    def random_generator(self):
        if self.__random is not None:
            return self.__random
        else:
            raise AttributeError(f'No sheet number set!')


    @cached_property
    def __uc_array(self):
        return np.repeat(self.uc_ids, self.uc_numbers)

    @property
    def uc_array(self):
        return self.__random.shuffle(self.__uc_array)

    def write_gro(self):
        filename: Path = self.get_filename(suffix='.gro')
        if filename.isfile():
            print(f'\n{sheetfile.name} already exists, creating backup.')
            self.backup(filename)
        sheet = self.uc_array
        np.save()

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
            'x': np.tile(np.arange(0, x_cells).repeat(uc_n_atoms), y_cells),
            'y': np.arange(0, y_cells).repeat(uc_n_atoms * x_cells)})
        xy_box = uc_dim[:2]

# TODO: add n_atoms and uc data to match data

    def backup(self, filename: Path):
        sheets_backup: Path = filename.with_suffix(f'{filename.suffix}.1')
        backups = filename.parent.glob(f'*.{filename.suffix}.*')
        for backup in reversed(backups):
            n_backup = int(backup.suffices[-1].strip('.'))
            new_backup = backup.with_suffix(f'{filename.suffix}.{n_backup + 1}')
            shutil.move(backup, new_backup)
        shutil.move(filename, sheets_backup)
