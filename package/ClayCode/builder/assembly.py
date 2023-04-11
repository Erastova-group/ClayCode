import shutil
import re
import tempfile
from functools import cached_property, partial, partialmethod
from typing import Optional, List, Tuple, Dict
from pathlib import Path

import MDAnalysis
import numpy as np
import pandas as pd
import logging

from MDAnalysis import Universe

from ClayCode.analysis.setup import check_insert_numbers
from ClayCode.builder.consts import GRO_FMT
from ClayCode.core.classes import FileFactory
from ClayCode.core.gmx import run_gmx_insert_mols
from ClayCode.core.lib import add_resnum, add_n_ions, write_insert_dat
from ClayCode.builder.topology import TopologyConstructorBase
from ClayCode.builder.solvent import Solvent

logger = logging.getLogger(Path(__file__).name)

class Box:
    ...

class InterlayerSolvent(Solvent):
    def __init__(self, x_):
        ...

class Builder:
    def __init__(self, build_args):
        self.args = build_args
        self.sheet = Sheet(self.args._uc_data,
                      uc_ids=self.args.sheet_uc_weights.index.values,
                      uc_numbers=self.args.sheet_uc_weights.values,
                      x_cells=self.args.x_cells,
                      y_cells=self.args.y_cells,
                      fstem=self.args.filestem,
                      outpath=self.args.outpath)
        self.top = TopologyConstructorBase(self.args._uc_data,
                                      self.args.ff)
        self.__solv = None

    def solvate_clay_sheets(self):
        solvent = Solvent(x_dim=self.sheet.dimensions[0],
                          y_dim=self.sheet.dimensions[1],
                          n_mols=self.args.n_waters,
                          z_dim=self.args.il_solv_height)
        spc_file = self.get_filename('spc')
        solvent.write(outname=spc_file, topology=self.top)
        self.__solv = spc_file
        self.add_il_ions()


    def write_sheet_crds(self):
        for sheet_id in range(self.args.n_sheets):
            self.sheet.n_sheet = sheet_id
            self.sheet.write_gro()
        self.sheet.n_sheet = None

    def write_sheet_top(self):
        for sheet_id in range(self.args.n_sheets):
            self.top.reset_molecules()
            self.sheet.n_sheet = sheet_id
            self.top.add_molecules(self.sheet.universe)
            self.top.write(self.sheet.get_filename(suffix='.top'))
        self.sheet.n_sheet = None

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
        fstem = f'{self.args.filestem}{sheetnum}{tcb_spec}'
        fstem = '_'.join([fstem, *arg_list])
        return (self.args.outpath / f'{fstem}.{suffix}')

    @property
    def il_solv(self):
        if self.__solv is not None:
            return Path(self.__solv)
        else:
            logger.info(f'No solvation specified')

    def add_il_ions(self):#, infile: Path,
                      # outfile: Path,
                      # add_mols: Optional[Dict[str, int]] = None,
                      # replace: str = 'SOL',
                      # dr: Tuple[int, int, int] = (10, 10, 1)):
        infile = self.__solv.with_suffix('.gro')
        outfile = self.get_filename('solv', 'ions', suffix='gro')
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=outfile.suffix) as temp_outfile:
            shutil.copy(infile, temp_outfile.name)
            # ion_itp_list = self.args.ff['ions'].itp_filelist
            # ion_itp_molecules = np.ravel(np.argwhere(np.array(ion_itp_list.order) == 6))
            # assert len(ion_itp_molecules) == 1
            # ion_itp = ion_itp_list[ion_itp_molecules[0]]
            # insert_u = Universe(str(ion_itp), topology_format='ITP', infer_system=True)
            #
            # insert_u.atoms.positions = np.zeros_like((insert_u.atoms.n_atoms, 3))
            # insert_u.dimensions = self.sheet.dimensions
            dr = self.sheet.dimensions[[0,1,2]] / 10
            if isinstance(self.args.n_il_ions, dict):
                for ion, n_ions in self.args.n_il_ions.items():
                    if n_ions != 0:
                        logger.info(
                            f"Inserting {n_ions} {ion} atoms"
                        )
                        with tempfile.NamedTemporaryFile(suffix='.gro') as ion_gro:
                            ion_u = Universe.empty(n_atoms=1, n_residues=1, n_segments=1,
                                                   atom_resindex=[0], residue_segindex=[0],
                                                   trajectory=True)
                            ion_u.add_TopologyAttr('name', [ion])
                            ion_u.add_TopologyAttr('resname', [ion])
                            ion_u.dimensions = np.array([*self.sheet.dimensions, 90, 90, 90])
                            ion_u.atoms.positions = np.zeros((3,))
                            # ion_u = insert_u.select_atoms(f'resname {ion}')
                            ion_u.atoms.write(ion_gro.name)
                            # determine positions for adding ions
                            with tempfile.NamedTemporaryFile(suffix='.dat') as posfile:
                                write_insert_dat(n_mols=n_ions, save=posfile.name)
                                assert posfile.is_file()
                                insert_out, insert_err = run_gmx_insert_mols(
                                    f=temp_outfile.name,
                                    ci=ion_gro,
                                    ip=posfile,
                                    nmol=n_ions,
                                    o=temp_outfile.name,
                                    replace='SOL',
                                    dr="{} {} {}".format(dr),
                                )
                            u = Universe(temp_outfile.name)
                            assert temp_outfile.is_file()
                            replace_check = check_insert_numbers(
                                add_repl="Added", searchstr=insert_err
                            )
                            if replace_check != n_ions:
                                raise ValueError(
                                    f"Number of inserted molecules ({replace_check}) does not match target number "
                                    f"({n_ions})!"
                                )
            # shutil.copy(temp_outfile.name, outfile)


    # def add_il_ions(self):
    #     solvent_crds = Path(self.il_solv).with_suffix('.gro')
    #     crdout = self.get_filename('spc', 'ions', suffix='gro')
    #     topout = crdout.with_suffix('.top')
    #     # tmpcrd = tempfile.NamedTemporaryFile(suffix='.gro')
    #     # tmptop = tempfile.NamedTemporaryFile(suffix='.top')
    #     shutil.copy(solvent_crds.with_suffix('.gro'), crdout)
    #     shutil.copy(solvent_crds.with_suffix('.top'), topout)
    #     for ion, n_atoms in self.args.n_il_ions.items():
    #         # if self.args.ion_df.loc[ion, 'charges'] < 0:
    #         #     nname = ion
    #         #     nq = self.args.ion_df.loc[ion, 'charges']
    #         #     nn = n_atoms
    #         #     pname = 'Na'
    #         #     pq = 1
    #         #     pn = 0
    #         # elif self.args.ion_df.loc[ion, 'charges'] > 0:
    #         #     pname = ion
    #         #     pq = self.args.ion_df.loc[ion, 'charges']
    #         #     pn = n_atoms
    #         #     nname = 'Cl'
    #         #     nq = -1
    #         #     nn = 0
    #         # else:
    #         #     raise ValueError('Invalid charge!')
    #
    #         add_n_ions(odir=self.args.outpath,
    #                    crdin=crdout,
    #                    topin=topout, topout=topout,
    #                    ion=ion, n_atoms=n_atoms)


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
        self.x_cells = x_cells
        self.y_cells = y_cells
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
        # n_atoms = np.array([self.uc_data.n_atoms[uc_id] for uc_id in self.uc_array]).reshape(self.x_cells, self.y_cells)
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
        uc_n_atoms = np.array([self.uc_data.n_atoms[uc_id] for uc_id in self.uc_array]).reshape(self.x_cells, self.y_cells)
        x_repeats = lambda n_atoms: self.__cells_shift(n_atoms=n_atoms, n_cells=self.x_cells)
        y_repeats = lambda n_atoms: self.__cells_shift(n_atoms=n_atoms, n_cells=self.y_cells)
        x_pos_shift = np.ravel(np.apply_along_axis(x_repeats, arr=uc_n_atoms, axis=0), order='F')
        y_pos_shift = np.ravel(np.apply_along_axis(y_repeats, arr=uc_n_atoms, axis=1), order='F')
        new_positions = filename.universe.atoms.positions
        new_positions[:, 0] += self.uc_dimensions[0] * x_pos_shift
        new_positions[:, 1] += self.uc_dimensions[1] * y_pos_shift
        filename.universe.atoms.positions = new_positions
        filename.universe.atoms.write(str(filename.resolve()))

    def __cells_shift(self, n_cells, n_atoms):
        shift = np.atleast_2d(np.arange(n_cells)).repeat(n_atoms, axis=1)
        return shift
    @staticmethod
    def format_dimensions(dimensions):
        return ''.join([f'{dimension:12.4f}' for dimension in dimensions])
    @cached_property
    def uc_dimensions(self):
        return self.uc_data.dimensions

    @property
    def universe(self):
        return MDAnalysis.Universe(str(self.get_filename(suffix='.gro')))

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


    # @cached_property
    # def xy_mask(self):
    #     xy_mask = pd.DataFrame({
    #         'x': np.tile(np.arange(0, self.x_cells).repeat(self.uc_n_atoms), self.y_cells),
    #         'y': np.arange(0, self.y_cells).repeat(self.uc_n_atoms * self.x_cells)})
    #     xy_box = uc_dim[:2]

# TODO: add n_atoms and uc data to match data

    def backup(self, filename: Path):
        sheets_backup: Path = filename.with_suffix(f'{filename.suffix}.1')
        backups = filename.parent.glob(f'*.{filename.suffix}.*')
        for backup in reversed(list(backups)):
            n_backup = int(backup.suffices[-1].strip('.'))
            new_backup = backup.with_suffix(f'{filename.suffix}.{n_backup + 1}')
            shutil.move(backup, new_backup)
        shutil.move(filename, sheets_backup)
