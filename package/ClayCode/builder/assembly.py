import os
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

from MDAnalysis import Universe, Merge
from MDAnalysis.units import constants

from ClayCode.analysis.setup import check_insert_numbers
from ClayCode.builder.consts import GRO_FMT
from ClayCode.core.classes import FileFactory, FileFactory, GROFile
from ClayCode.core.gmx import run_gmx_insert_mols, run_gmx_solvate, run_gmx_genion_conc
from ClayCode.core.lib import add_resnum, add_ions_n_mols, write_insert_dat, center_clay, add_ions_neutral, \
    add_ions_conc, select_outside_clay_stack
from ClayCode.builder.topology import TopologyConstructorBase
from ClayCode.builder.solvent import Solvent
from ClayCode.core.utils import get_header, get_subheader

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
        self.__stack = None
        self.__box_ext = False
        logger.info(get_header(f'Building {self.args.name} model'
                    ))
        logger.info(
            f'{self.args.n_sheets} sheets\n'
            f'Sheet dimensions: '
                    f'{self.sheet.x_cells * self.sheet.uc_dimensions[0]:.2f} A X {self.sheet.y_cells * self.sheet.uc_dimensions[1]:.2f} A '
            f'({self.sheet.x_cells} unit cells X {self.sheet.y_cells} unit cells)\n'
                    f'Box height: {self.args.box_height:.1f} A')


    @property
    def extended_box(self):
        return self.__box_ext

    def solvate_clay_sheets(self):
        logger.info(get_subheader(f'2. Generating interlayer solvent.'))
        solvent = Solvent(x_dim=self.sheet.dimensions[0],
                          y_dim=self.sheet.dimensions[1],
                          n_mols=self.args.n_waters,
                          z_dim=self.args.il_solv_height)
        spc_file = self.get_filename('interlayer', suffix='.gro')
        solvent.write(outname=spc_file, topology=self.top)
        self.il_solv = spc_file
        logger.info(f'Writing interlayer sheet to {self.il_solv.name!r}\n')

    def rename_il_solv(self):
        # crdout = self.get_filename('solv', 'ions', 'iSL', suffix='.gro')
        il_u = Universe(str(self.il_solv))
        il_resnames = il_u.residues.resnames
        il_resnames = list(map(lambda resname: re.sub('SOL', 'iSL', resname), il_resnames))
        il_u.residues.resnames = il_resnames
        self.il_solv.universe = il_u
        self.il_solv.write()
        # il_u.atoms.write(str(crdout))
        # topout = crdout.top
        self.top.reset_molecules()
        self.top.add_molecules(il_u)
        self.top.write(self.il_solv.top)
        # self.top.write(topout)
        # self.il_solv = crdout

    def remove_il_solv(self):
        logger.info(f'Removing interlayer solvent')
        # crdout = self.get_filename('ions', suffix='.gro')
        il_u = Universe(str(self.il_solv))
        il_atoms = il_u.select_atoms('not resname SOL iSL')
        il_atoms.write(str(self.il_solv))
        # shutil.move(self.il_solv, crdout)
        # self.il_solv = crdout
        # topout = crdout.top
        self.top.reset_molecules()
        self.top.add_molecules(il_u)
        # self.top.write(topout)
        self.top.write(self.il_solv.top)

    def extend_box(self):
        if type(self.args.box_height) in [int, float]:
            if self.args.box_height > self.stack.universe.dimensions[2]:
                logger.info(f'Extending simulation box to {self.args.box_height:.1f} A')
                self.__box_ext = True
                ext_boxname = self.get_filename('ext', suffix='.gro')
                box_u = self.stack.universe
                box_u.universe.dimensions[2] = self.args.box_height
                # box_u = center_clay(box_u, crdout=ext_boxname)
                # box_u.atoms.pack_into_box(box_u.dimensions, inplace=True)
                box_u.atoms.write(ext_boxname)
                shutil.copy(self.stack.top,
                            ext_boxname.top)
                self.stack = ext_boxname
                logger.info(f'Saving extended box as {self.stack.stem!r}\n')
            else:
                self.__box_ext = False

    def remove_SOL(self):
        box_u = self.stack.universe
        box_u = box_u.select_atoms('not resname SOL')
        box_u.write(str(self.stack))
        self.top.reset_molecules()
        self.top.add_molecules(box_u)
        self.top.write(self.stack.top)

    def solvate_box(self):
        if self.extended_box is True:
            logger.info('Adding bulk solvation:')
            solv_box_crd = self.get_filename('solv', suffix='.gro')
            # shutil.copy(self.stack, solv_box_crd)
            # shutil.copy(self.stack.top, solv_box_crd.top)
            # self.remove_SOL()
            run_gmx_solvate(p=self.stack.top,
                            pp=solv_box_crd.top,
                            cp=self.stack,
                            radius=0.105,
                            scale=0.57,
                            o=solv_box_crd,
                            maxsol=0,
                            box='{} {} {}'.format(*self.stack.universe.dimensions[:3] * 0.10))
            self.stack = solv_box_crd
            solv_box_u = self.stack.universe
            not_sol = solv_box_u.select_atoms(f'not resname SOL')
            sol = solv_box_u.select_atoms(f'resname SOL')
            sol = self.select_atoms_outside_clay(sol, extra=1)
            logger.info(f'\tInserted {sol.n_atoms} {np.unique(sol.resnames)[0]} molecules')
            solv_box_u = not_sol + sol
            solv_box_u.write(str(self.stack))
            self.top.reset_molecules()
            self.top.add_molecules(solv_box_u)
            self.top.write(self.stack.top)
            logger.info(f'Saving solvated box as {self.stack.stem!r}\n')
        else:
            logger.info('Skipping bulk solvation.\n')

    def remove_bulk_ions(self):
        stack_u = self.stack.universe
        ion_sel = ' '.join(self.args.ff['ions']['atomtypes'].df['at-type'])
        il_ions = stack_u.select_atoms(f'resname {ion_sel}')
        il_ions = self.select_atoms_outside_clay(il_ions, extra=0)
        stack_atoms = stack_u.atoms - il_ions
        self.stack.universe = stack_atoms
        self.stack.write()
        self.top.reset_molecules()
        self.top.add_molecules(stack_atoms)
        self.top.write(self.stack.top)

    @property
    def clay(self):
        return self.stack.universe.select_atoms(f'resname {self.args._uc_stem}*')


    def select_atoms_outside_clay(self, atomgroup, extra=0):
        atom_group = select_outside_clay_stack(atom_group=atomgroup,
                                  clay=self.clay,
                                  extra=extra)
        # max_clay = self.clay_max
        # min_clay = self.clay_min
        # ag = atomgroup.select_atoms(f'(prop z > {max_clay + extra}) or (prop z < {min_clay - extra})')
        # ag = ag.residues.atoms
        return atom_group

    @property
    def clay_min(self):
        return np.min(self.clay.positions[:, 2])

    @property
    def clay_max(self):
        return np.max(self.clay.positions[:, 2])


    def add_bulk_ions(self):
        if self.extended_box is True:
            logger.info(f'Adding bulk ions:')
            outcrd = self.get_filename('solv', 'ions', suffix='.gro')
            shutil.copy(self.stack, outcrd)
            self.stack = outcrd
            self.remove_bulk_ions()
            self.top.reset_molecules()
            self.top.add_molecules(self.stack.universe)
            self.top.write(self.stack.top)
            ion_df = self.args.bulk_ion_df
            pion = self.args.default_bulk_pion[0]
            # pq = int(self.args.default_bulk_pion[1])
            nion = self.args.default_bulk_nion[0]
            # nq = int(self.args.default_bulk_nion[1])
            bulk_x, bulk_y, bulk_z = self.stack.universe.dimensions[:3]
            bulk_z -= np.abs(self.clay_max - self.clay_min)
            for ion, values in ion_df.iterrows():
                charge, conc = values
                n_ions = np.rint(bulk_z * bulk_x * bulk_y * constants['N_Avogadro'] * conc * 1E-27).astype(int) # 1 mol/L = 10^-27 mol/A
                logger.info(f'\tAdding {conc} mol/L ({n_ions} atoms) {ion} to bulk')
                replaced = add_ions_n_mols(odir=self.args.outpath,
                              crdin=self.stack,
                              topin=self.stack.top,
                              topout=self.stack.top,
                              ion=ion,
                              charge=int(charge),
                              n_atoms=n_ions
                              )
                logger.info(f"\t\tReplaced {replaced} SOL molecules with {ion}")
            logger.info(f"\tNeutralising with {pion} and {nion}")
            add_ions_neutral(odir=self.args.outpath,
                             crdin=self.stack,
                             topin=self.stack.top,
                             topout=self.stack.top,
                             nion=nion,
                             # nq=nq,
                             pion=pion,
                             # pq=pq)
                             )
            logger.info(f"\t\tReplaced {replaced} SOL molecules")
            logger.info(f'Saving solvated box with ions as {self.stack.stem!r}')
        else:
            logger.info('\tSkipping bulk ion addition.')



    def stack_sheets(self):
        try:
            il_crds = self.il_solv
            il_u = Universe(str(il_crds))
            il_solv = True
        except AttributeError:
            il_solv = False
        sheet_universes = []
        sheet_heights = []
        if il_solv is not False:
            logger.info(get_subheader('3. Assembling box'))
            logger.info('Combining clay sheets and interlayer')
        else:
            logger.info(f'Combining clay sheets\n')
        for sheet_id in range(self.args.n_sheets):
            self.sheet.n_sheet = sheet_id
            sheet_u = self.sheet.universe.copy()
            if il_solv is not False:
                il_u_copy = il_u.copy()
                if sheet_id == self.args.n_sheets - 1:
                    il_u_copy.residues.resnames = list(map(lambda resname: re.sub('iSL', 'SOL', resname),
                                                           il_u_copy.residues.resnames))
                il_u_copy.atoms.translate([0, 0, sheet_u.dimensions[2] + 1])
                new_dimensions = sheet_u.dimensions
                sheet_u = Merge(sheet_u.atoms, il_u_copy.atoms)
                sheet_u.dimensions = new_dimensions
                sheet_u.dimensions[2] = sheet_u.dimensions[2] + il_u_copy.dimensions[2] + 1
                sheet_u.atoms.translate([0, 0, sheet_id * (sheet_u.dimensions[2] + 1)])
                sheet_u.dimensions[2] = sheet_u.dimensions[2] + 1
            else:
                sheet_u.atoms.translate([0, 0, sheet_id * sheet_u.dimensions[2]])
            sheet_universes.append(sheet_u.atoms.copy())
            sheet_heights.append(sheet_u.dimensions[2])
        combined = Merge(*sheet_universes)
        combined.dimensions = sheet_u.dimensions
        new_dimensions = combined.dimensions
        new_dimensions[2] = np.sum(sheet_heights)
        new_dimensions[3:] = [90., 90., 90.]
        combined.dimensions = new_dimensions
        crdout = self.get_filename(suffix='.gro')
        # combined = center_clay(combined,
        #             crdout=str(crdout),
        #             uc_name=self.args._uc_stem)
        combined.atoms.pack_into_box(box=combined.dimensions, inplace=True)
        combined.atoms.write(str(crdout))
        topout = crdout.top
        self.top.reset_molecules()
        self.top.add_molecules(combined)
        self.top.write(str(topout))
        self.stack = crdout
        logger.info(f'Saving sheet stack as {self.stack.stem!r}\n')
    @property
    def stack(self):
        if self.__stack is not None:
            return self.__stack
        else:
            logger.info('No sheet stack filename defined.')

    @stack.setter
    def stack(self, stack):
        self.__stack = FileFactory(Path(stack).with_suffix('.gro'))

    def write_sheet_crds(self):
        logger.info(get_subheader(f'1. Generating clay sheets.'))
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

    def get_filename(self, *solv_ion_args, suffix=None, sheetnum: Optional[int]=None, tcb_spec=None) -> Path:
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
        try:
            suffix = suffix.strip('.')
            suffix = f'.{suffix}'
        except AttributeError:
            suffix = ''
        path = FileFactory(self.args.outpath / f'{fstem}{suffix}')
        return path

    @property
    def il_solv(self):
        if self.__solv is not None:
            return self.__solv
        else:
            logger.info(f'No solvation specified')

    @il_solv.setter
    def il_solv(self, il_solv):
        self.__solv = FileFactory(Path(il_solv).with_suffix('.gro'))

    def add_il_ions(self):
        if self.__solv is None:
            self.solvate_clay_sheets()
        logger.info(f'Adding interlayer ions:')# to {self.il_solv.name!r}')
        infile = self.il_solv
        # outfile = self.get_filename('solv', 'ions', suffix='gro')
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=self.il_solv.suffix) as temp_outfile:
            shutil.copy(infile, temp_outfile.name)
            dr = self.sheet.dimensions[:3] / 10
            dr[-1] *= 0.4
            if isinstance(self.args.n_il_ions, dict):
                for ion, n_ions in self.args.n_il_ions.items():
                    if n_ions != 0:
                        logger.info(
                            f"\tInserting {n_ions} {ion} atoms"
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
                                assert Path(posfile.name).is_file()
                                insert_out, insert_err = run_gmx_insert_mols(
                                    f=temp_outfile.name,
                                    ci=ion_gro.name,
                                    ip=posfile.name,
                                    nmol=n_ions,
                                    o=temp_outfile.name,
                                    replace='SOL',
                                    dr="{} {} {}".format(*dr),
                                )
                            center_clay(crdname=temp_outfile.name,
                                        crdout=temp_outfile.name,
                                        uc_name=ion)
                            _ = Universe(temp_outfile.name)
                            assert Path(temp_outfile.name).is_file()
                            replace_check = check_insert_numbers(
                                add_repl="Added", searchstr=insert_err
                            )
                            if replace_check != n_ions:
                                raise ValueError(
                                    f"Number of inserted molecules ({replace_check}) does not match target number "
                                    f"({n_ions})!"
                                )
            added_ions_u = Universe(temp_outfile.name)
            # solv_u = Universe(self.__solv.with_suffix('.gro'))
            self.top.reset_molecules()
            self.top.add_molecules(added_ions_u)
            self.top.write(self.il_solv.top)
            shutil.copy(temp_outfile.name, self.il_solv)
        # self.il_solv = outfile


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
    #         add_ions_n_mols(odir=self.args.outpath,
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

    @property
    def filename(self):
        return self.get_filename(suffix='.gro')

    def write_gro(self):
        filename: Path = self.filename
        if filename.is_file():
            logger.debug(f'\n{filename.parent}/{filename.name} already exists, creating backup.')
            self.backup(filename)
        gro_df = self.uc_data.gro_df
        sheet_df = pd.concat([gro_df.filter(regex=f'[A-Z0-9]+{uc_id}', axis=0) for uc_id in self.uc_array])
        sheet_df.reset_index(['atom-id'], inplace=True)
        sheet_df['atom-id'] = np.arange(1, len(sheet_df) + 1)
        sheet_df = sheet_df.loc[:, ['at-type', 'atom-id', 'x', 'y', 'z']]
        sheet_n_atoms = len(sheet_df)
        with open(filename, 'w') as grofile:
            grofile.write(f'{self.fstem} sheet {self.n_sheet}\n{sheet_n_atoms}\n')
            for idx, entry in sheet_df.reset_index().iterrows():
                line = entry.to_list()
                grofile.write(GRO_FMT.format(*re.split(r'(\d+)', line[0], maxsplit=1)[1:], *line[1:]))
            grofile.write(f'{self.format_dimensions(self.dimensions / 10)}\n')
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
        logger.info(f'Writing sheet {self.n_sheet} to {filename.name}')
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
