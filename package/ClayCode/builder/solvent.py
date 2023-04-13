import re
import shutil
import textwrap
from pathlib import Path
import subprocess as sp
import logging
from typing import Optional, Dict, Tuple

from ClayCode import MDP
from ClayCode.analysis.setup import check_insert_numbers
from ClayCode.core import lib
from ClayCode.core import gmx
from MDAnalysis import Universe
from MDAnalysis.units import constants

from ClayCode.core.gmx import run_gmx_solvate

logger = logging.getLogger(Path(__file__).stem)

class Solvent:
    solv_density = 1000E-24  # g/L 1L = 10E24 nm^3
    mw_sol = 18

    def __init__(self, x_dim=None, y_dim=None, z_dim=None, n_mols=None, n_ions=None):
        self.x_dim = x_dim
        self.y_dim = y_dim
        if z_dim is None and n_mols is not None:
            self.n_mols = n_mols
            self.z_dim = self.get_solvent_sheet_height(self.n_mols)
        elif n_mols is None and z_dim is not None:
            self.z_dim = z_dim / 10
            self.n_mols = self.get_solvent_sheet_mols(self.z_dim)
        else:
            raise ValueError(f'No sheet height or number of molecules specified')
        if n_ions is None:
            self.n_ions = 0
        else:
            self.n_ions = n_ions
            self.n_mols += self.n_ions
        self.n_mols = int(self.n_mols)

    def get_solvent_sheet_height(self, mols_sol):
        z_dim = (self.mw_sol * mols_sol) / (constants['N_Avogadro'] * self.x_dim / 10 * self.y_dim / 10 * self.solv_density)
        return z_dim

    def get_sheet_solvent_mols(self, z_dim):
        mols_sol = (z_dim * constants['N_Avogadro'] * self.x_dim / 10 * self.y_dim / 10 * self.solv_density) / (self.mw_sol)
        return round(mols_sol, 0)

    def top_str(self):
        return f'SOL\t{self.n_mols}\n'

    def write(self, outname, topology):
        spc_groname = Path(outname).with_suffix('.gro')
        spc_topname = Path(outname).with_suffix('.top')
        topology.write(spc_topname)
        _, solv = run_gmx_solvate(cs='spc216', maxsol=self.n_mols,
                                   o=spc_groname, p=spc_topname, scale=0.00010,
                                   box=f'{self.x_dim / 10} {self.y_dim / 10} {self.z_dim}')
        self.check_solvent_nummols(solv)
        logger.debug(f'Saving solvent sheet as {outname.stem!r}')



    # def write_spc_topfile(self, outtop, clay_ff_path):
        # """Generate header str for simulation box topology file."""
        # ff_head = textwrap.dedent(f"""
        #                                       ; include params for ClayFF_Fe
        #                                       #include "{clay_ff_path}/forcefield.itp"
        #                                       #include "{clay_ff_path}/ffnonbonded.itp"\n
        #                                       #include "{clay_ff_path}/ffbonded.itp"\n
        #                                        """)
        # solv_head = textwrap.dedent(f"""
        #                             ; include params for solvent
        #                             #include "{clay_ff_path}/interlayer_spc.itp"
        #                             #include "{clay_ff_path}/spc.itp"
        #                             """)
        # mol_head = textwrap.dedent(f"""
        #                    [ system ]
        #                    SPC water
        #                    [ molecules ]
        #                    ; Compound        #mols
        #                    """)
        # with open(outtop, 'w') as topfile:
        #     topfile.write(ff_head + solv_head + mol_head)

    def check_solvent_nummols(self, solvate_stderr):
        added_wat = re.search(r'(?<=Number of solvent molecules:)\s+(\d+)',
                              solvate_stderr).group(1)
        if int(added_wat) < self.n_mols:
            raise ValueError(f'With chosen box height, GROMACS was only able to '
                             f'insert {added_wat} instead of {self.n_mols} water '
                             f'molecules.\nIncrease box size!')






