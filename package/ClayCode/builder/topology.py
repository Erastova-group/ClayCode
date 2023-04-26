from functools import (
    cached_property,
    wraps,
)
from pathlib import Path
import logging

from MDAnalysis import ResidueGroup

from ..core.classes import ITPList

logger = logging.getLogger(Path(__file__).name)


class TopologyConstructorBase:
    def __init__(self, uc_data, ff):
        self.ff = ff
        self.uc_data = uc_data
        self.__mol_str = ""

    @cached_property
    def _ff_head(self):
        ff_head_str = "; selection FF params for clay, water and ions\n"
        ff_itps = ITPList(
            [*self.ff["clay"].itp_filelist, *self.ff["ions"].itp_filelist]
        )
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
        mol_head_str = (
            "[ system ]\n"
            f" {self.uc_data.name}\n"
            "[ molecules ]\n"
            "; Compound        nmols\n"
        )
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
        try:
            if type(universe) != ResidueGroup:
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
        except UnboundLocalError:
            logger.debug(f"Empty Universe, not adding any molecules to topology")
        # print(self.__mol_str)

    def reset_molecules(self):
        self.__mol_str = ""

    @property
    def mol_str(self):
        return self.__mol_str

    def write(self, fname):
        fname = Path(fname).with_suffix(".top")
        with open(fname, "w") as topfile:
            topfile.write(self.header + self.mol_str)
