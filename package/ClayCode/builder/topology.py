import codecs
import logging
import os
from functools import cached_property, wraps
from pathlib import Path
from typing import List, Optional, Union

import MDAnalysis
from ClayCode.core.classes import ClayFFData, File, GROFile, ITPFile, ITPList
from ClayCode.core.gmx import GMXCommands
from ClayCode.core.utils import execute_shell_command
from MDAnalysis import AtomGroup, ResidueGroup

logger = logging.getLogger(__name__)


class TopologyConstructor:
    def __init__(self, uc_data, ff, gmx_commands=None):
        self._definitions = {}
        self.ff = ff
        self.uc_data = uc_data
        self.__mol_str = ""
        self._gmx_commands = gmx_commands

    @cached_property
    def _ff_head(self):
        ff_head_str = "; selection FF params for clay, water and ions\n"
        ff_itps = ITPList(
            [*self.ff["clay"].itp_filelist, *self.ff["ions"].itp_filelist]
        )
        for itp_file in ff_itps:
            ff_head_str += f'#include "{itp_file}"\n'
        return ff_head_str + "\n"

    def generate_restraints(
        self,
        crd: GROFile,
        name: str,
        ag: AtomGroup,
        gmx_commands=None,
        **kwargs,
    ) -> None:
        """Add a restraint definition.
        :param crd: GROFile instance to use for running gmx genrestr.
        :type crd: GROFile
        :param gmx_commands: GMXCommands instance to use for running gmx genrestr.
        :type gmx_commands: GMXCommands
        :param name: Name of the restraint definition.
        :type name: str
        :param ag: AtomGroup to restrain.
        :type ag: AtomGroup
        :param kwargs: Additional arguments for gmx genrestr.
        :type kwargs: Dict[str, Any]
        :return: None"""
        if gmx_commands is None:
            gmx_commands = self.gmx_commands
        index_name = File(name).with_suffix(".ndx")
        ag.write(index_name, name=name)
        restr_file = index_name.with_suffix(".itp")
        gmx_commands.run_gmx_genrestr(
            f=crd, n=index_name, o=restr_file, **kwargs
        )
        return restr_file

    @property
    def gmx_commands(self):
        if self._gmx_commands is None:
            self._gmx_commands = GMXCommands()
        return self._gmx_commands

    @gmx_commands.setter
    def gmx_commands(
        self, gmx_alias="gmx", mdp_template=None, mdp_defaults={}
    ):
        self._gmx_commands = GMXCommands(
            gmx_alias=gmx_alias,
            mdp_template=mdp_template,
            mdp_defaults=mdp_defaults,
        )

    @cached_property
    def _uc_head(self):
        uc_head_str = "; include clay unit cell topology\n"
        for uc_itp in sorted(self.uc_data.group_itp_filelist):
            distcontr_file = (
                uc_itp.parent
                / "constraints"
                / uc_itp.with_stem(f"{uc_itp.stem}_distconstr").name
            )
            if not distcontr_file.exists():
                logger.info(f"Generating restraints for {uc_itp.stem}")
                os.makedirs(distcontr_file.parent, exist_ok=True)
                uc_universe = GROFile(uc_itp.with_suffix(".gro")).universe
                restraint_dict = {}
                t_sheet = uc_universe.atoms.select_atoms(
                    f"name {'*|'.join(map(lambda x: x.upper(), ClayFFData().clayff_elements['T']))}* O[XB]*"
                )
                restr_file = self.generate_restraints(
                    uc_universe.filename,
                    distcontr_file.with_stem(
                        f"{uc_itp.stem}_T_sheet_disconst"
                    ),
                    t_sheet,
                    disre="",
                )
                restraint_dict["TSheets"] = restr_file
                restr_file = self.generate_restraints(
                    uc_universe.filename,
                    distcontr_file.with_stem(f"{uc_itp.stem}_clay_disconst"),
                    uc_universe.atoms,
                    disre="",
                )
                restraint_dict["ClaySheets"] = restr_file
                not_oh = uc_universe.atoms.select_atoms("not name OH* HO*")
                restr_file = self.generate_restraints(
                    uc_universe.filename,
                    distcontr_file.with_stem(
                        f"{uc_itp.stem}_clay_no_OH_sheet_disconst"
                    ),
                    not_oh,
                    disre="",
                )
                restraint_dict["ClayNoOHSheets"] = restr_file
                with open(distcontr_file, "w", encoding="utf-8") as f:
                    for (
                        restraint_def,
                        restraint_file,
                    ) in restraint_dict.items():
                        f.write(f"#ifdef {restraint_def}\n")
                        f.write(
                            execute_shell_command(
                                f"tail +3 {restraint_file}"
                            ).stdout
                        )
                        # with open(restraint_file, 'r', encoding='utf-32') as restr_f:
                        #     for line in restr_f:
                        #         try:
                        #             f.write(line)
                        #         except UnicodeDecodeError:
                        #             logger.error(f"Could not read {restraint_file}")
                        # f.write(f"\t#include {restr_file}\n")
                        f.write("#endif\n")
                        os.remove(restraint_file)
                        os.remove(restraint_file.with_suffix(".ndx"))
            uc_head_str += f'#include "{uc_itp}"\n'
            uc_head_str += f'#include "{distcontr_file}"\n'
        # uc_head_str += "".join(
        #     list(
        #         map(
        #             lambda uc_itp: f"""#include "{uc_itp}"\n""",
        #             sorted(self.uc_data.group_itp_filelist),
        #         )
        #     )
        # )
        return uc_head_str + "\n"

    @property
    def definitions(self):
        return self._definitions

    def add_definition(
        self,
        name: str,
        actions: Union[List[str], str],
        else_actions: Optional[Union[str, List[str]]] = None,
    ):
        if not isinstance(name, str):
            logger.error(f"Definition name must be a string, not {type(name)}")
        else:
            if actions in [None, ""] and name in self._definitions:
                self._definitions.pop(name)
            elif isinstance(actions, list):
                actions = "\n".join(actions)
            self._definitions[name] = f"#ifdef {name}\n{actions}"
            if else_actions is not None:
                if not isinstance(else_actions, str):
                    else_actions = "\n".join(else_actions)
                self._definitions[name] += f"\n#else\n{else_actions}\n"
            self._definitions[name] += "#endif\n"

    def remove_definition(self, name: str):
        if name in self._definitions:
            self._definitions.pop(name)

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
                    print(
                        f"No filename selected, writing topology to {self.filename}"
                    )
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

    def add_molecules(self, universe: MDAnalysis.Universe) -> None:
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
            logger.debug(
                "Empty Universe, not adding any molecules to topology"
            )
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
            if self.definitions:
                topfile.write("[ intermolecular_interactions ]\n")
                topfile.write("\n".join(self.definitions.values()))
