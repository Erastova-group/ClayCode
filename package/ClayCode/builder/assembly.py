#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
r""":mod:`ClayCode.builder.assembly` --- Assembly of clay models
==============================================================="""
from __future__ import annotations

import copy
import itertools
import logging
import math
import os
import re
import shutil
import sys
import tempfile
from functools import cached_property, partialmethod
from pathlib import Path
from typing import Any, Callable, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import unicodeit
from ClayCode.builder.claycomp import InterlayerIons, UCData
from ClayCode.builder.topology import TopologyConstructor
from ClayCode.core.classes import (
    Dir,
    FileFactory,
    GROFile,
    TOPFile,
    set_mdp_freeze_groups,
    set_mdp_parameter,
)
from ClayCode.core.consts import ANGSTROM, LINE_LENGTH
from ClayCode.core.gmx import (
    GMXCommands,
    add_gmx_args,
    check_box_lengths,
    gmx_command_wrapper,
)
from ClayCode.core.lib import (
    add_ions_n_mols,
    add_ions_neutral,
    add_resnum,
    center_clay,
    check_insert_numbers,
    get_system_n_atoms,
    run_em,
    select_outside_clay_stack,
    write_insert_dat,
)
from ClayCode.core.utils import backup_files, get_header, get_subheader
from ClayCode.data.consts import FF, GRO_FMT, MDP, MDP_DEFAULTS
from MDAnalysis import AtomGroup, Merge, ResidueGroup, Universe
from MDAnalysis.lib.mdamath import triclinic_box, triclinic_vectors
from MDAnalysis.units import constants
from numpy._typing import NDArray

__all__ = ["Builder", "Sheet", "Solvent"]

logger = logging.getLogger(__name__)


class Builder:
    """Clay model builder class.
    :param build_args: Arguments for building clay model.
    :type build_args: Type["BuildArgs"]"""

    __tmp_outpath: Type[
        tempfile.TemporaryDirectory
    ] = tempfile.TemporaryDirectory()
    __tmp_file: Type[
        tempfile.NamedTemporaryFile
    ] = tempfile.NamedTemporaryFile(dir=__tmp_outpath.name, delete=False)

    def __init__(self, build_args: Type["BuildArgs"]):
        self.args = build_args
        self.sheet = Sheet(
            self.args._uc_data,
            uc_ids=self.args.sheet_uc_weights.index.values,
            uc_numbers=self.args.sheet_uc_weights.values,
            x_cells=self.args.x_cells,
            y_cells=self.args.y_cells,
            fstem=self.args.filestem,
            outpath=self.args.outpath,
            debug=self.args.debug_run,
        )
        self.top = TopologyConstructor(self.args._uc_data, self.args.ff)
        self.__il_solv: Union[None, GROFile] = None
        self.__il: Union[None, GROFile] = None
        self.__stack: Union[None, GROFile] = None
        self.__box_ext: bool = False
        logger.info(get_header(f"Building {self.args.name} model"))
        logger.finfo(f"{self.args.n_sheets} sheets")
        x_dim: float = self.sheet.x_cells * self.sheet.uc_dimensions[0]
        y_dim: float = self.sheet.y_cells * self.sheet.uc_dimensions[1]
        logger.finfo(
            kwd_str="Sheet dimensions: ",
            message=f"{x_dim:.2f} {ANGSTROM} X {y_dim:.2f} {ANGSTROM} "
            f"({self.sheet.x_cells} unit cells X {self.sheet.y_cells} unit cells)",
        )
        if self.args.box_height > self.sheet.uc_dimensions[2]:
            logger.finfo(
                kwd_str="Box height: ",
                message=f"{self.args.box_height:.1f} {ANGSTROM}",
            )
        else:
            logger.finfo("Will set box height to clay height")
        self.gmx_commands: GMXCommands = GMXCommands(
            gmx_alias=self.args.gmx_alias
        )
        self.em_prms = self.gmx_commands.mdp_defaults.copy()
        self.em_prms.update(self.args.mdp_parameters["EM"])
        check_box_lengths(
            self.em_prms, [self.sheet.dimensions[0], self.sheet.dimensions[1]]
        )

    def get_solvated_il(self):
        self.construct_solvent(
            solvate=self.args.il_solv,
            ion_charge=self.args.match_charge["tot"],
            ion_add_func=self.add_il_ions,
            solvate_add_func=self.solvate_clay_sheets,
            solvent_remove_func=self.remove_il_solv,
            il_rename_func=self.rename_il,
            backup=self.args.backup,
            solvent=True,
            ions=False,
        )

    def get_il_ions(self, sheet_id=None):
        self.construct_solvent(
            solvate=self.args.il_solv,
            ion_charge=self.args.match_charge["tot"],
            solvate_add_func=self.solvate_clay_sheets,
            ion_add_func=self.add_il_ions,
            solvent_remove_func=self.remove_il_solv,
            solvent_rename_func=self.rename_il_solv,
            il_rename_func=self.rename_il,
            backup=self.args.backup,
            ions=True,
            solvent=False,
        )

    @property
    def extended_box(self) -> bool:
        """
        :return: Whether a bulk space has been added to the clay stack.
        :rtype: Bool"""
        return self.__box_ext

    def solvate_clay_sheets(self, backup: bool = False) -> None:
        """Generate interlayer solvent sheets.
        :param backup: Whether to back up existing files.
        :type backup: bool
        :return: None"""
        logger.info(get_subheader("2. Generating interlayer solvent."))
        solvent: Solvent = Solvent(
            x_dim=self.sheet.dimensions[0],
            y_dim=self.sheet.dimensions[1],
            n_mols=self.args.n_waters,
            z_dim=self.args.il_solv_height,
            gmx_commands=self.gmx_commands,
            z_padding=self.args.z_padding,
        )
        spc_file: GROFile = self.get_filename("interlayer", suffix=".gro")
        if backup:
            backup_files(self.args.outpath / spc_file.name)
            backup_files(self.args.outpath / spc_file.top.name)
        solvent.write(spc_name=spc_file, topology=self.top)
        self.il_solv: GROFile = spc_file

    @staticmethod
    def construct_solvent(
        solvate: bool,
        ion_charge: Union[int, float],
        solvate_add_func: Callable[[bool], Any],
        ion_add_func: Callable[[], Any],
        solvent_remove_func: Callable[[], Any],
        solvent_rename_func: Optional[Callable[[], Any]] = None,
        il_rename_func: Optional[Callable[[], Any]] = None,
        backup: bool = False,
        solvent=True,
        ions=True,
    ) -> None:
        """Construct solvent workflow.
        :param solvate: Whether to solvate the clay stack.
        :type solvate: bool
        :param ion_charge: Charge of the system.
        :type ion_charge: Union[int, float]
        :param solvate_add_func: Function to add solvent.
        :type solvate_add_func: Callable[[bool], Any]
        :param ion_add_func: Function to add ions.
        :type ion_add_func: Callable[[], Any]
        :param solvent_remove_func: Function to remove solvent.
        :type solvent_remove_func: Callable[[], Any]
        :param solvent_rename_func: Function to rename solvent.
        :type solvent_rename_func: Optional[Callable[[], Any]]
        :param backup: Whether to backup existing files.
        :type backup: bool
        :return: None"""
        if not solvate and ion_charge == 0:
            pass
        elif solvate or ion_charge != 0:
            if solvent:
                solvate_add_func(backup=backup)
            if ion_charge != 0 and ions:
                ion_add_func()
                if not solvate:
                    solvent_remove_func()
            if ion_charge == 0 and solvent and solvate:
                solvent_rename_func()
            elif (
                ion_charge != 0
                and ions
                and solvate
                and il_rename_func is not None
            ):
                il_rename_func()

    def rename_solv(self, solv: Union["il", "il_solv"]) -> None:
        """Rename interlayer solvent residues from 'SOl' to 'iSL'.
        :return: None"""
        gro_file = getattr(self, solv)
        il_u: Universe = Universe(str(gro_file))
        il_resnames: NDArray = il_u.residues.resnames
        il_resnames: list = list(
            map(lambda resname: re.sub("SOL", "iSL", resname), il_resnames)
        )
        il_u.residues.resnames: NDArray = il_resnames

        gro_file.universe: Universe = il_u
        gro_file.write(topology=self.top)
        setattr(self, solv, gro_file)

    rename_il_solv = partialmethod(rename_solv, solv="il_solv")
    rename_il = partialmethod(rename_solv, solv="il")

    # def rename_il_solv(self) -> None:
    #     """Rename interlayer solvent residues from 'SOl' to 'iSL'.
    #     :return: None"""
    #     il_u: Universe = Universe(str(self.il_solv))
    #     il_resnames: NDArray = il_u.residues.resnames
    #     il_resnames: list = list(
    #         map(lambda resname: re.sub("SOL", "iSL", resname), il_resnames)
    #     )
    #     il_u.residues.resnames: NDArray = il_resnames
    #     self.il_solv.universe: Universe = il_u
    #     self.il_solv.write(topology=self.top)
    #     self.il_solv = self.il_solv

    def run_em(
        self,
        freeze_clay: Optional[
            Union[List[Union[Literal["Y"], Literal["N"]]], bool]
        ] = False,
        # ["Y", "Y", "Y"],
        backup=False,
    ) -> None:
        """Run energy minimisation.
        :param freeze_clay: Whether to freeze the clay stack during energy minimisation. If a list of three booleans is provided, the corresponding dimensions will be frozen. If True, all three dimensions will be frozen. If False, no dimensions will be frozen.
        :type freeze_clay: Optional[Union[List[Union[Literal["Y"], Literal["N"]]], bool]]
        :param backup: Whether to back up existing files.
        :type backup: bool
        :return: None"""
        logger.info(get_subheader("Minimising energy"))
        # outgro = None
        outname_list = [f"{self.stack.stem.strip('_pre_em').strip('_em')}_em"]
        ndx_list = [None]
        if freeze_clay:
            if isinstance(freeze_clay, bool):
                freeze_dims_list = [["Y", "Y", "Y"]]
            else:
                freeze_dims_list = [freeze_clay]
            freeze_grps_list = [np.unique(self.clay.residues.resnames)]
        else:
            logger.finfo(
                f"1. Energy minimisation with frozen clay hydroxyl atom positions."
            )
            freeze_grps = "clay_hydroxyl_groups"
            freeze_dims = ["Y", "Y", "Y"]
            self.gmx_commands.run_gmx_make_ndx_with_new_sel(
                f=self.stack,
                o=self.stack.with_suffix(".ndx"),
                sel_str="r "
                + " ".join(np.unique(self.clay.residues.resnames))
                + " & ! a OH* HO*",
                sel_name=freeze_grps,
            )
            freeze_grps_list = [freeze_grps, None]
            freeze_dims_list = [freeze_dims, None]
            outname_list.insert(
                0, f"{self.stack.stem.strip('_pre_em').strip('_em')}_pre_em"
            )
            ndx_list.insert(0, self.stack.with_suffix(".ndx"))
        for freeze_grps, freeze_dims, outname, ndx in zip(
            freeze_grps_list, freeze_dims_list, outname_list, ndx_list
        ):
            result = run_em(
                crdin=self.stack,
                topin=self.stack.top,
                odir=self.args.outpath,
                outname=outname,
                gmx_commands=self.gmx_commands,
                freeze_grps=freeze_grps,
                freeze_dims=freeze_dims,
                ndx=ndx,
            )
            outgro = GROFile(self.args.outpath / f"{outname}.gro")
            if result is not None and outgro.exists():
                self.stack = outgro
            else:
                logger.ferror(f"Energy minimisation failed.")
                outgro = None
                break
            # freeze_grps = None
            # freeze_dims = None
        # if outgro is not False:
        #     result = run_em(
        #         crdin=self.stack,
        #         topin=self.stack.top,
        #         odir=self.args.outpath,
        #         outname=f"{self.stack.stem.strip('_pre_em').strip('_em')}_em",
        #         gmx_commands=self.gmx_commands,
        #         freeze_grps=freeze_grps,
        #         freeze_dims=freeze_dims,
        #     )
        #     if result is not None:
        #         outgro = self.args.outpath / f"{outname}.gro"
        #         if GROFile(outgro).exists():
        #             self.stack = outgro
        #         else:
        #             outgro = False
        #             logger.ferror(f"Energy minimisation failed.")
        if outgro is not None:
            outpath = Dir(self.args.outpath)
            crd_top_files = [
                *outpath.gro_filelist,
                *outpath.itp_filelist,
                *outpath._get_filelist(ext=".top"),
                *outpath._get_filelist(ext=".csv"),
                *outpath._get_filelist(ext=".mdp"),
                *outpath._get_filelist(ext=".edr"),
                *outpath._get_filelist(ext=".trr"),
                *outpath._get_filelist(ext=".log"),
            ]
            em_files = []
            backups = []
            for file in outpath.iterdir():
                if file not in crd_top_files and not file.is_dir():
                    file.unlink(missing_ok=True)
                else:
                    if file.stem.split("_")[-1] == "em":
                        if outpath.name != "EM":
                            em_path = outpath / "EM"
                            os.makedirs(em_path, exist_ok=True)
                            if backup:
                                backups.append(
                                    backup_files(
                                        new_filename=em_path / file.name
                                    )
                                )
                            new_file = em_path / file.name
                            try:
                                shutil.copy2(file, new_file)
                            except shutil.SameFileError:
                                pass
                            else:
                                file.unlink(missing_ok=True)
                                file = new_file
                        if file.suffix == ".gro":
                            self.stack = file
                            logger.finfo(
                                f"Writing final output from energy minimisation to {str(file.parent)!r}:"
                            )
                        em_files.append(file.name)
            if backups:
                logger.finfo(
                    "\t"
                    + "\n\t".join([backup for backup in backups if backup]),
                    initial_linebreak=True,
                )
            em_files = "'\n\t - '".join(sorted(em_files))
            logger.info(f"\t - '{em_files}'")
        return result

    def conclude(self):
        """Conclude model setup.
        Copy final files to output directory.
        :return: None"""
        logger.info(get_subheader("Finishing up"))
        self.stack: GROFile = self.args.outpath / self.stack.name
        add_resnum(crdin=self.stack, crdout=self.stack)
        self.__tmp_outpath.cleanup()
        logger.debug(
            f"Writing final coordinates and topology to {self.stack.name!r} and {self.stack.top.name!r}"
        )
        logger.set_file_name(final="builder")
        logger.finfo(
            f"Log for this setup written to {str(logger.logfilename)!r}"
        )
        logger.info(get_header(f"{self.args.name} model setup complete"))

    def remove_il_solv(self) -> None:
        """Remove interlayer solvent if interlayer needs to have ions but no solvent.
        :return: None"""
        logger.finfo("Removing interlayer solvent", initial_linebreak=True)
        il_u: Universe = self.il.universe  # Universe(str(self.il_solv))
        il_atoms: AtomGroup = il_u.select_atoms("not resname SOL iSL")
        self.il.universe = il_atoms
        self.il.write(topology=self.top)
        self.il = self.il

    def extend_box(self, backup) -> None:
        """Extend simulation box to specified height.
        :param backup: Whether to back up existing files.
        :type backup: bool
        :return: None"""
        if type(self.args.box_height) in [int, float]:
            if self.args.box_height > self.stack.universe.dimensions[2]:
                logger.finfo(
                    f"Extending simulation box to {self.args.box_height:.1f} {ANGSTROM}"
                )
                self.remove_SOL()
                self.center_clay_in_box()
                self.__box_ext: bool = True
                ext_boxname: GROFile = self.get_filename("ext", suffix=".gro")
                if backup:
                    backup_files(self.args.outpath / ext_boxname.name)
                    backup_files(self.args.outpath / ext_boxname.top.name)
                self.stack.reset_universe()
                box_u: Universe = self.stack.universe
                box_u.universe.dimensions[2] = self.args.box_height
                self.stack: GROFile = ext_boxname
                self.stack.universe = box_u
                self.stack.write(topology=self.top)
                logger.finfo("Centering clay in box", initial_linebreak=False)
                self.center_clay_in_box()
                logger.finfo(
                    f"Saving extended box as {self.stack.stem!r}\n",
                    initial_linebreak=False,
                )
            else:
                self.__box_ext: bool = False
        check_box_lengths(self.em_prms, self.stack.universe.dimensions[:3])

    def remove_SOL(self) -> None:
        """Remove solvent molecules from clay stack.
        :return: None"""
        box_u: Universe = self.stack.universe
        box_u: AtomGroup = box_u.select_atoms("not resname SOL")
        self.stack.universe = box_u
        self.stack.write(topology=self.top)
        add_resnum(crdin=self.stack, crdout=self.stack)
        self.stack.write(topology=self.top)

    def solvate_box(self, extra=1.5, backup=False) -> None:
        """Solvate bulk space.
        :param extra: Offset for solvation.
        :type extra: float
        :param backup: Whether to back up existing files.
        :type backup: bool
        :return: None"""

        if self.extended_box is True:
            logger.finfo("Adding bulk solvation:")
            solv_box_crd: GROFile = self.get_filename("solv", suffix=".gro")
            if backup:
                backup_files(self.args.outpath / solv_box_crd.name)
                backup_files(self.args.outpath / solv_box_crd.top.name)
            self.remove_SOL()
            self.gmx_commands.run_gmx_solvate(
                p=self.stack.top,
                pp=solv_box_crd.top,
                cp=self.stack,
                radius=0.105,
                scale=0.57,
                o=solv_box_crd,
                maxsol=0,
                box="{} {} {}".format(
                    *self.stack.universe.dimensions[:3] * 0.10
                ),
            )
            solv_box_u: Universe = solv_box_crd.universe.copy()
            not_sol: AtomGroup = solv_box_u.select_atoms("not resname SOL")
            sol: AtomGroup = solv_box_u.select_atoms("resname SOL")
            _sol = self.select_molecules_outside_clay(sol, extra=extra)
            logger.finfo(
                f"\tInserted {_sol.n_atoms} {np.unique(_sol.resnames)[0]} molecules"
            )
            sol = _sol
            solv_box_u: AtomGroup = not_sol + sol
            solv_box_crd.universe: Union[
                Universe, AtomGroup, ResidueGroup
            ] = solv_box_u
            solv_box_crd.write(self.top)
            self.stack: GROFile = solv_box_crd
            # self.stack.universe = solv_box_u
            self.stack.write(self.top)
            logger.finfo(f"Saving solvated box as {self.stack.stem!r}\n")
        else:
            logger.finfo("Skipping bulk solvation.\n")

    @cached_property
    def __ion_sel_str(self) -> str:
        """String of ion residue names.
        :return: String of ion residue names.
        :rtype: str"""
        return " ".join(self.args.ff["ions"]["atomtypes"].df["at-type"])

    def remove_bulk_ions(self):
        """Remove ions from bulk space.
        :return: None"""
        stack_u = self.stack.universe
        il_ions = stack_u.select_atoms(f"resname {self.__ion_sel_str}")
        il_ions = self.select_molecules_outside_clay(il_ions, extra=0)
        stack_atoms = stack_u.atoms - il_ions
        self.stack.universe = stack_atoms
        self.stack.write(topology=self.top)

    @property
    def clay(self):
        """Clay atoms.
        :return: AtomGroup of clay atoms.
        :rtype: AtomGroup"""
        return self.stack.universe.select_atoms(
            f"resname {self.args.uc_stem}*"
        )

    def select_molecules_outside_clay(
        self, atomgroup: AtomGroup, extra: Union[int, float] = 0
    ) -> AtomGroup:
        """Select molecules outside clay stack.
        :param atomgroup: AtomGroup to select from.
        :type atomgroup: AtomGroup
        :param extra: Offset to add to clay boundaries.
        :type extra: Union[int, float]
        :return: AtomGroup of molecules outside clay stack.
        :rtype: AtomGroup"""
        atom_group: AtomGroup = select_outside_clay_stack(
            atom_group=atomgroup, clay=self.clay, extra=extra
        )
        residue_groups = atom_group.split("residue")
        for residue_group in residue_groups:
            if residue_group.n_atoms != residue_group.residues.atoms.n_atoms:
                atom_group -= residue_group
        return atom_group

    @property
    def clay_min(self) -> float:
        """Minimum z-coordinate of clay atoms.
        :return: Minimum z-coordinate of clay atoms.
        :rtype: float"""
        return np.min(self.clay.positions[:, 2])

    @property
    def clay_max(self) -> float:
        """Maximum z-coordinate of clay atoms.
        :return: Maximum z-coordinate of clay atoms.
        :rtype: float"""

        return np.max(self.clay.positions[:, 2])

    @staticmethod
    def get_formatted_ion_charges(ion_type, ion_charge) -> str:
        """Get superscript formatted ion charges.
        :param ion_type: Ion type.
        :type ion_type: str
        :param ion_charge: Ion charge.
        :type ion_charge: float
        :return: formatted ion charges.
        :rtype: str
        """
        if ion_charge != 0:
            ion_charge_str = f"{ion_charge:+.0f}"
            if abs(ion_charge) == 1:
                ion_charge_str = ion_charge_str[0]
            else:
                ion_charge_str = ion_charge_str[1:] + ion_charge_str[0]
            ion_charge_str = unicodeit.replace(
                "^" + "^".join([i for i in ion_charge_str])
            )
        else:
            ion_charge_str = ""
        return ion_charge_str

    def add_bulk_ions(self, backup=False) -> None:
        """Add bulk ions to bulk space.
        :param backup: Whether to back up existing files.
        :type backup: bool
        """
        if self.extended_box is True:
            logger.finfo("Adding bulk ions:")
            outcrd: GROFile = self.get_filename("solv", "ions", suffix=".gro")
            if backup:
                backup_files(self.args.outpath / outcrd.name)
                backup_files(self.args.outpath / outcrd.top.name)
            shutil.copy(self.stack, outcrd)
            outcrd.write(topology=self.top)
            self.stack: GROFile = outcrd
            logger.debug(
                f"before n_atoms: {self.stack.universe.atoms.n_atoms}"
            )
            self.remove_bulk_ions()
            logger.debug(f"after n_atoms: {self.stack.universe.atoms.n_atoms}")
            # TODO: use monovalent bulk ion from Ions class
            ion_df: pd.DataFrame = self.args.bulk_ion_df
            pion: str = self.args.default_bulk_pion[0]
            nion: str = self.args.default_bulk_nion[0]
            bulk_x, bulk_y, bulk_z = self.stack.universe.dimensions[:3]
            bulk_z -= np.abs(self.clay_max - self.clay_min)
            replaced: int = 0
            ion_charge = 0
            for ion, values in ion_df.iterrows():
                charge, conc = values
                n_ions: int = np.rint(
                    bulk_z
                    * bulk_x
                    * bulk_y
                    * constants["N_Avogadro"]
                    * conc
                    * 1e-27
                ).astype(
                    int
                )  # 1 mol/L = 10^-27 mol/A
                logger.finfo(
                    f"\tAdding {conc} mol/L ({n_ions} atoms) {ion}{self.get_formatted_ion_charges(ion, charge)} ions to bulk"
                )
                logger.debug(
                    f"before n_atoms: {self.stack.universe.atoms.n_atoms}"
                )
                ion_charge += int(charge * n_ions)
                # TODO: use ion df with actual numbers that consider charge
                replaced += add_ions_n_mols(
                    odir=self.__tmp_outpath.name,
                    crdin=self.stack,
                    topin=self.stack.top,
                    ion=ion,
                    charge=int(charge),
                    n_atoms=n_ions,
                    gmx_commands=self.gmx_commands,
                )
                logger.debug(
                    f"after n_atoms: {self.stack.universe.atoms.n_atoms}"
                )
                self.stack.reset_universe()
                self.stack.write(self.top)
            excess_charge = int(self.args.il_ions.clay_charge + ion_charge)
            logger.finfo(f"Neutralising charge:", initial_linebreak=True)
            if excess_charge != 0:
                neutral_bulk_ions = InterlayerIons(
                    excess_charge,
                    ion_ratios=self.args.bulk_ions.df["conc"].to_dict(),
                    n_ucs=1,
                    neutral=True,
                )
                for ion, values in neutral_bulk_ions.df.iterrows():
                    charge, n_ions = values
                    replaced += add_ions_n_mols(
                        odir=self.__tmp_outpath.name,
                        crdin=self.stack,
                        topin=self.stack.top,
                        ion=ion,
                        charge=int(charge),
                        n_atoms=n_ions,
                        gmx_commands=self.gmx_commands,
                    )
                    logger.debug(
                        f"after n_atoms: {self.stack.universe.atoms.n_atoms}"
                    )
                    logger.finfo(
                        f"Added {n_ions} {ion}{self.get_formatted_ion_charges(ion, charge)} ions to bulk",
                        indent="\t",
                    )
                    self.stack.reset_universe()
                    self.stack.write(self.top)
            logger.debug(f"n_atoms: {self.stack.universe.atoms.n_atoms}")
            logger.finfo(f"Replaced {replaced} SOL molecules", indent="\t")
            logger.finfo(
                f"Saving solvated box with ions as {self.stack.stem!r}",
                initial_linebreak=True,
            )
            self.stack.reset_universe()
            self.stack.write(self.top)
            processed_top = Path("processed.top")
            processed_top.unlink(missing_ok=True)
        else:
            logger.finfo("\tSkipping bulk ion addition.")

    def stack_sheets(self, extra=2.0, backup=False) -> None:
        """Stack clay sheets.
        :param extra: Offset for stacking.
        :type extra: float
        :param backup: Whether to back up existing files.
        :type backup: bool
        :return: None"""
        try:
            il_crds: GROFile = self.il_solv
            il_u: Universe = il_crds.universe

            il_u = self.unwrap_il_solv(il_u)
            self.il_solv.universe = il_u
            il_solv = True
        except AttributeError:
            il_solv = False
        sheet_universes = []
        sheet_heights = []
        if il_solv is not False:
            logger.info(get_subheader("3. Assembling box"))
            logger.finfo("Combining clay sheets and interlayer:")
        else:
            logger.finfo("Combining clay sheets")
        for sheet_id in range(self.args.n_sheets):
            logger.finfo(
                f"Sheet {sheet_id + 1} of {self.args.n_sheets}:",
                initial_linebreak=False,
                indent="\t",
            )
            self.sheet.n_sheet = sheet_id
            sheet_u = self.sheet.universe.copy()
            sheet_u.dimensions[2] = sheet_u.dimensions[2] + extra
            if il_solv is not False:
                self.get_il_ions(sheet_id=sheet_id)
                il_u_copy = self.il.universe.copy()
                il_u_copy = self.unwrap_il_solv(il_u_copy)
                il_ions = il_u_copy.select_atoms("not resname SOL iSL")
                il_ions.positions = np.roll(il_ions.positions, 3, axis=0)
                if sheet_id == self.args.n_sheets - 1:
                    il_u_copy.residues.resnames = list(
                        map(
                            lambda resname: re.sub("iSL", "SOL", resname),
                            il_u_copy.residues.resnames,
                        )
                    )
                il_u_copy.atoms.translate([0, 0, sheet_u.dimensions[2]])
                new_dimensions: NDArray = sheet_u.dimensions
                sheet_u: Universe = Merge(sheet_u.atoms, il_u_copy.atoms)
                sheet_u.dimensions = new_dimensions
                sheet_u.dimensions[2] = (
                    sheet_u.dimensions[2] + il_u_copy.dimensions[2] + extra
                )
            else:
                sheet_u.dimensions[2] = sheet_u.dimensions[2] + (2 * extra)
            sheet_u.atoms.translate([0, 0, sheet_id * sheet_u.dimensions[2]])
            sheet_universes.append(sheet_u.atoms.copy())
            sheet_heights.append(sheet_u.dimensions[2])
        combined: Universe = Merge(*sheet_universes)
        combined.dimensions = sheet_u.dimensions
        new_dimensions = combined.dimensions
        new_dimensions[2] = np.sum(sheet_heights)
        new_dimensions[3:] = [90.0, 90.0, 90.0]
        combined.dimensions = new_dimensions
        combined.atoms.pack_into_box(box=combined.dimensions, inplace=True)
        crdout: GROFile = self.get_filename(suffix=".gro")
        if backup:
            backup_files(self.args.outpath / crdout.name)
            backup_files(self.args.outpath / crdout.top.name)
        crdout.universe: Universe = combined
        logger.finfo(
            kwd_str=f"Clay stack height: ",
            message=f"{combined.dimensions[2]:2.2f} {ANGSTROM}",
            initial_linebreak=True,
        )
        crdout.write(self.top)
        add_resnum(crdin=crdout, crdout=crdout)
        self.stack: GROFile = crdout
        logger.finfo(f"Saving sheet stack as {self.stack.stem!r}\n")

    def unwrap_il_solv(self, universe: Universe) -> Universe:
        """Unwrap interlayer solvent.
        :param universe: Universe to unwrap.
        :type universe: Universe
        :return: Universe with unwrapped interlayer solvent.
        :rtype: Universe"""
        if (
            "SOL" in universe.residues.resnames
            or "iSL" in universe.residues.resnames
        ):
            for residue in universe.residues:
                if residue.resname in ["SOL", "iSL"]:
                    residue.atoms.guess_bonds()
                    if (
                        len(residue.atoms.bonds) not in [2, 3]
                        or residue.atoms.n_atoms != 3
                    ):
                        logger.ferror(
                            f"Found number of bonds {len(residue.atoms.bonds)} < 2"
                        )
                        sys.exit(4)
            sol = universe.select_atoms("resname iSL SOL")
            sol.positions = sol.unwrap(compound="residues")
        return universe

    def __path_getter(self, property_name) -> GROFile:
        """Get path to file.
        :param property_name: Name of property.
        :type property_name: str
        :return: Path to file.
        :rtype: GROFile"""
        path = getattr(self, f"__{property_name}")
        if path is not None:
            return path
        else:
            logger.debug(f"No {property_name} filename defined.")

    @property
    def stack(self) -> GROFile:
        """Clay stack GRO filename.
        :return: Clay stack GRO filename.
        :rtype: GROFile"""
        return self.__path_getter("stack")

    @stack.setter
    def stack(self, stack: Union[Path, str, GROFile]) -> None:
        """Set clay stack GRO filename.
        :param stack: Clay stack GRO filename.
        :type stack: Union[Path, str, GROFile]
        :return: None"""
        self.__path_setter_copy("stack", stack, backup=False)

    def write_sheet_crds(self, backup=False) -> None:
        """Write clay sheet coordinates.
        :param backup: Whether to back up existing files.
        :type backup: bool
        :return: None"""
        logger.info(get_subheader("1. Generating clay sheets."))
        for sheet_id in range(self.args.n_sheets):
            self.sheet.n_sheet: int = sheet_id
            self.sheet.write_gro(backup=backup)
        self.sheet.n_sheet = None

    def write_sheet_top(self) -> None:
        """Write clay sheet topology to TOP file.
        :return: None"""
        for sheet_id in range(self.args.n_sheets):
            self.top.reset_molecules()
            self.sheet.n_sheet: int = sheet_id
            self.top.add_molecules(self.sheet.universe)
            self.top.write(self.sheet.get_filename(suffix=".top"))
        self.sheet.n_sheet = None

    def get_filename(
        self,
        *solv_ion_args,
        suffix=None,
        sheetnum: Optional[int] = None,
        tcb_spec=None,
    ) -> Union[GROFile, TOPFile]:
        """Get filename for coordinates/topology.
        :param solv_ion_args: solvation/ion keywords.
        :type solv_ion_args: Any
        :param suffix: Filename suffix.
        :type suffix: Optional[str]
        :param sheetnum: Sheet number.
        :type sheetnum: Optional[int]
        :param tcb_spec: top, center bottom unit cell type specifier.
        :type tcb_spec: Optional[str]
        :return: Filename.
        :rtype: Union[GROFile, TOPFile]"""
        if sheetnum is not None:
            sheetnum: str = f"_{int(sheetnum)}"
        else:
            sheetnum: str = ""
        if tcb_spec is not None:
            if tcb_spec in ["T", "C", "B"]:
                tcb_spec: str = f"_{tcb_spec}"
            else:
                raise ValueError(
                    f'{tcb_spec} was given for "tcb". Accepted '
                    '"tcb_spec" values are "T", "C", "B".'
                )
        else:
            tcb_spec = ""
        solv_ion_list: list = ["solv", "ions"]
        arg_list: list = [s for s in solv_ion_list if s in solv_ion_args]
        other_args: set = set(solv_ion_args) - set(arg_list)
        for a in sorted(other_args):
            arg_list.append(a)
        fstem: str = f"{self.args.filestem}{sheetnum}{tcb_spec}"
        fstem: str = "_".join([fstem, *arg_list])
        try:
            suffix: str = suffix.strip(".")
            suffix: str = f".{suffix}"
        except AttributeError:
            suffix: str = ""
        logger.debug(
            f"{self.__tmp_outpath.name} exists: {Path(self.__tmp_outpath.name).is_dir()}"
        )
        path: Union[TOPFile, GROFile] = FileFactory(
            f"{self.__tmp_outpath.name}/{fstem}{suffix}"
        )
        return path

    @property
    def il_solv(self) -> GROFile:
        """Interlayer solvent GRO filename.
        :return: Interlayer solvent GRO filename.
        :rtype: GROFile"""

        return self.__path_getter("il_solv")

    @il_solv.setter
    def il_solv(self, il_solv: Union[Path, str, GROFile]) -> None:
        """Set interlayer solvent GRO filename.
        :param il_solv: Interlayer solvent GRO filename.
        :type il_solv: Union[Path, str, GROFile]
        :return: None"""
        self.__path_setter_copy("il_solv", il_solv)

    @property
    def il(self) -> GROFile:
        """Interlayer GRO filename.
        :return: Interlayer GRO filename.
        :rtype: GROFile"""
        return self.__path_getter("il")

    @il.setter
    def il(self, il: Union[Path, str, GROFile]) -> None:
        """Set interlayer GRO filename.
        :param il: Interlayer GRO filename.
        :type il: Union[Path, str, GROFile]
        :return: None"""
        self.__path_setter_copy("il", il)

    def __path_setter_copy(
        self, property_name: str, file: Union[Path, str, GROFile], backup=False
    ) -> None:
        """Set path to file.
        :param property_name: Name of property.
        :type property_name: str
        :param file: Path to file.
        :type file: Union[Path, str, GROFile]
        :param backup: Whether to back up existing files.
        :type backup: bool
        :return: None"""
        path: GROFile = getattr(self, property_name, None)
        if file is None:
            path = file
        else:
            file = GROFile(Path(file).with_suffix(".gro"))
            if path is None:
                path = file
            elif file.is_file() and path.is_file():
                if file.newer(path):
                    path = file
            elif file.is_file():
                path = file
        # path already set, copy to new path
        if path is not None:
            new_path = GROFile(
                self.args.outpath / file.with_suffix(".gro").name
            )
            try:
                shutil.copy(path, new_path)
                logger.debug(
                    f"\nResetting {property_name}\nCopied {path.name} to {new_path.parent.name}\n"
                )
            except shutil.SameFileError:
                pass
            try:
                shutil.copy(path.top, new_path.top)
            except FileNotFoundError:
                GROFile(new_path).write(topology=self.top)
            except shutil.SameFileError:
                pass
            finally:
                logger.debug(
                    f"Copied {path.top.name} to {new_path.parent.name}\n"
                )
                path = new_path
                path.description = f"{path.stem.split('_')[0]} " + " ".join(
                    property_name.split("_")
                )
        setattr(self, f"__{property_name}", path)

    def add_il_ions(self) -> None:
        """Add interlayer ions.
        :return: None"""
        logger.finfo(
            "Adding interlayer ions:", initial_linebreak=False, indent="\t\t"
        )
        infile: GROFile = self.il_solv
        add_resnum(crdin=infile, crdout=infile)
        with tempfile.NamedTemporaryFile(
            suffix=self.il_solv.suffix
        ) as temp_outfile:
            temp_gro: GROFile = GROFile(temp_outfile.name)
            shutil.copy2(infile, temp_gro)
            dr: NDArray = self.sheet.dimensions[:3] / 10
            dr[-1] *= 0.4
            if isinstance(self.args.n_il_ions, dict):
                for ion, n_ions in self.args.n_il_ions.items():
                    if n_ions != 0:
                        ion_charge = self.args.il_ions.df.loc[ion, "charges"]
                        ion_charge = self.get_formatted_ion_charges(
                            ion, ion_charge
                        )
                        logger.finfo(
                            f"Inserting {n_ions} {ion}{ion_charge} atoms",
                            indent="\t\t\t",
                        )
                        with tempfile.NamedTemporaryFile(
                            suffix=".gro"
                        ) as ion_gro:
                            ion_u: Universe = Universe.empty(
                                n_atoms=1,
                                n_residues=1,
                                n_segments=1,
                                atom_resindex=[0],
                                residue_segindex=[0],
                                trajectory=True,
                            )
                            ion_u.add_TopologyAttr("name", [ion])
                            ion_u.add_TopologyAttr("resname", [ion])
                            ion_u.dimensions = np.array(
                                [*self.sheet.dimensions, 90, 90, 90]
                            )
                            ion_u.atoms.positions = np.zeros((3,))
                            ion_u.atoms.write(ion_gro.name)
                            # determine positions for adding ions
                            with tempfile.NamedTemporaryFile(
                                suffix=".dat"
                            ) as posfile:
                                write_insert_dat(
                                    n_mols=n_ions, save=posfile.name
                                )
                                assert Path(posfile.name).is_file()
                                (
                                    insert_err,
                                    insert_out,
                                ) = self.gmx_commands.run_gmx_insert_mols(
                                    f=temp_gro,
                                    ci=ion_gro.name,
                                    ip=posfile.name,
                                    nmol=n_ions,
                                    o=temp_gro,
                                    replace="SOL",
                                    dr="{} {} {}".format(*dr),
                                )
                            center_clay(
                                crdname=temp_gro, crdout=temp_gro, uc_name=ion
                            )
                            _ = Universe(temp_gro)
                            assert Path(temp_gro).is_file()
                            replace_check: int = check_insert_numbers(
                                add_repl="Added", searchstr=insert_err
                            )
                            if replace_check != n_ions:
                                raise ValueError(
                                    f"Number of inserted molecules ({replace_check}) does not match target number "
                                    f"({n_ions})!"
                                )
            # if sheet_num is not None:
            #     numstr = f"_{sheet_num}"
            # else:
            #     numstr = ""
            il = self.get_filename(f"interlayer_ions", suffix=".gro")
            il.universe: Universe = temp_gro.universe
            il.write(topology=self.top)
            self.il: GROFile = il

    def center_clay_in_box(self) -> None:
        """Center clay in box.
        :return: None"""
        center_clay(self.stack, self.stack, uc_name=self.args.uc_stem)
        self.stack.reset_universe()


class Sheet:
    """Clay sheet class.
    :param uc_data: Unit cell data.
    :type uc_data: UCData
    :param uc_ids: Unit cell IDs.
    :type uc_ids: List[int]
    :param uc_numbers: Unit cell numbers.
    :type uc_numbers: List[int]
    :param x_cells: Number of unit cells in x-direction.
    :type x_cells: int
    :param y_cells: Number of unit cells in y-direction.
    :type y_cells: int
    :param fstem: Filestem.
    :type fstem: str
    :param outpath: Output path.
    :type outpath: Path
    :param n_sheet: Sheet number.
    :type n_sheet: int
    :return: None"""

    def __init__(
        self,
        uc_data: UCData,
        uc_ids: List[int],
        uc_numbers: List[int],
        x_cells: int,
        y_cells: int,
        fstem: str,
        outpath: Path,
        n_sheet: int = None,
        debug: bool = False,
    ):
        self.debug = debug
        self.uc_data: UCData = uc_data
        self.uc_ids: list = uc_ids
        self._uc_charges: list = uc_data.tot_charge[uc_ids]
        self.uc_numbers: list = uc_numbers
        self.dimensions: NDArray = self.uc_data.dimensions[:3] * [
            x_cells,
            y_cells,
            1,
        ]
        self.x_cells: int = x_cells
        self.y_cells: int = y_cells
        self.fstem: str = fstem
        self.outpath: Path = outpath
        self.__n_sheet = None
        self.n_sheet = n_sheet
        self.__random = None
        self._res_n_atoms = None

    def __adjust_z_to_bbox(self):
        """Adjust z-dimension to bounding box.
        :return: None"""
        u_file = self.filename
        u = self.filename.universe
        u.atoms.translate([0, 0, self.uc_data.bbox_z_shift])
        triclinic_dims = triclinic_vectors(u.dimensions)
        triclinic_dims[2, 2] = self.uc_data.bbox_height
        new_dims = u.dimensions
        new_dims[2] = triclinic_box(*triclinic_dims)[
            2
        ]  # self.uc_data.bbox_height
        u.dimensions = new_dims
        u_file.universe = u
        u_file.write()

    def get_filename(self, suffix: str) -> Union[GROFile, TOPFile]:
        return FileFactory(
            self.outpath / f"{self.fstem}_{self.n_sheet}{suffix}"
        )

    @property
    def n_sheet(self) -> Union[int, None]:
        """Sheet number.
        :return: Sheet number.
        :rtype: Union[int, None]"""
        if self.__n_sheet is not None:
            return self.__n_sheet
        else:
            raise AttributeError("No sheet number set!")

    @n_sheet.setter
    def n_sheet(self, n_sheet: int):
        """Set sheet number.
        :param n_sheet: Sheet number.
        :type n_sheet: int
        :return: None"""
        if type(n_sheet) == int:
            self.__n_sheet: int = n_sheet
            self.__random = np.random.default_rng(n_sheet)
        else:
            if n_sheet is not None:
                logger.error(f"Got {n_sheet}: Sheet number must be integer!\n")
            self.__n_sheet = None
            self.__random = None

    @property
    def random_generator(self) -> Union[None, np.random._generator.Generator]:
        """Random number generator.
        :return: Random number generator.
        :rtype: Union[None, np.random._generator.Generator]"""
        if self.__random is not None:
            return self.__random
        else:
            raise AttributeError("No sheet number set!")

    @property
    def uc_array(self) -> NDArray:
        """Unit cell array.
        :return: Unit cell index array.
        :rtype: NDArray"""
        uc_array: NDArray = np.repeat(self.uc_ids, self.uc_numbers)
        return sorted(uc_array)

    @property
    def uc_charges(self) -> NDArray:
        uc_charges = self._uc_charges[self.uc_array]
        return uc_charges

    @property
    def filename(self) -> GROFile:
        """Sheet filename.
        :return: Sheet filename.
        :rtype: GROFile"""
        return self.get_filename(suffix=".gro")

    def write_gro(self, backup: bool = False) -> None:
        """Write sheet coordinates.
        :param backup: Whether to back up existing files.
        :type backup: bool, default=False
        :return: None"""

        filename: GROFile = self.filename
        filename.description = (
            f'{self.filename.stem.split("_")[0]} sheet {self.n_sheet}'
        )
        if filename.is_file() and backup:
            logger.debug(
                f"\n{filename.parent}/{filename.name} already exists, creating backup."
            )
            backup_files(filename)
        gro_df: pd.DataFrame = self.uc_data.gro_df
        uc_array = False
        logger.finfo(
            f"Getting unit cell arrangement for sheet {self.n_sheet}:",
            initial_linebreak=True,
        )
        while uc_array is False:
            uc_array = self.get_uc_sheet_array()
        logger.finfo(f"Unit cell arrangement:", initial_linebreak=True)
        for line in uc_array.T:
            logger.finfo(" ".join(map(str, line)), indent="\t")
        sheet_df = pd.concat(
            [
                gro_df.filter(regex=f"[A-Z]([A-Z]|[0-9]){uc_id}", axis=0)
                for uc_id in uc_array.flatten(order="C")
            ]
        )
        sheet_df.reset_index(["atom-id"], inplace=True)
        sheet_df["atom-id"] = np.arange(1, len(sheet_df) + 1)
        sheet_df = sheet_df.loc[:, ["at-type", "atom-id", "x", "y", "z"]]
        sheet_n_atoms: int = len(sheet_df)
        sheet_df = sheet_df.astype(
            {
                "at-type": str,
                "atom-id": int,
                "x": float,
                "y": float,
                "z": float,
            }
        )
        with open(filename, "w") as grofile:
            grofile.write(
                f"{self.fstem} sheet {self.n_sheet}\n{sheet_n_atoms}\n"
            )
            for idx, entry in sheet_df.reset_index().iterrows():
                line: list = entry.to_list()
                grofile.write(
                    GRO_FMT.format(
                        *re.split(r"(\d+)", line[0], maxsplit=1)[1:], *line[1:]
                    )
                )
            grofile.write(f"{self.format_dimensions(self.dimensions / 10)}\n")
        add_resnum(
            crdin=filename, crdout=filename, res_n_atoms=self.res_n_atoms
        )
        uc_n_atoms: NDArray = np.array(
            [self.uc_data.n_atoms[uc_id] for uc_id in uc_array]
        ).astype(np.int32)
        y_repeats: Callable = lambda n_atoms: self._cells_shift(
            n_atoms=n_atoms, n_cells=self.y_cells, axis=1
        )
        x_repeats: Callable = lambda n_atoms: self._cells_shift(
            n_atoms=n_atoms, n_cells=self.x_cells, axis=1
        )
        y_pos_shift: NDArray = np.ravel(
            np.apply_along_axis(y_repeats, arr=uc_n_atoms, axis=1), order="C"
        )
        x_pos_shift: NDArray = np.ravel(
            np.apply_along_axis(x_repeats, arr=uc_n_atoms, axis=0), order="C"
        )
        new_universe = filename.universe
        new_positions: NDArray = new_universe.atoms.positions
        new_positions[:, 0] += (
            triclinic_vectors(self.uc_dimensions)[0, 0] * x_pos_shift
        )
        new_positions[:, 1] += (
            triclinic_vectors(self.uc_dimensions)[1, 1] * y_pos_shift
        )
        new_universe.atoms.positions = new_positions
        logger.finfo(
            f"Writing sheet {self.n_sheet} to {filename.name}",
            initial_linebreak=True,
        )
        filename.universe = new_universe
        filename.write()
        self.__adjust_z_to_bbox()
        if np.any(filename.universe.dimensions[3:5] != 90):
            self.shift_sheet()

    def get_charge_groups(self) -> Tuple[int, NDArray]:
        for charge_group in (
            self.uc_charges.sort_values(
                ascending=bool(np.min(self.uc_charges))
            )
            .round(0)
            .unique()
        ):
            uc_array = self.uc_charges[
                self.uc_charges.round(0) == charge_group
            ].index.values
            # self.random_generator.shuffle(uc_array)
            n_ucs: NDArray = len(uc_array)
            if n_ucs != 0:
                yield charge_group, n_ucs, uc_array

    def get_occ_counts(self, axis_id: int, free: NDArray) -> NDArray:
        occ_cols = np.select([free], [1], 0)
        occ_counts = np.sum(occ_cols, axis=0)
        diag_counts = np.sum(
            np.array(
                [
                    *np.fromiter(
                        self.get_all_diagonals(occ_cols), dtype=np.ndarray
                    )
                ]
            ),
            axis=1,
        )
        opposite_diag_counts = np.flip(
            np.sum(
                np.array(
                    [
                        *np.fromiter(
                            self.get_all_diagonals(np.flip(occ_cols, axis=1)),
                            dtype=np.ndarray,
                        )
                    ]
                ),
                axis=1,
            )
        )
        return occ_counts, diag_counts, opposite_diag_counts

    def get_uc_sheet_array(self):
        pm = "\u00B1"
        logdict = {
            0: "right diagonal",
            1: "left diagonal",
            2: "columns",
        }
        # TODO: move counts to bottom and initialise with 0 for all
        max_dict = {self.x_cells: 0, self.y_cells: 1}
        max_ax_len = max(max_dict.keys())
        other_ax_len = min(max_dict.keys())
        remaining_add = {}
        uc_ids = np.empty((max_ax_len, other_ax_len), dtype=object)
        prev_id = {}
        idxs_mask = np.full((self.x_cells, self.y_cells), fill_value=np.NaN)
        remainder_choices = np.arange(max_ax_len)
        self.random_generator.shuffle(remainder_choices)
        lines = {}
        # if self.debug:
        symbols = np.array(["x", "o", "+", "#", "-", "*"])
        symbols = itertools.cycle(symbols)
        symbol_arr = np.full((self.x_cells, self.y_cells), fill_value=" ")
        symbol_dict = {}
        for charge_group_id, (
            charge,
            charge_group_n_ucs,
            charge_group,
        ) in enumerate(self.get_charge_groups()):
            # if self.debug:
            symbol = next(symbols)
            symbol_dict[symbol] = charge
            uc_array = charge_group.copy()
            self.random_generator.shuffle(uc_array)
            remaining_add[charge_group_id] = 0
            n_per_line = charge_group_n_ucs // max_ax_len
            per_line_remainder = charge_group_n_ucs % max_ax_len
            n_per_col = charge_group_n_ucs // other_ax_len
            per_col_remainder = charge_group_n_ucs % other_ax_len
            # per_diag_col_remainder = charge_group_n_ucs % (other_ax_len)
            # per_opp_diag_col_remainder = charge_group_n_ucs % (other_ax_len)
            lines[charge_group_id], remainder_choices = np.split(
                remainder_choices, [per_line_remainder]
            )
            prev = np.array([])
            for axis_id in range(max_ax_len):
                if charge_group_id == 0:
                    prev_id[axis_id] = 0
                n_add_ucs = n_per_line
                n_col_ucs = n_per_col
                if axis_id in lines[charge_group_id]:
                    n_add_ucs += 1
                if n_add_ucs == 0:
                    prev = np.array([])
                    continue
                free = np.isnan(idxs_mask[axis_id])
                occ_cols = np.select([idxs_mask == charge], [1], 0)
                occ_counts = np.sum(occ_cols, axis=0)
                diag_counts = np.sum(
                    np.array(
                        [
                            *np.fromiter(
                                self.get_all_diagonals(occ_cols),
                                dtype=np.ndarray,
                            )
                        ]
                    ),
                    axis=1,
                )
                opposite_diag_counts = np.flip(
                    np.sum(
                        np.array(
                            [
                                *np.fromiter(
                                    self.get_all_diagonals(
                                        np.flip(occ_cols, axis=1)
                                    ),
                                    dtype=np.ndarray,
                                )
                            ]
                        ),
                        axis=1,
                    )
                )
                idx_choices = None
                counts = np.array(
                    [
                        np.roll(diag_counts, axis_id),
                        np.roll(opposite_diag_counts, -axis_id),
                        occ_counts,
                    ]
                )
                combined_counts = np.rint(
                    np.mean(
                        [
                            np.roll(diag_counts, axis_id),
                            np.roll(opposite_diag_counts, -axis_id),
                            occ_counts,
                        ],
                        axis=0,
                    )
                )
                free_cols = np.logical_and(
                    free,
                    combined_counts < n_col_ucs,
                )
                init_i = np.array([0, 0, 0])
                minmax = itertools.cycle([min, max])
                cycle_count = 0
                # increase max allowed counts if not enough free columns
                while free_cols[free_cols].size < n_add_ucs:
                    free_cols = np.logical_and(
                        free,
                        combined_counts
                        < n_col_ucs + next(minmax)(1, per_col_remainder),
                    )
                    cycle_count += 1
                    if cycle_count > 4:
                        return False
                extra_remainder = np.zeros_like(init_i, dtype=np.int32)
                # while init occ < max allowed occ and no or not enough idxs selected
                while np.any(
                    np.less(
                        init_i, extra_remainder + per_col_remainder + n_per_col
                    )
                ) and (
                    idx_choices is None
                    or (idx_choices.flatten().size < n_add_ucs)
                ):
                    # number of free cols == n_add_ucs (don't look further)
                    if free[free].flatten().size == n_add_ucs:
                        idx_choices = np.argwhere(free).flatten()
                        break
                    allowed_cols = free_cols.copy()
                    idx_choices = None
                    occ_devs = np.std(counts, axis=1)
                    order = np.argsort(occ_devs)[::-1]
                    if self.debug:
                        logger.finfo("Occupancy deviations:")
                        self._log_occ_devs(
                            logdict, pm, counts, occ_devs, order
                        )
                    prev_choices = None
                    # get allowed idxs starting from count with highest deviation
                    for occ_id in order:
                        (
                            idx_choices,
                            allowed_cols,
                            init_i[occ_id],
                        ) = self.get_idxs(
                            init_i[occ_id],
                            n_col_ucs,
                            n_add_ucs,
                            counts[occ_id],
                            free_cols,
                            intersect_idxs=idx_choices,
                            intersect_allowed=allowed_cols,
                            remainder=per_col_remainder
                            + extra_remainder[occ_id],
                        )
                        # found idxs length == n_add_ucs
                        if idx_choices.flatten().size == n_add_ucs:
                            # print(
                            #     occ_id,
                            #     f": stopping with {idx_choices.flatten()}, n_add_ucs = {n_add_ucs}",
                            # )
                            break
                        # if found idxs length < n_add_ucs, use previous idxs if available
                        elif (
                            idx_choices.flatten().size
                            < n_add_ucs
                            <= prev_choices.flatten().size
                            and prev_choices is not None
                        ):
                            idx_choices = prev_choices
                        prev_choices = idx_choices
                    # if still not enough idxs found, abort
                    if idx_choices.flatten().size < n_add_ucs:
                        return False
                    # if more idxs found than necessary and not first row, select idxs with lowest counts
                    # across all occ counters and try to use only idxs with lowest counts if possible
                    elif (
                        idx_choices.flatten().size > n_add_ucs
                        and np.unique(counts).size > 1
                    ):
                        min_count = np.argwhere(
                            np.min(counts, axis=0) == np.min(counts)
                        ).flatten()
                        try:
                            if (
                                min_count.size >= n_add_ucs
                                and min_count.size != combined_counts.size
                                and np.all(min_count != idx_choices)
                            ):
                                intersect_idxs = np.intersect1d(
                                    idx_choices, min_count
                                ).flatten()
                                if intersect_idxs.size >= n_add_ucs:
                                    idx_choices = intersect_idxs
                        except ValueError:
                            pass

                    # if idx_choices.flatten().size > n_add_ucs:
                    #     if prev.flatten().size != 0:
                    #         _, prev_idxs, _ = np.intersect1d(
                    #             idx_choices.flatten(), prev.flatten(), assume_unique=True, return_indices=True
                    #         )
                    #         if prev_idxs.flatten().size != 0:
                    #             remove_idxs = np.random.choice(prev_idxs, idx_choices.size - n_add_ucs, replace=False)
                    #             idx_choices = np.delete(idx_choices, remove_idxs)
                    # break if enough idxs found
                    if idx_choices.flatten().size >= n_add_ucs:
                        if self.debug:
                            logger.finfo(f"Row {axis_id}:", indent="\t")
                            logger.finfo(
                                f"Adding {n_add_ucs} from {idx_choices.flatten()}",
                                indent="\t\t",
                            )
                        break
                    else:
                        if np.any(
                            np.greater_equal(
                                init_i,
                                extra_remainder
                                + per_col_remainder
                                + n_per_col,
                            )
                        ):
                            if np.any(
                                np.greater_equal(
                                    init_i,
                                    np.sort(counts, axis=1)[:, n_add_ucs - 1],
                                )
                            ):
                                init_i[
                                    np.argwhere(
                                        np.greater_equal(
                                            init_i,
                                            np.sort(counts, axis=1)[
                                                :, n_add_ucs - 1
                                            ],
                                        )
                                    )
                                ] -= 1

                            else:
                                extra_remainder[
                                    np.argwhere(
                                        np.greater_equal(
                                            init_i,
                                            extra_remainder
                                            + per_col_remainder
                                            + n_per_col,
                                        )
                                    )
                                ] = 1
                        # init_i = np.apply_along_axis(lambda arr: np.where(sorted(arr)[:n_add_ucs], )
                        continue  # if idx_choices.flatten().size == 0:  #     return False  # if np.any(extra_remainder == 0) and idx_choices.flatten().size < n_col_ucs + per_col_remainder and np.all(init_i == n_per_col + per_col_remainder):  #     if np.all(init_i + extra_remainder == n_per_col + per_col_remainder):  #     extra_remainder[np.intersect1d(order, np.argwhere(extra_remainder == 0))[-1]] += 1  #     continue  #     else:  #     return False  # elif (  #     idx_choices.flatten().size == prev.flatten().size  #     and np.equal(  #         idx_choices.flatten(), prev.flatten()  #     ).all()) and np.all(init_i + extra_remainder <= n_per_col + per_col_remainder):  #         if np.all(init_i + extra_remainder == n_per_col + per_col_remainder):  #             extra_remainder = 1  #         continue  # elif np.min(init_i) < n_per_col - per_col_remainder - 1:  #     new_init_i = np.where(init_i == min(min(init_i), n_per_col - per_col_remainder - 1), init_i + 1, init_i)  #     if not np.equal(new_init_i, init_i).all():  #         init_i = new_init_i  #         continue  #     elif n_col_ucs <= n_per_col:  #         n_col_ucs = n_per_col + 1  #         continue  # elif n_col_ucs <= n_per_col:  #     n_col_ucs += 1  # elif n_add_ucs == idx_choices.flatten().size:  #     if self.debug:  #         logger.finfo(f"Row {axis_id}:", indent="\t")  #         logger.finfo(  #             f"Adding {n_add_ucs} from {idx_choices.flatten()}",  #             indent="\t\t",  #         )  #     break  # elif np.any(init_i == extra_remainder + n_per_col + per_col_remainder - 1):  #     init_i_idxs = np.argwhere(init_i < n_per_col + per_col_remainder + extra_remainder - 1)  #     if init_i_idxs.size == 0:  #         init_i_idxs = np.argwhere(init_i < n_per_col + per_col_remainder + extra_remainder)  #         remainder_idxs = np.argwhere(extra_remainder == 0)  #         order_idxs = np.intersect1d(init_i_idxs, remainder_idxs, assume_unique=True)  #         if order_idxs.size == 0:  #             return False  #         order_idxs = np.intersect1d(order, order_idxs, assume_unique=True)  #         extra_remainder[order[order_idxs[-1]]] = 1  #     else:  #         order_idxs = np.intersect1d(order, init_i_idxs, assume_unique=True)  # init_i[order[order_idxs[-1]]] += 1

                    # elif extra_remainder == 0 and np.all(init_i >= n_per_col + per_col_remainder - 1):  #     extra_remainder = 1  #     continue  # elif np.any(init_i < extra_remainder + n_per_col + per_col_remainder) and np.any(extra_remainder == 0):

                    # free_cols = np.logical_and(  #     free,  #     combined_counts  #     < n_col_ucs + max(1, per_col_remainder),  # )
                if (
                    idx_choices is None
                    or idx_choices.flatten().size < n_add_ucs
                ):
                    return False
                # if
                # continuous_choices = np.intersect1d(
                #     idx_choices,
                #     idx_choices[
                #         [
                #             *(
                #                 idx_choices[:-1] + 1
                #                 == np.roll(idx_choices, -1)[:-1]
                #             ),
                #             *(
                #                 idx_choices[-1:] - 1
                #                 == np.roll(idx_choices, 1)[-1:]
                #             ),
                #         ]
                #     ],
                #     assume_unique=True,
                #     return_indices=True,
                # )[1]
                # if n_add_ucs <= continuous_choices.size // 2 and n_add_ucs > 1:
                #     p = np.zeros_like(idx_choices, dtype=np.float_)
                #     start_idx = self.random_generator.choice(
                #         [0, 1], 1, replace=False
                #     )[0]
                #     p[np.sort(continuous_choices)[start_idx::2]] = 1
                #     # pass
                #     if p[0] == 1 and p[-1] == 1:
                #         p[
                #             self.random_generator.choice(
                #                 [0, -1], 1, replace=False
                #             )[0]
                #         ] = 0
                #     p = np.divide(p, np.sum(p), where=p != 0)
                # else:
                p = np.full_like(
                    idx_choices,
                    np.divide(1, idx_choices.size),
                    dtype=np.float_,
                )
                idx_sel = None
                if idx_choices.size == n_add_ucs:
                    idx_sel = idx_choices
                else:
                    while idx_sel is None or (
                        idx_sel.size == prev.size
                        and np.equal(np.sort(idx_sel), prev).all()
                        and free_cols[free_cols].size != idx_sel.size
                    ):
                        idx_sel = self.random_generator.choice(
                            idx_choices.flatten(),
                            n_add_ucs,
                            replace=False,
                            p=p,
                        )
                        if idx_sel.size == idx_choices.size:
                            idx_sel = idx_choices
                if self.debug:
                    logger.finfo(f"Selected {idx_sel}", indent="\t")
                    logger.finfo(
                        f"Occupancy counts:\n"
                        + "\n\t".join([f"{c}" for c in counts.tolist()]),
                        indent="\t",
                    )
                idxs_mask[axis_id, idx_sel] = charge
                uc_ids[axis_id, idx_sel], uc_array = np.split(
                    uc_array, [n_add_ucs]
                )
                # if self.debug:
                symbol_arr[axis_id, idx_sel] = symbol
                if idx_sel.size != 0:
                    prev = np.sort(idx_sel)
            counts = np.array(
                [
                    np.roll(diag_counts, axis_id),
                    np.roll(opposite_diag_counts, -axis_id),
                    occ_counts,
                ]
            )
            occ_devs = np.std(counts, axis=1)
            # make sure charges are evenly distributed
            if np.any(occ_devs > 1):
                return False
            if self.debug:
                order = np.argsort(occ_devs)[::-1]
                logger.finfo(
                    f"Mean occupancies for charge group {charge_group_id} (q = {charge:+2.1f}):"
                )
                self._log_occ_devs(logdict, pm, counts, occ_devs, order)
        if max_dict[max_ax_len] == 1:
            uc_ids = uc_ids.T
        else:
            symbol_arr = symbol_arr.T
        # if self.debug:
        logger.finfo("Added charges:", indent="\t")
        for k, v in symbol_dict.items():
            logger.finfo(kwd_str=f"{k}: ", message=f"{v:2.1f}", indent="\t\t")
        logger.finfo(
            "Final charge arrangement:", initial_linebreak=True, indent="\t"
        )
        for line in symbol_arr:
            logger.finfo("  ".join(line), indent="\t\t")
        return uc_ids

    def _log_occ_devs(self, logdict, pm, counts, occ_devs, order):
        logstr = list(
            map(
                lambda x, y, z: f"\t{logdict[x]:15}: {y:.1f} {pm} {z:.1f}",
                np.sort(order),
                np.mean(counts[np.argsort(order)], axis=1),
                occ_devs[np.argsort(order)],
            )
        )
        logger.finfo("\n".join(logstr))

    @staticmethod
    def _get_order(counts):
        occ_devs = np.std(counts, axis=1)
        order = np.argsort(occ_devs)[::-1]
        # if self.debug:
        logger.finfo("Occupancy deviations:")
        pm = "\u00B1"
        logdict = {
            0: "right diagonal",
            1: "left diagonal",
            2: "columns",
        }
        logstr = list(
            map(
                lambda x, y, z: f"\t{logdict[x]:15}: {y:.1f} {pm} {z:.1f}",
                np.sort(order),
                np.mean(counts[np.argsort(order)], axis=1),
                occ_devs[np.argsort(order)],
            )
        )
        return logstr, order

    def _get_counts(self, axis_id, charge, idxs_mask):
        free = np.isnan(idxs_mask[axis_id])
        occ_cols = np.select([idxs_mask == charge], [1], 0)
        occ_counts = np.sum(occ_cols, axis=0)
        diag_counts = np.sum(
            np.array(
                [
                    *np.fromiter(
                        self.get_all_diagonals(occ_cols),
                        dtype=np.ndarray,
                    )
                ]
            ),
            axis=1,
        )
        opposite_diag_counts = np.flip(
            np.sum(
                np.array(
                    [
                        *np.fromiter(
                            self.get_all_diagonals(np.flip(occ_cols, axis=1)),
                            dtype=np.ndarray,
                        )
                    ]
                ),
                axis=1,
            )
        )
        idx_choices = None
        counts = np.array(
            [
                np.roll(diag_counts, axis_id),
                np.roll(opposite_diag_counts, -axis_id),
                occ_counts,
            ]
        )
        combined_counts = np.rint(
            np.mean(
                [
                    np.roll(diag_counts, axis_id),
                    np.roll(opposite_diag_counts, -axis_id),
                    occ_counts,
                ],
                axis=0,
            )
        )
        return combined_counts, counts, free, idx_choices

    def _init_uc_array(
        self,
        charge_group,
        charge_group_id,
        charge_group_n_ucs,
        lines,
        max_ax_len,
        other_ax_len,
        remainder_choices,
        remaining_add,
    ):
        uc_array = charge_group.copy()
        self.random_generator.shuffle(uc_array)
        remaining_add[charge_group_id] = 0
        n_per_line = charge_group_n_ucs // max_ax_len
        per_line_remainder = charge_group_n_ucs % max_ax_len
        n_per_col = charge_group_n_ucs // other_ax_len
        per_col_remainder = charge_group_n_ucs % other_ax_len
        # per_diag_col_remainder = charge_group_n_ucs % (other_ax_len)
        # per_opp_diag_col_remainder = charge_group_n_ucs % (other_ax_len)
        lines[charge_group_id], remainder_choices = np.split(
            remainder_choices, [per_line_remainder]
        )
        return n_per_col, n_per_line, per_col_remainder, uc_array

    def get_all_diagonals(self, arr):
        x_dim, y_dim = arr.shape
        arr_p = np.pad(arr, ((0, 0), (0, x_dim)), mode="wrap")
        for d in range(y_dim):
            yield np.diagonal(arr_p, offset=d)

    def get_weights(
        self,
        idxs_mask,
        axis_id,
        idx_choices,
        n_ucs,
        charge_group_id=0,
        all_same=False,
    ):
        arr = np.abs(idxs_mask[axis_id - 1, :])
        idx_choices = idx_choices.astype(int)
        if (
            (axis_id == 0 and charge_group_id == 0)
            or n_ucs == len(idx_choices)
            or all_same
        ):
            p = np.ones_like(arr, dtype=np.float_)
            if n_ucs != len(idx_choices) and not all_same:
                start_id = int(self.random_generator.choice([0, 1], 1))
                p[start_id::2] -= 0.75
        else:
            p = np.divide(1, arr, where=arr > 0, out=np.full_like(arr, 1.5))
        p = np.ravel(np.where(p < 0, 0, p)[idx_choices])
        if n_ucs != len(idx_choices) and not all_same:
            for pi in range(len(p)):
                if p[pi] == np.roll(p, 1)[pi]:
                    p[pi] -= (
                        0.75 * p[pi]
                    )  # np.divide(0.5, p[pi], where=p[pi] > 0, out=np.full_like(p[pi], 0))
        return np.divide(p, np.sum(p))

    @staticmethod
    def choose_idx(
        occ_counts,
        max_n_per_col,
        free_cols,
        diag_axis_id=None,
        n_remainder=0,
        min_length=0,
    ):
        # if diag_axis_id is not None:
        #     occ_counts = np.roll(occ_counts, diag_axis_id)
        allowed_cols = occ_counts < max_n_per_col
        idx_choices = np.argwhere(
            np.logical_and(free_cols.flatten(), allowed_cols.flatten())
        ).flatten()
        if n_remainder > 0:
            extra = 0
            while len(idx_choices) < min_length and extra <= n_remainder:
                add_allowed = occ_counts <= max_n_per_col + extra
                if len(add_allowed[add_allowed == True].flatten()) != 0:
                    add_idxs = np.argwhere(
                        np.logical_and(
                            free_cols.flatten(), add_allowed.flatten()
                        )
                    ).flatten()
                    add_idxs = np.random.choice(
                        add_idxs,
                        min(n_remainder, len(add_idxs)),
                        replace=False,
                    )
                    idx_choices = np.union1d(idx_choices, add_idxs)
                extra += 1
        return idx_choices, allowed_cols

    def get_idxs(
        self,
        init_i: int,
        per_col_ucs: int,
        per_line_ucs: int,
        occ_counts,
        free_cols,
        intersect_idxs=None,
        intersect_allowed=None,
        remainder=0,
    ):
        n_remaining = 0
        init_i = max(1, init_i)
        for i in range(init_i, per_col_ucs + remainder + 1):
            if i == per_col_ucs:
                n_remaining = remainder
            new_idx_choices, allowed_cols = self.choose_idx(
                occ_counts,
                max_n_per_col=i,
                free_cols=free_cols,
                n_remainder=n_remaining,
                min_length=per_line_ucs,
            )
            if intersect_idxs is not None:
                idx_choices = np.intersect1d(
                    new_idx_choices.flatten(), intersect_idxs.flatten()
                )
            else:
                idx_choices = new_idx_choices
            if intersect_allowed is not None:
                allowed_cols = np.where(
                    intersect_allowed == allowed_cols, intersect_allowed, False
                )
            if len(idx_choices.flatten()) < per_line_ucs:
                init_i = i
                continue
            else:
                return (
                    idx_choices.flatten(),
                    allowed_cols,
                    min(i, np.sort(occ_counts)[-per_line_ucs]),
                )
        return (
            idx_choices.flatten(),
            allowed_cols,
            min(init_i, np.sort(occ_counts)[-per_line_ucs]),
        )

    def shift_sheet(self):
        u = self.filename.universe
        u.atoms.translate(
            np.array(
                [
                    u.dimensions[0] * math.sin(u.dimensions[4]),
                    u.dimensions[1] * math.sin(u.dimensions[5]),
                    0,
                ]
            )
            * self.n_sheet
        )
        self.filename.universe = u
        self.filename.write()

    @property
    def res_n_atoms(self) -> pd.Series:
        if self._res_n_atoms is None:
            self._res_n_atoms = self.get_system_n_atoms()
        return self._res_n_atoms

    @res_n_atoms.setter
    def res_n_atoms(self) -> pd.Series:
        self._res_n_atoms = self.get_system_n_atoms()

    def get_system_n_atoms(self) -> pd.Series:
        return get_system_n_atoms(crds=self.universe, write=False)

    def _cells_shift(
        self, n_cells: int, n_atoms: int, axis: Literal[0, 1] = 1
    ) -> NDArray:
        """Get shift for unit cells in sheet.
        :param n_cells: Number of cells.
        :type n_cells: int
        :param n_atoms: Number of atoms.
        :type n_atoms: int
        :return: Shift.
        :rtype: NDArray"""
        shift: NDArray = np.atleast_2d(np.arange(n_cells)).repeat(
            n_atoms, axis=axis
        )
        return shift

    @staticmethod
    def format_dimensions(dimensions: NDArray) -> str:
        """Format dimension string for GRO file.
        :param dimensions: Dimensions.
        :type dimensions: NDArray
        :return: Formatted dimensions.
        :rtype: str"""
        return "".join([f"{dimension:12.4f}" for dimension in dimensions])

    @cached_property
    def uc_dimensions(self) -> NDArray:
        """Unit cell dimensions.
        :return: Unit cell dimensions.
        :rtype: NDArray"""

        return self.uc_data.dimensions

    @property
    def universe(self) -> Universe:
        """Sheet universe.
        :return: Sheet universe.
        :rtype: Universe"""

        return (
            self.filename.universe
        )  # Universe(str(self.get_filename(suffix=".gro")))

    # TODO: add n_atoms and uc data to match data

    def backup(self, filename: Path) -> None:
        """Backup files.
        :param filename: Filename.
        :type filename: Path
        :return: None"""
        sheets_backup: Path = filename.with_suffix(f"{filename.suffix}.1")
        backups = filename.parent.glob(f"*.{filename.suffix}.*")
        for backup in reversed(list(backups)):
            n_backup: int = int(backup.suffices[-1].strip("."))
            new_backup: Path = backup.with_suffix(
                f"{filename.suffix}.{n_backup + 1}"
            )
            shutil.move(backup, new_backup)
        shutil.move(filename, sheets_backup)


class Solvent:
    """Solvent class.
    :param x_dim: X-dimension.
    :type x_dim: Optional[Union[int, float]]
    :param y_dim: Y-dimension.
    :type y_dim: Optional[Union[int, float]]
    :param z_dim: Z-dimension.
    :type z_dim: Optional[Union[int, float]]
    :param n_mols: Number of molecules.
    :type n_mols: Optional[Union[int]]
    :param n_ions: Number of ions.
    :type n_ions: Optional[Union[int]]
    :param z_padding: Z-padding.
    :type z_padding: float
    :param min_height: Minimum height.
    :type min_height: float
    :return: None"""

    solv_density = 1000e-27  # g/L 1L = 10E27 A^3
    mw_sol = 18

    @add_gmx_args
    def __init__(
        self,
        x_dim: Optional[Union[int, float]] = None,
        y_dim: Optional[Union[int, float]] = None,
        z_dim: Optional[Union[int, float]] = None,
        n_mols: Optional[Union[int]] = None,
        n_ions: Optional[Union[int]] = None,
        z_padding: float = 0.4,
        min_height: float = 1.5,
    ):
        self.x_dim = float(x_dim)
        self.y_dim = float(y_dim)
        self.min_height = float(min_height)
        if z_dim is None and n_mols is not None:
            self.n_mols = int(n_mols)
            self._z_dim = self.get_solvent_sheet_height(self.n_mols)
        elif n_mols is None and z_dim is not None:
            self._z_dim = float(z_dim)
            self.n_mols = self.get_sheet_solvent_mols(self._z_dim)
        else:
            raise ValueError(
                "No sheet height or number of molecules specified"
            )

        self._z_padding = 0
        self._z_padding_increment = z_padding

        if n_ions is None:
            self.n_ions = 0
        else:
            self.n_ions = n_ions
            self.n_mols += self.n_ions
        self.n_mols: int = int(self.n_mols)

    @property
    def z_dim(self) -> float:
        """Z-dimension.
        :return: Z-dimension.
        :rtype: float"""
        return self._z_dim + self._z_padding

    @property
    def universe(self) -> Universe:
        """Solvent universe.
        :return: Universe.
        :rtype: Universe"""
        universe = getattr(self, "__universe", None)
        return universe

    @property
    def topology(self) -> TopologyConstructor:
        """Solvent topology.
        :return: Topology.
        :rtype: TopologyConstructor"""
        top = getattr(self, "__top", None)
        return top

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.n_mols} molecules, {self.x_dim:.2f} X {self.y_dim:.2f} X {self.z_dim:.2f} {ANGSTROM}))"

    def __str__(self) -> str:
        return self.__repr__()

    def get_solvent_sheet_height(self, mols_sol: int) -> float:
        """Get solvent sheet height from number of solvent molecules..
                :param mols_sol: Number of solvent molecules.
        :type mols_sol: int
        """
        z_dim = (self.mw_sol * mols_sol) / (
            constants["N_Avogadro"]
            * self.x_dim
            * self.y_dim
            * self.solv_density
        )
        return z_dim

    def get_sheet_solvent_mols(self, z_dim: Union[float, int]) -> int:
        """Get number of solvent molecules from solvent sheet height.
        :param z_dim: Solvent sheet height.
        :type z_dim: Union[float, int]
        """

        mols_sol = (
            z_dim
            * constants["N_Avogadro"]
            * self.x_dim
            * self.y_dim
            * self.solv_density
        ) / (self.mw_sol)
        return round(mols_sol, 0)

    def top_str(self) -> str:
        return f"SOL\t{self.n_mols}\n"

    def write(
        self, spc_name: GROFile, topology: Optional[TopologyConstructor] = None
    ) -> None:
        """Write solvent sheet to GRO file
        :param spc_name: GROFile object or path to GRO file
        :param topology: TopologyConstructor object
        :return: None
        """
        if spc_name.__class__.__name__ != "GROFile":
            spc_gro: GROFile = GROFile(spc_name)
        else:
            spc_gro: GROFile = spc_name
        spc_top: TOPFile = spc_gro.top
        spc_gro.universe = Universe.empty(n_atoms=0)
        spc_gro.write(topology=topology)
        logger.finfo(f"Adding interlayer solvent to {spc_name.name!r}:")
        while True:
            if self._z_padding > 5:
                raise Exception(
                    f"\nUnsuccessful solvation after expanding interlayer by {self._z_padding} {ANGSTROM}.\nSomething odd is going on..."
                )

            logger.finfo(
                f"\tAttempting solvation with interlayer height = {self.z_dim:.2f} {ANGSTROM}"
            )
            if self._z_dim < self.min_height:
                self._z_dim = self.min_height
            solv, out = self.gmx_commands.run_gmx_solvate(
                cs="spc216",
                maxsol=self.n_mols,
                o=spc_gro,
                p=spc_top,
                scale=0.57,
                v="",
                box=f"{self.x_dim / 10} {self.y_dim / 10} {(self.z_dim / 10)}",
            )

            # check if a sufficient number of water molecules has been added
            # if not, expand z-axis by 0.5 A and try again
            try:
                self.check_solvent_nummols(solv)
            except Exception as e:
                logger.finfo(kwd_str="\t\t", message=f"{e}")
                self._z_padding += self._z_padding_increment
                logger.finfo(
                    f"\t\tIncreasing box size by {self._z_padding} {ANGSTROM}"
                )
                continue
            else:
                break

        logger.debug(
            f"Saving solvent sheet as {spc_gro.stem!r}", initial_linebreak=True
        )
        self.__universe: Universe = spc_gro.universe
        self.__top: TopologyConstructor = topology

    def check_solvent_nummols(self, solvate_stderr: str) -> None:
        """Find number of inserted water molecules from GROMAX stderr output
        :param solvate_stderr: GROMACS solvate stderr output
        :type solvate_stderr: str
        :return: None"""
        added_wat: str = re.search(
            r"(?<=Number of solvent molecules:)\s+(\d+)", solvate_stderr
        ).group(1)
        if int(added_wat) < self.n_mols:
            raise ValueError(
                "With chosen box height, GROMACS was only able to "
                f"insert {added_wat} instead of {self.n_mols} water "
                f"molecules."
            )


#
# if __name__ == "__main__":
#     gc = GMXCommands(gmx_alias="gmx_mpi")
#     gc.run_gmx_make_ndx_with_new_sel(
#         f=Path("/storage/new_clays/Na/NAu-1-fe/NAu-1-fe_7_5_solv_ions.gro"),
#         o=Path("index.ndx"),
#         sel_str="r T2* & ! a OH* HO*",
#         sel_name="new_sel",
#     )
