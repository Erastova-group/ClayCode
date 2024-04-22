#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r""":mod:`ClayCode.__main__` --- Main module for ClayCode
===========================================================
This module provides the main entry point for ClayCode.
"""

from __future__ import annotations

import logging
import sys

from ClayCode import ArgsFactory, BuildArgs, ClayCodeLogger, SiminpArgs, parser
from ClayCode.builder.utils import select_input_option
from ClayCode.core.parsing import AnalysisArgs, DataArgs, PlotArgs
from ClayCode.data.ucgen import UCWriter
from ClayCode.plot.plots import AtomTypeData2D, Data, Data2D

__all__ = ["run"]

from ClayCode.plot.plots import GaussHistPlot, HistPlot, HistPlot2D, LinePlot

logging.setLoggerClass(ClayCodeLogger)

logger: ClayCodeLogger = logging.getLogger(__name__)


def run_builder(args: BuildArgs):
    from ClayCode.builder import Builder

    extra_il_space = {True: 1.5, False: 1}
    clay_builder = Builder(args)
    clay_builder.get_solvated_il()
    # clay_builder.write_sheet_crds(backup=clay_builder.args.backup)
    # clay_builder.solvate_il = lambda: clay_builder.construct_solvent(
    #     solvate=args.il_solv,
    #     ion_charge=args.match_charge["tot"],
    #     solvate_add_func=clay_builder.solvate_clay_sheets,
    #     ion_add_func=clay_builder.add_il_ions,
    #     solvent_remove_func=clay_builder.remove_il_solv,
    #     solvent_rename_func=clay_builder.rename_il_solv,
    #     backup=clay_builder.args.backup,
    # )
    logger.finfo(f"Wrote interlayer sheet to {clay_builder.il_solv.name!r}")
    completed = False
    while completed is False:
        clay_builder.write_sheet_crds(backup=clay_builder.args.backup)
        clay_builder.stack_sheets(
            extra=extra_il_space[args.il_solv], backup=clay_builder.args.backup
        )
        clay_builder.extend_box(backup=clay_builder.args.backup)
        if clay_builder.extended_box:
            clay_builder.construct_solvent(
                solvate=args.bulk_solv,
                ion_charge=args.bulk_ion_conc,
                solvate_add_func=clay_builder.solvate_box,
                ion_add_func=clay_builder.add_bulk_ions,
                solvent_remove_func=clay_builder.remove_SOL,
                backup=clay_builder.args.backup,
                solvent_rename_func=lambda: None,  # .rename_il_solv,
            )
        completed = clay_builder.run_em(
            backup=clay_builder.args.backup,
            freeze_clay=clay_builder.args.em_freeze_clay,
            constrain_distances=clay_builder.args.em_constrain_clay_dist,
            n_runs=clay_builder.args.em_n_runs,
        )
        if completed is None:
            if clay_builder.extended_box and (
                args.bulk_solv or args.bulk_ion_conc != 0.0
            ):
                completed = select_input_option(
                    instance_or_manual_setup=True,
                    query="Repeat bulk setup? [y]es/[n]o\n",
                    options=["y", "n"],
                    result=None,
                    result_map={"y": False, "n": None},
                )
                # if repeat == "y":
                #     completed = False
                #     logger.finfo(get_subheader("Repeating solvation"))
            if completed is False:
                logger.finfo("Repeating bulk setup.\n", initial_linebreak=True)
            else:
                logger.finfo(
                    "Finishing setup without energy minimisation.\n",
                    initial_linebreak=True,
                )
        else:
            logger.finfo(
                completed,
                initial_linebreak=True,
                fix_sentence_endings=False,
                expand_tabs=False,
                replace_whitespace=False,
            )
            logger.debug("\nFinished setup!")
    clay_builder.conclude()
    if isinstance(args, SiminpArgs):
        print("siminp")
    return 0


def add_new_uc_type(args):
    uc_writer = UCWriter(
        args.ingro,
        args.uc_type,
        args.outpath,
        default_solv=args.default_solv,
        substitutions=args.substitutions,
    )
    uc_writer.write_new_uc(args.uc_name)
    uc_writer.add_single_substitutions()
    uc_writer.add_double_substitutions()
    uc_writer.add_triple_substitutions()


def plot_data(args):
    if args.odir is False:
        plsave = False
        odir = None
    else:
        plsave = True
        odir = args.odir
    if args.data2d is True:
        data = Data2D(
            indir=args.idir,
            zdir=args.zdir,
            namestem=args.namestem,
            cutoff=args.cutoff,
            bins=args.bins,
            ions=args.ions,
            aas=args.aas,
            clays=args.clays,
            load=False,
            odir=odir,
            nameparts=1,
            atoms=args.atoms,
            zstem="zdist",
            zname="zdens",
            analyses=[args.analysis],
        )
        atoms = args.atoms
    else:
        atoms = args.atoms
        if args.atypes is not None:
            data = AtomTypeData2D(
                indir=args.datadir,
                cutoff=args.cutoff,
                bins=args.bins,
                new_bins=args.new_bins,
                odir=args.idir,
                namestem=args.namestem,
                nameparts=1,
                ions=args.ions,
                aas=args.aas,
                clays=args.clays,
                atomnames="atype",
                analysis=args.analysis,
                atoms=atoms,
                save=True,  # args.save,
                overwrite=args.overwrite,
                group_all=True,
                load=args.load,  # '/storage/results.p'
            )
            if args.new_bins is not None:
                args.bins = args.new_bins
            atoms = None
        data = Data(
            indir=args.idir,
            analysis=args.analysis,
            namestem=args.namestem,
            cutoff=args.cutoff,
            bins=args.bins,
            use_rel_data=args.use_rel,
            ions=args.ions,
            aas=args.aas,
            clays=args.clays,
            group_all=args.grouped,
            atoms=atoms,
            atomname=args.atomname,
            load=False,
        )

        data.ignore_density_sum = args.ignore_sum
    logger.info(f"Start")
    plot_method = {"i": lambda x: x.plot_ions, "o": lambda x: x.plot_other}

    if args.data2d is False:
        if args.gbars is True:
            plot = GaussHistPlot(data, add_missing_bulk=args.add_bulk)
            for s in args.sel:
                plot_method[s](plot)(
                    x=args.x,
                    y=args.y,
                    bars=args.plsel,
                    dpi=200,
                    columnlabel=rf"{args.xlabel}",  # r"distance from surface (\AA)",
                    rowlabel=rf"{args.ylabel}",  # 'closest atom type',# r"$\rho_z$ ()",
                    plsave=plsave,
                    odir=args.odir,
                    ylim=args.ymax,
                    plot_table=args.table,
                )
        elif args.lines:
            plot = LinePlot(data)
            for s in args.sel:
                plot_method[s](plot)(
                    x=args.x,
                    y=args.y,
                    lines=args.plsel,
                    ylim=args.ymax,
                    dpi=200,
                    columnlabel=rf"{args.xlabel}",  # r"distance from surface (\AA)",
                    rowlabel=rf"{args.ylabel}",  # 'closest atom type',# r"$\rho_z$ ()",
                    plsave=plsave,
                    xlim=args.xmax,
                    odir=args.odir,  # "/storage/plots/aadist_u/",
                    edges=args.edges,
                    figsize=args.figsize,
                    colours=args.colours,
                )
        elif args.bars:
            plot = HistPlot(data)  # , add_missing_bulk=args.add_missing_bulk)
            for s in args.sel:
                plot_method[s](plot)(
                    x=args.x,
                    y=args.y,
                    bars=args.plsel,
                    dpi=200,
                    columnlabel=rf"{args.xlabel}",  # r"distance from surface (\AA)",
                    rowlabel=rf"{args.ylabel}",  # 'closest atom type',# r"$\rho_z$ ()",
                    plsave=plsave,
                    odir=args.odir,
                    ylim=args.ymax,
                    plot_table=args.table,
                )
    else:
        plot = HistPlot2D(data, col_sel=["rdens"], select="ions")
        plot.plot_ions(x="aas", y="clays", bars="atoms", col_sel=["rdens"])


def run():
    """Run ClayCode."""
    args = parser.parse_args(sys.argv[1:])
    args_factory = ArgsFactory()
    args = args_factory.init_subclass(args)
    if isinstance(args, SiminpArgs):
        args.write_runs()
    elif isinstance(args, DataArgs):
        add_new_uc_type(args)
    elif isinstance(args, BuildArgs):
        run_builder(args)
    elif isinstance(args, PlotArgs):
        plot_data(args)
    elif isinstance(args, AnalysisArgs):
        pass
    else:
        logger.ferror(f"Unknown args type: {args}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run())
