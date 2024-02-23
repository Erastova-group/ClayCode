#!/usr/bin/env python3
from __future__ import annotations

import logging
import sys

from ClayCode import ArgsFactory, BuildArgs, ClayCodeLogger, SiminpArgs, parser
from ClayCode.builder.utils import select_input_option
from ClayCode.core.parsing import DataArgs
from ClayCode.data.ucgen import UCWriter

__all__ = ["run"]


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
            )
        completed = clay_builder.run_em(
            backup=clay_builder.args.backup,
            freeze_clay=clay_builder.args.em_freeze_clay,
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


def run():
    """Run ClayCode."""
    args = parser.parse_args(sys.argv[1:])
    args_factory = ArgsFactory()
    args = args_factory.init_subclass(args)
    if isinstance(args, SiminpArgs):
        args.write_runs()
    if isinstance(args, DataArgs):
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
    elif isinstance(args, BuildArgs):
        run_builder(args)
    return 0


if __name__ == "__main__":
    sys.exit(run())
