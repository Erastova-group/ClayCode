#!/usr/bin/env python3
import logging
import sys
import textwrap

from ClayCode.builder.utils import select_input_option
from ClayCode.core.consts import LINE_LENGTH
from ClayCode.core.parsing import ArgsFactory, BuildArgs, SiminpArgs, parser

__all__ = ["run"]

logger = logging.getLogger(__name__)


def run():
    args = parser.parse_args(sys.argv[1:])
    # logger.setLevel(args.DEBUG)
    args_factory = ArgsFactory()
    args = args_factory.init_subclass(args)
    # file_handler = logging.FileHandler(filename=args.)
    # logger.addHandler(file_handler)
    if isinstance(args, BuildArgs):
        from ClayCode.builder import Builder

        extra_il_space = {True: 1.5, False: 1}
        clay_builder = Builder(args)
        clay_builder.write_sheet_crds()
        clay_builder.construct_solvent(
            solvate=args.il_solv,
            ion_charge=args.match_charge["tot"],
            solvate_add_func=clay_builder.solvate_clay_sheets,
            ion_add_func=clay_builder.add_il_ions,
            solvent_remove_func=clay_builder.remove_il_solv,
            solvent_rename_func=clay_builder.rename_il_solv,
        )
        clay_builder.stack_sheets(extra=extra_il_space[args.il_solv])
        clay_builder.extend_box()
        completed = False
        while completed is False:
            if clay_builder.extended_box:
                clay_builder.construct_solvent(
                    solvate=args.bulk_solv,
                    ion_charge=args.bulk_ion_conc,
                    solvate_add_func=clay_builder.solvate_box,
                    ion_add_func=clay_builder.add_bulk_ions,
                    solvent_remove_func=clay_builder.remove_SOL,
                )
                clay_builder.center_clay_in_box()
            completed = clay_builder.run_em()
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
                    #     logger.info(get_subheader("Repeating solvation"))
                if completed is False:
                    logger.info("\nRepeating bulk setup.\n")
                else:
                    logger.info(
                        "\nFinishing setup without energy minimisation.\n"
                    )
            else:
                logger.info(f"\n{textwrap.fill(completed, width=LINE_LENGTH)}")
                logger.debug("\nFinished setup!")
        clay_builder.conclude()
        if isinstance(args, SiminpArgs):
            print("siminp")


if __name__ == "__main__":
    sys.exit(run())
