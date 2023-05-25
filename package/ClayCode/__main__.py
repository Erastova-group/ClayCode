#!/usr/bin/env python3
import logging
import sys

from ClayCode.builder.utils import select_input_option
from ClayCode.core.log import logger
from ClayCode.core.parsing import ArgsFactory, BuildArgs, parser
from ClayCode.core.utils import get_subheader

__all__ = ["run"]


def run():
    args = parser.parse_args(sys.argv[1:])
    logger.setLevel(args.DEBUG)
    args_factory = ArgsFactory()
    args = args_factory.init_subclass(args)
    # file_handler = logging.FileHandler(filename=args.)
    # logger.addHandler(file_handler)
    if isinstance(args, BuildArgs):
        from ClayCode.builder import Builder

        clay_builder = Builder(args)
        clay_builder.write_sheet_crds()
        if args.il_solv is False and args.match_charge["tot"] == 0:
            pass
        elif args.il_solv is True or args.match_charge["tot"] != 0:
            clay_builder.solvate_clay_sheets()
        if args.match_charge["tot"] != 0:
            clay_builder.add_il_ions()
            if args.il_solv is False:
                # clay_builder.rename_il_solv()
                clay_builder.remove_il_solv()
            else:
                clay_builder.rename_il_solv()
        clay_builder.stack_sheets()
        clay_builder.extend_box()
        completed = False
        while completed is False:
            if args.bulk_solv is True:
                clay_builder.solvate_box()
            if not args.bulk_ion_conc == 0.0:
                clay_builder.add_bulk_ions()
            clay_builder.center_clay_in_box()
            completed = clay_builder.run_em()
            if completed is None:
                if clay_builder.extended_box is True:
                    completed = select_input_option(
                        instance_or_manual_setup=True,
                        query="Repeat solvation setup? [y]es/[n]o\n",
                        options=["y", "n"],
                        result=None,
                        result_map={"y": False, "n": None},
                    )
                    # if repeat == "y":
                    #     completed = False
                    #     logger.info(get_subheader("Repeating solvation"))
                if completed is False:
                    logger.info("\nRepeating solvation setup.\n")
                else:
                    logger.info(
                        "\nFinishing setup without energy minimisation.\n"
                    )
            else:
                logger.debug("\nFinished setup!\n")
        clay_builder.conclude()


if __name__ == "__main__":
    sys.exit(run())
