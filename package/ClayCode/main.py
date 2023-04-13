# from config.params import PRMS
import logging


logging.basicConfig(format='%(message)s', level=logging.INFO)
from ClayCode.core.parsing import parser, ArgsFactory, BuildArgs


if __name__ == '__main__':
    args = parser.parse_args(
        ["builder", "-f", "/storage/claycode/package/ClayCode/builder/tests/data/input.yaml"])
    args_factory = ArgsFactory()
    args = args_factory.init_subclass(args)

    if isinstance(args, BuildArgs):
        from ClayCode.builder import Builder
        clay_builder = Builder(args)
        clay_builder.write_sheet_crds()
        if args.il_solv is False and args.match_charge['tot'] == 0:
            pass
        elif args.il_solv is True or args.match_charge['tot'] != 0:
            solvent_sheet = clay_builder.solvate_clay_sheets()
        if args.match_charge['tot'] != 0:
            clay_builder.add_il_ions()
            if args.il_solv is False:
                clay_builder.remove_il_solv()
            else:
                clay_builder.rename_il_solv()
        clay_builder.stack_sheets()
        clay_builder.extend_box()
        if args.bulk_solv is True:
            clay_builder.solvate_box()
        if not args.bulk_ion_conc == 0.0:
            clay_builder.add_bulk_ions()


    # if PRMS.build:
    #     from package.builder.builder import ModelBuilder
    #     ModelBuilder().run()
    #
    # if PRMS.siminp:
    #     from package.siminp.siminp import SiminpWriter
    #     SiminpWriter().run()
    #
    # # builder = BuildParams()
    # # if builder.builder == "new":
    # #     if not builder.hasattr("uc_dict"):
    # #         ions_ff = ForceField(builder.FF['ions'])
    # #         clay_ff = ForceField(builder.FF['clay'])
    # #         exp = ExpComposition(builder._target_comp,
    # #                              ions_ff)
    # #         ratios = ElementRatios(
    # #             builder._target_comp,
    # #             builder.clay_type,
    # #             builder.x_cells,
    # #             builder.y_cells,
    # #             builder.outpath,
    # #             builder.sysname,
    # #         )
