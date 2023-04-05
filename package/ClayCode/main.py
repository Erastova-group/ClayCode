# from config.params import PRMS
import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)
from ClayCode.core.parsing import parser, ArgsFactory, BuildArgs

if __name__ == '__main__':
    args = parser.parse_args(
        ["builder", "-f", "/storage/claycode/package/ClayCode/builder/tests/data/input.yaml"]
    )
    args_factory = ArgsFactory()
    args = args_factory.init_subclass(args)

    if isinstance(args, BuildArgs):
        from ClayCode.builder import Builder
        modelbuilder = Builder(args)
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
