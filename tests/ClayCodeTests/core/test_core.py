from pathlib import Path

import pytest
from ClayCode import ArgsFactory, parser
from ClayCode.core.parsing import (
    AddMolsArgs,
    AnalysisArgs,
    BuildArgs,
    CheckArgs,
    DataArgs,
    EditArgs,
    PlotArgs,
    SiminpArgs,
)


@pytest.mark.parametrize(
    ("submodule", "argclass"),
    [
        ("builder", BuildArgs),
        # ("edit", EditArgs),
        # ("check", CheckArgs),
        ("analysis", AnalysisArgs),
        ("siminp", SiminpArgs),
        ("data", DataArgs),
        # ("addmols", AddMolsArgs),
        ("plot", PlotArgs),
    ],
)
def test_args_factory(submodule, argclass):
    args = ArgsFactory()
    parse_args = parser.parse_args(
        [f"{submodule}", "-f", str(Path(__file__).parent / f"data/empty.yaml")]
    )
    with pytest.raises(SystemExit, match="2"):
        args = args.init_subclass(parse_args)
        args.__class__ == argclass
    return args
