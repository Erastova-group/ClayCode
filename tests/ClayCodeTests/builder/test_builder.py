from pathlib import Path

import pytest
from ClayCode import YAMLFile
from ClayCode.__main__ import run, run_builder
from ClayCode.core.parsing import ArgsFactory, BuildArgs, parser


@pytest.fixture()
def BuildPrms(filename):
    args = ArgsFactory()
    parse_args = parser.parse_args(
        ["builder", "-f", str(Path(__file__).parent / f"data/{filename}")]
    )
    args = args.init_subclass(parse_args)
    return args
    # return '-f', root / f'data/{filename}'
    # return {'yaml_file': (root / f'data/{filename}').resolve()}
    # file = YAMLFile(root / f'data/{filename}').resolve()
    # print(file)
    # return BuildArgs({'yaml_file': file})


@pytest.mark.parametrize(
    ("filename", "name"),
    [("input_SWy-1.yaml", "SWy-1"), ("input_SWa-1.yaml", "SWa-1")],
)
def test_builder(filename, name, BuildPrms):
    assert (
        type(BuildPrms) == BuildArgs
    ), f"Unexpected type: {type(BuildPrms)}, expected: {BuildArgs}"
    complete = run_builder(BuildPrms)
    assert complete == 0, f"Unexpected return value: {complete}, expected: 0"
