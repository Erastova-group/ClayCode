import shutil
from io import StringIO
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest
from ClayCode.__main__ import run_builder
from ClayCode.builder import BulkIons, InterlayerIons
from ClayCode.core.classes import Dir, ForceField, YAMLFile
from ClayCode.core.parsing import BuildArgs, buildparser
from importlib_resources import files

# @pytest.fixture
# def mock_input(caplog, capsys, inputs):
#     input_dict = {'Could not guess Ti charge.\nEnter Ti charge value: ': '4'}
#     captured = capsys.readouterr()
#     try:
#         input = input_dict[captured.out]
#     except KeyError:
#         input = '\n'
#     return StringIO(f'{input}\n')


@pytest.fixture
def generate_input_data(tmp_path):
    builder_test_data = files("ClayCodeTests.builder").joinpath("data")
    tutorial_files = builder_test_data.parents[3] / "Tutorial"
    builder_files = tutorial_files / "builder"
    for file in builder_files.glob("*.csv"):
        new_file = builder_test_data.joinpath(file.name)
        shutil.copy(file, new_file)
    for file in builder_files.glob("*.yaml"):
        if file.suffix == ".yaml":
            new_file = builder_test_data.joinpath(file.name)
            shutil.copy(file, new_file)
            new_file: YAMLFile = YAMLFile(new_file)
            yaml_dict = new_file.data
            if "OUTPATH" in yaml_dict:
                yaml_dict["OUTPATH"] = tmp_path
            yaml_dict["MATCH_TOLERANCE"] = 0.25
            new_file.data = yaml_dict


def mock_init_temp_inout(inf, outf, new_tmp_dict, which):
    inp = Path(inf)
    outp = Path(outf)
    if inf == outf:
        temp_outp = inp.with_stem(f"{inf.stem}_temp")
        outp = outp.parent / temp_outp.name
        new_tmp_dict[which] = True
    else:
        temp_outp = None
    if type(inf) == str:
        inp = str(inp.resolve())
    if type(outf) == str:
        temp_outp = str(outp.resolve())
    return inp, outp, new_tmp_dict, temp_outp


@pytest.fixture
def outpath(tmp_path):
    return Dir(tmp_path, check=False)


def read_yaml_file(filename, tmp_path):
    file = Path(__file__).parent / f"data/{filename}"
    file = YAMLFile(file)
    yaml_dict = file.data
    for k, v in zip(["OUTPATH", "MATCH_TOLERANCE"], [tmp_path, 0.25]):
        if k in yaml_dict:
            yaml_dict[k] = v
    return yaml_dict


@pytest.fixture
def TestBuildArgs(
    filename, tmp_path, monkeypatch, caplog, inputs, generate_input_data
):
    p = buildparser.parse_args(
        ["-f", str(Path(__file__).parent / f"data/{filename}")]
    )
    data = p.__dict__
    data.update({"OUTPATH": tmp_path, "MATCH_TOLERANCE": 0.25})
    monkeypatch.setattr("sys.stdin", StringIO(f"{inputs}\n"))
    args = BuildArgs(data)
    return args


@pytest.mark.parametrize(
    ("filename", "name", "inputs"),
    [
        ("input_SWa-1.yaml", "SWa-1", "4"),
        ("input_SWy-1.yaml", "SWy-1", None),
        ("input_IMt-1.yaml", "IMt-1", None),
        ("input_KGa-1.yaml", "KGa-1", "4"),
        ("input_NAu-1.yaml", "NAu-1", None),
    ],
)
@mock.patch("ClayCode.core.lib.init_temp_inout", mock_init_temp_inout)
def test_build_args_init(
    filename, name, inputs, tmp_path, monkeypatch, caplog, TestBuildArgs
):
    args = TestBuildArgs
    assert args.outpath == tmp_path / name
    assert args.outpath.is_dir()
    assert (
        Path(args.data["yaml_file"])
        == Path(__file__).parent / f"data/{filename}"
    )
    assert (
        Path(args.data["CLAY_COMP"])
        == Path(__file__).parent / f"data/exp_clay.csv"
    )
    assert type(args.match_df) == pd.Series
    assert type(args.target_df) == pd.Series
    assert type(args.uc_df) == pd.DataFrame
    assert type(args.ion_df) == pd.DataFrame
    assert type(args.x_cells) == int
    assert type(args.y_cells) == int
    assert type(args.il_ions) == InterlayerIons
    assert type(args.bulk_ions) == BulkIons
    assert type(args.default_bulk_nion) == tuple
    assert type(args.bulk_ions.df) == pd.DataFrame
    assert type(args.il_ions.df) == pd.DataFrame
    assert type(args.ff) == dict
    assert type(args.ff["clay"]) == ForceField
    assert ("clay" and "ions" and "water") in args.ff.keys()
    assert type(args.gmx_alias) == str
    assert not args.uc_df.isna().any().any()
    assert not args._uc_data.gro_df.isna().any().any()

    # @pytest.mark.parametrize(
    #     ("filename", "name"),
    #     [
    #         # ("input_SWa-1.yaml", "SWa-1"),
    #         ("input_SWy-1.yaml", "SWy-1"),
    #         ("input_IMt-1.yaml", "IMt-1"),
    #     ],
    # )
    # def test_run_builder(filename, name, tmp_path, monkeypatch, TestBuildArgs):
    run_builder(TestBuildArgs)
    assert (tmp_path / f"{name}/EM").is_dir()
