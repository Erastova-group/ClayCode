from pathlib import Path

import pytest
from ClayCode.builder.claycomp import TargetClayComposition, UCData
from ClayCode.core.classes import ForceField
from ClayCode.core.consts import FF, UCS
from importlib_resources import files

TEST_DIR = files("ClayCode.builder.tests")
TEST_DATA = TEST_DIR.joinpath("data")
TEST_CSV = list(Path(TEST_DATA).glob("exp*.csv"))[0]
CLAYFF = list(FF.glob("Clay*.ff"))[0]


@pytest.fixture
def ff():
    return ForceField(CLAYFF)


@pytest.fixture
def uc_data(request):
    return UCData(UCS / request["uc_name"], uc_stem=request["uc_stem"], ff=ff)


@pytest.fixture
def target_comp(request):
    return TargetClayComposition(
        name=request["name"],
        csv_file=TEST_CSV,
        uc_data=uc_data,
        uc_name=request["uc_name"],
        uc_stem=request["uc_stem"],
        ff=ff,
    )


@pytest.fixture()
def gen_data():
    data = []
    names, values = (
        ("name", "uc_name", "uc_stem"),
        [
            ("NAu-1-fe", "D21", "D2"),
            ("IMt-1", "D21", "D2"),
            ("KGa-1", "D11", "D1"),
        ],
    )
    for dataset in values:
        data.append(dict(zip(names, dataset)))
    print(data)
    parameters = pytest.mark.parametrize("data", data)


@pytest.mark.parametrize(gen_data, indirect=True)
class TestClayComp:
    def test_reduce_charge(self, name, uc_name, uc_stem):
        target = target_comp(name, uc_name, uc_stem)
        print(target)

    def test_set_charge(self):
        assert False

    def test_clay_df(self):
        assert False

    def test_ion_df(self):
        assert False

    def test_split_fe_occupancies(self):
        assert False

    def test__get_charges(self):
        assert False

    def test_get_charges(self):
        assert False

    def test_occupancies(self):
        assert False

    def test_non_charged_sheet_df(self):
        assert False

    def test_charged_sheet_df(self):
        assert False

    def test_correct_charged_occupancies(self):
        assert False

    def test_occ_correction_df(self):
        assert False

    def test_correct_uncharged_occupancies(self):
        assert False

    def test_get_ion_numbers(self):
        assert False
