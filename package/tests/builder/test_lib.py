import re

import pytest
from ClayCode.core.classes import (
    add_mdp_parameter,
    set_mdp_freeze_groups,
    set_mdp_parameter,
)
from ClayCode.core.consts import MDP
from ClayCode.core.utils import get_first_item_as_int


@pytest.fixture
def input_em_mdp():  # self):
    with open(MDP / "2022/mdp_prms.mdp", "r") as mdp_file:
        mdp = mdp_file.read()
        return mdp


@pytest.mark.parametrize(
    ("parameter", "value", "result"),
    [("emtol", 5555.12, True), ("nsteps", 100, True), ("nstepSS", 100, False)],
)
def test_set_mdp_parameter(input_em_mdp, parameter, value, result):
    em_str = set_mdp_parameter(
        parameter=parameter, mdp_str=input_em_mdp, value=value
    )
    assert (
        re.search(
            rf"{parameter}\s*?=\s*?{value}\s*?\n",
            em_str,
            flags=re.MULTILINE | re.DOTALL,
        )
        is not None
    ) is result


@pytest.mark.parametrize(
    ("parameter", "value", "value1", "result"),
    [
        ("freezedim", "Y Y Y", "N N N", True),
        ("freezegrps", "A B C", "D E F1", True),
    ],
)
def test_add_mdp_parameter(input_em_mdp, parameter, value, value1, result):
    em_str = set_mdp_parameter(
        parameter=parameter, mdp_str=input_em_mdp, value=value
    )
    assert (
        re.search(
            rf"{parameter}\s*?=\s*?{value}\s*?\n",
            em_str,
            flags=re.DOTALL | re.MULTILINE,
        )
        is not None
    )
    em_str = add_mdp_parameter(
        parameter=parameter, mdp_str=em_str, value=value1
    )
    assert (
        re.search(
            rf"{parameter}\s*?=\s*?{value}\s*{value1}\s*?\n",
            em_str,
            flags=re.DOTALL | re.MULTILINE,
        )
        is not None
    ) is result


@pytest.mark.parametrize(
    ("uc_names", "freeze_dims", "result"),
    [
        pytest.param(
            ["D201", "D202"],
            ["Y", "Y", "X"],
            None,
            marks=pytest.mark.xfail(
                raises=ValueError, reason="Invalid dimension"
            ),
        ),
        pytest.param(
            ["D201", "D202"],
            ["X", "X"],
            None,
            marks=pytest.mark.xfail(
                raises=ValueError, reason="Invalid number of freeze parameters"
            ),
        ),
        (["D201", "D202"], ["Y", "Y", "Y"], True),
    ],
)
def test_set_mdp_freeze_clay(uc_names, input_em_mdp, freeze_dims, result):
    em_str = set_mdp_freeze_groups(
        input_em_mdp, uc_names=uc_names, freeze_dims=freeze_dims
    )
    dim_str = " ".join(list(freeze_dims) * len(uc_names))
    uc_str = " ".join(uc_names)
    assert (
        re.search(
            rf"freezedim\s*?=\s*?{dim_str}\s*?\n",
            em_str,
            flags=re.DOTALL | re.MULTILINE,
        )
        is not None
    ) is result
    assert (
        re.search(
            rf"freezegrps\s*?=\s*?{uc_str}\s*?\n",
            em_str,
            flags=re.DOTALL | re.MULTILINE,
        )
        is not None
    ) is result


@pytest.mark.parametrize(
    ("sequence", "result"),
    [
        ([1, 2], 1),
        ([1.00, 2, 3], 1),
        ([1.00], 1),
        pytest.param(
            ["a", "b"],
            None,
            marks=pytest.mark.xfail(raises=TypeError, reason="Wrong type"),
        ),
        pytest.param(
            "a",
            None,
            marks=pytest.mark.xfail(raises=TypeError, reason="Wrong type"),
        ),
        pytest.param(
            "aaaab",
            None,
            marks=pytest.mark.xfail(raises=TypeError, reason="Wrong type"),
        ),
    ],
)
def test_get_first_item_as_int(sequence, result):
    first_item = get_first_item_as_int(sequence)
    assert isinstance(first_item, int)
    assert first_item == result
