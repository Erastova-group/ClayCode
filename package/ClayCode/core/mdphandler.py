from __future__ import annotations

"""Utilities for manipulating GROMACS run parameter (.MDP options) data"""
from collections import UserDict
from typing import Dict, Union

from caseless_dictionary import CaselessDict
from ClayCode import AnyDict, StrNum
from ClayCode.core.consts import MDP_DEFAULTS


class MDPOptions(CaselessDict):
    def __init__(self):
        pass


class MDPOptionsCollection(UserDict):
    def __new__(
        cls,
        mdp_options_data: [
            Union[Dict[StrNum, StrNum], Dict[StrNum, AnyDict[StrNum, StrNum]]]
        ],
    ):
        ...

    def __init__(self, mdp_options_data):
        super().__init__(mdp_options_data)
