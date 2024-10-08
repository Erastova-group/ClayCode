from __future__ import annotations

import itertools
from typing import Any, Dict, List, Type

import pandas as pd
import yaml
from caseless_dictionary import CaselessDict
from importlib_resources import files

# from ClayCode.core.classes import YAMLFile

DATA: Type["PosixPath"] = files("ClayCode.data.data")
FF: Type["PosixPath"] = DATA.joinpath("FF")
AA: Type["PosixPath"] = DATA.joinpath("AA")
MDP: Type["PosixPath"] = DATA.joinpath("MDP")
CLAYS: Type["PosixPath"] = DATA.joinpath("CLAYS")
UCS: Type["PosixPath"] = DATA.joinpath("UCS")

CLAY_FF: Type["PosixPath"] = FF / "ClayFF_Fe"
CLAYFF_AT_TYPES: Type["PosixPath"] = UCS / "clay_at_types.yaml"

CLAYFF_AT_CHARGES: Type["PosixPath"] = UCS / "clay_charges.yaml"


USER_DATA: Type["PosixPath"] = files("ClayCode.data.user_data")
USER_MDP: Type["PosixPath"] = USER_DATA.joinpath("MDP")
USER_FF: Type["PosixPath"] = USER_DATA.joinpath("FF")
USER_UCS: Type["PosixPath"] = USER_DATA.joinpath("UCS")
USER_CLAYS: Type["PosixPath"] = USER_DATA.joinpath("CLAYS")
ALL_UCS: List[Type["PosixPath"]] = [
    f
    for f in itertools.chain.from_iterable([UCS.iterdir(), USER_UCS.iterdir()])
    if f.is_dir()
]

# UC_CHARGE_OCC: Clay unit cell charge and occupancy data from 'charge_occ.csv'
UC_CHARGE_OCC = (
    pd.read_csv(UCS / "charge_occ.csv")
    .fillna(method="ffill")
    .set_index(["sheet", "value"])
)
try:
    USER_CHARGE_OCC: pd.DataFrame = (
        pd.read_csv(USER_UCS / "charge_occ.csv")
        .fillna(method="ffill")
        .set_index(["sheet", "value"])
    )
except FileNotFoundError:
    pass
else:
    UC_CHARGE_OCC: pd.DataFrame = UC_CHARGE_OCC._append(
        USER_CHARGE_OCC, verify_integrity=True
    )
GRO_FMT: str = "{:>5s}{:<5s}{:5s}{:5d}{:8.3f}{:8.3f}{:8.3f}\n"
FILE_SEARCHSTR_LIST: List[str] = [""]
ITP_KWDS = TOP_KWDS = {
    "defaults": ["nbfunc", "comb-rule", "gen-pairs", "fudgeLJ", "fudgeQQ"],
    "atomtypes": [
        "at-type",
        "at-number",
        "mass",
        "charge",
        "ptype",
        "sigma",
        "epsilon",
    ],
    "bondtypes": ["ai", "aj", "func", "b0", "kb"],
    "pairtypes": ["ai", "aj", "V", "W"],
    "angletypes": ["ai", "aj", "ak", "theta0", "ktheta"],
    "dihedraltypes": ["ai", "aj", "ak", "al", "phi0", "phitheta"],
    "constrainttypes": ["ai", "aj", "b0"],
    "nonbond_params": ["ai", "aj", "V", "W"],
    "moleculetype": ["res-name", "n-excl"],
    "atoms": [
        "id",
        "at-type",
        "res-number",
        "res-name",
        "at-name",
        "charge-nr",
        "charge",
        "mass",
    ],
    "bonds": ["ai", "aj", "funct", "b0", "kb"],
    "pairs": ["ai", "aj", "funct", "theta0", "ktheta"],
    "angles": ["ai", "aj", "ak"],
    "dihedrals": ["ai", "aj", "ak", "al"],
    "system": ["sys-name"],
    "molecules": ["res-name", "mol-number"],
    "settles": ["at-type", "func", "doh", "dhh"],
    "exclusions": ["ai", "aj", "ak"],
    "position_restraints": ["i", "funct", "fcx", "fcy", "fcz"],
}


def set_globals() -> Dict[str, Dict[str, Any]]:
    """
    Combine '*._KWD' dictionaries and add datatype mapping
    :return: Combined keyword dictionary
    :rtype: Dict[str, Dict[str, str]]
    """
    import re

    combined_dict = {}

    def global_dict(global_key: str) -> Dict[str, Any]:
        return globals()[global_key]

    def del_global(global_key: str):
        globals().__delitem__(global_key)

    kwds = sorted(
        re.findall(r"[A-Z]+_KWDS", " ".join(globals().keys())), reverse=True
    )
    for kwd_dict in kwds:
        kwd = kwd_dict.split("_")[0]
        new_dict = {}
        for key, vals in global_dict(kwd_dict).items():
            new_dict[key] = {}
            for val in vals:
                new_dict[key][val] = global_dict("DTYPES")[val]
        combined_dict[f".{kwd.lower()}"] = new_dict
    del_global("DTYPES")
    return combined_dict


DTYPES = {
    "at-type": "str",
    "at-number": "int32",
    "ptype": "str",
    "sigma": "float64",
    "epsilon": "float64",
    "id": "int32",
    "res-number": "int32",
    "res-name": "str",
    "at-name": "str",
    "charge-nr": "int32",
    "charge": "float64",
    "mass": "float64",
    "FF": "str",
    "itp": "str",
    "ai": "int16",
    "aj": "int16",
    "ak": "int16",
    "al": "int16",
    "k0": "float64",
    "b0": "float64",
    "kb": "float64",
    "theta0": "float64",
    "ktheta": "float64",
    "phi0": "float64",
    "phitheta": "float64",
    "V": "str",
    "W": "str",
    "nbfunc": "int16",
    "func": "int16",
    "comb-rule": "int16",
    "gen-pairs": "str",
    "fudgeLJ": "float32",
    "fudgeQQ": "float32",
    "n-excl": "int16",
    "doh": "float32",
    "dhh": "float32",
    "funct": "int16",
    "sys-name": "str",
    "mol-number": "int32",
    "fcx": "int32",
    "fcy": "int32",
    "fcz": "int32",
    "i": "int32",
}
GRO_KWDS: Dict[str, str] = {}
MDP_KWDS: Dict[str, str] = {}
KWD_DICT = set_globals()

with open(MDP / "defaults.yaml", "r+") as yaml_file:
    MDP_DEFAULTS = yaml.safe_load(yaml_file)
    for k, v in MDP_DEFAULTS.items():
        MDP_DEFAULTS[k]: CaselessDict = CaselessDict(
            {ki: vi for ki, vi in v.items()}
        )
