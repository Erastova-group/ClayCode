from typing import Dict

ITP_KWDS = {
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
    "bondtypes": ["ai", "aj", "b0", "kb"],
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
    "nonbond_params": ["ai", "aj", "V", "W"],
}
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
}

# GRO_KWDS = {"titel": ["sys-name"],
#             "n-atoms": ["n-atoms"],
#             "coordinates":
#                 ["res-number",
#                        "res-name", "at-name", "at-number",
#                        "x", "y", "y", "vx", "vy", "vz",
#                        "box"]}

GRO_KWDS = {}
MDP_KWDS = {}
TOP_KWDS = ITP_KWDS


def set_globals() -> Dict[str, Dict[str, str]]:
    """
    Combine '*._KWD' dictionaries and add datatype mapping
    :return: Combined keyword dictionary
    :rtype: Dict[str, Dict[str, str]]
    """
    import re

    combined_dict = {}
    global_dict = lambda key: globals()[key]

    # set_global = lambda key, value: globals().__setitem__(key, value)

    del_global = lambda key: globals().__delitem__(key)
    # set_global('KWD_DICT', {})
    kwds = sorted(re.findall(r"[A-Z]+_KWDS", " ".join(globals().keys())), reverse=True)
    for kwd_dict in kwds:
        kwd = kwd_dict.split("_")[0]
        # assert len(dicts) % 2 == 0, ValueError(f'Expected even number of KWD and DTYPE dictionaries.')
        new_dict = {}
        for key, vals in global_dict(kwd_dict).items():
            new_dict[key] = {}
            for val in vals:
                new_dict[key][val] = global_dict("DTYPES")[val]
        combined_dict[f".{kwd.lower()}"] = new_dict
        del_global(kwd_dict)
    del_global("DTYPES")
    return combined_dict


KWD_DICT = set_globals()
