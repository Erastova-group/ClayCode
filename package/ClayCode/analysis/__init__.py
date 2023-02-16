# from functools import wraps
#
from datetime import datetime, timezone
from pathlib import Path
import logging
import logging

__all__ = [
    "plots",
    "multidist",
    "veldist",
    "analysisbase",
    "lib",
    "zdist",
    "ph",
    "setup",
    "utils",
    "exec_time",
    "exec_date",
    "AA",
    "FF",
    "MDP",
    "CLAYS",
    "IONS",
    "SOL",
    "SOL_DENSITY",
    "setup",
    "utils",
]

import MDAnalysis

tpr_logger = logging.getLogger("MDAnalysis.topology.TPRparser").setLevel(
    level=logging.WARNING
)

PATH = Path(__file__)
DATA = (PATH.parent / "../data").resolve()
AA = (DATA / "AA").resolve()
FF = (DATA / "FF").resolve()
MDP = (DATA / "MDP").resolve()
CLAYS = (DATA / "CLAYS").resolve()
UCS = (DATA / "UCS").resolve()

IONS = ["Cl", "Na", "Ca", "K", "Mg", "Cs"]
SOL_DENSITY = 1000  # g L-1
SOL = "SOL"

shandler = logging.StreamHandler()
shandler.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.INFO,
    format="%(name)-7s - %(message)s",  # %(levelname)s -
    datefmt="%Y/%m/%d",
    handlers=[shandler],
)

logger = logging.getLogger("ClayAnalysis")
logger.debug(f"Using MDAnalysis {MDAnalysis.__version__}")

exec_time = datetime.now(timezone.utc).strftime("%y%m%d-%H%M")
exec_date = datetime.now(timezone.utc).strftime("%y%m%d")

FILE_SEARCHSTR_LIST = ["_7", "_06", "_n", "_neutral"]

# AA = Path('/storage/aa_test/aa.csv')  # pl.Path(__file__).parent / "AA/aa.csv"
# IONS = ['Cl', 'Na', 'Ca', 'K', 'Mg', 'Cs']

# import MDAnalysis
# import numpy as np
# import re
# import warnings
#
# warnings.filterwarnings('ignore', category=DeprecationWarning)
#
# class ClayFFAtomAttributes:
#     Masses = {
#         'Fe': 55.850,
#         'Si': 28.090,
#         'O': 16.000,
#         'H': 1.008,
#         'Al': 26.980,
#         'Mg': 24.310,
#         'Na':23,
#         'Cl':35,
#         'K': 1,
#         'Ca': 1,
#         'Mg': 2
#
# }
#     Types = {
#     'feo': ['feo', 'Fe', 3],
#     'fet': ['fet', 'Fe', 2],
#     'fe2': ['fe2', 'Fe', 2],
#     'st': ['st', 'Si', 4],
#     'at': ['at', 'Al', 3],
#     'ao': ['ao', 'Al', 3],
#     'mgo': ['mgo', 'Mg', 2],
#     'obts': ['obts', 'O', -2],
#     'obos': ['obos', 'O', -2],
#     'obt': ['obts', 'O', -2],
#     'obo': ['obos', 'O', -2],
#     'ob': ['ob', 'O', -2],
#     'obs': ['obss', 'O', -2],
#     'ohs': ['ohs', 'O', -2],
#     'ho': ['ho', 'H', 1],
#     'oh': ['oh', 'O', -2],
#     'hw': ['HW', 'H', 1],
#     'ow': ['OW', 'O', -2],
#     'na': ['Na', 'Na', 1],
#     'cl': ['Cl', 'Cl', -1],
#     'ca': ['Ca', 'Ca', 2],
#     'mg': ['Mg', 'Mg', 2],
#     'k': ['K', 'K', 1]
# }
#
#     def __init__(self, name):
#         self.name = name
#         self.__type = None
#
#
#     @property
#     def type(self):
#         if self.__type is None:
#             try:
#                 type_match = re.search(r'|'.join(ClayFFAtomAttributes.Types.keys()),
#                                self.name.lower()).group(0)
#                 self.__type = ClayFFAtomAttributes.Types[type_match][0]
#             except:
#                 pass
#         return self.__type
#
#
#     @property
#     def element(self):
#         return ClayFFAtomAttributes.Types[self.type.lower()][1]
#
#
#     @property
#     def mass(self):
#         return ClayFFAtomAttributes.Masses[self.element]
#
#     @property
#     def charge(self):
#         return ClayFFAtomAttributes.Types[self.type.lower()][2]
#
# def add_method(cls):
#     def decorator(func):
#         @wraps(func)
#         def wrapper(self, *args, **kwargs):
#             return func(self, *args, **kwargs)
#
#         setattr(cls, func.__name__, wrapper)
#
#     return decorator
#
# @add_method(MDAnalysis.Universe)
# def elements(self):
#     elements_dict = dict({k: v} for (k, v, _) in ClayFFAtomAttributes.Types.values())
#     elements = np.full_like(crdin.atoms.names, None)
#
#     for el, element in enumerate(self.atoms):
#         print(element)
#         try:
#             elements[el] = elements_dict[element.type.lower()]
#             print(elements[el])
#         except:
#             pass
#     return elements
#

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
