""" Constants for model setup with ClayCode (:mod:`ClayCode.builder.consts`)
===============================================================================

Adds:
- Get default setup parameters from ``defaults.yaml``
- Charge and occupation information for different unit cell types from ``data/UCS/charge_occ.csv``

"""
from pathlib import Path

import yaml
from importlib_resources import files

__all__ = [
    "BUILD_DEFAULTS",
]

# BUILD_DEFAULTS: default parameters for model setup from 'defaults.yaml'
defaults_file = Path(__file__).parent / "config/defaults.yaml"
with open(defaults_file, "r") as file:
    BUILD_DEFAULTS = yaml.safe_load(file)
    BUILD_DEFAULTS["UC_INDEX_RATIOS"] = {}


BUILDER_DATA = files("ClayCode.builder.data")
