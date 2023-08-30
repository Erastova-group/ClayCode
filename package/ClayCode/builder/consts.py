""" Constants for model setup with ClayCode (:mod:`ClayCode.builder.consts`)
===============================================================================

Adds:
- Get default setup parameters from ``defaults.yaml``
- Charge and occupation information for different unit cell types from ``data/UCS/charge_occ.csv``

"""
from pathlib import Path

import pandas as pd
import yaml
from ClayCode.core.consts import UCS
from importlib_resources import files

__all__ = ["BUILD_DEFAULTS", "UC_CHARGE_OCC"]

# BUILD_DEFAULTS: default parameters for model setup from 'defaults.yaml'
defaults_file = Path(__file__).parent / "config/defaults.yaml"
with open(defaults_file, "r") as file:
    BUILD_DEFAULTS = yaml.safe_load(file)
    BUILD_DEFAULTS["UC_INDEX_RATIOS"] = {}

# UC_CHARGE_OCC: Clay unit cell charge and occupancy data from 'charge_occ.csv'
UC_CHARGE_OCC = (
    pd.read_csv(UCS / "charge_occ.csv")
    .fillna(method="ffill")
    .set_index(["sheet", "value"])
)

BUILDER_DATA = files("ClayCode.builder.data")
