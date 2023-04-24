""" Constants for model setup with ClayCode (:mod:`ClayCode.builder.consts`)
===============================================================================

Adds:
- Get default setup parameters from ``defaults.yaml``
- Charge and occupation information for different unit cell types from ``data/UCS/charge_occ.csv``

"""
import importlib_resources
import pandas as pd
import yaml
from pathlib import Path
from ClayCode import UCS
import importlib_resources

__all__ = ["BUILD_DEFAULTS", "UC_CHARGE_OCC"]

# BUILD_DEFAULTS: default parameters for model setup from 'defaults.yaml'
BUILD_CONFIG_DEFAULTS = importlib_resources.files('ClayCode.builder').joinpath('config/defaults.yaml')
with open(BUILD_CONFIG_DEFAULTS, "r") as file:
    BUILD_DEFAULTS = yaml.safe_load(file)
    for prm in ["UC_INDEX_LIST", "UC_RATIOS_LIST"]:
        BUILD_DEFAULTS[prm] = []

# UC_CHARGE_OCC: Clay unit cell charge and occupancy data from 'charge_occ.csv'
UC_CHARGE_OCC = (
    pd.read_csv(UCS / "charge_occ.csv")
    .fillna(method="ffill")
    .set_index(["sheet", "value"])
)
