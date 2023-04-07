""" Constants for model setup with ClayCode (:mod:`ClayCode.builder.consts`)
===============================================================================

Adds:
- Get default setup parameters from ``defaults.yaml``
- Charge and occupation information for different unit cell types from ``data/UCS/charge_occ.csv``

"""
import pandas as pd
import yaml
from pathlib import Path
from ClayCode import UCS

__all__ = ['BUILD_DEFAULTS',
           'UC_CHARGE_OCC',
           'GRO_FMT']

# BUILD_DEFAULTS: default parameters for model setup from 'defaults.yaml'
defaults_file = Path(__file__).parent / 'config/defaults.yaml'
with open(defaults_file, 'r') as file:
    BUILD_DEFAULTS = yaml.safe_load(file)
    for prm in ['UC_INDEX_LIST', 'UC_RATIOS_LIST']:
        BUILD_DEFAULTS[prm] = []

# UC_CHARGE_OCC: Clay unit cell charge and occupancy data from 'charge_occ.csv'
UC_CHARGE_OCC = pd.read_csv(UCS / 'charge_occ.csv').fillna(method='ffill').set_index(['sheet', 'value'])

GRO_FMT = '{:>5s}{:<5s}{:5s}{:5d}{:8.3f}{:8.3f}{:8.3f}\n'
