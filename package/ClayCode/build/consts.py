""" Constants for model setup with ClayCode (:mod:`ClayCode.build.consts`)
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
           'UC_CHARGE_OCC']

# BUILD_DEFAULTS: default parameters for model setup from 'defaults.yaml'
defaults_file = Path('defaults.yaml')
with open(defaults_file, 'r') as file:
    BUILD_DEFAULTS = yaml.safe_load(file)

# UC_CHARGE_OCC: Clay unit cell charge and occupancy data from 'charge_occ.csv'
UC_CHARGE_OCC = pd.read_csv(UCS / 'charge_occ.csv').fillna(method='ffill').set_index(['sheet', 'value'])
