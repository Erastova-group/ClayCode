#!/bin/env python3
# -*- coding: UTF-8 -*-
from pathlib import Path

import yaml

__all__ = ["ADDMOLS_DEFAULTS", "ADDTYPES"]

defaults_file = Path(__file__).parent / "config/defaults.yaml"
with open(defaults_file, "r") as file:
    ADDMOLS_DEFAULTS = yaml.safe_load(file)

with open(Path(__file__).parent / "config/addtypes.yaml", "r") as file:
    ADDTYPES = yaml.safe_load(file)
