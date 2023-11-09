#!/bin/env python3
# -*- coding: UTF-8 -*-
from pathlib import Path

import yaml

__all__ = ["ADDMOLS_DEFAULTS"]

defaults_file = Path(__file__).parent / "config/defaults.yaml"
with open(defaults_file, "r") as file:
    ADDMOLS_DEFAULTS = yaml.safe_load(file)
