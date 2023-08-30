from pathlib import Path

import yaml

__all__ = ["SIMINP_DEFAULTS"]

# SIMINP_DEFAULTS: default parameters for model setup from 'defaults.yaml'
defaults_file = Path(__file__).parent / "config/defaults.yaml"
with open(defaults_file, "r") as file:
    SIMINP_DEFAULTS = yaml.safe_load(file)
