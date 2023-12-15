from pathlib import Path

import yaml
from importlib_resources import files

__all__ = ["SIMINP_DEFAULTS", "REMOVE_WATERS_SCRIPT", "DSPACE_RUN_SCRIPT"]

# SIMINP_DEFAULTS: default parameters for model setup from 'defaults.yaml'
defaults_file = Path(__file__).parent / "config/defaults.yaml"
with open(defaults_file, "r") as file:
    SIMINP_DEFAULTS = yaml.safe_load(file)

REMOVE_WATERS_SCRIPT = files("ClayCode.siminp.scripts").joinpath(
    "remove_waters.sh"
)
DSPACE_RUN_SCRIPT = files("ClayCode.siminp.scripts").joinpath("dspace_run.sh")
