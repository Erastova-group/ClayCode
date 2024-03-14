from pathlib import Path

import yaml
from importlib_resources import files

__all__ = [
    "ANALYSIS_DEFAULTS",
    "PE_DATA",
    "ANALYSIS_DATA",
]
ANALYSIS_DATA = files("ClayCode.analysis.data")
PE_DATA: Path = ANALYSIS_DATA / "peaks_edges"

# ANALYSIS_DEFAULTS: default parameters for analysis from 'defaults.yaml'
defaults_file = Path(__file__).parent / "config/defaults.yaml"
with open(defaults_file, "r") as file:
    ANALYSIS_DEFAULTS = yaml.safe_load(file)
