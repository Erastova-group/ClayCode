from pathlib import Path

import yaml

__all__ = [
    "PLOT_DEFAULTS",
]

# PLOT_DEFAULTS: default parameters for plotting from 'defaults.yaml'
defaults_file = Path(__file__).parent / "config/defaults.yaml"
with open(defaults_file, "r") as file:
    PLOT_DEFAULTS = yaml.safe_load(file)
