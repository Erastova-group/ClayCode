from pathlib import Path

from importlib_resources import files

ANALYSIS_DATA = files("ClayCode.analysis.data")
PE_DATA: Path = ANALYSIS_DATA / "peaks_edges"
