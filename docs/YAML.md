System specifications YAML file

The first section contains general parameters that are required for the model construction.[^2]
If the directives in the optional section are not given by the user, `ClayCode` will use default values.

[^2]: Currently only dioctahedral 2:1 unit cell type available

```yaml
# =============================================================================
# General specifications for clay model construction
# =============================================================================

# =============================================================================
# Required Parameters
# =============================================================================

OUTPATH: /path/to/output/directory
# name of system
SYSNAME: NAu-1-fe

# name of CSV file with target stoichiometry
CLAY_COMP: /path/to/clay_comp.csv

# clay type available options in 'clay_units' directory:
# Dioctahedral 2:1 - D21
# Trioctahedral 2:1 - T21
# Dioctahedral 1:1 - D11
# Trioctahedral 1:1 - T11
# layered double hydroxide 3:1 - L31
# layered double hydroxide 2:1 - L21
# Sepiolite - SEP

CLAY_TYPE: D21


# =============================================================================
# Optional: Clay Sheet and Interlayer Specifications
# =============================================================================

# number of unit cells in x direction (Default 7)
X_CELLS: 7

# number of unit cells in y direction (Default 5)
Y_CELLS: 5

# number of unit cells in z direction (Default 3)
N_SHEETS: 3

# ----------------------------------------------------------------------------
# Optional: Unit Cell Composition and Ratios input
# if not given, these will be calculated from data in the CLAY_COMP CSV file
# (Default UC_INDEX_LIST: [], UC_RATIOS_LIST: [])
# -----------------------------------------------------------------------------

# required UCs to builder system as list
# UC_INDEX_LIST: [1]

# probability list of unit cells in system
# p(tot) = 1.00
# UC_RATIOS_LIST: [1]

# -----------------------------------------------------------------------------

# interlayer solvent present or not (Default True)
IL_SOLV: True

# -----------------------------------------------------------------------------
# Optional: Interlayer Ion Specification Options (comment the other options!)
# (Default UC_WATERS: 20)
# -----------------------------------------------------------------------------

# 1. Number of water molecules that should be added per ion (ION_WATERS)
# a. as ion species dictionary for hydration number
# ION_WATERS = {'Ca': 12,
#               'Na': 12
#               }

# b. as ion species int for hydration number
# ION_WATERS = 12

# 3. OR per unit cell
UC_WATERS: 20

# 4. OR for a target d-spacing value in A
# SPACING_WATERS = 10.0

# =============================================================================
# Optional: Simulation Box Specifications
# =============================================================================

# full simulation box height in A (Default 150.0)
BOX_HEIGHT: 150.0

# =============================================================================
# Optional: Solvation and Bulk Ions Specifications
# =============================================================================

# Bulk solvent added or not (Default: True)
BULK_SOLV: True

# Ion species and concentration in mol/L name to add in bulk solvent
# (Default: BULK_IONS:
#  Na: 0.1
#  Cl: 0.1)

BULK_IONS:
  Na: 0.1
  Cl: 0.1

# -----------------------------------------------------------------------------
# bash alias for used GROMACS version
# -----------------------------------------------------------------------------

GMX: gmx_mpi
```
