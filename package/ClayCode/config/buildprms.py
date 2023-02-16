"""Global variables for ClayCode."""

# =============================================================================
# General SystemParameter Specifications
# =============================================================================

# name of system
SYSNAME = 'NAu-1-new'

# =============================================================================
# General specifications for clay model construction
# =============================================================================

# specify whether new clay model should be constructed:
# 'new' - a new clay model is constructed
# 'load' - sheet coordinates or unit cell sequences are loaded from existent .gro or .npy files
# {'load': [X, Y]}
# with X - 'np' or 'gro', Y - '.npy' or '.gro' filename
# False - no clay model is constructed

BUILD = 'new'

# name of .csv file with target stoichiometry
CLAY_COMP = 'exp_4.csv'


# clay type available options:
# Dioctahedral 2:1 - D21
# Trioctahedral 2:1 - T21
# Dioctahedral 1:1 - D11
# Trioctahedral 1:1 - T11
# layered double hydroxide 3:1 - L31
# layered double hydroxide 2:1 - L21
# Sepiolite - SEP


# unit cell folder in 'ucs' directory
CLAY_TYPE = 'D21'

# =============================================================================
# Clay Sheet and Interlayer Specifications
# =============================================================================

# number of unit cells in x direction
X_CELLS = 7

# number of unit cells in y direction
Y_CELLS = 5

# number of unit cells in z direction
N_SHEETS = 3

# ----------------------------------------------------------------------------
# Optional Unit Cell Composition and Ratios input
# -----------------------------------------------------------------------------

# required UCs to build system as list
# if not given will be calculated from data in exp.csv
# UC_NUMLIST = [1]

# probability list of unit cells in system
# p(tot) = 1.00
# if not given will be calculated from data in exp.csv
# UC_RATIOS = [1]
# -----------------------------------------------------------------------------

# interlayer solvent present or not
IL_SOLV = True

# -----------------------------------------------------------------------------
# Interlayer Ion Specification Options (comment the other options!)
# -----------------------------------------------------------------------------

# 1. Number of water molecules that should be added per ion (ION_WATERS)
# a. as ion species dictionary for hydration number
# ION_WATERS = {'Ca': 12,
#               'Na': 12
#               }

# b. as ion species int for hydration number
# ION_WATERS = 12

# 3. OR per unit cell (BOX_WATERS)
UC_WATERS = 30  # 24

# 4. OR for a target d-spacing value in nm
# SPACING_WATERS = 1.0

# =============================================================================
# Simulation Box Specifications
# =============================================================================

# full simulation box height in nm
BOX_HEIGHT = 15.0

# =============================================================================
# Solvation and Bulk Ions Specifications
# =============================================================================

# Bulk solvent added or not
BULK_SOLV = False

# Ion species and concentration in mol/L name to add in bulk solvent
BULK_IONS = {'Na': 0.1}

# =============================================================================
# Smulation Runs Specifications
# =============================================================================

SIMINP = 'EM'

# GRO =

# TOP =

# SIMINP_OPTIONS = 'mdruns.csv'

# select where EM and EQ should be run
MDRUNS_REMOTE = True

# Target d-spacing in A
D_SPACE = 19.7

REMOVE_STEPS = 1000
# Water molecules to be removed at a time during d-spacing equilibration runs

# Absolute number of water molecules per interlayer space
SHEET_WAT = 2

# Number of water molecules per unit cell
# UC_WAT = 0.1

# Percentage of water molecules from interlayer
# PERCENT_WAT = 5
