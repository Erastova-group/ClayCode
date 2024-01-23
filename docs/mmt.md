# Montmorillonite

## Clay composition

Wyoming montmorillonite is widely studied and used smectite. It is available for purchase from Source Clays of [The Clay Minerals Society](https://www.clays.org), and is identified as SWy-1, SWy-2 or SWy-3, depending on the batch. 

This is a well-characterised clay with the structure listed under "Physical and Chemical Data of Source Clays" at the [clays.org](https://www.clays.org/sourceclays_data/), as:

**Swy-1:**    \(\mathrm {Ca_{0.12} Na_{0.32} K_{0.05} ~\left[ Si_{7.98} Al_{0.02}\right]^{-0.02} \left[ Al_{3.01} Fe^{III}_{0.41} Mn_{0.01} Mg_{0.54} Ti_{0.02}\right]^{-0.52} }\), <br/> with unbalanced charge of 0.05.

Ti is often identified during clay analysis, but is attributed to the TiO inclusions, therefore, we will omit it in the input. Meanwhile, ClayFF force field does not have parameters for Mn, which is shown at the level of detection and so we will also omit this entry.

The provided  `clay_comp.csv` file contains an entry `SWy-1` which corresponds to this Wyoming montmorillonite clay. 


## Construction of the model

Let's examine the `Swy1.yaml`, which is provided in the `Tutorial` directory. This file contains all of the information necessary to build the clay model.

Please consult [YAML](YAML.md) for the parameter description


### Required parameters 


```yaml

# =============================================================================
# General specifications for clay model construction
# =============================================================================

# =============================================================================
# Required Parameters
# =============================================================================

OUTPATH: .

# name of system to call according to CLAY_COMP (exp_clay.csv)
# compositions currently included are: 'NAu-1-fe' 'NAu-2-fe' 'NG-1'  'SWa-1' 'LDH31' 'IMt-1' 'KGa-1'
SYSNAME: SWy-1

# specify whether new clay model should be constructed:
# new - a new clay model is constructed
# load - sheet coordinates or unit cell sequences are loaded from existent .gro or .npy files
# load: [X, Y]
# with X - 'np' or 'gro', Y - '.npy' or '.gro' filename
# False - no clay model is constructed
BUILD: new

# name of .csv file with target stoichiometry
CLAY_COMP: exp_clay.csv

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
# if not given, these will be calculated from data in the CLAY_COMP .csv file
# (Default UC_INDEX_LIST: [], UC_RATIOS_LIST: [])
# -----------------------------------------------------------------------------

# required UCs to builder system as list
# UC_INDEX_LIST: [1]

# probability list of unit cells in system
# p(tot): 1.00
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
# ION_WATERS: {'Ca': 12,
#               'Na': 12
#               }

# b. as ion species int for hydration number
# ION_WATERS : 12

# 3. OR per unit cell
#UC_WATERS: 25

# 4. OR for a target d-spacing value in A
SPACING_WATERS: 20

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
#  Na: 0.1)

BULK_IONS:
  Na: 0.1
  Cl: 0.05

# =============================================================================
# Optional: Simulation Runs Specifications
# =============================================================================

# Generate scripts and '.mdp' files for simulation runs.
# Available options:
# EM - energy minimisation
# EQ - equilibration
# D-SPACE - d-spacing equilibration
# P - production
# SIMINP: [EM, D-SPACE]

# select where EM and EQ should be run (Default: False)
# MDRUNS_REMOTE: False

# -----------------------------------------------------------------------------
# d-spacing equilibration options
# -----------------------------------------------------------------------------

# Target d-spacing in A
# D_SPACE: 19.5

# Water molecules to be removed at a time during d-spacing equilibration runs
# REMOVE_STEPS: 1000

# Absolute number of water molecules per interlayer space
# SHEET_WAT: 2

# Number of water molecules per unit cell
# UC_WAT: 0.1

# Percentage of water molecules from interlayer
# PERCENT_WAT: 5

# -----------------------------------------------------------------------------
# bash alias for used GROMACS version
# -----------------------------------------------------------------------------

GMX: gmx

```


run as


```shell
ClayCode builder -f path/to/input_Clay.yaml
```



