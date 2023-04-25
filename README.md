# ClayCode
Hannah Pollak, Matteo Degiacomi, Valentina Erastova

University of Edinburgh, 2023

## Installation

This package requires python version >=3.9.

With conda, you can create a working environment, then activate it:
```shell
conda create -n py39 python=3.9
conda activate py39
````

```shell
gh repo clone Erastova-group/ClayCode
cd ClayCode
python3 pip install .
````


## Usage

ClayCode is called in the following way:

```shell
ClayCode <option> <option arguments>
```

The available options are:
  - `builder`
  - <del>`siminp`</del>
  - <del>`analysis`</del>
  - <del>`plot`</del>
  - <del>`check`</del>
  - <del>`edit`</del>

## builder

Builds clay models based on average target occupancies for a specified unit cell type.

`ClayCode.builder` will match a combination of differently substituted unit cells to fit the average target occupancies specified by the user.

### Usage

Arguments for `builder` are:
  - `-f`: [System specifications YAML file](#1-system-specifications-yaml-file)
  - `-comp`: [Clay composition in CSV format](#2-clay-composition-in-csv-format) (can also be given in system specifications YAML)

#### Example:

```shell
ClayCode builder -f path/to/input.yaml
```

### Required input files:

The build specifications are provided in YAML format:

#### 1. System specifications YAML file

The first section contains general parameters that are required for the model construction.[^1]
If the directives in the optional section are not given by the user, `ClayCode` will use default values.

[^1]: Currently only dioctahedral 2:1 unit cell type available

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

# 4. OR for a target d-spacing value in nm
# SPACING_WATERS = 1.0

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



#### 2. Clay composition in CSV format:
`CLAY_COMP` specifies the path of a CVS file with the following information:

   - First column: `sheet`
     - for tetrahedral (`T`) and octahedral (`O`) occupancies
     - interlayer ion ratios (`I`) for interlayer ions if the total later charge is non-zero
     - `T`, `O` and total (`tot`) average unit cell charge (`C`)

   - Second column with `element`:
     - ClayFF/ion atom types and charge categories
   
   - Target occupancies, header is `SYSNAME`

Example of a `clay_comp.csv` file for NAu-1 and NAu-2 nontronite clays:

Total occupancies for each dioctahedral/trioctahedral tetrahedral sheet should sum up to 4.
For octahedral sheets, the total occupancies should amount to 4/3 for dioctahedral/trioctahedral unit cell types.
For example, the occupancies of a dioctahedral 2:1 clay (TOT) should have T occupancies of 8, and O occupancies of 4.

There is no need to specify how much of the total iron is Fe3+ or Fe2+ if at least two values among the total, T and O unit cell charge are specified.
`ClayCode.builder` can perform the splitting between the two iron species.

Experimental stoichiometries do not necessarily sum up to interger occupancies. `ClayCode.builder` will first process the target composition such that the occupancies match those expected for the specified unit cell type.

Interlayer ions will be added to compensate the total charge imbalance resulting from substitutions. 
The ratio of these ions can be specified in the `I` section. The sum of all ion contributions should sum to 1.
Only ion species of the opposite sign to the layer charge will be considered.

| **sheet** | **element** | **NAu\-1\-fe** | **NAu\-2\-fe** |
|:----------|:------------|---------------:|---------------:|
| **T**     | **st**      | 6\.98          |          7\.55 |
|           | **at**      | 0\.95          |          0\.16 |
|           | **fet**     | 0\.07          |          0\.29 |
| **O**     | **fe\_tot** | 3\.61          |          3\.54 |
|           | **ao**      | 0\.36          |          0\.34 |
|           | **mgo**     | 0\.04          |          0\.04 |
| **I**     | **Ca**      | 1              |           0\.5 |
|           | **Na**      | 0              |           0\.5 |
|           | **Mg**      | 0              |              0 |
|           | **K**       | 0              |              0 |
|           | **Cl**      | 0              |              0 |
| **C**     | **T**       | \-1\.02        |        \-0\.45 |
|           | **O**       | \-0\.03        |        \-0\.27 |
|           | **tot**     | \-1\.05        |        \-0\.72 |


### Output files (inside `<OUTPATH>` directory)

1. Sheet coordinates `<SYSNAME>_<X_CELLS>_<Y_CELLS>_<sheetnumber>.gro`
2. Interlayer coordinates and topology (with solvent and/or ions) (if interlayer is solvated and/or clay has non-zero charge) `<SYSNAME>_<X_CELLS>_<Y_CELLS>_interlayer.gro` and `<SYSNAME>_<X_CELLS>_<Y_CELLS>_interlayer.top`
3. (Solvated) clay sheet stack coordinates and topology `<SYSNAME>_<X_CELLS>_<Y_CELLS>.gro` and `<SYSNAME>_<X_CELLS>_<Y_CELLS>.top`
4. Sheet stack coordinates and topology in extended box (if specfied box height > height of clay sheet stack) `<SYSNAME>_<X_CELLS>_<Y_CELLS>_ext.gro` and `<SYSNAME>_<X_CELLS>_<Y_CELLS>_ext.top`
5. Sheet stack coordinates and topology in extended and solvated box (if bulk solvation is specified and specfied box height > height of clay sheet stack) `<SYSNAME>_<X_CELLS>_<Y_CELLS>_solv.top` and `<SYSNAME>_<X_CELLS>_<Y_CELLS>_solv.top`
6. Sheet stack coordinates and topology in extended and solvated box with ions (if bulk solvation is specified and specified box height > height of clay sheet stack) `<SYSNAME>_<X_CELLS>_<Y_CELLS>_solv_ions.top` and `<SYSNAME>_<X_CELLS>_<Y_CELLS>_solv_ions.top`
7. Sheet stack coordinates and topology in box after energy minimisation `<SYSNAME>_<X_CELLS>_<Y_CELLS>_<sheetnumber>_em.gro` and `<SYSNAME>_<X_CELLS>_<Y_CELLS>_<sheetnumber>_em.gro`

## <del>siminp</del>
Not yet available
## <del>analysis</del>
Not yet available
## <del>plot</del>
Not yet available
## <del>check</del>
Not yet available
## <del>edit</del>
Not yet available
