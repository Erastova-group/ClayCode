# Nontronites

In this Tutorial we look at the two South Australian nontronite clays, part of the Source Clays of [The Clay Minerals Society](https://www.clays.org). These are commonly known as **NAu-1** or the Uley green nontronite and **NAu-2** or Uley brown nontronite.


Their structures are listed under "Physical and Chemical Data of Source Clays" at the [clays.org](https://www.clays.org/sourceclays_data/), as:

**NAu-1:**   \(\mathrm{M^{+1} \left[Si_{7}Al_{1}\right] \left[Al_{0.58}Fe_{3.38}Mg_{0.05}\right]} \) <br/.>
**NAu-2:**   \( \mathrm{M^{+0.97} \left[Si_{7.57}Al_{0.01}\right] \left[Al_{0.52}Fe_{3.32}Mg_{0.7}\right]} \), <br/.>
where \(\mathrm{M^{charge}}\) indicates interlayer counter-balancing cationic charge.



## Assigning iron distribution and valence

Iron can be present as either Fe(II) or Fe(III) and found in both, the tetrahedral and the octohedral sheets, as well as a charge compensating cation in the interlayer space space. 
Furthermore, in many samples, such as NAu-2, ferrihydrydes are adsorbed to clay surfaces.[@Frost2002]
While methods such as near-infrared (NIR), extended X-ray adsorption fine structures (EXAFS), Moessbauer spectroscopy, X-ray diffraction (XRD) and thermogravimetric analysis (TGA) have been used to determine the charge distributions of Fe in the sample,[@Gates2002, @Baron2017, @Decarreau2014, Ding2002] the exact compossition of these nontronites remains matter of debate.

Conventionally, when Al and Fe(III), are present in a structure, because of their larger ionic radius, Fe(III) is assigned complitely to the octahedral sheet. Only once all octahedral spaces are occupied by Fe(III), the structure is assumed to also have Fe(III) in tetrahedral positions.
However, this assignment is not necessarily correct and due to the detection limits of commonly used spectroscopic or chemical analysis techniques tetrahedral Fe(III) contents below 5% often remain undetected.[@Baron2017] 

To this end, after a review of the literature, we use the stuctures identified by Gates *et al*[@Gates2002]. We noth that the octohedral occupancy fro NAu-2 is 3.93 instead of full 4. ClayCode will address this when generating the stucture.  Furthermore, we assign \(\mathrm{Ca^{2+}}\) as counterbalancing ion to all the systems. Resulting structures are, therefore, as follows: 


**NAu-1:** \(\mathrm{0.525 ~Ca^{2+} \left[Si_{6.98}Al_{0.95}Fe_{0.07}\right]^{-1.02} \left[Al_{0.36}Fe_{3.61}Mg_{0.04}\right]^{-0.03}}\)

**NAu-2:** \(\mathrm{0.36 ~Ca^{2+} \left[Si_{7.55}Al_{0.16}Fe_{0.29}\right]^{-0.45} \left[Al_{0.34}Fe_{3.54}Mg_{0.05}\right]^{-0.27}}\)


`Tutorial` directory provides the `clay_comp.csv` file which contains the above clay structures. 
See [CSV](CSV.md) for details on this file.


| **sheet** | **element** | **NAu\-1\-fe** | **NAu\-2\-fe** |
|:----------|:------------|---------------:|---------------:|
| **T**     | **Si**      | 6.98           |           7.55 |
| **T**     | **Al**      | 0.95           |           0.16 |
| **T**     | **Fe**      | 0.07           |           0.29 |
| **O**     | **Fe**      | 3.61           |           3.54 |
| **O**     | **Al**      | 0.36           |           0.34 |
| **O**     | **Mg**      | 0.04           |           0.05 |
| **I**     | **Ca**      | 0.525          |           0.36 |
| **I**     | **Na**      | 0              |              0 |
| **I**     | **Mg**      | 0              |              0 |
| **I**     | **K**       | 0              |              0 |
| **I**     | **Cl**      | 0              |              0 |
| **C**     | **T**       | \-1.02         |         \-0.45 |
| **C**     | **O**       | \-0.03         |         \-0.27 |
| **C**     | **tot**     | \-1.05         |         \-0.72 |


## System specification

We now create the system specification [YAML](YAML.md) file, which will contain the information necessary to build the clay structures of or interest. 

The files  `NAu-1.yaml` and  `NAu-2.yaml` are provided.


The first section contains general parameters that are required for the model construction.
If the directives in the optional section are not given by the user, `ClayCode` will use default values.


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



```shell
ClayCode builder -f path/to/input_Clay.yaml
```




