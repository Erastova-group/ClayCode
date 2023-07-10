# `ClayCode.builder` System Specifications YAML file 

`ClayCode.builder` reads the parameters specified in this file.


## Requiered Parameters

The first section contains general parameters that are required for the model construction.


`OUTPATH`  - output directory path [str] <br/>
Directory in which output folder is created.


`SYSNAME` - clay system name [str] <br/>
Th name for clay type, must to match a column header in the [CSV](CSV.md) file, the created output folder will also have this name. Avoid spaces and unusual symbols.


`CLAY_COMP` - filename.csv [str] <br/>
Name of CSV file where the target stoichiometry for the system specified under `SYSNAME`.


`CLAY_TYPE` - unit cell type [str] <br/>
Clay unit cell type, available in `data/UCS` directory. <br/>
Such as:<br/>
Dioctahedral 2:1 - D21  <br/>
Trioctahedral 2:1 - T21  <br/>
Dioctahedral 1:1 - D11  <br/>
Trioctahedral 1:1 - T11  <br/>
layered double hydroxide 3:1 - L31  <br/>
layered double hydroxide 2:1 - L21  <br/>
Sepiolite - SEP  <br/>
*Note*: Only D21, D11 and L31 are supplied with current release. 


**Example:**

```yaml
# output directory path
OUTPATH: /path/to/output/directory

# name of system
SYSNAME: NAu-1-fe

# name of CSV file with target stoichiometry
CLAY_COMP: /path/to/clay_comp.csv

# clay type
CLAY_TYPE: D21
# Dioctahedral 2:1 - D21
```



## Optional Parameters

This is an optional parameter section, if the directives are not given by the user, `ClayCode.builder` will use default values.

### Clay Sheet Size

`X_CELLS` - Unit cell number [int] (Default: 7)<br/>
Number of unit cells in x-direction. <br/>
*Note*: Future version will have the option to specify target x-dimensions `X_DIM`

`Y_CELLS` - Unit cell number [int] (Default: 5) <br/>
Number of unit cells in y-direction. <br/>
*Note*: Future version will have the option to specify target y-dimensions `Y_DIM`

`N_SHEETS` - Unit cell number [int] (Default: 3) <br/>
Number of sheets of 1 unit cell thickness to be stacked in z-direction.<br/>


**Example:**

```yaml
# number of unit cells in x direction (Default 7)
X_CELLS: 7

# number of unit cells in y direction (Default 5)
Y_CELLS: 5

# number of unit cells in z direction (Default 3)
N_SHEETS: 3
```

### Clay Composition 

`ClayCode.builder` calculates the number and type of unit cells necessary to match the desired compossition, as given in [CSV](CSV.md) file. <br/>
If, instead, the user would like to specify this manually, the following parameters can be used:

`UC_INDEX_LIST` - unit cell list [list] <br/>
required unit cells, as specified in `data/UCS` directory, to use for building the system

`UC_RATIOS_LIST` - probabilities [list]<br/>
 probability list for the above unit cells, total probability must be 1.00


**Example:**

The following script will use the dioctohedral 2:1 unit cells, which are both neutral smectite unit cells, with the latter having 1 octohedral Al substitution for the Fe3.

This will produce a neutal smectite with the following stoicheometry: \(\mathrm{Si_8 \left[Al_{3.8}Fe_{0.2}\right]}\).

```yaml
# required UCs to builder system as list
UC_INDEX_LIST: [D221 D228]

# probability list of unit cells in system
UC_RATIOS_LIST: [0.8 0.2]
```


### Interlayer Solvent and Ions

`IL_SOLV` - interlayer solvent presence [bool] (Default: *True*)<br/>
If *True* interlayer space should be solvated, otherwise *False* will produce non-hydrated clay.

There are three options to handle interlayer water and interlayer ions, **only one should be specified**:

1. 
`ION_WATERS` - number of waters per ion [int, Dict[str: int]] <br/>
Number of water molecules that should be added per ion. <br/> 
If a single number [int] is given, all of the ions will be hydrated by the same number of waters. <br/>
To specify number of waters per ion-type the entry should be a citionary, then the total number of water molecules will be mapped to the specifications given for each of the  interlayer occupancies `I` in the [CSV](CSV.md) entry.<br/>
<br/>
**Example:**<br/>
A given number of waters will be added for each cation, which quantity is specified in the clay stoicheometry [CSV](CSV.md) file.

          ION_WATERS: 
                Ca: 12
                Na: 12
                K: 10
          

2. 
`UC_WATERS` - number of water molecules per unit cell [int] <br/>
The specified number of water molecules multiplied by the number of unit cells is added into each of the interlayer spaces.

3. 
`SPACING_WATERS` - hydrated interleyer spacing, in A [float] (Default 10.0 A) <br/>
Target interlayer spacing filled with water, final value may vary due to the water rearrangement when in contact with clay surface and packing around ions.



### Simulation Box Specifications

`BOX_HEIGHT` - simulation box height in A [float]  <br/>
Size of the final simulation box in z-direction in A, note the clay layers are possitioned in xy-plane.


`BULK_SOLV` - bulk solvent presence [bool] (Default: *True*)<br/>
If *True* the box space will be filled with water, otherwise *False* will keep the box empty. This is useful if further plan is to add other species, such as oil, into the system.

`BULK_IONS` - type of ions and their concentration in mol/L [int, Dict[str: int]] <br/>
Ions to be added into the bulk solvated space. <br/>
*Note*: Gromacs will not be happy if the system is not neutral!

**Example:**

This will create a simulation box of 120 A hight, filled with 0.1 M NaCl solution.

```yaml

# full simulation box height in A 
BOX_HEIGHT: 120.0

# Bulk solvent added or not (Default: True)
BULK_SOLV: True

BULK_IONS:
  Na: 0.1
  Cl: 0.1
```

### GROMACS version specification

Sometimes more than one version of GROMACS may be installed, this allows user to specify the one to use. 


`GMX` - bash alias for your favourite gromacs version [str]

**Example:**

To use your MPI-compiled version:

```yaml
GMX: gmx_mpi
```


***

# `ClayCode.siminp` System Specifications YAML file 

`ClayCode.simpinp` reads the parameters specified in this file and generates simulation inputs for GROMACS.

Coming soon...