# User Guide 


## Dependencies

ClayCode is compatible with UNIX operating systems. 

It relies on the following python libraries: NumPy (1.21.2), Pandas (1.3.4) and MDAnalysis (2.0.0).

Furthermore, solvent molecules and ions are added as GROMACS subprocesses. Therefore, in order to execute builder, a local GROMACS installation is required.



## Data files

The data files store information necasary to construct the clay stuctiures for MD simulation. 

The files are stored within `ClayCode/data/` directory:

```shell
ClayCode
│
...
│
└── data
   ├── FF
   │    └── ClayFF_Fe.ff
   │    └── Ions.ff
   ├── MDP
   ├── UCS
   │   ├── D11
   │   ├── D21
   │   └── LDH31
   └── user


```
where:

* `data/FF` contains force fields files, as dictated by Gromacs format, currently included:
		-  ClayFF force field[@Cygan2004, @Cygan2021] (with added Fe parameters based on personal communication with Andrey Kalinichev) in a directory `ClayFF_Fe.ff`, and
		-  Ions by Pengfei Li [@li2016advances, @smith2023consequences] in `Ions.ff`, default is IOD-type, HFE and CN also included.
* `data/UCS` contains unit cell structures in .GRO format and their corresponding .ITP, topology assigned to ClayFF force field. The files are grouped per type, where `D21` us dioctohedral 2:1 clay, `D11` is dioctohedral 1:1 and `LDH31` is a layered double hydroxide. To include new UCs, see [Adding Unit Cells](#Adding_UC).
* `data/MDP` contains Gromacs version-specific .MDP files for energy minimisation and equilibration.
* `data/user` reserved for user files.





## Clay Composition

The file in .CSV format containing the reduced unit cell structure, including partial atomic ocupancies, charge balancing ions and layer charges for each clay listed.

See full details in the [Input files: CSV](CSV.md)

It is also possible to supply clay compossioton within the [.YAML](YAML.md) input only. See [Pyrophyllite](pyr.md) as an example



## Input Parameters 

System specification for the set-up are done given in .YAML format. See full details in [Input files: YAML](YAML.md)

Parameters

<img src="https://raw.githubusercontent.com/Erastova-group/ClayCode/main/docs/assets/input_illustration.png" height="600">



## Output

See [Output files](output.md)

***

<a name="Adding_UC"></a>
## Adding Unit Cells


Use of ClayCode should not be dictated only by the Unite Cells provided with this release. To add a new unit cell, one needs to:

1 - Obtain a crystal structure. <br/>
We recommend downloading .cif from the [American Mineralogist Crystal Structure Database](http://rruff.geo.arizona.edu/AMS/amcsd.php).

2 - Convert it to full ocupancy expanded structure .gro (or .pdb). <br/>
We recommend using one of the the following [OpenBabel](http://openbabel.org/wiki/Main_Page), [Avogadro](https://avogadro.cc)[@hanwell2012avogadro] (not Avogadro2) or [Mercury](https://www.ccdc.cam.ac.uk/solutions/software/mercury/) by CCDC (licence needed). We prefer Avogadro.

3 - Manually rename the atoms in the .gro to have unique names. <br/>

4 - Create an include topology file (.itp), please reffer to [GROMACS manual](https://manual.gromacs.org/current/reference-manual/topologies/topology-file-formats.html). <br/>
Assign each unique atom name in the .gro to an atom type, as given in the `ClayFF.ff/atomtypes.atp`:

```
hw      1.008        ; water hydrogen
ho      1.008        ; hydroxyl hydrogen
ow      16.00        ; water hydrogen
oh      16.00        ; hydroxyl oxygen
ob      16.00        ; bridging oxygen
obos    16.00        ; bridging oxygen with octahedral substitution
obts    16.00        ; bridging oxygen with tetrahedral substitution
obss    16.00        ; bridging oxygen with double substitution 
ohs     16.00        ; hydroxyl oxygen with substitution
st      28.09        ; tetrahedral silicon
ao      26.98        ; octahedral aluminum
at      26.98        ; tetrahedral aluminum
mgo     24.31        ; octahedral magnesium
mgh     24.31        ; hydroxide magnesium
cao     40.08        ; octahedral calcium
cah     40.08        ; hydroxide calcium 
feo     55.85        ; octahedral iron (III)
fe2     55.85        ; octahedral iron (II)
lio     6.941        ; octahedral lithium
```

### Example UC.gro and UC.itp 

A unit cell for Dioctohedral 1:1 (Kaolinite-type) with stocheometery `D101.gro`:
```
Dioctahedral 1:1 unit cell  1
   34
    1D101   AO1    1   0.061   0.433   0.332
    1D101   AO2    2   0.321   0.283   0.332
    1D101   AO3    3   0.320   0.880   0.332
    1D101   AO4    4   0.064   0.730   0.332
    1D101   ST1    5   0.237   0.749   0.065
    1D101   ST2    6   0.500   0.594   0.067
    1D101   ST3    7   0.493   0.301   0.065
    1D101   ST4    8   0.244   0.147   0.067
    1D101   OB1    9   0.225   0.751   0.226
    1D101   OB2   10   0.255   0.135   0.227
    1D101   OB3   11   0.258   0.000   0.000
    1D101   OB4   12   0.359   0.651   0.021
    1D101   OB5   13   0.360   0.236   0.001
    1D101   OB6   14   0.480   0.304   0.226
    1D101   OB7   15   0.510   0.582   0.227
    1D101   OB8   16   0.002   0.447   0.000
    1D101   OB9   17   0.100   0.204   0.021
    1D101  OB10   18   0.104   0.683   0.001
    1D101   OH1   19   0.223   0.413   0.232
    1D101   OH2   20   0.123   0.581   0.433
    1D101   OH3   21   0.164   0.855   0.431
    1D101   OH4   22   0.162   0.306   0.434
    1D101   OH5   23   0.480   0.860   0.232
    1D101   OH6   24   0.379   0.134   0.433
    1D101   OH7   25   0.420   0.408   0.431
    1D101   OH8   26   0.420   0.753   0.434
    1D101   HO1   27   0.530   0.940   0.233
    1D101   HO2   28   0.410   0.129   0.527
    1D101   HO3   29   0.400   0.434   0.522
    1D101   HO4   30   0.137   0.264   0.519
    1D101   HO5   31   0.272   0.497   0.233
    1D101   HO6   32   0.150   0.576   0.527
    1D101   HO7   33   0.136   0.880   0.522
    1D101   HO8   34   0.400   0.712   0.519
   0.51540   0.89420   0.63910
```
and corresponding `D101.itp`:

```
[ moleculetype ]
; name		nrexcl
   D101		1

[ atoms ]
;   nr       type  resnr residue  atom   cgnr     charge       mass  typeB    chargeB      massB
; residue   1  KAO rtp  KAO  q  0.0
     1         ao      1     D101   AO1      1      1.575      26.98   ;
     2         ao      1     D101   AO2      2      1.575      26.98   ;
     3         ao      1     D101   AO3      3      1.575      26.98   ;
     4         ao      1     D101   AO4      4      1.575      26.98   ;
     5         st      1     D101   ST1      5        2.1      28.09   ;
     6         st      1     D101   ST2      6        2.1      28.09   ;
     7         st      1     D101   ST3      7        2.1      28.09   ;
     8         st      1     D101   ST4      8        2.1      28.09   ;
     9         ob      1     D101   OB1      9      -1.05         16   ;
    10         ob      1     D101   OB2     10      -1.05         16   ;
    11         ob      1     D101   OB3     11      -1.05         16   ;
    12         ob      1     D101   OB4     12      -1.05         16   ;
    13         ob      1     D101   OB5     13      -1.05         16   ;
    14         ob      1     D101   OB6     14      -1.05         16   ;
    15         ob      1     D101   OB7     15      -1.05         16   ;
    16         ob      1     D101   OB8     16      -1.05         16   ;
    17         ob      1     D101   OB9     17      -1.05         16   ;
    18         ob      1     D101  OB10     18      -1.05         16   ;
    19         oh      1     D101   OH1     19      -0.95         16   ;
    20         oh      1     D101   OH2     20      -0.95         16   ;
    21         oh      1     D101   OH3     21      -0.95         16   ;
    22         oh      1     D101   OH4     22      -0.95         16   ;
    23         oh      1     D101   OH5     23      -0.95         16   ;
    24         oh      1     D101   OH6     24      -0.95         16   ;
    25         oh      1     D101   OH7     25      -0.95         16   ;
    26         oh      1     D101   OH8     26      -0.95         16   ;
    27         ho      1     D101   HO1     27      0.425      1.008   ;
    28         ho      1     D101   HO2     28      0.425      1.008   ;
    29         ho      1     D101   HO3     29      0.425      1.008   ;
    30         ho      1     D101   HO4     30      0.425      1.008   ;
    31         ho      1     D101   HO5     31      0.425      1.008   ;
    32         ho      1     D101   HO6     32      0.425      1.008   ;
    33         ho      1     D101   HO7     33      0.425      1.008   ;
    34         ho      1     D101   HO8     34      0.425      1.008   ;

[ bonds ]
; i	j	funct	length	force.c.					
19 31   1     0.1    463532.808
20 32   1     0.1    463532.808
21 33   1     0.1    463532.808
22 30   1     0.1    463532.808
23 27   1     0.1    463532.808
24 28   1     0.1    463532.808
25 29   1     0.1    463532.808
26 34   1     0.1    463532.808
```

