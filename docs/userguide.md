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
   ├── MDP
   ├── UCS
   │   ├── D11
   │   ├── D21
   │   └── LDH31
   └── user


```
where:

* `FF` contains force fields files, currently storing ClayFF force field with added Fe parameters in a directory `ClayFF_Fe.ff`, as dictated by Gromacs format.
* `UCS` contains unit cell structures in .GRO format and their corresponding .ITP, topology assigned to ClayFF force field. The files are grouped per type, where `D21` us dioctohedral 2:1 clay, `D11` is dioctohedral 1:1 and `LDH31` is a layered double hydroxide.
* `MDP` contains Gromacs version specific .MDP files for energy minimisation and equilibration.



## Clay Composition

The file in .CSV format containing the reduced unit cell structure, including partial atomic ocupancies, charge balancing ions and layer charges for each clay listed.
See full details in the [Input files CSV](CSV.md)



## Input Parameters 

System specification for the set-up are done given in .YAML format. See full details in [Input files YAML](YAML.md)

Parameters

<img src="https://raw.githubusercontent.com/Erastova-group/ClayCode/main/docs/assets/input_illustration.png" height="600">



## Output

See [Output files](output.md)

