developers guide coming soon ...





## Dependencies:


ClayCode is compatible with UNIX operating systems. 

It relies on the following python libraries: NumPy (1.21.2), Pandas (1.3.4) and MDAnalysis (2.0.0).

Furthermore, solvent molecules and ions are added as GROMACS subprocesses. Therefore, in order to execute builder, a local GROMACS installation is required.


# Code structure:

Modules:

* ClayCode.builder
* ClayCode.siminp
* ClayCode.config
* ClayCode.data
* ClayCode.tests

Environemnt set up files:

* setup.py
* ...


The code structure is as follows:
```shell
ClayCode
│
├── builder
│   ├── config
│   └── tests
│       └── data
├── config
├── core
├── data
│   ├── FF
│   │   ├── ClayFF_Fe.ff
│   ├── MDP
│   ├── UCS
│   │   ├── D11
│   │   ├── D21
│   │   └── LDH31
│   └── user
└── siminp

```


