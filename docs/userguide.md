## user guide outline





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
```
ClayCode
│   │ 
│ 	└───config
│ 
└───Package
│     │ 	
│     └───main.py
│     │ 	
│     └───builder
│     │ 	│ 
│     │ 	└───
│     │     	
│     └───siminp
│     │ 	│ 
│     │ 	└───
│     │ 	
│     └───...
│   
└─── Smth..
│ 	│ 
│ 	   
└─── Smth..
	│  
```



# Data files

The main two data files are the clay structure file, unit cell structures, and the force field file


## Clay composition

See details in the [CSV](CSV.md)

## Clay Unite Cell structures

The unit cells are D211...
T21...


## Force fields 

ClayFF force field
Note on the Fe




## Input parameters 

See details in the  [YAML](YAML.md)

### Parameters

<img src="https://raw.githubusercontent.com/Erastova-group/ClayCode/main/docs/assets/input_illustration.png" height="600">



## Output

See [Output files](output.md)

