## user guide outline

is compatible with \ac{UNIX} operating systems. 

It relies on the following python libraries: NumPy (1.21.2) \cite{harris2020array}, 

Pandas (1.3.4) and MDAnalysis (2.0.0) 

Furthermore, solvent molecules and ions are added as \ac{GROMACS} subprocesses. Therefore, in order to execute \cc*[builder], a local \ac{GROMACS} installation is required.


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


## Force fields 




## Input files:


<img src="https://raw.githubusercontent.com/Erastova-group/ClayCode/main/docs/assets/input_illustratiion.png"  width="400" height="400">

