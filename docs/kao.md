Kaolinite 


**NOTE** treatment of the Ti in the structure and what is the origin of the Ti, why to ignore


Kaolin KGa-1(KGa-1b), (low-defect)

ORIGIN: Tuscaloosa formation? (Cretaceous?) (stratigraphy uncertain)
County of Washington, State of Georgia, USA
LOCATION: 32°58′ N-82°53′ W approximately, topographic map Tabernacle, Georgia N 3252.5-W 8252.5/7.5, Collected from face of Coss-Hodges pit, October 3,1972.
CHEMICAL COMPOSITION (%): SiO2: 44.2, Al2O3: 39.7, TiO2: 1.39, Fe2O3: 0.13,FeO: 0.08, MnO: 0.002, MgO: 0.03, CaO: n.d., Na2O: 0.013, K2O: 0.05, F:0.013,P2O5: 0.034, Loss on heating: -550°C: 12.6; 550-1000°C: 1.18.
CATION EXCHANGE CAPACITY (CEC): 2.0 meq/100g
SURFACE AREA: N2 area: 10.05 +/- 0.02 m2 /g
THERMAL ANALYSIS: DTA: endotherm at 630oC, exotherm at 1015oC, TG: dehydroxylation weight loss 13.11% (theory 14%) indicating less than 7% impurities.
INFRARED SPECTROSCOPY: Typical spectrum for well crystallized kaolinite,however not as well crystallized as a typical China clay from Cornwall,as judged from the intensity of the 3669 cm-1 band. Splitting of the 1100cm- 1 band is due to the presence of coarse crystals.

STRUCTURE:(Mg.02 Ca.01 Na.01 K.01)[Al3.86 Fe(III).02 Mntr Ti.11][Si3.83Al.17] O10(OH)8, Octahedral charge:.11, Tetrahedral charge:-.17,Interlayer charge:-.06, Unbalanced charge:0.00



The supplied `exp_clay.csv` file and contains clay structures corresponding to the **Source Clays** listed under "Physical and Chemical Data of Source Clays" [clays.org](https://www.clays.org/sourceclays_data/).




Removing invalid atom types:

```

	sheet - at-type: occupancy
	  O   -   Mn   :      0.00, 	  O   -   Ti   :      0.11

Could not guess Ti charge.
Enter Ti charge value:  (or exit with 'e')

```
>4
```
Assuming Ti charge of 4.

Getting sheet occupancies:
	Found 'O' sheet occupancies of 3.8800/4.0000 (-0.1200)
	Found 'T' sheet occupancies of 4.0000/4.0000 (+0.0000)

Adjusting values to match expected occupancies:
	old occupancies -> new occupancies per unit cell:
	sheet - atom type : occupancies  (difference)
		'O'   -    'ao'   : 3.8600 -> 3.9800 (+0.1200)

Splitting total iron content (0.0200) to match charge.

	Will use the following target composition:
		'T'   -    'at'   : 0.1700
		'T'   -    'st'   : 3.8300
		'O'   -    'ao'   : 3.9800
		'O'   -   'feo'   : 0.0200

Writing new target clay composition to 'KGa-1_exp_df.csv'

```


```shell
ClayCode builder -f path/to/input_Clay.yaml
```



