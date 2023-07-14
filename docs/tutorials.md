
# Tutorials

In this section we show how to set up the following clay systems:


 * Wyoming [Montmorillonite](mmt.md),
 * [Illite](imt.md),
 * [Pyrophyllite](pyr.md),
 * Brown and green Uley [nontronites](nont.md),
 * [Ferruginous Smectite](swa.md),
 * [Kaolinite](kao.md),
 * [Layered double hydroxide](ldh.md).

All the necessary input files (the `.CSV` and corresponding `.YAML`  files) can be found in the `Tutorial` directory.


The supplied `exp_clay.csv` file and contains clay structures corresponding to the **Source Clays** listed under "Physical and Chemical Data of Source Clays" [clays.org](https://www.clays.org/sourceclays_data/).

* 1:1 (TO) dioctohedral smectite:

    * Kaolin KGa-1;

* 2:1 (TOT) dioctohedral smectite:

    * Na-Montmorillonite (Wyoming) SWy-1,
    * Illite IMt-1,
    * Ferruginous Smectite SWa-1,  
    * Nontronite NAu-1-fe,
    * Nontronite NAu-2-fe,
    * Nontronite NG-1; 

* example Al-Mg 3:1 layered double hydroxide:

    * LDH31.

Note, that Pyrophyllite composition is not given in the .CSV file, and it s fully described in the .YAML.


To run ClayCode:

```shell
ClayCode builder -f path/to/input_Clay.yaml
```

Your output will be a directory containing all the files for simulation:)


