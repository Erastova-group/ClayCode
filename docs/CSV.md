# Clay composition in CSV format

## Including into Input

Inclusion of the CSV file into the [INPUT .YAML file](YAML.md):

- the path to the CSV file is specified under `CLAY_COMP`, 
- target occupancies from the CSV file are given by `SYSNAME`, corresponding to the clay name

e.g.:

```yaml
CLAY_COMP: /path/to/clay_comp.csv
SYSNAME: NAu-1-fe

```


## File Structure


- First column: `sheet`
     - for tetrahedral (`T`) and octahedral (`O`) occupancies;
     - interlayer ion  (`I`) ocupansies, if the total charge (`tot` second column) is non-zero;
     - `T`, `O` and total (`tot`) average unit cell charge (`C`).

- Second column with `element`:
    - Atom names (if atom is not supported by ClayFF.FF, `ClayCode.builder` will try to assign it to the supported atom of same charge and re-assign the ocupansies);
    - charges for the `T` layer, `O` layer and `tot` total;
    - The oxidation state of clay atoms can be specified by adding the charge after the element name (e.g. Fe2).
   

### Rules

**Occupancies:**

 - Total occupancies for each dioctahedral/trioctahedral tetrahedral (`T`) sheet should sum up to 4.
 - For octahedral (`O`) sheets, the total occupancies should amount to 4/3 for dioctahedral/trioctahedral unit cell types, respectively.
 - For example, the occupancies of a dioctahedral 2:1 clay (TOT) should have `T` occupancies of 8, and `O` occupancies of 4;
 - Experimental stoichiometries do not necessarily sum up to interger occupancies. `ClayCode.builder` will first process the target composition such that the occupancies match those expected for the specified [unit cell type](userguide.md## Data files).

 **Interlayer Ions:**

- Interlayer ions will be added to compensate the total charge imbalance resulting from substitutions.
- The ratio of these ions can be specified in the `I` section. The sum of all ion contributions should sum to 1.
- Only ion species of the opposite sign to the layer charge will be considered.


**Iron 2+/3+:**

- There is no need to specify how much of the total iron is Fe3+ or Fe2+ if at least two values among the total, `T` and `O` unit cell charge are specified.
- `ClayCode.builder` can perform the splitting between the two iron species.




## Supplied CSV file

The supplied file within directory `Tutorial` is `exp_clay.csv` and contains clay structures corresponding to the **Source Clays** listed under "Physical and Chemical Data of Source Clays" [clays.org](https://www.clays.org/sourceclays_data/).

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



## Example 

Example from `clay_comp.csv` file for NAu-1 and NAu-2 [nontronite](nont.md) clays:


| **sheet** | **element** | **NAu\-1\-fe** | **NAu\-2\-fe** |
|:----------|:------------|---------------:|---------------:|
| **T**     | **Si**      | 6.98          |          7.55 |
| **T**     | **Al**      | 0.95          |          0.16 |
| **T**     | **Fe3**     | 0.07          |          0.29 |
| **O**     | **Fe**      | 3.61          |          3.54 |
| **O**     | **Al**      | 0.36          |          0.34 |
| **O**     | **Mg**      | 0.04          |          0.05 |
| **I**     | **Ca**      | 0.525         |          0.36 |
| **I**     | **Na**      | 0             |             0 |
| **I**     | **Mg**      | 0             |             0 |
| **I**     | **K**       | 0             |             0 |
| **I**     | **Cl**      | 0             |             0 |
| **C**     | **T**       | \-1.02        |        \-0.45 |
| **C**     | **O**       | \-0.03        |        \-0.27 |
| **C**     | **tot**     | \-1.05        |        \-0.72 |


