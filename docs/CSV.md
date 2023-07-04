Clay composition in CSV format:

`CLAY_COMP` specifies the path of a CVS file with the following information:

   - First column: `sheet`
     - for tetrahedral (`T`) and octahedral (`O`) occupancies
     - interlayer ion ratios (`I`) for interlayer ions if the total later charge is non-zero
     - `T`, `O` and total (`tot`) average unit cell charge (`C`)

   - Second column with `element`:
     - (ClayFF) atom types and charge categories
       - The oxidation state of clay atoms can be specified by addig the charge after the element name (e.g. Fe2)
   
   - Target occupancies, header is `SYSNAME`

Example of a `clay_comp.csv` file for NAu-1 and NAu-2 nontronite clays:

Total occupancies for each dioctahedral/trioctahedral tetrahedral sheet should sum up to 4.
For octahedral sheets, the total occupancies should amount to 4/3 for dioctahedral/trioctahedral unit cell types.
For example, the occupancies of a dioctahedral 2:1 clay (TOT) should have T occupancies of 8, and O occupancies of 4.

There is no need to specify how much of the total iron is Fe3+ or Fe2+ if at least two values among the total, T and O unit cell charge are specified.
`ClayCode.builder` can perform the splitting between the two iron species.

Experimental stoichiometries do not necessarily sum up to interger occupancies. `ClayCode.builder` will first process the target composition such that the occupancies match those expected for the specified unit cell type.

Interlayer ions will be added to compensate the total charge imbalance resulting from substitutions. 
The ratio of these ions can be specified in the `I` section. The sum of all ion contributions should sum to 1.
Only ion species of the opposite sign to the layer charge will be considered.

| **sheet** | **element** | **NAu\-1\-fe** | **NAu\-2\-fe** |
|:----------|:------------|---------------:|---------------:|
| **T**     | **Si**      | 6\.98          |          7\.55 |
| **T**     | **Al**      | 0\.95          |          0\.16 |
| **T**     | **Fe3**     | 0\.07          |          0\.29 |
| **O**     | **Fe**      | 3\.61          |          3\.54 |
| **O**     | **Al**      | 0\.36          |          0\.34 |
| **O**     | **Mg**      | 0\.04          |          0\.04 |
| **I**     | **Ca**      | 1              |           0\.5 |
| **I**     | **Na**      | 0              |           0\.5 |
| **I**     | **Mg**      | 0              |              0 |
| **I**     | **K**       | 0              |              0 |
| **I**     | **Cl**      | 0              |              0 |
| **C**     | **T**       | \-1\.02        |        \-0\.45 |
| **C**     | **O**       | \-0\.03        |        \-0\.27 |
| **C**     | **tot**     | \-1\.05        |        \-0\.72 |
