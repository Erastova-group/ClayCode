# Montmorillonite

## Clay composition

Wyoming montmorillonite is widelly studied and used smectite. It is available for purchace from Source Clays of [The Clay Minerals Society](https://www.clays.org), and is identified as SWy-1, SWy-2 or SWy-3, depending on the batch. 

This is a well-characterised clay with the structure listed under "Physical and Chemical Data of Source Clays" at the [clays.org](https://www.clays.org/sourceclays_data/), as:

**Swy-1:**    \(\mathrm {Ca_{0.12} Na_{0.32} K_{0.05} ~\left[ Si_{7.98} Al_{0.02}\right]^{-0.02} \left[ Al_{3.01} Fe^{III}_{0.41} Mn_{0.01} Mg_{0.54} Ti_{0.02}\right]^{-0.52} }\), <br/> with unbalanced charge of 0.05.

Ti is often identified during clay analysis, but is attributed to the TiO inclussions, therefore, we will omit it in the input. Meanwhile, ClayFF force field does not have parameters for Mn, which is shown at the level of detection and so we will also omit this entry.

The provided  `clay_comp.csv` file contains an entry `SWy-1` which corresponds to this Wyoming montmorillonite clay. 


## Construction of the model

The file  `Swy-1.yaml` is provided in the `Tutorial` directory.
