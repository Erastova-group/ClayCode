.. _input_files:

Input Files
============

There are two types of input files required to run :code:`ClayCode.builder`. A mandatory :ref:`.yaml file <input_files_yaml>` and an optional :ref:`.csv file <input_files_csv>`. It should be noted that if a .csv file is omitted, these parameters should be in the .yaml file. See the :ref:`Pyrophyllite tutorial <pyro_tutorial>` for an example.

CSV File
---------

For the .csv file to be read by :code:`ClayCode.builder`, the relative path needs to be referenced in the .yaml file. See :ref:`Input files: YAML <input_files_yaml>` more for information.
A CSV file is supplied with :code:`ClayCode` within the :code:`Tutorial` directory. The file (:code:`exp_clay.csv`) contains clay structures from the `Clay Mineral Society`_'s Source Clays. :cite:p:`Sourceclays, Clays`


File Structure
~~~~~~~~~~~~~~~

The file should be comprised of at least 3 columns:

- First column - :code:`sheet` with each row containing one of the following options:

   - :code:`T` for tetrahedral occupancies

   - :code:`O` for octahedral occupancies

   - :code:`I` for interlayer ion occupancies (if the total layer charge is non-zero)

   - :code:`C` for the tetrahedral, octahedral and total unit cell charges


- Second column - :code:`element` containing either:

   - Atom names

      - The oxidation state of clay atoms can be specified by adding the charge after the element name (e.g. Fe2)
      - If the specified atom is not supported by :code:`ClayFF.ff`, :code:`ClayCode.builder` will try to assign it to a supported atom of same charge and re-assign the occupancies

   - Whether the charge (:code:`sheet: C`) is for the :code:`T` layer, :code:`O` layer or the :code:`tot` total


- Subsequent columns - the clay name(s) with each row giving the corresponding value for that clay. One of these should match the :code:`SYSNAME` option in the corresponding .yaml file.


Iron Oxidation State
*********************

There is no need to specify how much of the total iron content is Fe\ :sup:`2+` or Fe\ :sup:`3+` if at least two
charge values are given (:code:`T` layer, :code:`O` layer or the :code:`tot` total). :code:`ClayCode.builder` will
determine the iron content splitting based on these values.


Rules
~~~~~~

Occupancies:
*************

- Tetrahedral (:code:`T`) unit cell occupancies should sum to **4** for each sheet

- Dioctahedral (:code:`O`) unit cell occupancies should sum to **4** for each sheet

- Trioctahedral (:code:`O`) unit cell occupancies should sum to **3** for each sheet

For example, a dioctahedral 2:1 clay (TOT) should have 8 tetrahedral occupancies and 4 octahedral occupancies

Experimental stoichiometries do not necessarily sum up to integer occupancies. :code:`ClayCode.builder` will first process the target composition such that the occupancies match those expected for the specified unit cell type.


Interlayer Ions
****************

- Interlayer ions will be added to compensate the total charge imbalance resulting from :code:`T` and :code:`O` sheet substitutions

- The sum of all ion contributions should be **1**

- Only ion species of the opposite sign to the layer charge will be considered

Example File
~~~~~~~~~~~~~

For the :ref:`Nontronite <non_tutorial>` clays NAu-1 and NAu-2, the CSV file would be written as follows:

.. table::
   :widths: auto

   ===== ======= ====== ======
   sheet element NAu-1  NAu-2
   ===== ======= ====== ======
   T     Si        6.98   7.55
   T     Al        0.95   0.16
   T     Fe3       0.07   0.29
   O     Al        0.36   0.34
   O     Fe        3.61   3.54
   O     Mg        0.04   0.05
   I     Ca       0.525   0.36
   C     T       \-1.02 \-0.45
   C     O       \-0.03 \-0.27
   C     tot     \-1.05 \-0.72
   ===== ======= ====== ======

YAML file
----------

For the .yaml file to be read by :code:`ClayCode.builder`, the relative path must be specified when calling the function.

Multiple YAML files are supplied with :code:`ClayCode` within the :code:`Tutorial` directory. The files are named by their corresponding clay system (e.g. :code:`NAu1.yaml`), and can be used as examples.

Required Parameters
~~~~~~~~~~~~~~~~~~~~

The required input parameters are general directives for model construction.

| :code:`OUTPUT`: /path/to/directory [ *str* ]
| Path to directory in which the output folder is created


| :code:`CLAY_COMP`: /path/to/file.csv [ *str* ]
| Path to input .csv file


| :code:`SYSNAME`: clay_name [ *str* ]
| Name of the clay system. It must match a column header in the :ref:`CSV <input_files_csv>` file. The created output folder will also have this name.


| :code:`CLAY_TYPE`: unit_cell [ *str* ]
| Clay unit cell type. Must match a directory in :code:`data/UCS/` (see :ref:`data files <data_files>`). For example, a Dioctahedral 2:1 unit cell is :code:`D21`.
| *Note:* Only D11, D21 and T21 are supplied with the current release.


Optional Parameters
~~~~~~~~~~~~~~~~~~~~

There are a number of optional parameters. If no directives are given by the user, :code:`ClayCode.builder` will use the default values.

Clay Sheet Size
****************

| :code:`X_CELLS`: x_unit_cells [ *int* ] (Default = 5)
| Number of unit cells in the x-direction

| :code:`Y_CELLS`: y_unit_cells [ *int* ] (Default = 5)
| Number of unit cells in the y-direction

| :code:`N_SHEETS`: sheet_number [ *int* ] (Default = 5)
| Number of sheets of 1 unit cell thickness to be stacked in the z direction

Clay Composition
*****************

:code:`ClayCode.builder` uses the supplied .csv file to calculate the number and type of unit cells necessary to match the desired composition. The way in which ClayCode will match this target composition can be specified.

| :code:`OCC_TOL`: occupation_tolerance [ *float* ] (Default = 0.1)
| The maximum occupancy/charge deviation for a unit cell that is adjusted without querying to match the expected values.
| *E.g.* with an expected :code:`csv:T` total unit cell occupancy of 4 and :code:`OCC_TOL` of 0.1, any composition with tetrahedral occupancies between 3.9 and 4.1 will be automatically adjusted to match the expected value.

| :code:`ZERO_THRESHOLD`: threshold [ *float* ] (Default = 0.05)
| The occupancy threshold below which the matched composition will be set to 0 if the element is not found in the force field or in the unit cell database.

| :code:`SEL_PRIORITY`: charge_correction [ *str* ] (Default = charges)
| The priority to use when correcting charges and substitution occupancies.
| *E.g.* An octahedral charge of -0.3 and a "mgo" occupancy of 0.4 is not possible, and one needs to be adjusted.
| There are two possible options:
| 1. :code:`charges`: Conserve specified charges and adjust substitution occupancies. *E.g. mgo: 0.4 -> 0.3*
| 2. :code:`occupancies`: Conserve occupancies and adjust charges. *E.g. -0.3 e -> -0.4 e*

| :code:`CHARGE_PRIORITY`: charge_correction [ *str* ] (Default = total_charge)
| The priority to use when individual sheet and total charge do not match.
| *E.g.* total charge = -1.00 but tetrahedral charge = -0.70 and octahedral charge = -0.40
| There are two possible options:
| 1. :code:`total_charge`: Conserve the total charge and adjust tetrahedral and octahedral occupancies. *E.g. tetrahedral charge: -0.70 -> -0 .65 and octahedral charge: -0.40 -> -0.35*
| 2. :code:`sheet_charges`: Conserve the sheet charges and adjust total charge. *E.g. total charge: -1.00 -> 1.10*

| :code:`MATCH_TOLERANCE`: tolerance [ *float* ] (Default = 0.02)
| The maximum total occupancy deviation from the corrected target stoichiometry.

No CSV File
*******************

The clay composition can also be supplied manually in the .yaml file using the following parameters. See the :ref:`Pyrophyllite tutorial <pyro_tutorial>` for an example.

| :code:`UC_INDEX_LIST`: unit_cells [ *list* ]
| List of unit cells to be used for building the model. These must have corresponding :code:`.gro` and :code:`.itp` files in the :code:`data/UCS/CLAY_TYPE` directory.

| :code:`UC_RATIOS_LIST`: cell_probabilities [ *list* ]
| List of probabilities for the above unit cells. The total probability must equal 1.

| :code:`IL_ION_RATIOS`: ions_and_ratios [ *dict[str: int]* ] (Default = Ca: 1 Cl: 1)
| The interlayer ions to be used and the ratio between them. The sum of all cation/anion contributions should be **1**. Only ion species of the opposite sign to the layer charge will be considered.

Interlayer Solvent and Ions
****************************

| :code:`IL_SOLV`: solvent_presence [ *bool* ] (Default = True)
| If false a non-hydrated clay will be created. If true the interlayer space will be solvated, and **one** further option is needed on how to handle it. The default option is :code:`UC_WATERS`: 20.

| 1. :code:`ION_WATERS`: waters_per_ion [ *int, dict[str: int]* ]
| The number of water molecules that should be added per interlayer ion. If a single integer is given, all ions will be hydrated by the same number of water molecules. To specify the number of waters per ion-type a dictionary should be given.

| 2. :code:`UC_WATERS`: waters_per_uc [ *int* ] (Default = 20)
| The number of water molecules to add per unit cell. The total number of water added is the number specified multiplied by the number of unit cells used to create the model.

| 3. :code:`SPACING_WATERS`: hydrated_spacing [ *float* ]
| The target hydrated interlayer spacing, in angstroms (Å). The final value may vary due to the water rearrangement when in contact with clay surface and packing around ions.

Simulation Box
***************

| :code:`BOX_HEIGHT`: z_box_length [ *float* ] (Default = 15.0)
| The size of the final simulation box along the z-axis, in nanometers (nm).
| *Note:* The clay layers are positioned in the xy-plane.

| :code:`BULK_SOLV`: solvent_presence [ *bool* ] (Default = True)
| If true the box space will be solvated. If false the box will be empty. This is useful if the further plan is to add other species, such as oil.

| :code:`BULK_IONS`: ion_type_conc [ *dict[str: int]* ] (Default = 'Na': 0.0 'Cl': 0.0)
| The type and concentration (in mol/L) of ions to to be added into the bulk space.
| *Note:* GROMACS will raise a warning if the system isn't neutral.

GROMACS version
****************

| :code:`GMX`: bash_alias [ *str* ]
| Allows the user to specify which version of GROMACS to use if they have multiple installed.

.. _`Clay Mineral Society`: https://www.clays.org

.. bibliography::
   :style: plain
   :filter: False

   Sourceclays
   Clays


.. toctree::
   :maxdepth: 1
   :hidden:

   ./input_files/csv
   ./input_files/yaml
