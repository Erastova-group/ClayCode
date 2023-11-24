.. _data_files:

Data Files
===========

The data files store the information necessary to construct the clay structures for MD simulation.

The files are stored in the :code:`ClayCode/package/ClayCode/data` directory: 

.. code-block:: bash

   data
   │
   ├── user
   └── data
       ├── FF
       ├── MDP
       └── UCS

Each subdirectory contains the data files of that kind where:

- :code:`data/FF/` contains force field files, as dictated by GROMACS format. Currently included is: 

   * :code:`ClayFF.ff` force fields :cite:p:`Cygan2004, Cygan2021` with added Fe parameters based on personal communication with Andrey Kalinichev. The interlayer water and solvent water are also included here.
   
   * :code:`Ions.ff` force fields by Pengfei Li. :cite:p:`Li2016, Smith2023` The default is IOD-type, with HFE and CN also included.
   
- :code:`data/UCS/` contains unit cell structures in .GRO format and their corresponding topology in .ITP format with information taken from the ClayFF force field. Currently included, grouped by type, is:

   * :code:`D21` is dioctohedral 2:1
   
   * :code:`D11` is dioctohedral 1:1
   
   * :code:`T21` is trioctohedral 2:1
   
  To include new UCS data see :ref:`Adding Unit Cells <adding_ucs>`.
  
- :code:`data/MDP/` contains version-specific GROMACS .MDP files for energy minimisation and equilibration.

- :code:`user/` reserved for user files.


.. _adding_ucs:

Adding Unit Cells
------------------

Use of ClayCode should not be dictated only by the Unit Cells provided with this release. To add a new unit cell, one needs to:

#. Obtain a crystal structure. We recommend downloading a .cif file from the `American Mineralogist Crystal Structure Database`_.

#. Convert it to a full occupancy expanded structure (.gro or .pdb). We recommend using one of the the following:
`OpenBabel`_, `Avogadro`_ :cite:p:`Thanwell2012` (not Avogadro2) or `Mercury by CCDC`_ (licence needed).

#. Manually rename the atoms in the .gro to have unique names.

#. Create an "include topology" file (.itp). Please refer to the `GROMACS manual`_ and assign each unique atom name in the .gro to an atom type, as given in :code:`ClayFF.ff/atomtypes.atp`.

.. _`American Mineralogist Crystal Structure Database`: http://rruff.geo.arizona.edu/AMS/amcsd.php

.. _`OpenBabel`: http://openbabel.org/wiki/Main_Page

.. _`Avogadro`: https://avogadro.cc/

.. _`Mercury by CCDC`: https://www.ccdc.cam.ac.uk/solutions/software/mercury/

.. _`GROMACS manual`: https://manual.gromacs.org/current/reference-manual/topologies/topology-file-formats.html

ClayFF.ff/atomtypes.atp
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash
    
    hw      1.008        ; water hydrogen
    ho      1.008        ; hydroxyl hydrogen
    ow      16.00        ; water hydrogen
    oh      16.00        ; hydroxyl oxygen
    ob      16.00        ; bridging oxygen
    obos    16.00        ; bridging oxygen with octahedral substitution
    obts    16.00        ; bridging oxygen with tetrahedral substitution
    obss    16.00        ; bridging oxygen with double substitution 
    ohs     16.00        ; hydroxyl oxygen with substitution
    st      28.09        ; tetrahedral silicon
    ao      26.98        ; octahedral aluminum
    at      26.98        ; tetrahedral aluminum
    mgo     24.31        ; octahedral magnesium
    mgh     24.31        ; hydroxide magnesium
    cao     40.08        ; octahedral calcium
    cah     40.08        ; hydroxide calcium 
    feo     55.85        ; octahedral iron (III)
    fe2     55.85        ; octahedral iron (II)
    lio     6.941        ; octahedral lithium

Example UC.gro
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash
    
    Dioctahedral 1:1 unit cell  1
   34
    1D101   AO1    1   0.061   0.433   0.332
    1D101   AO2    2   0.321   0.283   0.332
    1D101   AO3    3   0.320   0.880   0.332
    1D101   AO4    4   0.064   0.730   0.332
    1D101   ST1    5   0.237   0.749   0.065
    1D101   ST2    6   0.500   0.594   0.067
    1D101   ST3    7   0.493   0.301   0.065
    1D101   ST4    8   0.244   0.147   0.067
    1D101   OB1    9   0.225   0.751   0.226
    1D101   OB2   10   0.255   0.135   0.227
    1D101   OB3   11   0.258   0.000   0.000
    1D101   OB4   12   0.359   0.651   0.021
    1D101   OB5   13   0.360   0.236   0.001
    1D101   OB6   14   0.480   0.304   0.226
    1D101   OB7   15   0.510   0.582   0.227
    1D101   OB8   16   0.002   0.447   0.000
    1D101   OB9   17   0.100   0.204   0.021
    1D101  OB10   18   0.104   0.683   0.001
    1D101   OH1   19   0.223   0.413   0.232
    1D101   OH2   20   0.123   0.581   0.433
    1D101   OH3   21   0.164   0.855   0.431
    1D101   OH4   22   0.162   0.306   0.434
    1D101   OH5   23   0.480   0.860   0.232
    1D101   OH6   24   0.379   0.134   0.433
    1D101   OH7   25   0.420   0.408   0.431
    1D101   OH8   26   0.420   0.753   0.434
    1D101   HO1   27   0.530   0.940   0.233
    1D101   HO2   28   0.410   0.129   0.527
    1D101   HO3   29   0.400   0.434   0.522
    1D101   HO4   30   0.137   0.264   0.519
    1D101   HO5   31   0.272   0.497   0.233
    1D101   HO6   32   0.150   0.576   0.527
    1D101   HO7   33   0.136   0.880   0.522
    1D101   HO8   34   0.400   0.712   0.519
   0.51540   0.89420   0.63910

Example UC.itp
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash
    
    [ moleculetype ]
    ; name      nrexcl
       D101     1

    [ atoms ]
    ;   nr       type  resnr residue  atom   cgnr     charge       mass  typeB    chargeB      massB
    ; residue   1  KAO rtp  KAO  q  0.0
         1         ao      1     D101   AO1      1      1.575      26.98   ;
         2         ao      1     D101   AO2      2      1.575      26.98   ;
         3         ao      1     D101   AO3      3      1.575      26.98   ;
         4         ao      1     D101   AO4      4      1.575      26.98   ;
         5         st      1     D101   ST1      5        2.1      28.09   ;
         6         st      1     D101   ST2      6        2.1      28.09   ;
         7         st      1     D101   ST3      7        2.1      28.09   ;
         8         st      1     D101   ST4      8        2.1      28.09   ;
         9         ob      1     D101   OB1      9      -1.05         16   ;
        10         ob      1     D101   OB2     10      -1.05         16   ;
        11         ob      1     D101   OB3     11      -1.05         16   ;
        12         ob      1     D101   OB4     12      -1.05         16   ;
        13         ob      1     D101   OB5     13      -1.05         16   ;
        14         ob      1     D101   OB6     14      -1.05         16   ;
        15         ob      1     D101   OB7     15      -1.05         16   ;
        16         ob      1     D101   OB8     16      -1.05         16   ;
        17         ob      1     D101   OB9     17      -1.05         16   ;
        18         ob      1     D101  OB10     18      -1.05         16   ;
        19         oh      1     D101   OH1     19      -0.95         16   ;
        20         oh      1     D101   OH2     20      -0.95         16   ;
        21         oh      1     D101   OH3     21      -0.95         16   ;
        22         oh      1     D101   OH4     22      -0.95         16   ;
        23         oh      1     D101   OH5     23      -0.95         16   ;
        24         oh      1     D101   OH6     24      -0.95         16   ;
        25         oh      1     D101   OH7     25      -0.95         16   ;
        26         oh      1     D101   OH8     26      -0.95         16   ;
        27         ho      1     D101   HO1     27      0.425      1.008   ;
        28         ho      1     D101   HO2     28      0.425      1.008   ;
        29         ho      1     D101   HO3     29      0.425      1.008   ;
        30         ho      1     D101   HO4     30      0.425      1.008   ;
        31         ho      1     D101   HO5     31      0.425      1.008   ;
        32         ho      1     D101   HO6     32      0.425      1.008   ;
        33         ho      1     D101   HO7     33      0.425      1.008   ;
        34         ho      1     D101   HO8     34      0.425      1.008   ;

    [ bonds ]
    ; i j   funct   length  force.c.                    
    19 31   1     0.1    463532.808
    20 32   1     0.1    463532.808
    21 33   1     0.1    463532.808
    22 30   1     0.1    463532.808
    23 27   1     0.1    463532.808
    24 28   1     0.1    463532.808
    25 29   1     0.1    463532.808
    26 34   1     0.1    463532.808

.. bibliography::
   :style: plain
   :filter: False

   Cygan2021
   Cygan2004
   Li2016
   Thanwell2012
   Smith2023
