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
   - :code:`ClayFF.ff` force fields [1]_ [2]_ with added Fe parameters based on personal communication with Andrey Kalinichev. The interlayer water and solvent water are also included here.
   - :code:`Ions.ff` force fields by Pengfei Li. [3]_ [4]_ The default is IOD-type, with HFE and CN also included.
- :code:`data/UCS/` contains unit cell structures in .GRO format and their corresponding topology in .ITP format with information taken from the ClayFF force field. Currently included, grouped by type, is:
   - :code:`D21` is dioctohedral 2:1
   - :code:`D11` is dioctohedral 1:1
   - :code:`T21` is trioctohedral 2:1
  To include new UCS data see :ref:`Adding Unit Cells <adding_ucs>`.
- :code:`data/MDP/` contains version-specific GROMACS .MDP files for energy minimisation and equilibration.
- :code:`user/` reserved for user files.


.. _adding_ucs:

Adding Unit Cells
------------------

Use of ClayCode should not be dictated only by the Unit Cells provided with this release. To add a new unit cell, one needs to:

#. Obtain a crystal structure. We recommend downloading a .cif file from the `American Mineralogist Crystal Structure Database`_.
#. Convert it to a full occupancy expanded structure (.gro or .pdb). We recommend using one of the the following: `OpenBabel`_, `Avogadro`_ [5]_ (not Avogadro2) or `Mercury by CCDC`_ (licence needed).
#. Manually rename the atoms in the .gro to have unique names. 
#. Create an "include topology" file (.itp). Please refer to the `GROMACS manual`_ and assign each unique atom name in the .gro to an atom type, as given in :code:`ClayFF.ff/atomtypes.atp`.

.. _`American Mineralogist Crystal Structure Database`: http://rruff.geo.arizona.edu/AMS/amcsd.php

.. _`OpenBabel`: http://openbabel.org/wiki/Main_Page

.. _`Avogadro`: https://avogadro.cc/

.. _`Mercury by CCDC`: https://www.ccdc.cam.ac.uk/solutions/software/mercury/

.. _`GROMACS manual`: https://manual.gromacs.org/current/reference-manual/topologies/topology-file-formats.html

.. [1] Randall T. Cygan, Jian Jie Liang, and Andrey G. Kalinichev. Molecular models of hydroxide, oxyhydroxide, and clay phases and the development of a general force field. *Journal of Physical Chemistry B*, 108(4):1255–1266, 1 2004. `doi:10.1021/JP0363287`_.

.. [2] Randall T. Cygan, Jeffery A. Greathouse, and Andrey G. Kalinichev. Advances in Clayff Molecular Simulation of Layered and Nanoporous Materials and Their Aqueous Interfaces. *Journal of Physical Chemistry C*, 125(32):17573–17589, 8 2021. `doi:10.1021/ACS.JPCC.1C04600`_.

.. [3] Pengfei Li. *Advances in metal ion modeling*. Michigan State University, 2016.

.. [4] Madelyn Smith, Zhen Li, Luke Landry, Kenneth M Merz Jr, and Pengfei Li. Consequences of overfitting the van der waals radii of ions. *Journal of Chemical Theory and Computation*, 19(7):2064–2074, 2023. `doi:10.1021/acs.jctc.2c01255`_.

.. [5] Marcus D Hanwell, Donald E Curtis, David C Lonie, Tim Vandermeersch, Eva Zurek, and Geoffrey R Hutchison. Avogadro: an advanced semantic chemical editor, visualization, and analysis platform. *Journal of cheminformatics*, 4(1):1–17, 2012. `doi:10.1186/1758-2946-4-17`_.

.. _`doi:10.1021/JP0363287`: https://doi.org/10.1021/jp0363287

.. _`doi:10.1021/ACS.JPCC.1C04600`: https://doi.org/10.1021/acs.jpcc.1c04600

.. _`doi:10.1021/acs.jctc.2c01255`: https://doi.org/10.1186/1758-2946-4-17

.. _`doi:10.1186/1758-2946-4-17`: https://doi.org/10.1186/1758-2946-4-17
