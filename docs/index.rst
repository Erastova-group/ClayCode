.. using logo instead of a Title
.. image:: images/Logo.png
   :alt ClayCode logo
   :align: center

.. buttons under heading
|Project Status: Active| |License: MIT|

.. |Project Status: Active| image:: https://www.repostatus.org/badges/latest/active.svg
   :target: https://www.repostatus.org/#active
.. |License: MIT| image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   
.. description of claycode
**ClayCode** generates clay mineral structures based upon their natural partially occupied unit cell compositions and assigns ClayFF forecefield parameters, generating all the input files ready for simulation with Gromacs software.


Documentation
==============

.. outline of docs
 
:ref:`Quickstart`
------------------
Installing and running ClayCode

:ref:`User Guide`
------------------
Comprehensive information on how to use ClayCode

:ref:`Tutorials`
----------------
Practical step-by-step how-to guides

:API:`API`
----------
Technical documentation for developers

Source Code
===========
|Project Status: Active| |GitHub commit activity (branch)| |GitHub issues| |GitHub pull requests|

.. |Project Status: Active| image:: https://www.repostatus.org/badges/latest/active.svg
   :target: https://www.repostatus.org/#active
.. |GitHub commit activity (branch)| image:: https://img.shields.io/github/commit-activity/m/Erastova-group/ClayCode
.. |GitHub issues| image:: https://img.shields.io/github/issues/Erastova-group/ClayCode
.. |GitHub pull requests| image:: https://img.shields.io/github/issues-pr/Erastova-group/ClayCode

The source code is available on `GitHub`_.

**Report bugs and give sugestions** for future improvements and new features via an `issue`_.

**Have you added new functionality to the code, or assigned new clay unit cells?** Please share with the wider community by `forking this project`_ and submitting a `pull request`_. Learn more how to contribute `here`_.

.. _`GitHub`: https://github.com/Erastova-group/ClayCode/tree/main
.. _`issue`: https://github.com/Erastova-group/ClayCode/issues
.. _`forking this project`: https://github.com/Erastova-group/ClayCode/fork
.. _`pull request`: https://github.com/Erastova-group/ClayCode/pulls
.. _`here`: https://docs.github.com/en/get-started/quickstart/contributing-to-projects

Citation
========
ClayCode is developed by Hannah Pollak, Matteo Degiacomi and Valentina Erastova, University of Edinburgh, 2023.

Please CITE us: HP, MTD, VE "ClayCode: setting up clay structures from simulation in Gromacs", DOI: XXX


.. toctree::
   :maxdepth: 1
   :hidden:
   
   Home <self>
   quickstart
   API_docs

.. toctree::
   :maxdepth: 1
   :caption: User Guide
   :hidden:
   
   ./user_guide/data_files
   ./user_guide/modules
   ./user_guide/input_files
   ./user_guide/adding_unit_cells
   ./user_guide/output_files
   
.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   :hidden:
   
   ./tutorials/montmorillonite
   ./tutorials/illite
   ./tutorials/pyrophylite
   ./tutorials/nontronite
   ./tutorials/fe_smectite
   ./tutorials/kaolinite
   ./tutorials/ldh
   
   
   
   