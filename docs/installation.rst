.. _installation:


Installation
=============

Dependencies
-------------

ClayCode is compatible with UNIX operating systems and requires `python`_ version 3.9, `pipx`_ and a local `GROMACS`_ installation. It relies on the following python libraries:

 - `NumPy`_ (:math:`\geq` 1.21.2)

 - `Pandas`_ (:math:`\geq` 1.3.4)

 - `MDAnalysis`_ (:math:`\geq` 2.0.0)


.. _`python`: https://docs.python.org/3/using/index.html
.. _`pipx`: https://pypa.github.io/pipx/
.. _`GROMACS`: https://manual.gromacs.org/documentation/current/install-guide/index.html
.. _`Numpy`: https://numpy.org/doc/stable/user/index.html
.. _`Pandas`: https://pandas.pydata.org/docs/getting_started/index.html
.. _`MDAnalysis`: https://userguide.mdanalysis.org/stable/index.html

ClayCode
---------

The package can be installed by cloning the source code repository and executing the bash installation script.

.. code-block:: bash

   git clone https://github.com/Erastova-group/ClayCode.git
   cd ClayCode
   bash install.sh