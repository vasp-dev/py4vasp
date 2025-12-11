py4vasp
=======

*py4vasp* is a Python interface to extract data from VASP calculations. It is
intended mainly to get a quick look at the data and provide the functionality to
export it into common formats that can be used by other more sophisticated
postprocessing tools. The second domain of application is for people that want to
write Python scripts based on the data calculated by VASP. This tool interfaces
directly with the new HDF5 file format and thereby avoids parsing issues
associated with the XML or OUTCAR files.

For these two groups of users, we provide a different level of access. The
simple routines used in the tutorials will read the data from the file directly
and then generate the requested plot. For script developers, we provide interfaces
to convert the data to Python dictionaries for further processing. If I/O access
limits the performance, you can lazily load the data only when needed.

Installation
------------
.. _PyPI: https://pypi.org/project/py4vasp

While this is not required to be able to run *py4vasp*, you may want to consider
creating a separate environment for installation to avoid interference with other
installed packages. [#environment]_
You can then install *py4vasp* from PyPI_ using the pip package installer

.. code-block:: bash

  pip install py4vasp

This will automatically download *py4vasp* and its required dependencies.

For a minimalistic setup where you use py4vasp as a library, you can install the
core package

.. code-block:: bash

  pip install py4vasp-core

The core package contains the same source code as the main package and does not
impact the usage. However, it does not install any of the dependencies of *py4vasp*
except for *numpy* and *h5py*. Hence, this core package is most suitable for
script developers that do not need all the visualization features of *py4vasp*.

Alternatively, you can obtain the code from GitHub and install it. This will give you
the most recent version with all bugfixes. However, some features may only work once
the next VASP version is released.

.. code-block:: bash

  git clone https://github.com/vasp-dev/py4vasp.git
  cd py4vasp
  pip install .

If these commands succeed, you should be able to use *py4vasp*. You can make a quick
test of your installation running the following command

.. code-block:: bash

  python -c "import py4vasp; print(py4vasp.__version__)"

This should print the version of *py4vasp* that you installed.

.. important::
  *py4vasp* extracts all information from the HDF5 output so you need to make
  sure to compile VASP adding ``-DVASP_HDF5`` to the ``CPP_OPTIONS`` in the
  *makefile.include*. You will also need to add the HDF5 library to the include
  (``INCS``) and linking (``LLIBS``) instructions. *py4vasp* also requires a
  VASP version > 6.2 and because py4vasp is developed alongside VASP, we
  recommend that you use versions of these two codes released about at the same
  time for maximum compatibility.

Quick start
-----------

.. _Jupyter: https://jupyter.org/

The user interface of *py4vasp* is optimized for usage inside a Jupyter_ environment
(Jupyter notebook or Jupyter lab), though it can be used in regular Python scripts
as well. To give you an illustrative example of what *py4vasp* can do, we assume
that you created a Jupyter notebook inside the directory of your VASP calculation.
In the VASP calculation, you computed the density of states (DOS) with orbital
projections (:tag:`LORBIT` = 11). You may now want to read the data from your
VASP calculation to post-process it further with a script. This can be achieved with

>>> import py4vasp
>>> dos = py4vasp.calculation.dos.read()

Under the hood, this will access the *vaspout.h5* file, because *py4vasp* knows where
the output is stored after you ran VASP. It will read the relevant tags from
the file and store them all in a Python dictionary. If you want to access particular
orbital projections, let's say the *p* orbitals, you can pass a ``selection = "p"`` as
an argument to the routine. More generally, you can check how to use a function with

>>> help(py4vasp.calculation.dos.read)

The most common use case for the DOS data may be to prepare a plot to get some
insight into the system of interest. Because of this, we provide an easy wrapper
for this particular functionality

>>> py4vasp.calculation.dos.plot()

This will return an interactive figure that you can use to investigate the DOS.
The *plot* command takes the same arguments as the read command. Note that this
requires a browser to work; if you execute this from within a interactive
environment, it may open a browser for you or you can enforce it by appending
`.show()`

>>> py4vasp.calculation.dos.plot().show()

The interface for the other quantities is very similar. Every quantity provides
a *read* function to get the raw data into Python and where it makes sense a
*plot* function visualizes the data. However, note that in particular, all data
visualized inside the structure require a Jupyter notebook to work.
All plots can be converted to csv files `to_csv` of pandas dataframes `to_frame`
for further refinement.

If your calculation is not in the root directory, you can create your own
instance

>>> from py4vasp import Calculation
>>> calc = Calculation.from_path("/path/to/your/VASP/calculation")

The attributes of the calculation correspond to different physical quantities that
you could have calculated with VASP. If you have an interactive session you can type
``calc.`` and then hit :kbd:`Tab` to get a list of all possible quantities. However
only the ones that you computed with VASP will give you any meaningful
result.

.. _tutorials: https://www.vasp.at/tutorials/latest

If you want to experience more features of *py4vasp*, we highly recommend taking
a look at the tutorials_ for VASP. Many of them use *py4vasp* to plot or analyze
the data produced by VASP, so this may give you an excellent starting point to learn
how you can apply *py4vasp* in your research.

.. rubric:: Optional dependencies

You can install mdtraj if you want to analyze molecular dynamics trajectories
beyond the pair correlation function. We recommend using conda for the installation
which we found to be more robust than pip.

.. toctree::
   :hidden:
   :glob:

   calculation/*
   
.. currentmodule:: py4vasp

.. autosummary::

   calculation
   Calculation


----------------------------------------------------------------------------------------

.. _venv: https://docs.python.org/3/tutorial/venv.html
.. _conda: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

.. [#environment] To create a separate *py4vasp* from the rest of your packages you
  can create an environment with venv_ or conda_. The advantage of the former is
  that it comes with your Python installation, the latter requires the installation
  of Miniconda or Anaconda. Nevertheless, it may be a good idea to go with *conda*
  in particular on Windows and macOS, because it can help install dependencies
  of *py4vasp*. Below you find the instructions on how to create the environment
  depending on your environment managing tool and your operating system.

  venv (Linux / MacOS)
    .. code-block:: bash

      python3 -m venv py4vasp-env
      source py4vasp-env/bin/activate.sh

  venv (Windows)
    .. code-block:: bash

      python3 -m venv py4vasp-env
      py4vasp-env\Scripts\activate.bat

  conda (Linux / MacOS / Windows)
    .. code-block:: bash

      conda create --name py4vasp-env python
      conda activate py4vasp-env

  .. note::
    You will need to run the activation part of the command again if you open a
    new shell.
