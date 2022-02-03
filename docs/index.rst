py4vasp
=======

*py4vasp* is a python interface to extract data from VASP calculations. It is
intended mainly to get a quick look at the data and provide functionality to
export it into common formats that can be used by other more sophisticated
postprocessing tools. A second domain of application is for people that want to
write python scripts based on the data calculated by VASP. This tool interfaces
directly with the new HDF5 file format and thereby avoids parsing issues
associated with the XML or OUTCAR files.

For these two groups of users, we provide a different level of access. The
simple routines used in the tutorials will read the data from the file directly
and then generate the requested plot. For script developers, we provide an
expert interface were the data is lazily loaded as needed with some greater
flexibility when the data file is opened and closed.

Installation
------------
.. _PyPI: https://pypi.org/project/py4vasp

While this is not required to be able to run *py4vasp*, you may want to consider
creating a separate a environment for installation to avoid interference with other
installed packages.[#environment]_
You can then install *py4vasp* from PyPI_ using the pip package installer

.. code-block:: bash

  pip install py4vasp

This will automatically download *py4vasp* as well as all the required dependencies.
However, we noticed that this approach is not fail-safe, because the installation
of the *mdtraj* dependency does not work on all operating systems. So in case that
the simple installation above fails, you may need to use *conda* to install *mdtraj*

.. code-block:: bash

  conda install -c conda-forge mdtraj
  pip install py4vasp

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
  VASP version > 6.2 and because py4vasp is developed alongside to VASP, we
  recommend that you use versions of these two codes released about at the same
  time for maximum compatibility.

Quickstart
----------

.. _Jupyter: https://jupyter.org/

The user interface of *py4vasp* is optimized for usage inside a Jupyter_ environment
(Jupyter notebook or Jupyter lab), though it can be used in regular Python scripts
as well. To give you an illustrative example of what *py4vasp* can do, we assume
that you created a Jupyter notebook inside the directory of your VASP calculation.
Then you access all the results of this calculation with

.. code-block:: python

  from py4vasp import Calculation
  calc = Calculation.from_path(".")

Naturally, if you created the notebook outside of the calculation directory, you
would replace the path ``.`` with the directory of the calculation.

The attributes of the calculation correspond to different physical quantities that
you could have calculated with VASP. If you have an interactive session you can type
``calc.`` and then hit :kbd:`Tab` to get a list of all possible quantities. However
only the ones that you actually computed with VASP will give you any meaningful
result.

.. _LORBIT: https://www.vasp.at/wiki/index.php/LORBIT

In the following, we will assume that you computed the density of states (DOS) with
orbital projections (LORBIT_ = 11). You may now want to read the data from your
VASP calculation to postprocess it further with a script. This can be achieved in
a single line of code

.. code-block:: python

  dos = calc.dos.read()

Under the hood, this will access the *vaspout.h5* file, because *py4vasp* knows that
the output is stored there after you ran VASP. It will read the relevant tags from
the file and store it all in a Python dictionary. If you want to access particular
orbital projections, let's say the *p* orbitals, you can pass a ``select = "p"`` as
argument to the routine. More generally, you can check how to use a function with

.. code-block:: python

  help(calc.dos.read)

The most common use case for the DOS data may be to prepare a plot to get some
insight into the system of interest. Because of this, we provide an easy wrapper
for this particular functionality

.. code-block:: python

  calc.dos.plot()

This will return an interactive figure that you can use to investigate the DOS.
Note that this requires a browser to work, which means it will open one if you
execute this inside a script instead of a Jupyter notebook. The *plot* command
takes the same arguments as the read command.

The interface for the other quantities is very similar. Every quantity provides
a *read* function to get the raw data into Python and were it makes sense a
*plot* function visualizes the data. However, note that in particular all data
visualized inside the structure require a Jupyter notebook to work.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules/index

----------------------------------------------------------------------------------------

.. _venv: https://docs.python.org/3/tutorial/venv.html
.. _conda: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

.. [#environment] To create a separate *py4vasp* from the rest of you packages you
  can create an environment with venv_ or conda_. The advantage of the former is
  that it comes with your python installation, the latter requires the installation
  of Miniconda or Anaconda. Nevertheless it may be a good idea to go with *conda*
  in particular on Windows and MacOS, because it can help installing dependencies
  of *py4vasp*. Below you find the instructions how to create the environment
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
    You will need to run the activation part of the command again, if you open a
    new shell.
