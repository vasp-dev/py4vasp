# py4vasp

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![test-full](https://github.com/vasp-dev/py4vasp/actions/workflows/test_full.yml/badge.svg)](https://github.com/vasp-dev/py4vasp/actions/workflows/test_full.yml)
[![test-core](https://github.com/vasp-dev/py4vasp/actions/workflows/test_core.yml/badge.svg)](https://github.com/vasp-dev/py4vasp/actions/workflows/test_core.yml)

> Please note that this document is intended mostly for developers that want to use
> the version of py4vasp provided on Github. If you just want to install py4vasp to
> use it, please follow the [official documentation](https://vasp.at/py4vasp/latest).

## Installation

We use the [poetry dependency manager](https://python-poetry.org/) which takes care of
all dependencies and maintains a virtual environment to check the code. If you want to
test something in the virtual environment, just use e.g. `poetry run jupyter-notebook`.

We recommend installing py4vasp in a conda environment to resolve issues related to
installing `mdtraj` with pip. To do this please use the following steps. The last step
will test whether everything worked
~~~shell
conda create --name py4vasp-env python=3.8
git clone git@github.com:vasp-dev/py4vasp.git
pip install poetry
poetry install
conda install -c conda-forge mdtraj
poetry run pytest
~~~
Note that this will install py4vasp into the conda environment. This isolates the code
from all packages you have installed in other conda environments. Using poetry makes
sure that when you modify the code all the relevant dependencies are tracked.

## py4vasp core

If you want to use py4vasp to develop your own scripts, you may want to limit the amount
of external dependencies. To this end, we provide alternative configuration files that
only install numpy, h5py, and the development dependencies. To install this core package
replace the configurations files in the root folder with the ones in the `core` folder
~~~shell
cp core/* .
~~~
Then you can install py4vasp with the same steps as above. Alternatively, since
py4vasp-core does not use mdtraj, you can also install everything in a virtual environment
mangaged by poetry
~~~shell
pip install poetry
poetry install
poetry run pytest
~~~
Note that some tests will be skipped because they require the external packages to run.
If you want to exclude even the development dependencies, you can run
~~~shell
poetry install --without dev
~~~
for the minimal installation.

## Code style

Code style is enforced, but is not something the developer should spend time on, so we
decided on using black and isort. Please run
~~~shell
black src tests
isort src tests
~~~
before committing the code. This will autoformat your code and sort the import
statements in a consistent order. If you would like this code formatting to be done
along with each commit, you can run
~~~shell
pre-commit install
~~~

## Contributing to py4vasp

We welcome contributions to py4vasp. To improve the code please follow this workflow

* Create an issue for the bugfix or feature you plan to work on, this gives the option
  to provide some input before work is invested.
* Implement your work in a fork of the repository and create a pull request for it.
  Please make sure to test your code thoroughly and commit the tests in the pull
  request in the tests directory.
* In the message to your merge request mention the issue the code attempts to solve.
* We will try to include your merge request rapidly when all the tests pass and your
  code is covered by tests.

Please limit the size of a pull request to approximately 200 lines of code
otherwise reviewing the changes gets unwieldy. Prefer splitting the work into
multiple smaller chunks if necessary.
