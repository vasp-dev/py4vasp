# py4vasp

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![tests](https://github.com/vasp-dev/py4vasp/actions/workflows/test.yml/badge.svg)](https://github.com/vasp-dev/py4vasp/actions/workflows/test.yml)

> Please note that this document is intended mostly for developers that want to use
> the version of py4vasp provided on Github. If you just want to install py4vasp to
> use it, please follow the [official documentation](https://vasp.at/py4vasp/latest).

## Installation

We use the [poetry dependency manager](https://python-poetry.org/) which takes care of
all dependencies and maintains a virtual environment to check the code. If you want to
test something in the virtual environment, just use e.g. ```poetry run jupyter-notebook```.

Using poetry installing and the code requires the following steps. The last step will
test whether everything worked
~~~shell
git clone git@github.com:vasp-dev/py4vasp.git
pip install poetry
poetry install
poetry run pytest
~~~
Note that this will install py4vasp into a virtual environment managed by poetry. This
isolates the code from all other packages you have installed and makes sure that when
you modify the code all the relevant dependencies are tracked.

Occasionally, we encountered errors when installing the *mdtraj* dependency in this
fashion, in particular on MacOS and Windows. If you notice the same behavior, we
recommend to manage your environment with *conda* and install *py4vasp* in the
following manner
~~~shell
git clone git@github.com:vasp-dev/py4vasp.git
conda create --name py4vasp-env python=3.8
conda activate py4vasp-env
conda install -c conda-forge poetry
conda install -c conda-forge mdtraj
poetry config virtualenvs.create false --local
poetry install
poetry run pytest
~~~

## Code style

Code style is enforced, but is not something the developer should spend time on, so we
decided on using the black formatter. Please run ```black .``` before committing the code.

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

Please limit the size of a the pull request to approximately 200 lines of code
otherwise reviewing the changes gets unwieldy. Prefer splitting the work into
multiple smaller chunks if necessary.
