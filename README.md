# py4vasp

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![tests](https://github.com/martin-schlipf/py4vasp/actions/workflows/test.yml/badge.svg)](https://github.com/martin-schlipf/py4vasp/actions/workflows/test.yml)

## Installation

We use the [poetry dependency manager](https://python-poetry.org/) which takes care of all dependencies and maintains a virtual environment to check the code. If you want to test something in the virtual environment, just use e.g. ```poetry run jupyter-notebook```.

Using poetry installing and the code requires the following steps. The last step will test whether everything worked
~~~shell
git clone git@github.com:vasp-dev/py4vasp.git
pip install poetry
poetry install
poetry run pytest
~~~

We occasionally encountered errors when installing the *mdtraj* dependency in this fashion, in particular on MacOS and Windows. If you notice the same behavior, we recommend to manage your environment with *conda* and install *py4vasp* in the following manner
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

Code style is enforced, but is not something the developer should spend time on, so we decided on using the black formatter. Just run ```black .``` before committing the code.

## Development workflow

* Create an issue for the bugfix or feature you are working on
* Create a branch prefixing it with the number of the issue
* Implement the bugfix or feature on the branch adding tests to check it
* Create a pull request and link it to the issue it resolves

A typical pull request should have up to approximately 200 lines of code otherwise reviewing the changes gets unwieldy.
