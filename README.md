# py4vasp

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![tests](https://github.com/martin-schlipf/py4vasp/workflows/tests/badge.svg)

## Installation

We use the [poetry dependency manager](https://python-poetry.org/) which takes care of all dependencies and maintains a virtual environment to check the code. If you want to test something in the virtual environment, just use e.g. ```poetry run jupyter-notebook```.

Using poetry installing and the code requires the following steps. The last step will test whether everything worked
~~~
git clone git@github.com:vasp-dev/py4vasp.git
pip install poetry
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
