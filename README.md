# py4vasp

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![test-full](https://github.com/vasp-dev/py4vasp/actions/workflows/test_full.yml/badge.svg)](https://github.com/vasp-dev/py4vasp/actions/workflows/test_full.yml)
[![test-core](https://github.com/vasp-dev/py4vasp/actions/workflows/test_core.yml/badge.svg)](https://github.com/vasp-dev/py4vasp/actions/workflows/test_core.yml)

> Please note that this document is intended mostly for developers that want to use
> the version of py4vasp provided on Github. If you just want to install py4vasp to
> use it, please follow the [official documentation](https://vasp.at/py4vasp/latest).

## Repository layout

The repository is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/)
that publishes three distributions from a single lockfile

| path | distribution | what it is |
|---|---|---|
| `.` | `py4vasp-core` | all the code (`src/py4vasp`), requiring only numpy and h5py |
| `packages/py4vasp` | `py4vasp` | a metadata-only wrapper pulling `py4vasp-core[all]` |
| `packages/backend` | `vasp-backend` | `vasp.backend`, the internal interface for our own tools |

pip extras can only *add* dependencies, never remove them. Since `pip install py4vasp`
has to give the largest dependency set and `pip install py4vasp-core` the smallest, the
code has to live in the distribution with the smallest set and `py4vasp` has to be a
superset of it. That is why the `py4vasp` distribution ships no modules of its own.

py4vasp reaches every dependency beyond numpy and h5py through the lazy proxies in
`py4vasp._util.import_`, so a `py4vasp-core` installation gains a feature the moment the
corresponding package becomes importable -- no reinstall, no code path of its own.

## Installation

We use the [uv package manager](https://docs.astral.sh/uv/) which takes care of
all dependencies and maintains a virtual environment to check the code. If you want to
test something in the virtual environment, just use e.g. `uv run jupyter-notebook`.

We recommend installing py4vasp in a conda environment to resolve issues related to
installing `mdtraj` with pip. To do this please use the following steps. The last step
will test whether everything worked
~~~shell
conda create --name py4vasp-env python=3.11
conda activate py4vasp-env
conda install conda-forge::uv
git clone git@github.com:vasp-dev/py4vasp.git
cd py4vasp
export VIRTUAL_ENV=$CONDA_PREFIX
uv sync --active --all-packages --all-extras --no-extra mdtraj
conda install conda-forge::mdtraj
uv run --active pytest
~~~
Note that this will install py4vasp into the conda environment. This isolates the code
from all packages you have installed in other conda environments. Using uv makes
sure that when you modify the code all the relevant dependencies are tracked.

`--all-packages` installs all three distributions in editable mode, `--all-extras` adds
every optional dependency, and `--no-extra mdtraj` leaves mdtraj to conda because
installing it with pip is unreliable.

## py4vasp core

If you want to use py4vasp to develop your own scripts, you may want to limit the amount
of external dependencies. Select the `py4vasp-core` distribution and none of its extras
to get an environment with nothing but numpy, h5py and the test tools
~~~shell
uv sync --package py4vasp-core --no-default-groups --group test
uv run --no-sync pytest
~~~
`uv sync` is exact, so this removes the optional dependencies again if they were
installed before; `make test-core` runs the same thing and syncs the full environment
back afterwards. Note that many tests will be skipped because they require the external
packages to run, and that the tests under `packages/` skip themselves entirely because
the distributions they cover are not installed.

To work on a single feature, ask for the extra that provides it instead, e.g.
~~~shell
uv sync --package py4vasp-core --extra plot
~~~
The available extras are `plot`, `structure`, `view`, `interactive`, `numeric`, `cli`,
`sphinx`, `mdtraj`, and `all` for everything except `sphinx` and `mdtraj`.

## Releasing

The version lives in `src/py4vasp/__init__.py`. The two wrapper distributions repeat it
because a metadata-only package has no module to read it from, and they pin
`py4vasp-core` exactly. `make release VERSION=0.12.0` (which runs
`scripts/set_version.py`) updates all of them at once and `tests/test_version.py` fails
if they ever drift apart.

Two things to keep in mind when cutting a release:

* A new minor series has to reset `__DB_SCHEMA__` in `src/py4vasp/_raw/models.py` to 0
  and record the schema snapshot again; see `tests/raw/test_schema_version.py`.
* Never republish a `py4vasp-core` version that is already on PyPI. `py4vasp` pins
  `py4vasp-core` exactly, so pip skips reinstalling core if the pin is already
  satisfied -- while still deleting the files an older, non-wrapper `py4vasp` recorded.

## Code style

Code style is enforced, but is not something the developer should spend time on, so we
decided on using black and isort. Please run
~~~shell
black src tests packages scripts
isort src tests packages scripts
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
