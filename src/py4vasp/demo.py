# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from pathlib import Path
from typing import Optional

import h5py

from py4vasp import _demo, exception
from py4vasp._calculation import Calculation
from py4vasp._raw.definition import DEFAULT_FILE
from py4vasp._raw.write import write

__all__ = ["calculation"]


def calculation(path: Path, selection: Optional[str] = None) -> Calculation:
    """Initialize example data in the given path and return a Calculation accessing it.

    Parameters
    ----------
    path
        Path where the calculation data will be generated. It must exist and be a
        directory. It must not contain existing calculation data.
    selection
        Optional choice of which data is generated. If not provided or None some default
        data is generated that is suitable for most examples.
    """
    path = Path(path)
    filename = path / DEFAULT_FILE
    _raise_error_if_demo_cannot_be_generated(path, filename)
    with h5py.File(filename, "w") as h5f:
        _generate_calculation_data(h5f, selection)
    return Calculation.from_path(path)


def _raise_error_if_demo_cannot_be_generated(path, filename):
    if not path.exists() or not path.is_dir():
        raise exception.IncorrectUsage(
            f"The path '{path}' does not exist or is not a directory."
        )
    if filename.exists():
        raise exception.IncorrectUsage(
            f"The path '{path}' already contains calculation data."
        )


def _generate_calculation_data(h5f, selection):
    if selection is None or selection == "default":
        _generate_default_data(h5f)
    elif selection == "spin_texture":
        _generate_spin_texture_data(h5f)
    else:
        raise exception.IncorrectUsage(
            f"The selection '{selection}' is not recognized. "
            "Available selections are: None (default), 'default', 'spin_texture'."
        )


def _generate_default_data(h5f):
    write(h5f, _demo.dos.Sr2TiO4("with_projectors"))
    write(h5f, _demo.band.multiple_bands("with_projectors"))
    write(h5f, _demo.energy.relax(randomize=True))
    write(h5f, _demo.structure.Sr2TiO4())


def _generate_spin_texture_data(h5f):
    pass
