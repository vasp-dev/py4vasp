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
        Path where the calculation data will be generated. It must not exist. This
        function will create the directory and create the data inside it. If a selection
        is given the generated data will be stored in a subdirectory of the given path.
    selection
        Optional choice of which data is generated. If not provided or None some default
        data is generated that is suitable for most examples.

    Returns
    -------
    A calculation that accesses the generated data.
    """
    path = _create_path_for_data(path, selection)
    filename = path / DEFAULT_FILE
    with h5py.File(filename, "w") as h5f:
        _generate_calculation_data(h5f, selection)
    return Calculation.from_path(path)


def _create_path_for_data(path, selection):
    if selection is None:
        path = Path(path)
    else:
        path = Path(path) / selection
    if path.exists():
        raise exception.IncorrectUsage(f"The path '{path}' already exists.")
    path.mkdir(parents=True)
    return path


def _generate_calculation_data(h5f, selection):
    generator = _DATA_GENERATORS.get(selection)
    if generator is not None:
        generator(h5f)
    else:
        available = ", ".join(filter(None, key) for key in _DATA_GENERATORS.keys())
        raise exception.IncorrectUsage(
            f"The selection '{selection}' is not recognized. "
            f"Available selections are: {available}. "
            "If no selection is given, some default data is generated."
        )


def _generate_default_data(h5f):
    write(h5f, _demo.band.multiple_bands("with_projectors"))
    write(h5f, _demo.dos.Sr2TiO4("with_projectors"))
    write(h5f, _demo.energy.relax(randomize=True))
    write(h5f, _demo.structure.Sr2TiO4())
    write(h5f, _demo.band.line_mode("no_labels"), selection="kpoints_opt")
    write(h5f, _demo.dos.Sr2TiO4("no_projectors"), selection="kpoints_opt")


def _generate_collinear_data(h5f):
    write(h5f, _demo.band.spin_polarized_bands("with_projectors"))
    write(h5f, _demo.dos.Fe3O4("with_projectors"))


def _generate_noncollinear_data(h5f):
    write(h5f, _demo.band.noncollinear_bands("with_projectors"))
    write(h5f, _demo.dos.Ba2PbO4("noncollinear"))


_DATA_GENERATORS = {
    None: _generate_default_data,
    "default": _generate_default_data,
    "collinear": _generate_collinear_data,
    "noncollinear": _generate_noncollinear_data,
}
