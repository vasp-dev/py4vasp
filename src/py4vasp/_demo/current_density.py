# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp import _demo, raw


def current_density(selection):
    if selection == "all":
        valid_indices = ("x", "y", "z")
    else:
        valid_indices = [selection]
    shape = (_demo.AXES, *_demo.GRID_DIMENSIONS)
    current_density = [_demo.wrap_random_data(shape) for _ in valid_indices]
    return raw.CurrentDensity(
        valid_indices=valid_indices,
        structure=_demo.structure.Fe3O4(),
        current_density=current_density,
    )
