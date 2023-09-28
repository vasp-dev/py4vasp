# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types
from unittest.mock import patch

import numpy as np
import pytest

from py4vasp._data import viewer3d
from py4vasp.data import Potential, Structure


@pytest.fixture(params=["total", "ionic", "hartree", "xc", "all"])
def included_parts(request):
    return request.param


@pytest.fixture(params=["Sr2TiO4", "Fe3O4 collinear", "Fe3O4 noncollinear"])
def reference_potential(raw_data, request, included_parts):
    raw_potential = raw_data.potential(f"{request.param} {included_parts}")
    potential = Potential.from_data(raw_potential)
    potential.ref = types.SimpleNamespace()
    potential.ref.output = get_expected_dict(raw_potential)
    return potential


def get_expected_dict(raw_potential):
    return {
        "structure": Structure.from_data(raw_potential.structure).read(),
        **separate_potential("total", raw_potential.total_potential),
        **separate_potential("xc", raw_potential.xc_potential),
        **separate_potential("hartree", raw_potential.hartree_potential),
        **separate_potential("ionic", raw_potential.ionic_potential),
    }


def separate_potential(potential_name, potential):
    if potential.is_none():
        return {}
    if len(potential) == 1:  # nonpolarized
        return {potential_name: potential[0].T}
    if len(potential) == 2:  # spin-polarized
        return {
            potential_name: potential[0].T,
            f"{potential_name}_up": potential[0].T + potential[1].T,
            f"{potential_name}_down": potential[0].T - potential[1].T,
        }
    return {
        potential_name: potential[0].T,
        f"{potential_name}_magnetization": np.moveaxis(potential[1:].T, -1, 0),
    }


def test_read(reference_potential, Assert):
    actual = reference_potential.read()
    assert actual.keys() == reference_potential.ref.output.keys()
    for key in actual:
        if key == "structure":
            continue
        Assert.allclose(actual[key], reference_potential.ref.output[key])


def test_plot_total_potential(reference_potential, Assert, not_core):
    obj = viewer3d.Viewer3d
    cm_init = patch.object(obj, "__init__", autospec=True, return_value=None)
    cm_cell = patch.object(obj, "show_cell")
    cm_surface = patch.object(obj, "show_isosurface")
    with cm_init as init, cm_cell as cell, cm_surface as surface:
        reference_potential.plot()
        init.assert_called_once()
        cell.assert_called_once()
        surface.assert_called_once()
        args, kwargs = surface.call_args
    Assert.allclose(args[0], reference_potential.ref.output["total"])
    assert kwargs == {"isolevel": 0.0, "color": "yellow", "opacity": 0.6}


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.potential("Fe3O4 collinear total")
    check_factory_methods(Potential, data)
