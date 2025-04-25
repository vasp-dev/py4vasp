# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import pytest

from py4vasp import _config, exception
from py4vasp._calculation.force import Force
from py4vasp._calculation.structure import Structure


@pytest.fixture
def Sr2TiO4(raw_data):
    return create_force_data(raw_data, "Sr2TiO4")


@pytest.fixture(params=["Sr2TiO4", "Fe3O4"])
def forces(raw_data, request):
    return create_force_data(raw_data, request.param)


@pytest.fixture(params=[-1, 0, slice(None), slice(1, 3)])
def steps(request):
    return request.param


def create_force_data(raw_data, structure):
    raw_forces = raw_data.force(structure)
    forces = Force.from_data(raw_forces)
    forces.ref = types.SimpleNamespace()
    forces.ref.structure = Structure.from_data(raw_forces.structure)
    forces.ref.forces = raw_forces.forces
    return forces


def test_read(forces, steps, Assert):
    actual = forces[steps].read() if steps != -1 else forces.read()
    reference_structure = forces.ref.structure[steps].read()
    Assert.same_structure(actual["structure"], reference_structure)
    Assert.allclose(actual["forces"], forces.ref.forces[steps])


@pytest.mark.parametrize("supercell", [None, 2, (3, 2, 1)])
def test_plot(forces, steps, supercell, Assert):
    structure_view = forces.ref.structure.plot(supercell)
    plot_method = forces[steps].plot if steps != -1 else forces.plot
    view = plot_method(supercell) if supercell else plot_method()
    Assert.same_structure_view(view, structure_view)
    assert len(view.ion_arrows) == 1
    arrows = view.ion_arrows[0]
    assert arrows.quantity.ndim == 3
    Assert.allclose(arrows.quantity, forces.force_rescale * forces.ref.forces[steps])
    assert arrows.label == "forces"
    assert arrows.color == _config.VASP_COLORS["purple"]
    assert arrows.radius == 0.2


def test_incorrect_access(Sr2TiO4):
    out_of_bounds = 999
    with pytest.raises(exception.IncorrectUsage):
        Sr2TiO4[out_of_bounds].read()
    with pytest.raises(exception.IncorrectUsage):
        Sr2TiO4["string instead of int"].read()
    with pytest.raises(exception.IncorrectUsage):
        Sr2TiO4[out_of_bounds].plot()
    with pytest.raises(exception.IncorrectUsage):
        Sr2TiO4["string instead of int"].plot()


def test_print_Sr2TiO4(Sr2TiO4, format_):
    actual, _ = format_(Sr2TiO4)
    ref_plain = """
POSITION                                       TOTAL-FORCE (eV/Angst)
-----------------------------------------------------------------------------------
     7.49659      3.28326      0.00000        63.000000     64.000000     65.000000
     4.12081      1.80478      0.00000        66.000000     67.000000     68.000000
     0.00000      0.00000      0.00000        69.000000     70.000000     71.000000
     9.77930      4.28301      0.00000        72.000000     73.000000     74.000000
     1.83822      0.80508      0.00000        75.000000     76.000000     77.000000
     0.55710     -1.27201      1.38866        78.000000     79.000000     80.000000
    -0.55710      1.27201      1.38866        81.000000     82.000000     83.000000
""".strip()
    assert actual == {"text/plain": ref_plain}
    #
    actual, _ = format_(Sr2TiO4[0])
    ref_plain = """
POSITION                                       TOTAL-FORCE (eV/Angst)
-----------------------------------------------------------------------------------
     7.49659      3.28326      0.00000         0.000000      1.000000      2.000000
     4.12081      1.80478      0.00000         3.000000      4.000000      5.000000
     0.00000      0.00000      0.00000         6.000000      7.000000      8.000000
     9.77930      4.28301      0.00000         9.000000     10.000000     11.000000
     1.83822      0.80508      0.00000        12.000000     13.000000     14.000000
     0.55710     -1.27201      1.38866        15.000000     16.000000     17.000000
    -0.55710      1.27201      1.38866        18.000000     19.000000     20.000000
""".strip()
    assert actual == {"text/plain": ref_plain}
    #
    actual, _ = format_(Sr2TiO4[1:3])
    ref_plain = """
POSITION                                       TOTAL-FORCE (eV/Angst)
-----------------------------------------------------------------------------------
     7.49659      3.28326      0.00000        42.000000     43.000000     44.000000
     4.12081      1.80478      0.00000        45.000000     46.000000     47.000000
     0.00000      0.00000      0.00000        48.000000     49.000000     50.000000
     9.77930      4.28301      0.00000        51.000000     52.000000     53.000000
     1.83822      0.80508      0.00000        54.000000     55.000000     56.000000
     0.55710     -1.27201      1.38866        57.000000     58.000000     59.000000
    -0.55710      1.27201      1.38866        60.000000     61.000000     62.000000
""".strip()
    assert actual == {"text/plain": ref_plain}


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.force("Fe3O4")
    check_factory_methods(Force, data)
