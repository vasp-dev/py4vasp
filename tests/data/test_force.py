import numpy as np
import pytest
import types
from unittest.mock import patch

import py4vasp.data as data
from py4vasp.data import Force, Structure
import py4vasp.exceptions as exception


@pytest.fixture
def Sr2TiO4(raw_data):
    raw_forces = raw_data.force("Sr2TiO4")
    forces = Force(raw_forces)
    forces.ref = types.SimpleNamespace()
    forces.ref.structure = Structure(raw_forces.structure)
    forces.ref.forces = raw_forces.forces
    return forces


@pytest.fixture
def Fe3O4(raw_data):
    raw_forces = raw_data.force("Fe3O4")
    forces = Force(raw_forces)
    forces.ref = types.SimpleNamespace()
    forces.ref.structure = Structure(raw_forces.structure)
    forces.ref.forces = raw_forces.forces
    return forces


def test_read_Sr2TiO4(Sr2TiO4, Assert):
    check_read_structure(Sr2TiO4.read(), Sr2TiO4.ref, -1, Assert)
    for steps in (slice(None), slice(1, 3), 0):
        check_read_structure(Sr2TiO4[steps].read(), Sr2TiO4.ref, steps, Assert)


def test_read_Fe3O4(Fe3O4, Assert):
    check_read_structure(Fe3O4.read(), Fe3O4.ref, -1, Assert)
    for steps in (slice(None), slice(1, 3), 0):
        check_read_structure(Fe3O4[steps].read(), Fe3O4.ref, steps, Assert)


def check_read_structure(actual, reference, steps, Assert):
    reference_structure = reference.structure[steps].read()
    for key in actual["structure"]:
        if key in ("elements", "names"):
            assert actual["structure"][key] == reference_structure[key]
        else:
            Assert.allclose(actual["structure"][key], reference_structure[key])
    Assert.allclose(actual["forces"], reference.forces[steps])


def test_plot_Sr2TiO4(Sr2TiO4, Assert):
    check_plot_forces(Sr2TiO4, -1, Assert)
    check_plot_forces(Sr2TiO4, 0, Assert)
    for steps in (slice(None), slice(1, 3)):
        with pytest.raises(exception.NotImplemented):
            Sr2TiO4[steps].plot()


def test_plot_Fe3O4(Fe3O4, Assert):
    check_plot_forces(Fe3O4, -1, Assert)
    check_plot_forces(Fe3O4, 0, Assert)


def check_plot_forces(forces, step, Assert):
    with patch("py4vasp.data.Structure._to_viewer3d") as plot:
        if step == -1:
            forces.plot()
        else:
            forces[step].plot()
        plot.assert_called_once()
        viewer = plot.return_value
        viewer.show_arrows_at_atoms.assert_called_once()
        args, _ = viewer.show_arrows_at_atoms.call_args
    Assert.allclose(args[0], forces.force_rescale * forces.ref.forces[step])
    Assert.allclose(args[1], np.array([0.3, 0.15, 0.35]))


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


def test_descriptor(Sr2TiO4, check_descriptors):
    descriptors = {
        "_to_dict": ["to_dict", "read"],
        "_to_viewer3d": ["to_viewer3d", "plot"],
        "_to_string": ["__str__"],
    }
    check_descriptors(Sr2TiO4, descriptors)


def test_from_file(raw_data, mock_file, check_read):
    raw_forces = raw_data.force("Sr2TiO4")
    with mock_file("force", raw_forces) as mocks:
        check_read(Force, mocks, raw_forces)
