# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types
from unittest.mock import patch

import pytest

from py4vasp import exception
from py4vasp.data import Structure, Velocity
from py4vasp._config import VASP_GRAY
from py4vasp._util import convert


@pytest.fixture
def Sr2TiO4(raw_data):
    raw_velocity = raw_data.velocity("Sr2TiO4")
    velocity = Velocity.from_data(raw_velocity)
    velocity.ref = types.SimpleNamespace()
    velocity.ref.structure = Structure.from_data(raw_velocity.structure)
    velocity.ref.velocities = raw_velocity.velocities
    return velocity


@pytest.fixture
def Fe3O4(raw_data):
    raw_velocity = raw_data.velocity("Fe3O4")
    velocity = Velocity.from_data(raw_velocity)
    velocity.ref = types.SimpleNamespace()
    velocity.ref.structure = Structure.from_data(raw_velocity.structure)
    velocity.ref.velocities = raw_velocity.velocities
    return velocity


def test_read_Sr2TiO4(Sr2TiO4, Assert):
    check_read_velocity(Sr2TiO4.read(), Sr2TiO4.ref, -1, Assert)
    for steps in (slice(None), slice(1, 3), 0):
        check_read_velocity(Sr2TiO4[steps].read(), Sr2TiO4.ref, steps, Assert)


def test_read_Fe3O4(Fe3O4, Assert):
    check_read_velocity(Fe3O4.read(), Fe3O4.ref, -1, Assert)
    for steps in (slice(None), slice(1, 3), 0):
        check_read_velocity(Fe3O4[steps].read(), Fe3O4.ref, steps, Assert)


def check_read_velocity(actual, reference, steps, Assert):
    reference_structure = reference.structure[steps].read()
    for key in actual["structure"]:
        if key in ("elements", "names"):
            assert actual["structure"][key] == reference_structure[key]
        else:
            Assert.allclose(actual["structure"][key], reference_structure[key])
    Assert.allclose(actual["velocities"], reference.velocities[steps])


def test_plot_Sr2TiO4(Sr2TiO4, Assert):
    check_plot_velocity(Sr2TiO4, -1, Assert)
    check_plot_velocity(Sr2TiO4, 0, Assert)
    for steps in (slice(None), slice(1, 3)):
        with pytest.raises(exception.NotImplemented):
            Sr2TiO4[steps].plot()


def test_plot_Fe3O4(Fe3O4, Assert):
    check_plot_velocity(Fe3O4, -1, Assert)
    check_plot_velocity(Fe3O4, 0, Assert)


def check_plot_velocity(velocity, step, Assert):
    with patch("py4vasp.data.Structure.plot") as plot:
        if step == -1:
            velocity.plot()
        else:
            velocity[step].plot()
        plot.assert_called_once()
        viewer = plot.return_value
        viewer.show_arrows_at_atoms.assert_called_once()
        args, _ = viewer.show_arrows_at_atoms.call_args
    Assert.allclose(args[0], velocity.velocity_rescale * velocity.ref.velocities[step])
    Assert.allclose(args[1], convert.to_rgb(VASP_GRAY))


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
    ref_plain = f"""{Sr2TiO4.ref.structure}

  63.0000000000000000   64.0000000000000000   65.0000000000000000
  66.0000000000000000   67.0000000000000000   68.0000000000000000
  69.0000000000000000   70.0000000000000000   71.0000000000000000
  72.0000000000000000   73.0000000000000000   74.0000000000000000
  75.0000000000000000   76.0000000000000000   77.0000000000000000
  78.0000000000000000   79.0000000000000000   80.0000000000000000
  81.0000000000000000   82.0000000000000000   83.0000000000000000"""
    assert actual == {"text/plain": ref_plain}
    #
    actual, _ = format_(Sr2TiO4[0])
    ref_plain = f"""{Sr2TiO4.ref.structure[0]}

   0.0000000000000000    1.0000000000000000    2.0000000000000000
   3.0000000000000000    4.0000000000000000    5.0000000000000000
   6.0000000000000000    7.0000000000000000    8.0000000000000000
   9.0000000000000000   10.0000000000000000   11.0000000000000000
  12.0000000000000000   13.0000000000000000   14.0000000000000000
  15.0000000000000000   16.0000000000000000   17.0000000000000000
  18.0000000000000000   19.0000000000000000   20.0000000000000000"""
    assert actual == {"text/plain": ref_plain}
    #
    actual, _ = format_(Sr2TiO4[1:3])
    ref_plain = f"""{Sr2TiO4.ref.structure[2]}

  42.0000000000000000   43.0000000000000000   44.0000000000000000
  45.0000000000000000   46.0000000000000000   47.0000000000000000
  48.0000000000000000   49.0000000000000000   50.0000000000000000
  51.0000000000000000   52.0000000000000000   53.0000000000000000
  54.0000000000000000   55.0000000000000000   56.0000000000000000
  57.0000000000000000   58.0000000000000000   59.0000000000000000
  60.0000000000000000   61.0000000000000000   62.0000000000000000"""
    assert actual == {"text/plain": ref_plain}
