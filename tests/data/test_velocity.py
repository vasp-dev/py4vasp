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
