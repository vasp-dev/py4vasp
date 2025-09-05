# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import pytest

from py4vasp import _config, exception
from py4vasp._calculation.structure import Structure
from py4vasp._calculation.velocity import Velocity


@pytest.fixture
def Sr2TiO4(raw_data):
    return create_velocity_data(raw_data, "Sr2TiO4")


@pytest.fixture(params=["Sr2TiO4", "Fe3O4"])
def velocities(raw_data, request):
    return create_velocity_data(raw_data, request.param)


@pytest.fixture(params=[-1, 0, slice(None), slice(1, 3)])
def steps(request):
    return request.param


def create_velocity_data(raw_data, structure):
    raw_velocity = raw_data.velocity(structure)
    velocity = Velocity.from_data(raw_velocity)
    velocity.ref = types.SimpleNamespace()
    velocity.ref.structure = Structure.from_data(raw_velocity.structure)
    velocity.ref.velocities = raw_velocity.velocities
    return velocity


def test_read(velocities, steps, Assert):
    actual = velocities.read() if steps == -1 else velocities[steps].read()
    reference_structure = velocities.ref.structure[steps].read()
    Assert.same_structure(actual["structure"], reference_structure)
    Assert.allclose(actual["velocities"], velocities.ref.velocities[steps])


@pytest.mark.parametrize("supercell", [None, 2, (3, 2, 1)])
def test_plot(velocities, steps, supercell, Assert):
    structure_view = velocities.ref.structure.plot(supercell)
    plot_method = velocities.plot if steps == -1 else velocities[steps].plot
    view = plot_method(supercell) if supercell else plot_method()
    Assert.same_structure_view(view, structure_view)
    assert len(view.ion_arrows) == 1
    arrows = view.ion_arrows[0]
    assert arrows.quantity.ndim == 3
    expected_velocities = velocities.velocity_rescale * velocities.ref.velocities[steps]
    Assert.allclose(arrows.quantity, expected_velocities)
    assert arrows.label == "velocities"
    assert arrows.color == _config.VASP_COLORS["gray"]
    assert arrows.radius == 0.2


def test_to_numpy(velocities, steps, Assert):
    actual = velocities.to_numpy() if steps == -1 else velocities[steps].to_numpy()
    Assert.allclose(actual, velocities.ref.velocities[steps])


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
    print(ref_plain)
    print(actual["text/plain"])
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


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.velocity("Fe3O4")
    check_factory_methods(Velocity, data)
