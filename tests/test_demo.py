# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import gc

import pytest

from py4vasp import Calculation, demo, exception


def test_creating_default_calculation(tmp_path):
    demo.calculation(tmp_path / "specific_example")


def test_creating_perovskite_calculation(tmp_path):
    # the "perovskite" selection pairs a structure with its symmetry so the
    # symmetry-derived structure examples have consistent data
    calculation = demo.calculation(tmp_path / "perovskite_example", "perovskite")
    assert calculation.structure.number_atoms() == 5


def test_creating_surface_calculation(tmp_path):
    # the "surface" selection is the only one with a vacuum region, which the surface
    # quantities need
    calculation = demo.calculation(tmp_path / "surface_example", "surface")
    assert calculation.structure.number_atoms() == 8


def test_creating_metal_calculation(tmp_path):
    # the "metal" selection is the only one with states at the Fermi energy
    calculation = demo.calculation(tmp_path / "metal_example", "metal")
    assert calculation.structure.number_atoms() == 1


def test_existing_path_is_rejected(tmp_path):
    with pytest.raises(exception.IncorrectUsage):
        demo.calculation(tmp_path)


def test_existing_path_message_says_how_to_reopen_the_data(tmp_path):
    # rerunning a notebook cell is the common way to get here
    path = tmp_path / "example"
    demo.calculation(path)
    with pytest.raises(exception.IncorrectUsage) as error:
        demo.calculation(path)
    message = str(error.value)
    assert f'Calculation.from_path("{path}")' in message
    assert "demo.calculation()" in message


def test_calculation_without_path_creates_new_directory():
    first = demo.calculation()
    second = demo.calculation()
    assert first.path().is_dir()
    assert first.path() != second.path()
    assert "total" in first.dos.read()


def test_selection_without_path():
    calculation = demo.calculation(selection="metal")
    assert calculation.structure.number_atoms() == 1


def test_quantity_outlives_the_temporary_calculation():
    # nothing but the quantity keeps the calculation's directory alive here
    dos = demo.calculation().dos
    gc.collect()
    assert "total" in dos.read()


def test_group_outlives_the_temporary_calculation():
    exciton = demo.calculation().exciton
    gc.collect()
    assert exciton.density.read()


def test_folder_removed_after_calculation_and_quantities_deleted():
    calculation = demo.calculation()
    path = calculation.path()
    dos = calculation.dos
    del calculation
    gc.collect()
    assert path.is_dir()
    del dos
    gc.collect()
    assert not path.exists()


def test_explicit_path_is_never_removed(tmp_path):
    path = tmp_path / "example"
    calculation = demo.calculation(path)
    del calculation
    gc.collect()
    assert path.is_dir()
    assert Calculation.from_path(path).dos.read()
