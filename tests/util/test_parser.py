import numpy as np
import pytest

from py4vasp._raw.data import Cell
from py4vasp._raw.data_wrapper import VaspData
from py4vasp._util.parser import ParsePoscar


@pytest.fixture
def cubic_BN():
    def _cubic_BN(scaling_factor=3.57, species_name_provided=True):
        if isinstance(scaling_factor, float) or isinstance(scaling_factor, int):
            scaling_factor = [scaling_factor]
        _comment_line = "Cubic BN"
        _scaling_factor = " ".join([str(x) for x in scaling_factor])
        _lattice = """0.0 0.5 0.5\n0.5 0.0 0.5\n0.5 0.5 0.0"""
        _species_names = "B N"
        _ions_per_species = "1 1"
        _ion_positions = """Direct\n0.0 0.0 0.0\n0.25 0.25 0.25"""
        if species_name_provided:
            componentwise_inputs = [
                _comment_line,
                _scaling_factor,
                _lattice,
                _species_names,
                _ions_per_species,
                _ion_positions,
            ]
        else:
            componentwise_inputs = [
                _comment_line,
                _scaling_factor,
                _lattice,
                _ions_per_species,
                _ion_positions,
            ]

        poscar_input_string = "\n".join(componentwise_inputs)
        return poscar_input_string, componentwise_inputs

    return _cubic_BN


def test_cubic_BN_fixture_defaults(cubic_BN):
    output_poscar_string, _ = cubic_BN()
    expected_poscar_string = """Cubic BN
3.57
0.0 0.5 0.5
0.5 0.0 0.5
0.5 0.5 0.0
B N
1 1
Direct
0.0 0.0 0.0
0.25 0.25 0.25"""
    assert output_poscar_string == expected_poscar_string


def test_cubic_BN_fixture_scaling_factor(cubic_BN):
    output_poscar_string, _ = cubic_BN(scaling_factor=[3.57, 3.57, 3.57])
    expected_poscar_string = """Cubic BN
3.57 3.57 3.57
0.0 0.5 0.5
0.5 0.0 0.5
0.5 0.5 0.0
B N
1 1
Direct
0.0 0.0 0.0
0.25 0.25 0.25"""
    assert output_poscar_string == expected_poscar_string


def test_cubic_BN_fixture_species_name_provided(cubic_BN):
    output_poscar_string, _ = cubic_BN(species_name_provided=False)
    expected_poscar_string = """Cubic BN
3.57
0.0 0.5 0.5
0.5 0.0 0.5
0.5 0.5 0.0
1 1
Direct
0.0 0.0 0.0
0.25 0.25 0.25"""
    assert output_poscar_string == expected_poscar_string


def test_comment_line(cubic_BN):
    output_poscar_string, componentwise_inputs = cubic_BN()
    comment_line = componentwise_inputs[0]
    parsed_comment_line = ParsePoscar(output_poscar_string).comment_line
    assert comment_line == parsed_comment_line


def test_cell(cubic_BN, Assert):
    output_poscar_string, componentwise_inputs = cubic_BN()
    _scaling_factor = componentwise_inputs[1]
    _lattice = componentwise_inputs[2]
    lattice_vectors = np.array([x.split() for x in _lattice.split("\n")], dtype=float)
    scaling_factor = np.array(_scaling_factor.split(), dtype=float)
    if len(scaling_factor) == 1:
        scaling_factor = scaling_factor[0]
    lattice_vectors = VaspData(lattice_vectors)
    expected_cell = Cell(lattice_vectors=lattice_vectors, scale=scaling_factor)
    output_cell = ParsePoscar(output_poscar_string).cell
    Assert.allclose(expected_cell.lattice_vectors, output_cell.lattice_vectors)
    Assert.allclose(expected_cell.scale, output_cell.scale)
