# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from dataclasses import dataclass

import numpy as np
import pytest

from py4vasp import exception, raw
from py4vasp._calculation._CONTCAR import CONTCAR
from py4vasp._calculation._stoichiometry import Stoichiometry
from py4vasp._calculation.structure import Structure
from py4vasp._util import parse


@dataclass
class GeneralPOSCAR(raw.CONTCAR):
    scaling_factor: str = "default"
    show_ion_types: bool = True
    ion_coordinate_system: str = "direct"
    velocity_coordinate_system: str = "cartesian"
    string_format: str = "default"

    def __str__(self):
        return "\n".join(self._line_generator())

    def _line_generator(self):
        yield self.system
        yield from self._cell_lines()
        yield from self._stoichiometry_lines()
        yield from self._selective_dynamic_line()
        yield from self._ion_position_lines()
        yield from self._lattice_velocity_lines()
        yield from self._ion_velocity_lines()

    def _cell_lines(self):
        yield from self._to_string(self._scale(first_line=True))
        structure = Structure.from_data(self.structure)
        yield from self._to_string(structure.lattice_vectors() / self._scale())

    def _stoichiometry_lines(self):
        if self.show_ion_types:
            stoichiometry = Stoichiometry.from_data(self.structure.stoichiometry)
            yield stoichiometry.to_POSCAR()
        else:
            yield from self._to_string(self.structure.stoichiometry.number_ion_types)

    def _selective_dynamic_line(self):
        if self.selective_dynamics.is_none():
            return
        yield self._formatted_string("selective_dynamics")

    def _ion_position_lines(self):
        yield self._formatted_string(self.ion_coordinate_system)
        if parse.first_char(self.ion_coordinate_system) in "cCkK":
            structure = Structure.from_data(self.structure)
            positions = structure.cartesian_positions() / self._scale()
        else:
            positions = self.structure.positions
        if self.selective_dynamics.is_none():
            yield from self._to_string(positions)
        else:
            for line in zip(positions, self.selective_dynamics):
                row = np.array(line, dtype=np.object_).flatten()
                yield from self._to_string(row)

    def _lattice_velocity_lines(self):
        if self.lattice_velocities.is_none():
            return
        yield self._formatted_string("lattice velocities and vectors")
        yield "1"  # lattice vectors initialized
        yield from self._to_string(self.lattice_velocities)
        structure = Structure.from_data(self.structure)
        yield from self._to_string(structure.lattice_vectors())

    def _ion_velocity_lines(self):
        if self.ion_velocities.is_none():
            return
        yield self._formatted_string(self.velocity_coordinate_system)
        if parse.first_char(self.velocity_coordinate_system) in "cCkK ":
            velocities = self.ion_velocities
        else:
            # TODO: check that this conversion is appropriate
            lattice_vectors = Structure.from_data(self.structure).lattice_vectors()
            velocities = self.ion_velocities @ np.linalg.inv(lattice_vectors)
        yield from self._to_string(velocities)

    def _scale(self, first_line=False):
        if self.scaling_factor == "default":
            return self.structure.cell.scale
        elif self.scaling_factor == "one":
            return 1.0
        elif self.scaling_factor == "split":
            return [2.0, 3.0, 4.0]
        elif self.scaling_factor == "volume":
            structure = Structure.from_data(self.structure)
            if first_line:
                return -structure.volume()
            else:
                return 1 / 1.1
        elif self.scaling_factor == "missing":
            if first_line:
                return []
            else:
                return 1.0
        elif self.scaling_factor == "too many":
            if first_line:
                return [2.0, 3.0, 4.0, 5.0]
            else:
                return 1.0
        elif self.scaling_factor == "negative":
            return [-1.0, -2.0, -3.0]
        else:
            raise NotImplemented

    def _to_string(self, rows):
        rows = np.atleast_1d(rows)
        if rows.ndim > 1:
            for row in rows:
                yield from self._to_string(row)
        else:
            to_str = lambda x: ("T" if x else "F") if isinstance(x, bool) else str(x)
            yield " ".join(to_str(value) for value in rows)

    def _formatted_string(self, string):
        if self.string_format == "default":
            return string
        elif self.string_format == "capitalize":
            return string.capitalize()
        elif self.string_format == "first letter":
            return parse.first_char(string)
        else:
            raise NotImplemented


@pytest.fixture(
    params=[
        ("Cubic SrTiO3", "SrTiO3"),
        ("With selective dynamics", "SrTiO3"),
        ("With lattice velocities", "SrTiO3"),
        ("With ion velocities", "SrTiO3"),
        ("With velocities", "ZnS"),
        ("Without ion types", "BN"),
        ("Capitalize keywords", "BN"),
        ("First letter only", "SrTiO3"),
        ("Ion velocity string is space", "BN"),
        ("Empty ion velocity string", "BN"),
        ("Multiple scaling factors", "BN"),
        ("Volume scaling factor", "SrTiO3"),
        ("Scaling factor set to 1", "BN"),
        ("Ion coordinates empty", "ZnS"),
        ("Cartesian ion positions", "ZnS"),
        ("Complex example with ion types", "BN"),
        ("Complex example without ion types", "BN"),
    ]
)
def example_poscar(raw_data, request):
    system, selection = request.param
    raw_structure = raw_data.structure(selection)
    common_part = {"structure": raw_structure, "system": system}
    if system in ("Cubic SrTiO3", "Hexagonal ZnS"):
        return raw.CONTCAR(**common_part)
    elif system == "With selective dynamics":
        selective_dynamics = raw.VaspData(np.random.choice([True, False], size=(5, 3)))
        return raw.CONTCAR(**common_part, selective_dynamics=selective_dynamics)
    elif system == "With lattice velocities":
        lattice_velocities = raw.VaspData(np.linspace(0, 0.2, 9).reshape(3, 3))
        return raw.CONTCAR(**common_part, lattice_velocities=lattice_velocities)
    elif system == "With ion velocities":
        ion_velocities = raw.VaspData(0.1 + 0.1 * raw_structure.positions)
        return raw.CONTCAR(**common_part, ion_velocities=ion_velocities)
    elif system == "With velocities":
        return raw.CONTCAR(
            **common_part,
            lattice_velocities=raw.VaspData(np.linspace(-1, 1, 9).reshape(3, 3)),
            ion_velocities=raw.VaspData(0.2 - 0.1 * raw_structure.positions),
        )
    elif system == "Without ion types":
        raw_structure.stoichiometry.ion_types = raw.VaspData(None)
        return GeneralPOSCAR(**common_part, show_ion_types=False)
    elif system == "Capitalize keywords":
        return GeneralPOSCAR(
            **common_part,
            selective_dynamics=raw.VaspData([[True, True, False], [False, True, True]]),
            string_format="capitalize",
        )
    elif system == "First letter only":
        return GeneralPOSCAR(
            **common_part,
            lattice_velocities=raw.VaspData(
                np.array([[0.0, -0.6, 0.2], [0.1, 0.3, -0.2], [0.2, -0.4, 0.4]])
            ),
            string_format="first letter",
        )
    elif system == "Ion velocity string is space":
        ion_velocities = raw.VaspData(np.array([[0.2, 0.4, -0.2], [0.4, 0.6, -0.3]]))
        return GeneralPOSCAR(
            **common_part, ion_velocities=ion_velocities, velocity_coordinate_system=" "
        )
    elif system == "Empty ion velocity string":
        ion_velocities = raw.VaspData(np.array([[0.2, 0.4, -0.2], [0.4, 0.6, -0.3]]))
        return GeneralPOSCAR(
            **common_part, ion_velocities=ion_velocities, velocity_coordinate_system=""
        )
    elif system == "Multiple scaling factors":
        return GeneralPOSCAR(**common_part, scaling_factor="split")
    elif system == "Volume scaling factor":
        return GeneralPOSCAR(**common_part, scaling_factor="volume")
    elif system == "Scaling factor set to 1":
        return GeneralPOSCAR(**common_part, scaling_factor="one")
    elif system == "Ion coordinates empty":
        return GeneralPOSCAR(**common_part, ion_coordinate_system="")
    elif system == "Cartesian ion positions":
        return GeneralPOSCAR(**common_part, ion_coordinate_system="cartesian")
    elif system == "Complex example with ion types":
        return GeneralPOSCAR(
            **common_part,
            selective_dynamics=raw.VaspData([[True, True, False], [False, True, True]]),
            string_format="capitalize",
            lattice_velocities=raw.VaspData(
                np.array([[0.0, -0.6, 0.2], [0.1, 0.3, -0.2], [0.2, -0.4, 0.4]])
            ),
            ion_velocities=raw.VaspData(np.array([[0.2, 0.4, -0.2], [0.4, 0.6, -0.3]])),
            scaling_factor="split",
        )
    elif system == "Complex example without ion types":
        raw_structure.stoichiometry.ion_types = raw.VaspData(None)
        return GeneralPOSCAR(
            **common_part,
            show_ion_types=False,
            ion_coordinate_system="cartesian",
            selective_dynamics=raw.VaspData([[True, True, False], [False, True, True]]),
            string_format="first letter",
            lattice_velocities=raw.VaspData(
                np.array([[0.0, -0.6, 0.2], [0.1, 0.3, -0.2], [0.2, -0.4, 0.4]])
            ),
            ion_velocities=raw.VaspData(np.array([[0.2, 0.4, -0.2], [0.4, 0.6, -0.3]])),
            scaling_factor="volume",
        )
    else:
        raise NotImplementedError


def test_parse_general_poscar(example_poscar, Assert):
    is_general_poscar = isinstance(example_poscar, GeneralPOSCAR)
    poscar_string = create_poscar_string(example_poscar, is_general_poscar)
    actual = parse.POSCAR(poscar_string)
    exact_match = not is_general_poscar
    Assert.same_raw_contcar(actual, example_poscar, exact_match)


def create_poscar_string(example_poscar, is_general_poscar):
    if is_general_poscar:
        return str(example_poscar)
    else:
        return str(CONTCAR.from_data(example_poscar))


@pytest.mark.parametrize("scaling_factor", ("missing", "too many", "negative"))
def test_error_in_scaling_factor(raw_data, scaling_factor):
    raw_poscar = GeneralPOSCAR(
        structure=raw_data.structure("ZnS"),
        system="wrong scaling factor",
        scaling_factor=scaling_factor,
    )
    poscar_string = str(raw_poscar)
    with pytest.raises(exception.ParserError):
        parse.POSCAR(poscar_string)


def test_error_velocities_in_fractional_coordinates(raw_data):
    raw_poscar = GeneralPOSCAR(
        structure=raw_data.structure("BN"),
        system="fractional velocities",
        velocity_coordinate_system="fractional",
        ion_velocities=raw.VaspData(np.array([[0.2, 0.4, -0.2], [0.4, 0.6, -0.3]])),
    )
    poscar_string = str(raw_poscar)
    with pytest.raises(exception.ParserError):
        parse.POSCAR(poscar_string)


@pytest.mark.parametrize("string, expected", (("", " "), (" ", " "), ("foo", "f")))
def test_first_char(string, expected):
    assert parse.first_char(string) == expected
