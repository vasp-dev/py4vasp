# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from dataclasses import dataclass

import numpy as np
import pytest

from py4vasp import exception, raw
from py4vasp._calculation._CONTCAR import CONTCAR
from py4vasp._calculation._stoichiometry import Stoichiometry
from py4vasp._calculation.structure import Structure
from py4vasp._raw.data_wrapper import VaspData
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
        if self.ion_coordinate_system[0] in "cCkK":
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
        if self.velocity_coordinate_system[0] in "cCkK ":
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
            return string[0]
        else:
            raise NotImplemented


STRUCTURE_SrTiO3 = raw.Structure(
    raw.Stoichiometry(number_ion_types=[1, 1, 3], ion_types=["Sr", "Ti", "O"]),
    raw.Cell(lattice_vectors=np.eye(3), scale=raw.VaspData(4.0)),
    positions=np.array(
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
    ),
)
STRUCTURE_ZnS = raw.Structure(
    raw.Stoichiometry(number_ion_types=[2, 2], ion_types=["Zn", "S"]),
    raw.Cell(
        lattice_vectors=np.array([[1.9, -3.3, 0.0], [1.9, 3.3, 0.0], [0, 0, 6.2]]),
        scale=raw.VaspData(1.0),
    ),
    positions=np.array(
        [
            [1 / 3, 2 / 3, 0.0],
            [2 / 3, 1 / 3, 0.5],
            [1 / 3, 2 / 3, 0.375],
            [2 / 3, 1 / 3, 0.875],
        ]
    ),
)
STRUCTURE_BN = raw.Structure(
    raw.Stoichiometry(number_ion_types=[1, 1], ion_types=["B", "N"]),
    raw.Cell(
        lattice_vectors=np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]),
        scale=raw.VaspData(3.63),
    ),
    positions=np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]),
)
EXAMPLE_POSCARS = (
    raw.CONTCAR(structure=STRUCTURE_SrTiO3, system="Cubic SrTiO3"),
    raw.CONTCAR(
        structure=STRUCTURE_SrTiO3,
        system="With selective dynamics",
        selective_dynamics=raw.VaspData(np.random.choice([True, False], size=(5, 3))),
    ),
    raw.CONTCAR(
        structure=STRUCTURE_SrTiO3,
        system="With lattice velocities",
        lattice_velocities=raw.VaspData(np.linspace(0, 0.2, 9).reshape(3, 3)),
    ),
    raw.CONTCAR(
        structure=STRUCTURE_SrTiO3,
        system="With ion velocities",
        ion_velocities=raw.VaspData(0.1 + 0.1 * STRUCTURE_SrTiO3.positions),
    ),
    raw.CONTCAR(structure=STRUCTURE_ZnS, system="Hexagonal ZnS"),
    raw.CONTCAR(
        structure=STRUCTURE_ZnS,
        system="With velocities",
        lattice_velocities=raw.VaspData(np.linspace(-1, 1, 9).reshape(3, 3)),
        ion_velocities=raw.VaspData(0.2 - 0.1 * STRUCTURE_ZnS.positions),
    ),
    GeneralPOSCAR(
        structure=STRUCTURE_BN,
        system="Without ion types",
        show_ion_types=False,
    ),
    GeneralPOSCAR(
        structure=STRUCTURE_BN,
        system="Capitalize keywords",
        selective_dynamics=raw.VaspData([[True, True, False], [False, True, True]]),
        string_format="capitalize",
    ),
    GeneralPOSCAR(
        structure=STRUCTURE_SrTiO3,
        system="First letter only",
        lattice_velocities=raw.VaspData(
            np.array([[0.0, -0.6, 0.2], [0.1, 0.3, -0.2], [0.2, -0.4, 0.4]])
        ),
        string_format="first letter",
    ),
    GeneralPOSCAR(
        structure=STRUCTURE_BN,
        system="Empty ion velocity string",
        ion_velocities=raw.VaspData(np.array([[0.2, 0.4, -0.2], [0.4, 0.6, -0.3]])),
        velocity_coordinate_system=" ",
    ),
    GeneralPOSCAR(
        structure=STRUCTURE_BN,
        system="Multiple scaling factors",
        scaling_factor="split",
    ),
    GeneralPOSCAR(
        structure=STRUCTURE_SrTiO3,
        system="Volume scaling factor",
        scaling_factor="volume",
    ),
    GeneralPOSCAR(
        structure=STRUCTURE_BN,
        system="Scaling factor set to 1",
        scaling_factor="one",
    ),
    GeneralPOSCAR(
        structure=STRUCTURE_ZnS,
        system="Cartesian ion positions",
        ion_coordinate_system="cartesian",
    ),
    GeneralPOSCAR(
        structure=STRUCTURE_BN,
        system="complex example with ion types",
        selective_dynamics=raw.VaspData([[True, True, False], [False, True, True]]),
        string_format="capitalize",
        lattice_velocities=raw.VaspData(
            np.array([[0.0, -0.6, 0.2], [0.1, 0.3, -0.2], [0.2, -0.4, 0.4]])
        ),
        ion_velocities=raw.VaspData(np.array([[0.2, 0.4, -0.2], [0.4, 0.6, -0.3]])),
        scaling_factor="split",
    ),
    GeneralPOSCAR(
        structure=STRUCTURE_BN,
        system="complex example without ion types",
        show_ion_types=False,
        ion_coordinate_system="cartesian",
        selective_dynamics=raw.VaspData([[True, True, False], [False, True, True]]),
        string_format="first letter",
        lattice_velocities=raw.VaspData(
            np.array([[0.0, -0.6, 0.2], [0.1, 0.3, -0.2], [0.2, -0.4, 0.4]])
        ),
        ion_velocities=raw.VaspData(np.array([[0.2, 0.4, -0.2], [0.4, 0.6, -0.3]])),
        scaling_factor="volume",
    ),
)


@pytest.mark.parametrize("raw_poscar", EXAMPLE_POSCARS)
def test_parse_general_poscar(raw_poscar, Assert):
    is_general_poscar = isinstance(raw_poscar, GeneralPOSCAR)
    poscar_string = create_poscar_string(raw_poscar, is_general_poscar)
    ion_types = determine_ion_types(raw_poscar, is_general_poscar)
    actual = parse.POSCAR(poscar_string, ion_types)
    exact_match = not is_general_poscar
    Assert.same_raw_contcar(actual, raw_poscar, exact_match)


def create_poscar_string(raw_poscar, is_general_poscar):
    if is_general_poscar:
        return str(raw_poscar)
    else:
        return str(CONTCAR.from_data(raw_poscar))


def determine_ion_types(raw_poscar, is_general_poscar):
    if is_general_poscar and not raw_poscar.show_ion_types:
        return raw_poscar.structure.stoichiometry.ion_types
    else:
        return None  # if ion types are in POSCAR we don't need to provide it to parser


@pytest.mark.parametrize("scaling_factor", ("missing", "too many", "negative"))
def test_error_in_scaling_factor(scaling_factor):
    raw_poscar = GeneralPOSCAR(
        structure=STRUCTURE_ZnS,
        system="wrong scaling factor",
        scaling_factor=scaling_factor,
    )
    poscar_string = str(raw_poscar)
    with pytest.raises(exception.ParserError):
        parse.POSCAR(poscar_string)


def test_error_no_species_provided():
    raw_poscar = GeneralPOSCAR(
        structure=STRUCTURE_BN, system="missing ion types", show_ion_types=False
    )
    poscar_string = str(raw_poscar)
    with pytest.raises(exception.ParserError):
        parse.POSCAR(poscar_string)


def test_error_velocities_in_fractional_coordinates():
    raw_poscar = GeneralPOSCAR(
        structure=STRUCTURE_BN,
        system="fractional velocities",
        velocity_coordinate_system="fractional",
        ion_velocities=raw.VaspData(np.array([[0.2, 0.4, -0.2], [0.4, 0.6, -0.3]])),
    )
    poscar_string = str(raw_poscar)
    with pytest.raises(exception.ParserError):
        parse.POSCAR(poscar_string)
