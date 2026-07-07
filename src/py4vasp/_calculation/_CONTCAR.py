# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

from py4vasp._calculation import _stoichiometry
from py4vasp._calculation.dispatch import (
    DataSource,
    merge_default,
    merge_strings,
    merge_to_database,
    quantity,
)
from py4vasp._calculation.structure import StructureHandler
from py4vasp._raw import data as raw
from py4vasp._raw.data_db import CONTCAR_DB
from py4vasp._third_party import view
from py4vasp._util import check, convert


class CONTCARHandler:
    """Handler for CONTCAR data — performs all data access and transformation."""

    def __init__(self, raw_contcar: raw.CONTCAR):
        self._raw_contcar = raw_contcar

    @classmethod
    def from_data(cls, raw_contcar: raw.CONTCAR) -> "CONTCARHandler":
        return cls(raw_contcar)

    def read(self) -> dict:
        return self.to_dict()

    def to_dict(self) -> dict:
        return {
            **self._structure().to_dict(),
            "system": convert.text_to_string(self._raw_contcar.system),
            **self._read("selective_dynamics"),
            **self._read("lattice_velocities"),
            **self._read("ion_velocities"),
        }

    def to_database(self) -> CONTCAR_DB:
        return CONTCAR_DB(
            system=(
                convert.text_to_string(self._raw_contcar.system)
                if not check.is_none(self._raw_contcar.system)
                else None
            ),
        )

    def to_view(self, supercell=None):
        return self._structure().to_view(supercell)

    def __str__(self) -> str:
        return "\n".join(self._line_generator())

    def _line_generator(self):
        cell = self._raw_contcar.structure.cell
        positions = self._raw_contcar.structure.positions
        selective_dynamics = self._raw_contcar.selective_dynamics
        yield convert.text_to_string(self._raw_contcar.system)
        yield from _cell_lines(cell)
        yield self._stoichiometry().to_POSCAR()
        if not selective_dynamics.is_none():
            yield "Selective dynamics"
        yield "Direct"
        yield from _ion_position_lines(positions, selective_dynamics)
        yield from _lattice_velocity_lines(self._raw_contcar.lattice_velocities, cell)
        yield from _ion_velocity_lines(self._raw_contcar.ion_velocities)

    def _structure(self):
        return StructureHandler.from_data(self._raw_contcar.structure)

    def _stoichiometry(self):
        return _stoichiometry.Stoichiometry.from_data(
            self._raw_contcar.structure.stoichiometry
        )

    def _read(self, key):
        data = getattr(self._raw_contcar, key)
        return {key: data[:]} if not data.is_none() else {}


@quantity("_CONTCAR")
class CONTCAR(view.Mixin):
    """CONTCAR contains structural restart-data after a relaxation or MD simulation.

    The CONTCAR contains the final structure of the VASP calculation. It can be used as
    input for the next calculation if desired. Depending on the particular setup the
    CONTCAR might contain additional information about the system such as the ion
    and lattice velocities."""

    def __init__(self, source, quantity_name="_CONTCAR"):
        self._source = source
        self._quantity_name = quantity_name

    @classmethod
    def from_data(cls, raw_contcar):
        return cls(source=DataSource(raw_contcar))

    def _handler_factory(self, raw):
        return CONTCARHandler.from_data(raw)

    def __str__(self, selection=None):
        return merge_strings(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            CONTCARHandler.__str__,
        )

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def read(self, selection=None) -> dict:
        """Extract the structural data and available additional data to a dictionary."""
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            CONTCARHandler.read,
        )

    def to_dict(self, selection=None) -> dict:
        """Alias for read()."""
        return self.read(selection=selection)

    def to_view(self, supercell=None, selection=None):
        """Generate a visualization of the final structure."""
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            CONTCARHandler.to_view,
            supercell,
        )

    def _to_database(self) -> dict:
        """Return {quantity[_selection]: handler_result} for database storage."""
        return merge_to_database(
            self._source,
            self._quantity_name,
            CONTCARHandler.from_data,
            CONTCARHandler.to_database,
        )


def _cell_lines(cell):
    yield _float_format(_cell_scale(cell.scale), scientific=False).lstrip()
    yield from _vectors_to_lines(cell.lattice_vectors)


def _cell_scale(scale):
    if not scale.is_none():
        return scale[()]
    else:
        return 1.0


def _ion_position_lines(positions, selective_dynamics):
    if selective_dynamics.is_none():
        yield from _vectors_to_lines(positions)
    else:
        yield from _vectors_and_flags_to_lines(positions, selective_dynamics)


def _lattice_velocity_lines(velocities, cell):
    if velocities.is_none():
        return
    yield "Lattice velocities and vectors"
    yield "1"  # lattice vectors initialized
    yield from _vectors_to_lines(velocities, scientific=True)
    lattice_vectors = _cell_scale(cell.scale) * cell.lattice_vectors
    yield from _vectors_to_lines(lattice_vectors, scientific=True)


def _ion_velocity_lines(velocities):
    if velocities.is_none():
        return
    yield "Cartesian"
    yield from _vectors_to_lines(velocities, scientific=True)


def _vectors_to_lines(vectors, scientific=False):
    for vector in vectors:
        yield _vector_to_line(vector, scientific)


def _vectors_and_flags_to_lines(vectors, flags):
    for vector, flag in zip(vectors, flags):
        yield f"{_vector_to_line(vector, scientific=False)}  {_flag_to_line(flag)}"


def _vector_to_line(vector, scientific):
    insert = "" if scientific else " "
    return insert.join(_float_format(x, scientific) for x in vector)


def _flag_to_line(flag):
    return " ".join(_bool_format(x) for x in flag)


def _float_format(number, scientific):
    if scientific:
        return f"{number:16.8e}"
    else:
        return f"{number:21.16f}"


def _bool_format(value):
    return "T" if value else "F"
