# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import data, raw
from py4vasp._data import base, structure
from py4vasp._util import convert


class CONTCAR(base.Refinery, structure.Mixin):
    "Access the final positions after the VASP calculation."

    def to_dict(self):
        return {
            **self._structure.read(),
            "system": convert.text_to_string(self._raw_data.system),
            **self._read("selective_dynamics"),
            **self._read("lattice_velocities"),
            **self._read("ion_velocities"),
        }

    def _read(self, key):
        data = getattr(self._raw_data, key)
        return {key: data} if not data.is_none() else {}

    def __str__(self):
        return "\n".join(self._line_generator())

    def _line_generator(self):
        cell = self._raw_data.structure.cell
        positions = self._raw_data.structure.positions
        selective_dynamics = self._raw_data.selective_dynamics
        yield convert.text_to_string(self._raw_data.system)
        yield from _cell_lines(cell)
        yield self._topology().to_POSCAR()
        if not selective_dynamics.is_none():
            yield "Selective dynamics"
        yield "Fractional ion positions"
        yield from _ion_position_lines(positions, selective_dynamics)
        yield from _lattice_velocity_lines(self._raw_data.lattice_velocities, cell)
        yield from _ion_velocity_lines(self._raw_data.ion_velocities)

    def _topology(self):
        return data.Topology.from_data(self._raw_data.structure.topology)


def _cell_lines(cell):
    yield _float_format(cell.scale, scientific=False).lstrip()
    yield from _vectors_to_lines(cell.lattice_vectors)


def _ion_position_lines(positions, selective_dynamics):
    if selective_dynamics.is_none():
        yield from _vectors_to_lines(positions)
    else:
        yield from _vectors_and_flags_to_lines(positions, selective_dynamics)


def _lattice_velocity_lines(velocities, cell):
    if velocities.is_none():
        return
    yield "Lattice velocities and vectors"
    yield from _vectors_to_lines(velocities, scientific=True)
    yield from _vectors_to_lines(cell.scale * cell.lattice_vectors, scientific=True)


def _ion_velocity_lines(velocities):
    if velocities.is_none():
        return
    yield "Cartesian ion velocities"
    yield from _vectors_to_lines(velocities, scientific=True)


def _vectors_to_lines(vectors, scientific=False):
    for vector in vectors:
        yield _vector_to_line(vector, scientific)


def _vectors_and_flags_to_lines(vectors, flags):
    for vector, flag in zip(vectors, flags):
        yield f"{_vector_to_line(vector, scientific=False)}  {_flag_to_line(flag)}"


def _vector_to_line(vector, scientific):
    return " ".join(_float_format(x, scientific) for x in vector)


def _flag_to_line(flag):
    return " ".join(_bool_format(x) for x in flag)


def _float_format(number, scientific):
    if scientific:
        return f"{number:16.8e}"
    else:
        return f"{number:21.16f}"


def _bool_format(value):
    return "T" if value else "F"
