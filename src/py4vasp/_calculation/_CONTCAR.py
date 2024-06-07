# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp import calculation
from py4vasp._third_party import view
from py4vasp._util import convert
from py4vasp.calculation import _base, _structure


class CONTCAR(_base.Refinery, view.Mixin, _structure.Mixin):
    """CONTCAR contains structural restart-data after a relaxation or MD simulation.

    The CONTCAR contains the final structure of the VASP calculation. It can be used as
    input for the next calculation if desired. Depending on the particular setup the
    CONTCAR might contain additional information about the system such as the ion
    and lattice velocities."""

    @_base.data_access
    def to_dict(self):
        """Extract the structural data and the available additional data to a dictionary.

        Returns
        -------
        dict
            Contains the final structure and information about the selective dynamics,
            lattice and ion velocities if available.
        """
        return {
            **self._structure.read(),
            "system": convert.text_to_string(self._raw_data.system),
            **self._read("selective_dynamics"),
            **self._read("lattice_velocities"),
            **self._read("ion_velocities"),
        }

    def _read(self, key):
        data = getattr(self._raw_data, key)
        return {key: data[:]} if not data.is_none() else {}

    @_base.data_access
    def to_view(self, supercell=None):
        """Generate a visualization of the final structure.

        Parameters
        ----------
        supercell : int or np.ndarray
            Scales all axes by the given factors.

        Returns
        -------
        View
            A 3d visualization of the structure.
        """
        return self._structure.plot(supercell)

    @_base.data_access
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
        yield "Direct"
        yield from _ion_position_lines(positions, selective_dynamics)
        yield from _lattice_velocity_lines(self._raw_data.lattice_velocities, cell)
        yield from _ion_velocity_lines(self._raw_data.ion_velocities)

    def _topology(self):
        return calculation.topology.from_data(self._raw_data.structure.topology)


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
