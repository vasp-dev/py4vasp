# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import exception
from py4vasp._config import VASP_GRAY
from py4vasp._util import convert, documentation, reader
from py4vasp.calculation import _base, _slice, _structure


@documentation.format(examples=_slice.examples("velocity"))
class Velocity(_slice.Mixin, _base.Refinery, _structure.Mixin):
    """The velocities describe the ionic motion during an MD simulation.

    The velocities of the ions are a metric for the temperature of the system. Most
    of the time, it is not necessary to consider them explicitly. VASP will set the
    velocities automatically according to the temperature settings (:tag:`TEBEG` and
    :tag:`TEEND`) unless you set them explicitly in the POSCAR file. Since the
    velocities are not something you typically need, VASP will only store them during
    the simulation if you set :tag:`VELOCITY` = T in the INCAR file. In that case you
    can read the velocities of each step along the trajectory. If you are only
    interested in the final velocities, please consider the :data:'~py4vasp.data.CONTCAR`
    class.

    {examples}
    """

    velocity_rescale = 200

    @_base.data_access
    def __str__(self):
        step = self._last_step_in_slice
        velocities = self._vectors_to_string(self._velocity[step])
        return f"{self._structure[step]}\n\n{velocities}"

    def _vectors_to_string(self, vectors):
        return "\n".join(self._vector_to_string(vector) for vector in vectors)

    def _vector_to_string(self, vector):
        return " ".join(self._element_to_string(element) for element in vector)

    def _element_to_string(self, element):
        return f"{element:21.16f}"

    @_base.data_access
    @documentation.format(examples=_slice.examples("velocity", "to_dict"))
    def to_dict(self):
        """Return the structure and ion velocities in a dictionary

        Returns
        -------
        dict
            The dictionary contains the ion velocities as well as the structural
            information for reference.

        {examples}
        """
        return {
            "structure": self._structure[self._steps].read(),
            "velocities": self._velocity[self._steps],
        }

    @_base.data_access
    @documentation.format(examples=_slice.examples("velocity", "plot"))
    def plot(self):
        """Plot the velocities as vectors in the structure.

        This is currently only implemented for a single step. So selecting multiple
        steps will raise an error.

        Returns
        -------
        Viewer3d
            Contains all atoms and the velocities are drawn as vectors.

        {examples}
        """
        self._raise_error_if_slice()
        velocities = self.velocity_rescale * self._velocity[self._steps]
        viewer = self._structure.plot()
        viewer.show_arrows_at_atoms(velocities, convert.to_rgb(VASP_GRAY))
        return viewer

    def _raise_error_if_slice(self):
        if self._is_slice:
            message = "Plotting velocities for multiple steps is not implemented."
            raise exception.NotImplemented(message)

    @property
    def _velocity(self):
        return _VelocityReader(self._raw_data.velocities)


class _VelocityReader(reader.Reader):
    def error_message(self, key, err):
        key = np.array(key)
        steps = key if key.ndim == 0 else key[0]
        return (
            f"Error reading the velocities. Please check if the steps "
            f"`{steps}` are properly formatted and within the boundaries. "
            "Additionally, you may consider the original error message:\n" + err.args[0]
        )
