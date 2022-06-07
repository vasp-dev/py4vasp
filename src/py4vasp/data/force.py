# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp.data import _base, _slice, Structure
import py4vasp.exceptions as exception
import py4vasp._util.documentation as _documentation
import py4vasp._util.reader as _reader

forces_docstring = f"""
The forces acting on the atoms for selected steps of the simulation.

You can use this class to analyze the forces acting on the atoms. In
particular, you can check whether the forces are small at the end of the
calculation.

{_slice.examples("force")}
""".strip()


@_documentation.add(forces_docstring)
class Force(_slice.Mixin, _base.Refinery):
    force_rescale = 1.5
    "Scaling constant to convert forces to Å."

    @_base.data_access
    def __str__(self):
        "Convert the forces to a format similar to the OUTCAR file."
        result = """
POSITION                                       TOTAL-FORCE (eV/Angst)
-----------------------------------------------------------------------------------
        """.strip()
        step = self._last_step_in_slice
        position_to_string = lambda position: " ".join(f"{x:12.5f}" for x in position)
        positions = self._structure[step].cartesian_positions()
        force_to_string = lambda force: " ".join(f"{x:13.6f}" for x in force)
        for position, force in zip(positions, self._force[step]):
            result += f"\n{position_to_string(position)}    {force_to_string(force)}"
        return result

    @_base.data_access
    @_documentation.add(
        f"""Read the forces and associated structural information for one or more
selected steps of the trajectory.

Returns
-------
dict
    Contains the forces for all selected steps and the structural information
    to know on which atoms the forces act.

{_slice.examples("force", "read")}"""
    )
    def to_dict(self):
        return {
            "structure": self._structure[self._steps].read(),
            "forces": self._force[self._steps],
        }

    @_base.data_access
    @_documentation.add(
        f"""Visualize the forces showing arrows at the atoms.

Returns
-------
Viewer3d
    Shows the structure with cell and all atoms adding arrows to the atoms
    sized according to the strength of the force.

{_slice.examples("force", "plot")}"""
    )
    def plot(self):
        self._raise_error_if_slice()
        forces = self.force_rescale * self._force[self._steps]
        color = [0.3, 0.15, 0.35]
        fig = self._structure.plot()
        fig.show_arrows_at_atoms(forces, color)
        return fig

    @property
    def _structure(self):
        return Structure.from_data(self._raw_data.structure)

    @property
    def _force(self):
        return _ForceReader(self._raw_data.forces)

    def _raise_error_if_slice(self):
        if self._is_slice:
            message = "Plotting forces for multiple steps is not implemented."
            raise exception.NotImplemented(message)


class _ForceReader(_reader.Reader):
    def error_message(self, key, err):
        key = np.array(key)
        steps = key if key.ndim == 0 else key[0]
        return (
            f"Error reading the forces. Please check if the steps "
            f"`{steps}` are properly formatted and within the boundaries. "
            "Additionally, you may consider the original error message:\n" + err.args[0]
        )
