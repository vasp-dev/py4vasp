# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import exception
from py4vasp._data import base, slice_, structure
from py4vasp._util import documentation, reader


@documentation.format(examples=slice_.examples("force"))
class Force(slice_.Mixin, base.Refinery, structure.Mixin):
    """The forces acting on the atoms for selected steps of the simulation.

    You can use this class to analyze the forces acting on the atoms. In
    particular, you can check whether the forces are small at the end of the
    calculation.

    {examples}
    """

    force_rescale = 1.5
    "Scaling constant to convert forces to Å."

    @base.data_access
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

    @base.data_access
    @documentation.format(examples=slice_.examples("force", "to_dict"))
    def to_dict(self):
        """Read the forces and associated structural information for one or more
        selected steps of the trajectory.

        Returns
        -------
        dict
            Contains the forces for all selected steps and the structural information
            to know on which atoms the forces act.

        {examples}
        """
        return {
            "structure": self._structure[self._steps].read(),
            "forces": self._force[self._steps],
        }

    @base.data_access
    @documentation.format(examples=slice_.examples("force", "to_graph"))
    def plot(self):
        """Visualize the forces showing arrows at the atoms.

        Returns
        -------
        Viewer3d
            Shows the structure with cell and all atoms adding arrows to the atoms
            sized according to the strength of the force.

        {examples}
        """
        self._raise_error_if_slice()
        forces = self.force_rescale * self._force[self._steps]
        color = [0.3, 0.15, 0.35]
        fig = self._structure.plot()
        fig.show_arrows_at_atoms(forces, color)
        return fig

    @property
    def _force(self):
        return _ForceReader(self._raw_data.forces)

    def _raise_error_if_slice(self):
        if self._is_slice:
            message = "Plotting forces for multiple steps is not implemented."
            raise exception.NotImplemented(message)


class _ForceReader(reader.Reader):
    def error_message(self, key, err):
        key = np.array(key)
        steps = key if key.ndim == 0 else key[0]
        return (
            f"Error reading the forces. Please check if the steps "
            f"`{steps}` are properly formatted and within the boundaries. "
            "Additionally, you may consider the original error message:\n" + err.args[0]
        )
