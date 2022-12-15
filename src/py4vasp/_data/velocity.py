# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp import exception
from py4vasp._config import VASP_GRAY
from py4vasp._data import base, slice_, structure
from py4vasp._util import convert


class Velocity(slice_.Mixin, base.Refinery, structure.Mixin):
    "The ion velocities for all steps of the calculation."
    velocity_rescale = 1

    def to_dict(self):
        return {
            "structure": self._structure[self._steps].read(),
            "velocities": self._raw_data.velocities[self._steps],
        }

    def plot(self):
        self._raise_error_if_slice()
        velocities = self.velocity_rescale * self._raw_data.velocities[self._steps]
        viewer = self._structure.plot()
        viewer.show_arrows_at_atoms(velocities, convert.to_rgb(VASP_GRAY))
        return viewer

    def _raise_error_if_slice(self):
        if self._is_slice:
            message = "Plotting velocities for multiple steps is not implemented."
            raise exception.NotImplemented(message)
