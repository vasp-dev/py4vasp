# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp._data import base, slice_, structure


class Velocity(slice_.Mixin, base.Refinery, structure.Mixin):
    "The ion velocities for all steps of the calculation."

    def to_dict(self):
        return {
            "structure": self._structure[self._steps].read(),
            "velocities": self._raw_data.velocities[self._steps],
        }
