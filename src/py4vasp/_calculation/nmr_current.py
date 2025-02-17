# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

from py4vasp._calculation import base, structure


class NMRCurrent(base.Refinery, structure.Mixin):
    """The NMR (Nuclear Magnetic Resonance) current refers to the electrical response
    induced in a detection coil by the precessing magnetic moments of nuclear spins in a
    sample. When the nuclei are exposed to a strong external magnetic field and excited
    by a radiofrequency pulse, they generate an oscillating magnetization that induces
    a weak but detectable voltage in the coil. This signal, known as the free induction
    decay (FID), contains information about the chemical environment, molecular
    structure, and dynamics of the sample.
    """

    @base.data_access
    def to_dict(self):
        return {"structure": self._structure.read()}
