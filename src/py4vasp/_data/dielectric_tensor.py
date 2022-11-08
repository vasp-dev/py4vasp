# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp._data import base
from py4vasp._util import convert


class DielectricTensor(base.Refinery):
    """The static dielectric tensor obtained from linear response."""

    @base.data_access
    def to_dict(self):
        """Read the dielectric tensor into a dictionary.

        Returns
        -------
        dict
        Contains the dielectric tensor and a string describing the method it
        was obtained.
        """
        return {
            "clamped_ion": self._raw_data.electron[:],
            "relaxed_ion": self._read_relaxed_ion(),
            "independent_particle": self._read_independent_particle(),
            "method": convert.text_to_string(self._raw_data.method),
        }

    @base.data_access
    def __str__(self):
        data = self.to_dict()
        return f"""
Macroscopic static dielectric tensor (dimensionless)
  {_description(data["method"])}
------------------------------------------------------
{_dielectric_tensor_string(data["clamped_ion"], "clamped-ion")}
{_dielectric_tensor_string(data["relaxed_ion"], "relaxed-ion")}
""".strip()

    def _read_relaxed_ion(self):
        if self._raw_data.ion.is_none():
            return None
        else:
            return self._raw_data.electron[:] + self._raw_data.ion[:]

    def _read_independent_particle(self):
        if self._raw_data.independent_particle.is_none():
            return None
        else:
            return self._raw_data.independent_particle[:]


def _dielectric_tensor_string(tensor, label):
    if tensor is None:
        return ""
    row_to_string = lambda row: 6 * " " + " ".join(f"{x:12.6f}" for x in row)
    rows = (row_to_string(row) for row in tensor)
    return f"{label:^55}".rstrip() + "\n" + "\n".join(rows)


def _description(method):
    if method == "dft":
        return "including local field effects in DFT"
    elif method == "rpa":
        return "including local field effects in RPA (Hartree)"
    elif method == "scf":
        return "including local field effects"
    elif method == "nscf":
        return "excluding local field effects"
    assert False  # unknown method
