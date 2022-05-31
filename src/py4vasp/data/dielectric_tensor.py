# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import py4vasp.data._base as _base
import py4vasp._util.convert as _convert


class DielectricTensor(_base.Refinery):
    """The static dielectric tensor obtained from linear response."""

    @_base.data_access
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
            "relaxed_ion": self._raw_data.electron[:] + self._raw_data.ion[:],
            "independent_particle": self._read_independent_particle(),
            "method": _convert.text_to_string(self._raw_data.method),
        }

    @_base.data_access
    def __str__(self):
        data = self.to_dict()
        return f"""
Macroscopic static dielectric tensor (dimensionless)
  {_description(data["method"])}
------------------------------------------------------
{_dielectric_tensor_string(data["clamped_ion"], "clamped-ion")}
{_dielectric_tensor_string(data["relaxed_ion"], "relaxed-ion")}
""".strip()

    def _read_independent_particle(self):
        if self._raw_data.independent_particle is None:
            return None
        else:
            return self._raw_data.independent_particle[:]


def _dielectric_tensor_string(tensor, label):
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
