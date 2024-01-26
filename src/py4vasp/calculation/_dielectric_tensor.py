# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp import exception
from py4vasp._util import convert
from py4vasp.calculation import _base


class DielectricTensor(_base.Refinery):
    """The dielectric tensor is the static limit of the :attr:`dielectric function<py4vasp.calculation.dielectric_function>`.

    The dielectric tensor represents how a material's response to an external electric
    field varies with direction. It is a symmetric 3x3 matrix, encapsulating the
    anisotropic nature of a material's dielectric properties. Each element of the
    tensor corresponds to the dielectric function along a specific crystallographic
    axis."""

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
            "relaxed_ion": self._read_relaxed_ion(),
            "independent_particle": self._read_independent_particle(),
            "method": convert.text_to_string(self._raw_data.method),
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
    message = f"The method {method} is not implemented in this version of py4vasp."
    raise exception.NotImplemented(message)
