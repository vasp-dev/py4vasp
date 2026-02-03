# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import exception
from py4vasp._calculation import base, cell
from py4vasp._demo import dielectric_tensor
from py4vasp._demo.dielectric_tensor import dielectric_tensor
from py4vasp._raw import data as raw_data
from py4vasp._util import check, convert
from py4vasp._util.tensor import symmetry_reduce, tensor_constants


class DielectricTensor(base.Refinery):
    """The dielectric tensor is the static limit of the :attr:`dielectric function<py4vasp.calculation.dielectric_function>`.

    The dielectric tensor represents how a material's response to an external electric
    field varies with direction. It is a symmetric 3x3 matrix, encapsulating the
    anisotropic nature of a material's dielectric properties. Each element of the
    tensor corresponds to the dielectric function along a specific crystallographic
    axis."""

    _raw_data: raw_data.DielectricTensor

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
    def _to_database(self, *args, **kwargs):
        tensor_reduced, isotropic_dielectric_constant, polarizability_2d = (
            None,
            None,
            None,
        )
        (
            tensor_reduced_ionic,
            isotropic_dielectric_constant_ionic,
            polarizability_2d_ionic,
        ) = (
            None,
            None,
            None,
        )
        try:
            tensor = self._read_relaxed_ion()
            isotropic_dielectric_constant, polarizability_2d = (
                self._calculate_dielectric_quantities(tensor)
            )
            tensor_reduced = list(symmetry_reduce(tensor))
        except:
            pass
        try:
            tensor = self._raw_data.ion[:]
            isotropic_dielectric_constant_ionic, polarizability_2d_ionic = (
                self._calculate_dielectric_quantities(tensor)
            )
            tensor_reduced_ionic = list(symmetry_reduce(tensor))
        except:
            pass

        method = (
            convert.text_to_string(self._raw_data.method)
            if not check.is_none(self._raw_data.method)
            else None
        )

        dielectric_tensor_db = {
            "dielectric_tensor": {
                "method": method,
                "tensor_reduced_ionic": tensor_reduced_ionic,
                "isotropic_dielectric_constant_ionic": isotropic_dielectric_constant_ionic,
                "polarizability_2d_ionic": polarizability_2d_ionic,
                "tensor_reduced_total": tensor_reduced,
                "isotropic_dielectric_constant_total": isotropic_dielectric_constant,
                "polarizability_2d_total": polarizability_2d,
            }
        }
        return dielectric_tensor_db

    @base.data_access
    def _calculate_dielectric_quantities(self, tensor: np.ndarray) -> float:
        # 2D polarizability for slab systems
        polarizability_2d = None
        is_2d_system = None
        try:
            if not (check.is_none(self._raw_data.cell)):
                final_cell = cell.Cell.from_data(self._raw_data.cell)
                if final_cell.is_2d_system:
                    is_2d_system = True
                    polarizability_2d = _calculate_2d_polarizability(
                        tensor, final_cell.lattice_vectors()
                    )
        except Exception:
            pass
        # 3D isotropic dielectric constant
        isotropic_dielectric_constant = None
        isotropic_dielectric_constant = float(np.mean(np.diag(tensor)))
        if is_2d_system is None:
            # unknown dimensionality of system
            return isotropic_dielectric_constant, None
        elif is_2d_system:
            # confirmed 2D
            return isotropic_dielectric_constant, polarizability_2d
        else:
            # confirmed 3D
            return isotropic_dielectric_constant, None

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
        if check.is_none(self._raw_data.ion):
            return None
        else:
            return self._raw_data.electron[:] + self._raw_data.ion[:]

    def _read_independent_particle(self):
        if check.is_none(self._raw_data.independent_particle):
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


def _find_vacuum_direction(lattice_vectors: np.ndarray) -> int:
    """
    Identify vacuum direction as the lattice vector with the largest length.
    """
    try:
        if lattice_vectors.shape != (3, 3):
            return None
        lengths = np.linalg.norm(lattice_vectors, axis=1)
        return int(np.argmax(lengths))
    except Exception:
        return None


def _calculate_2d_polarizability(
    dielectric_tensor: np.ndarray, lattice_vectors: np.ndarray
) -> float:
    """
    Compute 2D polarizability (alpha_2D) for a slab system with unknown vacuum direction.
    """
    try:
        vacuum_dir = _find_vacuum_direction(lattice_vectors)
        if vacuum_dir is None:
            return None

        # In-plane directions
        in_plane_dirs = [i for i in range(3) if i != vacuum_dir]

        eps_parallel = np.mean([dielectric_tensor[i, i] for i in in_plane_dirs])

        l_vacuum = np.linalg.norm(lattice_vectors[vacuum_dir])

        alpha_2d = (l_vacuum / (4.0 * np.pi)) * (eps_parallel - 1.0)
        return alpha_2d
    except Exception:
        return None
