# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp._calculation import base, cell
from py4vasp._raw import data as raw_data
from py4vasp._util import check
from py4vasp._util.tensor import symmetry_reduce, tensor_constants


class PiezoelectricTensor(base.Refinery):
    """The piezoelectric tensor is the derivative of the energy with respect to strain and field.

    The piezoelectric tensor represents the coupling between mechanical stress and
    electrical polarization in a material. VASP computes the piezoelectric tensor with
    a linear response calculation. The piezoelectric tensor is a 3x3 matrix that relates
    the three components of stress to the three components of polarization.
    Specifically, it describes how the application of mechanical stress induces an
    electric polarization and, conversely, how an applied electric field results in
    a deformation.

    The piezoelectric tensor helps to characterize the efficiency and anisotropy of the
    piezoelectric response. A large piezoelectric tensor is useful e.g. for sensors
    and actuators. Moreover, the tensor's symmetry properties are coupled to the crystal
    structure and symmetry. Therefore a mismatch of the symmetry properties between
    calculations and experiment can reveal underlying flaws in the characterization of
    the crystal structure.
    """

    _raw_data: raw_data.PiezoelectricTensor

    @base.data_access
    def __str__(self):
        data = self.to_dict()
        return f"""Piezoelectric tensor (C/m²)
         XX          YY          ZZ          XY          YZ          ZX
---------------------------------------------------------------------------
{_tensor_to_string(data["clamped_ion"], "clamped-ion")}
{_tensor_to_string(data["relaxed_ion"], "relaxed-ion")}"""

    @base.data_access
    def to_dict(self):
        """Read the ionic and electronic contribution to the piezoelectric tensor
        into a dictionary.

        It will combine both terms as the total piezoelectric tensor (relaxed_ion)
        but also give the pure electronic contribution, so that you can separate the
        parts.

        Returns
        -------
        dict
            The clamped ion and relaxed ion data for the piezoelectric tensor.
        """
        electron_data = self._raw_data.electron[:]
        return {
            "clamped_ion": electron_data,
            "relaxed_ion": electron_data + self._raw_data.ion[:],
        }

    @base.data_access
    def _to_database(self, *args, **kwargs):
        data = self.to_dict()

        tensor_x = None
        tensor_y = None
        tensor_z = None
        reduced_tensor_x = None
        reduced_tensor_y = None
        reduced_tensor_z = None

        main_tensor = data["relaxed_ion"] - data["clamped_ion"]

        try:
            tensor_x = main_tensor[0]
            reduced_tensor_x = list(symmetry_reduce(tensor_x))
        except:
            pass

        try:
            tensor_y = main_tensor[1]
            reduced_tensor_y = list(symmetry_reduce(tensor_y))
        except:
            pass

        try:
            tensor_z = main_tensor[2]
            reduced_tensor_z = list(symmetry_reduce(tensor_z))
        except:
            pass

        # Piezoelectric stress tensor e_ij (C/m^2)
        e_tensor = _extract_tensor(main_tensor)

        plane_2d, tensor_2d = None, None
        try:
            if not check.is_none(self._raw_data.cell):
                cCell = cell.Cell.from_data(self._raw_data.cell)
                results_2d = _compute_2d_piezoelectric(
                    e_tensor, cCell.lattice_vectors()
                )
                plane_2d = results_2d["2d_plane"]
                tensor_2d = results_2d["2d_tensor"]
        except Exception as e:
            pass

        # TODO make sure that for dielectric_tensor, piezoelectric_tensor, elastic_modulus:
        # - the retrieved tensors are what we expect (xx, yy etc. order seems shuffled in piezo)
        # - we use the correct tensors (ionic, electronic, total)
        # - we can project into 2d as suggested by Zahed
        # (we can check the OUTCAR end - CTRL+END)

        e11, e22, e33, e_avg_abs, e_rms, e_frobenius = (
            None,
            None,
            None,
            None,
            None,
            None,
        )
        try:
            (
                e11,
                e22,
                e33,
                e_avg_abs,
                e_rms,
                e_frobenius,
            ) = _compute_bulk_quantities(e_tensor)
        except:
            pass
        return {
            "piezoelectric_tensor": {
                "3d_tensor_reduced_x": reduced_tensor_x,
                "3d_tensor_reduced_y": reduced_tensor_y,
                "3d_tensor_reduced_z": reduced_tensor_z,
                "3d_piezoelectric_stress_coefficient_x": e11,
                "3d_piezoelectric_stress_coefficient_y": e22,
                "3d_piezoelectric_stress_coefficient_z": e33,
                "3d_mean_absolute": e_avg_abs,
                "3d_rms": e_rms,
                "3d_frobenius_norm": e_frobenius,
                "2d_tensor_reduced": tensor_2d,
                "2d_plane": plane_2d,
            }
        }


def _tensor_to_string(tensor, label):
    compact_tensor = symmetry_reduce(tensor.T).T
    line = lambda dir_, vec: dir_ + " " + " ".join(f"{x:11.5f}" for x in vec)
    directions = (" x", " y", " z")
    lines = (line(dir_, vec) for dir_, vec in zip(directions, compact_tensor))
    return f"{label:^75}".rstrip() + "\n" + "\n".join(lines)


def _extract_tensor(raw_tensor):
    voigt_indices = {
        (0, 0): 0,  # XX
        (1, 1): 1,  # YY
        (2, 2): 2,  # ZZ
        (1, 2): 3,  # YZ
        (0, 2): 4,  # ZX
        (0, 1): 5,  # XY
    }
    C_voigt = np.zeros((3, 6))
    for i in range(3):
        for j in range(i, 3):
            for k in range(3):
                column = voigt_indices.get((i, j))
                C_voigt[k, column] = raw_tensor[i, j, k]

    return C_voigt


def _compute_2d_piezoelectric(e_tensor: np.ndarray, lattice: np.ndarray) -> dict:
    """
    Convert 3D piezoelectric stress tensor (C/m^2) to 2D (C/m).
    """
    vac_dir = cell._find_vacuum_direction(lattice)
    if vac_dir is None:
        return {
            "2d_plane": None,
            "2d_tensor": None,
        }
    in_plane = [i for i in range(3) if i != vac_dir]

    l_vac = np.linalg.norm(lattice[vac_dir]) * 1e-10  # Å → m

    e2d = np.zeros((2, 6))
    for idx, i in enumerate(in_plane):
        for j in range(6):
            e2d[idx, j] = e_tensor[i, j] * l_vac

    conversion_list = ["a", "b", "c"]
    return {
        "2d_plane": str.join("", [conversion_list[l] for l in in_plane]),
        "2d_tensor": e2d,
    }


def _compute_bulk_quantities(
    e_tensor: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    """
    Scalar bulk measures from a VASP piezoelectric stress tensor (3x6).
    Units: C/m^2

    e11: Piezoelectric stress coefficient: polarization along x induced by normal strain along x (∂Px/∂εxx).

    e22: Piezoelectric stress coefficient: polarization along y induced by normal strain along y (∂Py/∂εyy).

    e33: Piezoelectric stress coefficient: polarization along z induced by normal strain along z (∂Pz/∂εzz);
        *** sign indicates direction of polarization relative to applied strain.

    e_avg_abs:
        - Mean absolute value of all piezoelectric tensor components;
        - a scalar descriptor of the overall piezoelectric response magnitude (not a fundamental constant).

    e_rms:
        - Root-mean-square of piezoelectric tensor components;
        - emphasizes larger tensor elements and provides a normalized measure of overall piezoelectric strength.

    e_frobenius:
        - Frobenius norm of the piezoelectric tensor;
        - rotation-invariant total magnitude of the piezoelectric response, commonly used for materials screening.

    """
    e11 = e_tensor[0, 0]
    e22 = e_tensor[1, 1]
    e33 = e_tensor[2, 2]
    e_avg_abs = np.mean(np.abs(e_tensor))
    e_rms = np.sqrt(np.mean(e_tensor**2))
    e_frobenius = np.linalg.norm(e_tensor)

    return e11, e22, e33, e_avg_abs, e_rms, e_frobenius
