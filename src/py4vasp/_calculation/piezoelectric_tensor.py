# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from contextlib import suppress

import numpy as np

from py4vasp import exception
from py4vasp._calculation import base, cell
from py4vasp._raw import data as raw_data
from py4vasp._raw.data_db import PiezoelectricTensor_DB
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
        reduced_tensor_x, reduced_tensor_y, reduced_tensor_z, tensor_2d = (
            [None, None, None] for _ in range(4)
        )
        in_plane = [[False, False, False] for _ in range(3)]
        e11, e22, e33, e_avg_abs, e_rms, e_frobenius = (
            [None, None, None] for _ in range(6)
        )

        total_tensor, electronic_tensor, ionic_tensor = None, None, None
        if not check.is_none(self._raw_data.ion) and not check.is_none(
            self._raw_data.electron
        ):
            total_tensor = self._raw_data.electron[:] + self._raw_data.ion[:]
        if not check.is_none(self._raw_data.electron):
            electronic_tensor = self._raw_data.electron[:]
        if not check.is_none(self._raw_data.ion):
            ionic_tensor = self._raw_data.ion[:]

        for idt, tensor in enumerate([total_tensor, ionic_tensor, electronic_tensor]):
            e_tensor = None
            # Piezoelectric stress tensor e_ij (C/m^2)
            with suppress(exception.Py4VaspError):
                e_tensor = _extract_tensor(
                    tensor
                )  # 3x6 tensor, column order: XX YY ZZ YZ ZX XY
                # write in default VASP order (rows: x,y,z; columns: XX,YY,ZZ,XY,YZ,ZX)
                reduced_tensor_x[idt] = e_tensor[0, (0, 1, 2, 5, 3, 4)].tolist()
                reduced_tensor_y[idt] = e_tensor[1, (0, 1, 2, 5, 3, 4)].tolist()
                reduced_tensor_z[idt] = e_tensor[2, (0, 1, 2, 5, 3, 4)].tolist()

                if not check.is_none(self._raw_data.cell):
                    cCell = cell.Cell.from_data(self._raw_data.cell)
                    in_plane[idt], lvac = _compute_2d_plane_and_conversion_factor(cCell)
                    if in_plane[idt] is not None and lvac is not None:
                        tensor_2d[idt] = e_tensor * lvac

            with suppress(exception.Py4VaspError):
                (
                    e11[idt],
                    e22[idt],
                    e33[idt],
                    e_avg_abs[idt],
                    e_rms[idt],
                    e_frobenius[idt],
                ) = _compute_bulk_quantities(e_tensor)

        return {
            "piezoelectric_tensor": PiezoelectricTensor_DB(
                total_3d_tensor_x=reduced_tensor_x[0],
                total_3d_tensor_y=reduced_tensor_y[0],
                total_3d_tensor_z=reduced_tensor_z[0],
                total_3d_piezoelectric_stress_coefficient_x=e11[0],
                total_3d_piezoelectric_stress_coefficient_y=e22[0],
                total_3d_piezoelectric_stress_coefficient_z=e33[0],
                total_3d_mean_absolute=e_avg_abs[0],
                total_3d_rms=e_rms[0],
                total_3d_frobenius_norm=e_frobenius[0],
                total_2d_tensor_x=(
                    (
                        tensor_2d[0][0]
                        if (in_plane[0] is not None and in_plane[0][0])
                        else None
                    )
                    if tensor_2d[0] is not None
                    else None
                ),
                total_2d_tensor_y=(
                    (
                        tensor_2d[0][1]
                        if (in_plane[0] is not None and in_plane[0][1])
                        else None
                    )
                    if tensor_2d[0] is not None
                    else None
                ),
                total_2d_tensor_z=(
                    (
                        tensor_2d[0][2]
                        if (in_plane[0] is not None and in_plane[0][2])
                        else None
                    )
                    if tensor_2d[0] is not None
                    else None
                ),
                ionic_3d_tensor_x=reduced_tensor_x[1],
                ionic_3d_tensor_y=reduced_tensor_y[1],
                ionic_3d_tensor_z=reduced_tensor_z[1],
                ionic_3d_piezoelectric_stress_coefficient_x=e11[1],
                ionic_3d_piezoelectric_stress_coefficient_y=e22[1],
                ionic_3d_piezoelectric_stress_coefficient_z=e33[1],
                ionic_3d_mean_absolute=e_avg_abs[1],
                ionic_3d_rms=e_rms[1],
                ionic_3d_frobenius_norm=e_frobenius[1],
                ionic_2d_tensor_x=(
                    (
                        tensor_2d[1][0]
                        if (in_plane[1] is not None and in_plane[1][0])
                        else None
                    )
                    if tensor_2d[1] is not None
                    else None
                ),
                ionic_2d_tensor_y=(
                    (
                        tensor_2d[1][1]
                        if (in_plane[1] is not None and in_plane[1][1])
                        else None
                    )
                    if tensor_2d[1] is not None
                    else None
                ),
                ionic_2d_tensor_z=(
                    (
                        tensor_2d[1][2]
                        if (in_plane[1] is not None and in_plane[1][2])
                        else None
                    )
                    if tensor_2d[1] is not None
                    else None
                ),
                electronic_3d_tensor_x=reduced_tensor_x[2],
                electronic_3d_tensor_y=reduced_tensor_y[2],
                electronic_3d_tensor_z=reduced_tensor_z[2],
                electronic_3d_piezoelectric_stress_coefficient_x=e11[2],
                electronic_3d_piezoelectric_stress_coefficient_y=e22[2],
                electronic_3d_piezoelectric_stress_coefficient_z=e33[2],
                electronic_3d_mean_absolute=e_avg_abs[2],
                electronic_3d_rms=e_rms[2],
                electronic_3d_frobenius_norm=e_frobenius[2],
                electronic_2d_tensor_x=(
                    (
                        tensor_2d[2][0]
                        if (in_plane[2] is not None and in_plane[2][0])
                        else None
                    )
                    if tensor_2d[2] is not None
                    else None
                ),
                electronic_2d_tensor_y=(
                    (
                        tensor_2d[2][1]
                        if (in_plane[2] is not None and in_plane[2][1])
                        else None
                    )
                    if tensor_2d[2] is not None
                    else None
                ),
                electronic_2d_tensor_z=(
                    (
                        tensor_2d[2][2]
                        if (in_plane[2] is not None and in_plane[2][2])
                        else None
                    )
                    if tensor_2d[2] is not None
                    else None
                ),
            )
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


def _compute_2d_piezoelectric(e_tensor: np.ndarray, cell_: cell.Cell) -> np.ndarray:
    """
    Convert 3D piezoelectric stress tensor (C/m^2) to 2D (C/m).
    """
    # TODO migrate finding vacuum direction to structure
    in_plane, l_vac = _compute_2d_plane_and_conversion_factor(cell_)
    return e_tensor[in_plane, :] * l_vac


def _compute_2d_plane_and_conversion_factor(
    cell_: cell.Cell,
) -> tuple[list[bool], float]:
    """
    Identify the 2D plane (in-plane directions) and compute the conversion factor
    from 3D piezoelectric tensor (C/m^2) to 2D (C/m).
    """
    vac_dir = cell_._find_likely_vacuum_direction()
    if vac_dir is None:
        return None, None
    in_plane = [i != vac_dir for i in range(3)]
    l_vac = np.linalg.norm(cell_.lattice_vectors()[vac_dir]) * 1e-10  # Å → m
    return in_plane, l_vac


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
    if (e_tensor is None) or (e_tensor.shape != (3, 6)):
        return None, None, None, None, None, None
    e11 = e_tensor[0, 0]
    e22 = e_tensor[1, 1]
    e33 = e_tensor[2, 2]
    e_avg_abs = np.mean(np.abs(e_tensor))
    e_rms = np.sqrt(np.mean(e_tensor**2))
    e_frobenius = np.linalg.norm(e_tensor)

    return e11, e22, e33, e_avg_abs, e_rms, e_frobenius
