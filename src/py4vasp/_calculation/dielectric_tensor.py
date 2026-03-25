# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from typing import Optional

import numpy as np

from py4vasp import exception
from py4vasp._calculation import base, cell
from py4vasp._raw import data as raw_data
from py4vasp._raw.data_db import DielectricTensor_DB
from py4vasp._util import check, convert
from py4vasp._util.tensor import symmetry_reduce

_TO_DATABASE_SUPPRESSED_EXCEPTIONS = (
    exception.Py4VaspError,
    AttributeError,
    TypeError,
    ValueError,
)


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
        encountered_errors = kwargs.get("encountered_errors")
        selection = kwargs.get("selection") or "default"
        error_key = f"dielectric_tensor:{selection}"

        tensor_reduced = [None, None, None]
        isotropic_dielectric_constant = [None, None, None]
        polarizability_2d = [None, None, None]

        total_tensor, ionic_tensor, electronic_tensor = None, None, None
        if not check.is_none(self._raw_data.electron):
            electronic_tensor = self._raw_data.electron[:]
        if not check.is_none(self._raw_data.ion):
            ionic_tensor = self._raw_data.ion[:]
        if not check.is_none(self._raw_data.ion) and not check.is_none(
            self._raw_data.electron
        ):
            total_tensor = self._raw_data.electron[:] + self._raw_data.ion[:]

        for idt, tensor in enumerate([total_tensor, ionic_tensor, electronic_tensor]):
            with base.suppress_and_record(
                encountered_errors,
                error_key,
                *_TO_DATABASE_SUPPRESSED_EXCEPTIONS,
                context=f"to_database.tensor[{idt}]",
            ):
                tensor_reduced[idt] = list(symmetry_reduce(tensor.T))
                (
                    isotropic_dielectric_constant[idt],
                    polarizability_2d[idt],
                ) = self._calculate_dielectric_quantities(
                    tensor,
                    encountered_errors=encountered_errors,
                    error_key=error_key,
                )

        method = (
            convert.text_to_string(self._raw_data.method)
            if not check.is_none(self._raw_data.method)
            else None
        )

        dielectric_tensor_db = {
            "dielectric_tensor": DielectricTensor_DB(
                method=method,
                total_3d_tensor=tensor_reduced[0],
                total_3d_isotropic_dielectric_constant=isotropic_dielectric_constant[0],
                total_2d_polarizability=polarizability_2d[0],
                ionic_3d_tensor=tensor_reduced[1],
                ionic_3d_isotropic_dielectric_constant=isotropic_dielectric_constant[1],
                ionic_2d_polarizability=polarizability_2d[1],
                electronic_3d_tensor=tensor_reduced[2],
                electronic_3d_isotropic_dielectric_constant=isotropic_dielectric_constant[
                    2
                ],
                electronic_2d_polarizability=polarizability_2d[2],
            )
        }
        return dielectric_tensor_db

    @base.data_access
    def _calculate_dielectric_quantities(
        self,
        tensor: np.ndarray,
        *,
        encountered_errors: Optional[dict[str, list[str]]] = None,
        error_key: Optional[str] = None,
    ) -> float:
        # 2D polarizability for slab systems
        # TODO migrate finding vacuum direction to structure
        polarizability_2d = None
        with base.suppress_and_record(
            encountered_errors,
            error_key,
            *_TO_DATABASE_SUPPRESSED_EXCEPTIONS,
            context="calculate_dielectric_quantities",
        ):
            if not (check.is_none(self._raw_data.cell)):
                final_cell = cell.Cell.from_data(self._raw_data.cell)
                if final_cell:
                    polarizability_2d = _calculate_2d_polarizability(
                        tensor,
                        final_cell,
                        encountered_errors=encountered_errors,
                        error_key=error_key,
                    )

        # 3D isotropic dielectric constant
        isotropic_dielectric_constant = None
        isotropic_dielectric_constant = float(np.mean(np.diag(tensor)))
        return isotropic_dielectric_constant, polarizability_2d

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


def _calculate_2d_polarizability(
    dielectric_tensor: np.ndarray,
    cell_: cell.Cell,
    *,
    encountered_errors: Optional[dict[str, list[str]]] = None,
    error_key: Optional[str] = None,
) -> float:
    """
    Compute 2D polarizability (alpha_2D) for a slab system with unknown vacuum direction.
    """
    with base.suppress_and_record(
        encountered_errors,
        error_key,
        *_TO_DATABASE_SUPPRESSED_EXCEPTIONS,
        context="calculate_2d_polarizability",
    ):
        vacuum_dir = cell_._find_likely_vacuum_direction()
        if vacuum_dir is None:
            return None

        eps_parallel = np.mean(
            [dielectric_tensor[i, i] for i in range(3) if i != vacuum_dir]
        )
        l_vacuum = np.linalg.norm(cell_.lattice_vectors()[vacuum_dir])

        alpha_2d = (l_vacuum / (4.0 * np.pi)) * (eps_parallel - 1.0)
        return alpha_2d
    return None
