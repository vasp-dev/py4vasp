# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from math import pow
from typing import Optional

import numpy as np

from py4vasp._calculation import base, structure
from py4vasp._raw import data as raw_data
from py4vasp._raw.data_db import ElasticModulus_DB
from py4vasp._util import check
from py4vasp._util.tensor import symmetry_reduce


class ElasticModulus(base.Refinery):
    """The elastic modulus is the second derivative of the energy with respect to strain.

    The elastic modulus, also known as the modulus of elasticity, is a measure of a
    material's stiffness and its ability to deform elastically in response to an
    applied force. It quantifies the ratio of stress (force per unit area) to strain
    (deformation) in a material within its elastic limit. You can use this class to
    extract the elastic modulus of a linear response calculation. There are two
    variants of the elastic modulus: (i) in the clamped-ion one, the cell is deformed
    but the ions are kept in their positions; (ii) in the relaxed-ion one the
    atoms are allowed to relax when the cell is deformed.
    """

    _raw_data: raw_data.ElasticModulus

    @base.data_access
    def to_dict(self):
        """Read the clamped-ion and relaxed-ion elastic modulus into a dictionary.

        Returns
        -------
        dict
            Contains the level of approximation and its associated elastic modulus.
        """
        return {
            "clamped_ion": self._raw_data.clamped_ion[:],
            "relaxed_ion": self._raw_data.relaxed_ion[:],
        }

    @base.data_access
    def _to_database(self, *args, **kwargs):
        volume_per_atom = None
        (
            bulk_modulus,
            shear_modulus,
            youngs_modulus,
            poisson_ratio,
            pugh_ratio,
            vickers_hardness,
            fracture_toughness,
        ) = ([None, None, None] for _ in range(7))
        compact_tensor = [None, None, None]
        if not check.is_none(self._raw_data.structure):
            structure_obj = structure.Structure.from_data(self._raw_data.structure)
            volume = structure_obj.volume()
            num_atoms = structure_obj.number_atoms()
            volume_per_atom = volume / num_atoms if num_atoms > 0 else None

        total_tensor, ionic_tensor, electronic_tensor = None, None, None
        if not check.is_none(self._raw_data.clamped_ion):
            electronic_tensor = self._raw_data.clamped_ion[:]
        if not check.is_none(self._raw_data.relaxed_ion):
            total_tensor = self._raw_data.relaxed_ion[:]
        if electronic_tensor is not None and total_tensor is not None:
            ionic_tensor = total_tensor - electronic_tensor

        for idt, tensor in enumerate([total_tensor, ionic_tensor, electronic_tensor]):
            voigt_tensor = None
            try:
                if not check.is_none(tensor):
                    compact_tensor[idt] = symmetry_reduce(symmetry_reduce(tensor).T).T
                    voigt_tensor = compact_tensor[idt] / 10.0  # converting kbar to GPa
                    compact_tensor[idt] = (
                        list([list(l) for l in compact_tensor[idt]])
                        if compact_tensor[idt] is not None
                        else None
                    )
            except:
                pass
            try:
                # Properties from elastic tensor
                (
                    bulk_modulus[idt],
                    shear_modulus[idt],
                    youngs_modulus[idt],
                    poisson_ratio[idt],
                    pugh_ratio[idt],
                    vickers_hardness[idt],
                    fracture_toughness[idt],
                ) = self._compute_elastic_properties(
                    voigt_tensor, volume_per_atom=volume_per_atom
                )
            except:
                pass

        return {
            "elastic_modulus": ElasticModulus_DB(
                total_3d_tensor=compact_tensor[0],  # [kbar]
                total_bulk_modulus=bulk_modulus[0],  # [GPa]
                total_shear_modulus=shear_modulus[0],  # [GPa]
                total_young_modulus=youngs_modulus[0],  # [GPa]
                total_poisson_ratio=poisson_ratio[0],  # []
                total_pugh_ratio=pugh_ratio[0],  # []
                total_vickers_hardness=vickers_hardness[0],  # [GPa]
                total_fracture_toughness=fracture_toughness[0],  # [MPa·m^0.5]
                ionic_3d_tensor=compact_tensor[1],
                ionic_bulk_modulus=bulk_modulus[1],
                ionic_shear_modulus=shear_modulus[1],
                ionic_young_modulus=youngs_modulus[1],
                ionic_poisson_ratio=poisson_ratio[1],
                ionic_pugh_ratio=pugh_ratio[1],
                ionic_vickers_hardness=vickers_hardness[1],
                ionic_fracture_toughness=fracture_toughness[1],
                electronic_3d_tensor=compact_tensor[2],
                electronic_bulk_modulus=bulk_modulus[2],
                electronic_shear_modulus=shear_modulus[2],
                electronic_young_modulus=youngs_modulus[2],
                electronic_poisson_ratio=poisson_ratio[2],
                electronic_pugh_ratio=pugh_ratio[2],
                electronic_vickers_hardness=vickers_hardness[2],
                electronic_fracture_toughness=fracture_toughness[2],
            )
        }

    @base.data_access
    def __str__(self):
        return f"""Elastic modulus (kBar)
Direction    XX          YY          ZZ          XY          YZ          ZX
--------------------------------------------------------------------------------
{_elastic_modulus_string(self._raw_data.clamped_ion[:], "clamped-ion")}
{_elastic_modulus_string(self._raw_data.relaxed_ion[:], "relaxed-ion")}"""

    def _compute_elastic_properties(
        self, voigt_tensor: np.ndarray, volume_per_atom: Optional[float] = None
    ) -> tuple:
        (
            bulk_modulus,
            shear_modulus,
            youngs_modulus,
            poisson_ratio,
            pugh_ratio,
            vickers_hardness,
            fracture_toughness,
        ) = (None, None, None, None, None, None, None)

        try:
            elastic_tensor = _ElasticTensor.from_array(voigt_tensor)

            try:
                bulk_modulus, shear_modulus, youngs_modulus, poisson_ratio = (
                    elastic_tensor.get_VRH()
                )
            except Exception as e:
                pass

            try:
                if shear_modulus is not None and bulk_modulus is not None:
                    pugh_ratio = (
                        shear_modulus / bulk_modulus
                        if (bulk_modulus != 0 and shear_modulus != 0)
                        else 0.0 if shear_modulus == 0 else None
                    )
            except Exception as e:
                pass

            try:
                vickers_hardness = elastic_tensor.get_hardness()
            except Exception as e:
                pass

            try:
                fracture_toughness = elastic_tensor.get_fracture_toughness(
                    volume_per_atom
                )
            except Exception as e:
                pass
        except Exception as e:
            pass

        return (
            bulk_modulus,
            shear_modulus,
            youngs_modulus,
            poisson_ratio,
            pugh_ratio,
            vickers_hardness,
            fracture_toughness,
        )


def _elastic_modulus_string(tensor, label):
    compact_tensor = symmetry_reduce(symmetry_reduce(tensor).T).T
    line = lambda dir_, vec: dir_ + 6 * " " + " ".join(f"{x:11.4f}" for x in vec)
    directions = ("XX", "YY", "ZZ", "XY", "YZ", "ZX")
    lines = (line(dir_, vec) for dir_, vec in zip(directions, compact_tensor))
    return f"{label:^79}".rstrip() + "\n" + "\n".join(lines)


class _ElasticTensor:
    """
    Elastic (stiffness) tensor utilities.

    Supports initialization from:
      - 6x6 Voigt matrix
      - 21-element upper-triangular row
    """

    # Indices of diagonal elements in flattened upper triangle
    _diag_indices = np.array([0, 6, 11, 15, 18, 20])

    # Common index groups for VRH averages
    _bulk_indices = np.array([[0, 0], [1, 1], [2, 2], [0, 1], [1, 2], [0, 2]])
    _shear_indices = np.array(
        [
            [0, 0],
            [1, 1],
            [2, 2],
            [0, 1],
            [1, 2],
            [0, 2],
            [3, 3],
            [4, 4],
            [5, 5],
        ]
    )

    def __init__(self, C: np.ndarray):
        if C.shape == (6, 6):
            self._tensor = C
            self._row = None
        elif C.shape == (21,):
            self._row = C
            self._tensor = None
        else:
            raise ValueError("Input must be a (6,6) or (21,) numpy array.")

        self._compliance = None

    # ---------- Constructors ----------

    @classmethod
    def from_array(cls, C: np.ndarray):
        return cls(C)

    # ---------- Tensor / row conversion ----------

    @staticmethod
    def _row_to_tensor(row: np.ndarray) -> np.ndarray:
        C = np.zeros((6, 6))
        C[np.diag_indices(6)] = row[_ElasticTensor._diag_indices]

        idx = _ElasticTensor._diag_indices.copy()
        for k in range(1, 6):
            idx = idx[:-1] + 1
            C += np.diag(row[idx], k=k) + np.diag(row[idx], k=-k)

        return C

    @staticmethod
    def _tensor_to_row(C: np.ndarray) -> np.ndarray:
        return C[np.triu_indices(6)]

    # ---------- Properties ----------

    @property
    def tensor(self) -> np.ndarray:
        if self._tensor is None:
            self._tensor = self._row_to_tensor(self._row)
        return self._tensor

    @property
    def row(self) -> np.ndarray:
        if self._row is None:
            self._row = self._tensor_to_row(self._tensor)
        return self._row

    @property
    def compliance_tensor(self) -> np.ndarray:
        if self._compliance is None:
            self._compliance = np.linalg.inv(self.tensor)
        return self._compliance

    # ---------- Internal helpers ----------

    @staticmethod
    def _weighted_sum(tensor, indices, coeffs):
        return sum(tensor[i, j] * c for (i, j), c in zip(indices, coeffs))

    # ---------- Mechanical properties ----------

    def get_VRH(self):
        """
        Voigt–Reuss–Hill averaged elastic moduli.

        Returns
        -------
        K, G, E, nu
        """
        Kv = (1 / 9) * self._weighted_sum(
            self.tensor,
            self._bulk_indices,
            [1, 1, 1, 2, 2, 2],
        )

        Gv = (1 / 15) * self._weighted_sum(
            self.tensor,
            self._shear_indices,
            [1, 1, 1, -1, -1, -1, 3, 3, 3],
        )

        Kr_inv = self._weighted_sum(
            self.compliance_tensor,
            self._bulk_indices,
            [1, 1, 1, 2, 2, 2],
        )

        Gr_inv = (1 / 15) * self._weighted_sum(
            self.compliance_tensor,
            self._shear_indices,
            [4, 4, 4, -4, -4, -4, 3, 3, 3],
        )

        K = 0.5 * (Kv + 1 / Kr_inv)
        G = 0.5 * (Gv + 1 / Gr_inv)
        E = 9 * K * G / (3 * K + G)
        nu = (3 * K - 2 * G) / (6 * K + 2 * G)

        return K, G, E, nu

    def get_hardness(self):
        """
        Empirical Vickers hardness (GPa).

        DOI: 10.1063/1.5113622
        """
        _, _, E, nu = self.get_VRH()
        return (
            0.096
            * E
            * (1 - 8.5 * nu + 19.5 * pow(nu, 2))
            / (1 - 7.5 * nu + 12.2 * pow(nu, 2) + 19.6 * pow(nu, 3))
        )

    def get_fracture_toughness(self, V0):
        """
        Empirical fracture toughness (MPa m^1/2).

        DOI: 10.1063/1.5113622
        """
        if check.is_none(V0):
            return None
        _, _, E, nu = self.get_VRH()
        return (
            1e-2
            * pow(8840, -0.5)
            * pow(V0, 1.0 / 6.0)
            * pow(
                (
                    E
                    * (1.0 - 13.7 * nu + 48.6 * pow(nu, 2.0))
                    / (1.0 - 15.2 * nu + 70.2 * pow(nu, 2.0) - 81.5 * pow(nu, 3.0))
                ),
                1.5,
            )
        )
