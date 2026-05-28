# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import raw
from py4vasp._calculation._dispersion import DispersionHandler
from py4vasp._calculation.dispatch import (
    DataSource,
    merge_default,
    merge_strings,
    quantity,
)
from py4vasp._raw.data_db import ExcitonEigenvector_DB
from py4vasp._util import check, convert


class ExcitonEigenvectorHandler:
    """Handler for BSE exciton eigenvector data."""

    def __init__(self, raw_exciton_eigenvector: raw.ExcitonEigenvector):
        self._raw_exciton_eigenvector = raw_exciton_eigenvector

    @classmethod
    def from_data(
        cls, raw_exciton_eigenvector: raw.ExcitonEigenvector
    ) -> "ExcitonEigenvectorHandler":
        return cls(raw_exciton_eigenvector)

    def __str__(self) -> str:
        shape = self._raw_exciton_eigenvector.bse_index.shape
        return f"""BSE eigenvector data:
    {shape[1]} k-points
    {shape[3]} valence bands
    {shape[2]} conduction bands"""

    def to_dict(self) -> dict:
        eigenvectors = convert.to_complex(self._raw_exciton_eigenvector.eigenvectors[:])
        dispersion = self._dispersion().to_dict()
        shifted_eigenvalues = (
            dispersion.pop("eigenvalues") - self._raw_exciton_eigenvector.fermi_energy
        )
        return {
            **dispersion,
            "bands": shifted_eigenvalues,
            "bse_index": self._raw_exciton_eigenvector.bse_index[:] - 1,
            "eigenvectors": eigenvectors,
            "fermi_energy": self._raw_exciton_eigenvector.fermi_energy,
            "first_valence_band": self._raw_exciton_eigenvector.first_valence_band[:]
            - 1,
            "first_conduction_band": self._raw_exciton_eigenvector.first_conduction_band[
                :
            ]
            - 1,
        }

    def to_database(self) -> dict:
        num_bands_valence = None
        num_bands_conduction = None
        num_kpoints = None
        if not check.is_none(self._raw_exciton_eigenvector.bse_index):
            bse_index = self._raw_exciton_eigenvector.bse_index[:]
            num_bands_conduction = np.shape(bse_index)[2]
            num_bands_valence = np.shape(bse_index)[3]
            num_kpoints = np.shape(bse_index)[1]
        return {
            "exciton_eigenvector": ExcitonEigenvector_DB(
                num_kpoints=num_kpoints,
                num_valence_bands=num_bands_valence,
                num_conduction_bands=num_bands_conduction,
            )
        }

    def _dispersion(self) -> DispersionHandler:
        return DispersionHandler.from_data(self._raw_exciton_eigenvector.dispersion)


@quantity("eigenvector", group="exciton")
class ExcitonEigenvector:
    """BSE can compute excitonic properties of materials.

    The Bethe-Salpeter Equation (BSE) accounts for electron-hole interactions
    involved in excitonic processes. For systems, where excitonic excitations
    matter the BSE method is an important tool. One can visualize excitonic
    contributions as so-called "fatbands" plots."""

    def __init__(self, source, quantity_name: str = "exciton_eigenvector"):
        self._source = source
        self._quantity_name = quantity_name

    @classmethod
    def from_data(
        cls, raw_exciton_eigenvector: raw.ExcitonEigenvector
    ) -> "ExcitonEigenvector":
        return cls(source=DataSource(raw_exciton_eigenvector))

    def _handler_factory(self, raw):
        return ExcitonEigenvectorHandler.from_data(raw)

    def __str__(self) -> str:
        return merge_strings(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            ExcitonEigenvectorHandler.__str__,
        )

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def read(self, selection: str | None = None) -> dict:
        """Read the data into a dictionary.

        Returns
        -------
        dict
            The dictionary contains the relevant k-point distances and labels as well as
            the electronic band eigenvalues. To produce fatband plots, use the array
            *bse_index* to access the relevant quantities of the BSE eigenvectors. Note
            that the dimensions of the bse_index array are **k** points, conduction
            bands, valence bands and that the conduction and valence band indices may
            be offset by first_valence_band and first_conduction_band, respectively.
        """
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            ExcitonEigenvectorHandler.to_dict,
        )

    def to_dict(self, selection: str | None = None) -> dict:
        """Convenient alias for :py:meth:`read`. Please read the documentation there."""
        return self.read(selection=selection)
