# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp import calculation
from py4vasp._calculation import base
from py4vasp._util import convert


class ExcitonEigenvector(base.Refinery):
    """BSE can compute excitonic properties of materials.

    The Bethe-Salpeter Equation (BSE) accounts for electron-hole interactions
    involved in excitonic processes. For systems, where excitonic excitations
    matter the BSE method is an important tool. One can visualize excitonic
    contributions as so-called "fatbands" plots. Here, the width of the band is
    adjusted such that the band is wider the larger the contribution is. This
    approach helps understanding of the electronic transitions and excitonic
    behavior in materials.
    """

    @base.data_access
    def __str__(self):
        shape = self._raw_data.bse_index.shape
        return f"""BSE eigenvector data:
    {shape[1]} k-points
    {shape[3]} valence bands
    {shape[2]} conduction bands"""

    @base.data_access
    def to_dict(self):
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
        eigenvectors = convert.to_complex(self._raw_data.eigenvectors[:])
        dispersion = self._dispersion.read()
        return {
            "kpoint_distances": dispersion["kpoint_distances"],
            "kpoint_labels": dispersion["kpoint_labels"],
            "bands": dispersion["eigenvalues"] - self._raw_data.fermi_energy,
            "bse_index": self._raw_data.bse_index[:] - 1,
            "eigenvectors": eigenvectors,
            "fermi_energy": self._raw_data.fermi_energy,
            "first_valence_band": self._raw_data.first_valence_band[:] - 1,
            "first_conduction_band": self._raw_data.first_conduction_band[:] - 1,
        }

    @property
    def _dispersion(self):
        return calculation._dispersion.from_data(self._raw_data.dispersion)
