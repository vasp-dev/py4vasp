# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import exception
from py4vasp._calculation import base


class Polarization(base.Refinery):
    """The static polarization describes the electric dipole moment per unit volume.

    Static polarization arises in a material in response to a constant external electric
    field. In VASP, we compute the linear response of the system when applying a
    :tag:`EFIELD`. Static polarization is a key characteristic of ferroelectric
    materials that exhibit a spontaneous electric polarization that persists even in
    the absence of an external electric field.

    Note that the polarization is only well defined relative to a reference
    system. The absolute value can change by a polarization quantum if some
    charge or ion leaves one side of the unit cell and reenters at the opposite
    side. Therefore you always need to compare changes of polarization.
    """

    @base.data_access
    def __str__(self):
        vec_to_string = lambda vec: " ".join(f"{x:11.5f}" for x in vec)
        return f"""
Polarization (|e|Å)
-------------------------------------------------------------
ionic dipole moment:      {vec_to_string(self._raw_data.ion[:])}
electronic dipole moment: {vec_to_string(self._raw_data.electron[:])}
""".strip()

    @base.data_access
    def to_dict(self):
        """Read electronic and ionic polarization into a dictionary

        Returns
        -------
        dict
            Contains the electronic and ionic dipole moments.
        """
        return {
            "electron_dipole": self._raw_data.electron[:],
            "ion_dipole": self._raw_data.ion[:],
        }

    @base.data_access
    def _to_database(self, *args, **kwargs):
        ionic_norm = None
        electronic_norm = None
        total_norm = None

        electron_dipole = None
        ion_dipole = None
        total_dipole = None

        try:
            electron_dipole = list(self._raw_data.electron[:])
            ion_dipole = list(self._raw_data.ion[:])
            total_dipole = list(self._raw_data.electron[:] + self._raw_data.ion[:])

            ionic_norm = np.linalg.norm(self._raw_data.ion[:])
            electronic_norm = np.linalg.norm(self._raw_data.electron[:])
            total_norm = np.linalg.norm(
                self._raw_data.electron[:] + self._raw_data.ion[:]
            )
        except exception.NoData:
            pass

        return {
            "polarization": {
                "polarization_norm_ionic": ionic_norm,
                "polarization_norm_electronic": electronic_norm,
                "polarization_norm_total": total_norm,
                "dipole_moment_ionic": ion_dipole,
                "dipole_moment_electronic": electron_dipole,
                "dipole_moment_total": total_dipole,
            }
        }
