# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
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
