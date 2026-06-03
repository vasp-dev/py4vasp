# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from contextlib import suppress

import numpy as np

from py4vasp import exception, raw
from py4vasp._calculation.dispatch import (
    _dispatch,
    DataSource,
    merge_to_database,
    merge_default,
    merge_strings,
    quantity,
)
from py4vasp._raw.data_db import Polarization_DB


class PolarizationHandler:
    """Handler for the polarization quantity. Works with exactly one raw.Polarization object."""

    def __init__(self, raw_polarization: raw.Polarization):
        self._raw_polarization = raw_polarization

    @classmethod
    def from_data(cls, raw_polarization: raw.Polarization) -> "PolarizationHandler":
        return cls(raw_polarization)

    def to_dict(self) -> dict:
        """Read electronic and ionic polarization into a dictionary.

        Returns
        -------
        dict
            Contains the electronic and ionic dipole moments.
        """
        return {
            "electron_dipole": self._raw_polarization.electron[:],
            "ion_dipole": self._raw_polarization.ion[:],
        }

    def __str__(self):
        vec_to_string = lambda vec: " ".join(f"{x:11.5f}" for x in vec)
        return f"""Polarization (|e|Å)
-------------------------------------------------------------
ionic dipole moment:      {vec_to_string(self._raw_polarization.ion[:])}
electronic dipole moment: {vec_to_string(self._raw_polarization.electron[:])}""".strip()

    def to_database(self) -> dict:
        ionic_norm = None
        electronic_norm = None
        total_norm = None

        electron_dipole = None
        ion_dipole = None
        total_dipole = None

        with suppress(exception.NoData):
            electron_dipole = list(self._raw_polarization.electron[:])
            ion_dipole = list(self._raw_polarization.ion[:])
            total_dipole = list(
                self._raw_polarization.electron[:] + self._raw_polarization.ion[:]
            )

            ionic_norm = np.linalg.norm(self._raw_polarization.ion[:])
            electronic_norm = np.linalg.norm(self._raw_polarization.electron[:])
            total_norm = np.linalg.norm(
                self._raw_polarization.electron[:] + self._raw_polarization.ion[:]
            )

        return Polarization_DB(
            total_dipole_norm=total_norm,
            total_dipole_moment=total_dipole,
            ionic_dipole_norm=ionic_norm,
            ionic_dipole_moment=ion_dipole,
            electronic_dipole_norm=electronic_norm,
            electronic_dipole_moment=electron_dipole,
        )


@quantity("polarization")
class Polarization:
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

    def __init__(self, source, quantity_name: str = "polarization"):
        self._source = source
        self._quantity_name = quantity_name

    @classmethod
    def from_data(cls, raw_polarization: raw.Polarization) -> "Polarization":
        """Create a Polarization dispatcher from raw data (convenience for testing)."""
        return cls(source=DataSource(raw_polarization))

    def read(self) -> dict:
        """Read electronic and ionic polarization into a dictionary.

        Returns
        -------
        dict
            Contains the electronic and ionic dipole moments.
        """
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            PolarizationHandler.from_data,
            PolarizationHandler.to_dict,
        )

    def to_dict(self, selection: str | None = None) -> dict:
        """Convenient alias for :py:meth:`read`."""
        return self.read()

    def __str__(self, selection=None):
        return merge_strings(
            self._source,
            self._quantity_name,
            selection,
            PolarizationHandler.from_data,
            PolarizationHandler.__str__,
        )

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def _to_database(self, selection=None) -> dict:
        """Return {quantity[_selection]: handler_result} for database storage."""
        return merge_to_database(
            self._source,
            self._quantity_name,
            selection,
            PolarizationHandler.from_data,
            PolarizationHandler.to_database,
        )
