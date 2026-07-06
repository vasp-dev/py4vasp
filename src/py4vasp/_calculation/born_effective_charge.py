# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np

from py4vasp import raw
from py4vasp._calculation.dispatch import (
    DataSource,
    _dispatch,
    merge_default,
    merge_strings,
    merge_to_database,
    quantity,
)
from py4vasp._calculation.structure import StructureHandler
from py4vasp._raw.data_db import BornEffectiveCharge_DB
from py4vasp._util import check


class BornEffectiveChargeHandler:
    """The Born effective charge tensors couple electric field and atomic displacement."""

    def __init__(self, raw_born_effective_charge: raw.BornEffectiveCharge):
        self._raw_born_effective_charge = raw_born_effective_charge

    @classmethod
    def from_data(
        cls, raw_born_effective_charge: raw.BornEffectiveCharge
    ) -> "BornEffectiveChargeHandler":
        return cls(raw_born_effective_charge)

    def __str__(self) -> str:
        data = self.to_dict()
        result = """
BORN EFFECTIVE CHARGES (including local field effects) (in |e|, cumulative output)
---------------------------------------------------------------------------------
        """.strip()
        generator = zip(data["structure"]["elements"], data["charge_tensors"])
        vec_to_string = lambda vec: " ".join(f"{x:11.5f}" for x in vec)
        for ion, (element, charge_tensor) in enumerate(generator):
            result += f"""
ion {ion + 1:4d}   {element}
    1 {vec_to_string(charge_tensor[0])}
    2 {vec_to_string(charge_tensor[1])}
    3 {vec_to_string(charge_tensor[2])}"""
        return result

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def read(self) -> dict:
        """Read structure information and Born effective charges into a dictionary."""
        return self.to_dict()

    def to_dict(self) -> dict:
        """Read structure information and Born effective charges into a dictionary.

        The structural information is added to inform about which atoms are included
        in the array. The Born effective charges array contains the mixed second
        derivative with respect to an electric field and an atomic displacement for
        all atoms and possible directions.

        Returns
        -------
        dict
            Contains structural information as well as the Born effective charges.
        """
        structure = StructureHandler.from_data(
            self._raw_born_effective_charge.structure
        )
        return {
            "structure": structure.to_dict(),
            "charge_tensors": self._raw_born_effective_charge.charge_tensors[:],
        }

    def to_database(self) -> dict:
        """Return Born effective charge data ready for database storage."""
        eigenvalue_max = None
        eigenvalue_max_index = None
        eigenvalue_min = None
        eigenvalue_min_index = None

        if not check.is_none(self._raw_born_effective_charge.charge_tensors):
            charge_tensors = self._raw_born_effective_charge.charge_tensors[:]
            traces = (
                charge_tensors[:, 0, 0]
                + charge_tensors[:, 1, 1]
                + charge_tensors[:, 2, 2]
            )
            eigenvalue_max = float(np.max(traces))
            eigenvalue_min = float(np.min(traces))
            eigenvalue_max_index = int(np.argmax(traces))
            eigenvalue_min_index = int(np.argmin(traces))

        return BornEffectiveCharge_DB(
            eigenvalue_min=eigenvalue_min,
            eigenvalue_min_index=eigenvalue_min_index,
            eigenvalue_max=eigenvalue_max,
            eigenvalue_max_index=eigenvalue_max_index,
        )


@quantity("born_effective_charge")
class BornEffectiveCharge:
    """The Born effective charge tensors couple electric field and atomic displacement.

    You can use this class to extract the Born effective charges of a linear
    response calculation. The Born effective charges describes the effective charge of
    an ion in a crystal lattice when subjected to an external electric field.
    These charges account for the displacement of the ion positions in response to the
    field, reflecting the distortion of the crystal structure. Born effective charges
    help understanding the material's response to external stimuli, such as
    piezoelectric and ferroelectric behavior.
    """

    def __init__(self, source, quantity_name: str = "born_effective_charge"):
        self._source = source
        self._quantity_name = quantity_name

    @classmethod
    def from_data(
        cls, raw_born_effective_charge: raw.BornEffectiveCharge
    ) -> "BornEffectiveCharge":
        """Create a BornEffectiveCharge dispatcher from raw data."""
        return cls(source=DataSource(raw_born_effective_charge))

    def __str__(self, selection=None) -> str:
        return merge_strings(
            self._source,
            self._quantity_name,
            selection,
            BornEffectiveChargeHandler.from_data,
            BornEffectiveChargeHandler.__str__,
        )

    def read(self) -> dict:
        """Read structure information and Born effective charges into a dictionary.

        The structural information is added to inform about which atoms are included
        in the array. The Born effective charges array contains the mixed second
        derivative with respect to an electric field and an atomic displacement for
        all atoms and possible directions.

        Returns
        -------
        dict
            Contains structural information as well as the Born effective charges.
        """
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            BornEffectiveChargeHandler.from_data,
            BornEffectiveChargeHandler.read,
        )

    def to_dict(self) -> dict:
        """Convenient alias for :py:meth:`read`."""
        return self.read()

    def _to_database(self) -> dict:
        """Return {quantity[_selection]: handler_result} for database storage."""
        return merge_to_database(
            self._source,
            self._quantity_name,
            BornEffectiveChargeHandler.from_data,
            BornEffectiveChargeHandler.to_database,
        )
