# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np

from py4vasp import raw
from py4vasp._calculation.dispatch import (
    DataSource,
    merge_default,
    merge_strings,
    merge_to_database,
    quantity,
)
from py4vasp._calculation.structure import StructureHandler
from py4vasp._raw.models import BornEffectiveChargeModel
from py4vasp._util import check, incar


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
            # like OUTCAR, each row is the field direction; vaspout.h5 stores it last
            field_x, field_y, field_z = charge_tensor.T
            result += f"""
ion {ion + 1:4d}   {element}
    1 {vec_to_string(field_x)}
    2 {vec_to_string(field_y)}
    3 {vec_to_string(field_z)}"""
        return result

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def read(self) -> dict:
        """Read structure information and Born effective charges into a dictionary."""
        return self.to_dict()

    def to_INCAR(self) -> str:
        charge_tensors = self._raw_born_effective_charge.charge_tensors[:]
        # VASP transposes every 3x3 block after reading it and expects the field
        # direction first; vaspout.h5 stores the field direction as the last axis
        rows = np.swapaxes(charge_tensors, 1, 2).reshape(len(charge_tensors), 9)
        return incar.tag_block("PHON_BORN_CHARGES", rows)

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

        return BornEffectiveChargeModel(
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

    def print(self, selection: str | None = None) -> None:
        """Print a string representation of this quantity.

        The output imitates the block of Born effective charges in the OUTCAR file, so
        you can compare the two directly. For every ion, the row index is the direction
        of the electric field and the column index the direction of the atomic
        displacement. This is the orientation of the PHON_BORN_CHARGES tag written by
        :py:meth:`to_INCAR` and the transpose of the array returned by :py:meth:`read`.

        Parameters
        ----------
        selection : str | None
            Select which source of the quantity is printed. If you select multiple
            sources, py4vasp prints one block per source.
        """
        print(self.__str__(selection))

    def selections(self) -> dict:
        """Returns possible alternatives for this particular quantity VASP can produce.

        The returned dictionary contains a single item with the name of the quantity
        mapping to all possible selections. Each of these selections may be passed to
        the other methods of this quantity to choose which output of VASP is used.

        Returns
        -------
        dict
            The key indicates this quantity and the value lists the possible choices
            for the selection argument of its other methods.
        """
        from py4vasp._raw import definition as raw_module

        return {self._quantity_name: list(raw_module.selections(self._quantity_name))}

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

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
        all atoms and possible directions. ``charge_tensors[ion, i, j]`` is the
        derivative with respect to the displacement in direction ``i`` and the electric
        field in direction ``j``, which is the transpose of the blocks VASP prints in
        the OUTCAR file. :py:meth:`print` and :py:meth:`to_INCAR` take care of that
        orientation for you.

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

    def to_INCAR(self, selection: str | None = None) -> str:
        """Format the Born effective charges as the PHON_BORN_CHARGES tag of an INCAR file.

        A phonon calculation of a polar material needs the Born effective charges to
        describe the long-range dipole-dipole interaction, e.g., the LO-TO splitting.
        Copy the returned text into the INCAR file of that calculation together with
        :py:meth:`~py4vasp._calculation.dielectric_tensor.DielectricTensor.to_INCAR`.
        The text ends with a newline, so you can concatenate it with other INCAR tags.

        Each line holds the 3x3 tensor of one ion, in the order of the ions in the
        POSCAR file of the linear-response calculation. Within a line, the nine numbers
        are the rows of the tensor with the electric field as row index and the atomic
        displacement as column index, which is the orientation VASP reads. You do not
        need to transpose anything yourself. Note that this is the transpose of the
        array returned by :py:meth:`read`, which stores the displacement first.

        VASP expects the charges of the atoms in the primitive cell of the phonon
        calculation and stops if the number of ions does not match. So compute the
        Born effective charges for the primitive cell, and build the supercell of the
        phonon calculation from that same POSCAR so that the order of the ions agrees.

        Parameters
        ----------
        selection : str | None
            Select the source of the Born effective charges, if VASP produced more than
            one. Most calculations only have the default source.

        Returns
        -------
        str
            The PHON_BORN_CHARGES tag with nine components for each ion.

        Examples
        --------
        First, we create some example data so that you can follow along. Please define a
        variable `path` with the path to a directory that does not exist yet.
        Alternatively, use your own data if you have run VASP.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        Each of the seven ions of Sr2TiO4 gets a line of the tag

        >>> print(calculation.born_effective_charge.to_INCAR())
        PHON_BORN_CHARGES =   2.470000   0.000000   0.000000   0.000000   2.470000   0.000000   0.000000   0.000000   2.690000 \\
                              2.470000   0.000000   0.000000   0.000000   2.470000   0.000000   0.000000   0.000000   2.690000 \\
                              6.940000   0.000000   0.000000   0.000000   6.940000   0.000000   0.000000   0.000000   5.650000 \\
                             -2.210000   0.000000   0.000000   0.000000  -2.210000   0.000000   0.000000   0.000000  -3.815000 \\
                             -2.210000   0.000000   0.000000   0.000000  -2.210000   0.000000   0.000000   0.000000  -3.815000 \\
                             -1.980000   0.000000   0.000000   0.000000  -5.480000   0.000000   0.000000   0.000000  -1.700000 \\
                             -5.480000   0.000000   0.000000   0.000000  -1.980000   0.000000   0.000000   0.000000  -1.700000
        """
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            BornEffectiveChargeHandler.from_data,
            BornEffectiveChargeHandler.to_INCAR,
        )

    def _to_database(self) -> dict:
        """Return {quantity[_selection]: handler_result} for database storage."""
        return merge_to_database(
            self._source,
            self._quantity_name,
            BornEffectiveChargeHandler.from_data,
            BornEffectiveChargeHandler.to_database,
        )
