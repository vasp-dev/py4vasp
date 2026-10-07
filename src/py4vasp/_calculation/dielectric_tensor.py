# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from typing import Optional

import numpy as np

from py4vasp import exception, raw
from py4vasp._calculation.cell import CellHandler
from py4vasp._calculation.dispatch import (
    DataSource,
    _dispatch,
    merge_default,
    merge_strings,
    merge_to_database,
    quantity,
)
from py4vasp._raw.models import DielectricTensorModel
from py4vasp._util import check, convert, error, incar, select
from py4vasp._util.tensor import symmetry_reduce

_TO_DATABASE_SUPPRESSED_EXCEPTIONS = (
    exception.Py4VaspError,
    AttributeError,
    TypeError,
    ValueError,
)


class DielectricTensorHandler:
    """Handler for the dielectric tensor quantity. Works with exactly one raw.DielectricTensor object."""

    def __init__(self, raw_dielectric_tensor: raw.DielectricTensor):
        self._raw_dielectric_tensor = raw_dielectric_tensor

    @classmethod
    def from_data(
        cls, raw_dielectric_tensor: raw.DielectricTensor
    ) -> "DielectricTensorHandler":
        return cls(raw_dielectric_tensor)

    def to_dict(self) -> dict:
        """Read the dielectric tensor into a dictionary.

        Returns
        -------
        dict
            Contains the dielectric tensor and a string describing the method it
            was obtained.
        """
        return {
            "clamped_ion": self._raw_dielectric_tensor.electron[:],
            "relaxed_ion": self._read_relaxed_ion(),
            "independent_particle": self._read_independent_particle(),
            "method": convert.text_to_string(self._raw_dielectric_tensor.method),
        }

    def __str__(self) -> str:
        data = self.to_dict()
        return f"""
Macroscopic static dielectric tensor (dimensionless)
  {_description(data["method"])}
------------------------------------------------------
{_dielectric_tensor_string(data["clamped_ion"], "clamped-ion")}
{_dielectric_tensor_string(data["relaxed_ion"], "relaxed-ion")}
""".strip()

    def to_INCAR(self, selection=None) -> str:
        choice = _parse_incar_selection(selection)
        tensor = self.to_dict()[choice]
        if tensor is None:
            message = f"The {choice} dielectric tensor was not computed in this VASP calculation."
            raise exception.NoData(message)
        # VASP transposes the tensor after reading it, so the rows of the INCAR are the
        # rows of the Fortran array, which is the transpose of the numpy array
        return incar.tag_block("PHON_DIELECTRIC", tensor.T)

    def to_database(self) -> dict:
        encountered_errors = {}
        error_key = "dielectric_tensor:default"

        tensor_reduced = [None, None, None]
        isotropic_constant = [None, None, None]
        polarizability_2d = [None, None, None]

        total_tensor, ionic_tensor, electronic_tensor = None, None, None
        if not check.is_none(self._raw_dielectric_tensor.electron):
            electronic_tensor = self._raw_dielectric_tensor.electron[:]
        if not check.is_none(self._raw_dielectric_tensor.ion):
            ionic_tensor = self._raw_dielectric_tensor.ion[:]
        if not check.is_none(self._raw_dielectric_tensor.ion) and not check.is_none(
            self._raw_dielectric_tensor.electron
        ):
            total_tensor = (
                self._raw_dielectric_tensor.electron[:]
                + self._raw_dielectric_tensor.ion[:]
            )

        for idt, tensor in enumerate([total_tensor, ionic_tensor, electronic_tensor]):
            with error.suppress_and_record(
                encountered_errors,
                error_key,
                *_TO_DATABASE_SUPPRESSED_EXCEPTIONS,
                context=f"to_database.tensor[{idt}]",
            ):
                tensor_reduced[idt] = list(symmetry_reduce(tensor.T))
                (
                    isotropic_constant[idt],
                    polarizability_2d[idt],
                ) = self._calculate_dielectric_quantities(
                    tensor,
                    encountered_errors=encountered_errors,
                    error_key=error_key,
                )

        method = (
            convert.text_to_string(self._raw_dielectric_tensor.method)
            if not check.is_none(self._raw_dielectric_tensor.method)
            else None
        )

        return DielectricTensorModel(
            method=method,
            total_3d_tensor=tensor_reduced[0],
            total_3d_isotropic_dielectric_constant=isotropic_constant[0],
            total_2d_polarizability=polarizability_2d[0],
            ionic_3d_tensor=tensor_reduced[1],
            ionic_3d_isotropic_dielectric_constant=isotropic_constant[1],
            ionic_2d_polarizability=polarizability_2d[1],
            electronic_3d_tensor=tensor_reduced[2],
            electronic_3d_isotropic_dielectric_constant=isotropic_constant[2],
            electronic_2d_polarizability=polarizability_2d[2],
        )

    # --- Private helpers ---

    def _read_relaxed_ion(self):
        if check.is_none(self._raw_dielectric_tensor.ion):
            return None
        else:
            return (
                self._raw_dielectric_tensor.electron[:]
                + self._raw_dielectric_tensor.ion[:]
            )

    def _read_independent_particle(self):
        if check.is_none(self._raw_dielectric_tensor.independent_particle):
            return None
        else:
            return self._raw_dielectric_tensor.independent_particle[:]

    def _calculate_dielectric_quantities(
        self,
        tensor: np.ndarray,
        *,
        encountered_errors: Optional[dict[str, list[str]]] = None,
        error_key: Optional[str] = None,
    ) -> tuple:
        polarizability_2d = None
        with error.suppress_and_record(
            encountered_errors,
            error_key,
            *_TO_DATABASE_SUPPRESSED_EXCEPTIONS,
            context="calculate_dielectric_quantities",
        ):
            if not (check.is_none(self._raw_dielectric_tensor.cell)):
                final_cell = CellHandler.from_data(
                    self._raw_dielectric_tensor.cell, steps=-1
                )
                if final_cell:
                    polarizability_2d = _calculate_2d_polarizability(
                        tensor,
                        final_cell,
                        encountered_errors=encountered_errors,
                        error_key=error_key,
                    )

        isotropic_dielectric_constant = float(np.mean(np.diag(tensor)))
        return isotropic_dielectric_constant, polarizability_2d


@quantity("dielectric_tensor")
class DielectricTensor:
    """The dielectric tensor is the static limit of the :attr:`dielectric function<py4vasp.calculation.dielectric_function>`.

    The dielectric tensor represents how a material's response to an external electric
    field varies with direction. It is a symmetric 3x3 matrix, encapsulating the
    anisotropic nature of a material's dielectric properties. Each element of the
    tensor corresponds to the dielectric function along a specific crystallographic
    axis.
    """

    def __init__(self, source, quantity_name: str = "dielectric_tensor"):
        self._source = source
        self._quantity_name = quantity_name

    @classmethod
    def from_data(
        cls, raw_dielectric_tensor: raw.DielectricTensor
    ) -> "DielectricTensor":
        """Create a DielectricTensor dispatcher from raw data (convenience for testing)."""
        return cls(source=DataSource(raw_dielectric_tensor))

    def _handler_factory(self, raw_data):
        return DielectricTensorHandler.from_data(raw_data)

    def read(self) -> dict:
        """Read the dielectric tensor into a dictionary.

        Returns
        -------
        dict
            Contains the dielectric tensor and a string describing the method it
            was obtained.
        """
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            DielectricTensorHandler.to_dict,
        )

    def to_dict(self) -> dict:
        """Convenient alias for :py:meth:`read`. Please read the documentation there."""
        return self.read()

    def to_INCAR(self, selection: str | None = None) -> str:
        """Format the dielectric tensor as the PHON_DIELECTRIC tag of an INCAR file.

        A phonon calculation of a polar material needs the dielectric tensor to
        describe the long-range dipole-dipole interaction, e.g., the LO-TO splitting.
        Copy the returned text into the INCAR file of that calculation together with
        :py:meth:`~py4vasp._calculation.born_effective_charge.BornEffectiveCharge.to_INCAR`.
        The text ends with a newline, so you can concatenate it with other INCAR tags.

        The tensor is written in the orientation VASP reads it, one row per line with
        a backslash continuing the tag onto the next line. You do not need to transpose
        anything yourself.

        Parameters
        ----------
        selection : str | None
            Choose which dielectric tensor is written. The default is the
            ``clamped_ion`` tensor, the electronic contribution only, which is the
            high-frequency dielectric constant ε∞ that the dipole-dipole interaction
            of phonons requires. Alternatively, select ``relaxed_ion`` for the static
            tensor including the ionic contribution or ``independent_particle`` for the
            electronic tensor without local field effects. Exactly one tensor can be
            selected; you may nest it inside the source, e.g. ``default(clamped_ion)``.

        Returns
        -------
        str
            The PHON_DIELECTRIC tag with the nine components of the tensor.

        Examples
        --------
        First, we create some example data so that you can follow along. Please define a
        variable `path` with the path to a directory that does not exist yet.
        Alternatively, use your own data if you have run VASP.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        Without a selection you obtain the clamped-ion tensor ε∞

        >>> print(calculation.dielectric_tensor.to_INCAR())
        PHON_DIELECTRIC =   4.620000   0.000000   0.000000 \\
                            0.000000   4.620000   0.000000 \\
                            0.000000   0.000000   4.350000

        Select the relaxed-ion tensor to include the ionic contribution

        >>> print(calculation.dielectric_tensor.to_INCAR("relaxed_ion"))
        PHON_DIELECTRIC =  37.120000   0.000000   0.000000 \\
                            0.000000  37.120000   0.000000 \\
                            0.000000   0.000000  18.150000
        """
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            DielectricTensorHandler.to_INCAR,
        )

    def print(self, selection: str | None = None) -> None:
        """Print a string representation of this quantity.

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

    def __str__(self, selection=None):
        return merge_strings(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            DielectricTensorHandler.__str__,
        )

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def _to_database(self) -> dict:
        """Return {quantity[_selection]: handler_result} for database storage."""
        return merge_to_database(
            self._source,
            self._quantity_name,
            DielectricTensorHandler.from_data,
            DielectricTensorHandler.to_database,
        )


def _dielectric_tensor_string(tensor, label):
    if tensor is None:
        return ""
    row_to_string = lambda row: 6 * " " + " ".join(f"{x:12.6f}" for x in row)
    rows = (row_to_string(row) for row in tensor)
    return f"{label:^55}".rstrip() + "\n" + "\n".join(rows)


_INCAR_TENSORS = ("clamped_ion", "relaxed_ion", "independent_particle")


def _parse_incar_selection(selection):
    tree = select.Tree.from_selection(selection)
    parts = [part for choice in tree.selections() for part in choice]
    if unknown := set(parts).difference(_INCAR_TENSORS):
        message = f"The selection {unknown} is not one of the dielectric tensors {_INCAR_TENSORS}."
        raise exception.IncorrectUsage(message)
    if len(parts) > 1:
        message = f"PHON_DIELECTRIC holds a single tensor, but you selected {parts}."
        raise exception.IncorrectUsage(message)
    return parts[0] if parts else "clamped_ion"


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
    cell_: CellHandler,
    *,
    encountered_errors: Optional[dict[str, list[str]]] = None,
    error_key: Optional[str] = None,
) -> float:
    """
    Compute 2D polarizability (alpha_2D) for a slab system with unknown vacuum direction.
    """
    with error.suppress_and_record(
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
