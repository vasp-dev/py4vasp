# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import copy
import pathlib

import numpy as np

from py4vasp import _config, exception, raw
from py4vasp._calculation import slice_
from py4vasp._calculation.dispatch import (
    DataSource,
    FileSource,
    merge_default,
    merge_strings,
    quantity,
    slice_steps,
)
from py4vasp._calculation.structure import StructureHandler
from py4vasp._raw.data_db import LocalMoment_DB
from py4vasp._third_party import view
from py4vasp._util import check, documentation, select

_index_note = """\
Notes
-----
The index order is different compared to the raw data when noncollinear calculations
are used. This routine returns the magnetic moments as (steps, orbitals, atoms,
directions)."""

_moment_selection = """\
selection : str
    If VASP was run with LORBMOM = T, the orbital moments are computed and the routine
    will default to the total moments. You can specify "spin" or "orbital" to select
    the individual contributions instead.
"""

_ORBITAL_PROJECTION = "orbital_projection"


class LocalMomentHandler:
    """Handler for local moment data."""

    length_moments = 1.5
    "Length in \u00c5 how a magnetic moment is displayed relative to the largest moment."

    def __init__(self, raw_local_moment: raw.LocalMoment, steps=None):
        self._raw_local_moment = raw_local_moment
        self._steps = steps

    @classmethod
    def from_data(cls, raw_local_moment: raw.LocalMoment, steps=None) -> "LocalMomentHandler":
        return cls(raw_local_moment, steps=steps)

    def __str__(self) -> str:
        if self._is_nonpolarized:
            return "not spin polarized"
        magmom = "MAGMOM = "
        moments_last_step = self.magnetic("spin")
        moments_to_string = lambda vec: " ".join(f"{moment:.2f}" for moment in vec)
        if moments_last_step.ndim == 1:
            return magmom + moments_to_string(moments_last_step)
        else:
            separator = " \\\n         "
            generator = (moments_to_string(vec) for vec in moments_last_step)
            return magmom + separator.join(generator)

    def read(self) -> dict:
        return self.to_dict()

    def to_dict(self) -> dict:
        return {
            _ORBITAL_PROJECTION: self.selections()[_ORBITAL_PROJECTION],
            "charge": self.projected_charge(),
            **self._add_total_magnetic_moment(),
            **self._add_spin_and_orbital_moments(),
        }

    def to_database(self) -> dict:
        spin_moments_orbitals = None
        if not check.is_none(self._raw_local_moment.spin_moments):
            if not self._is_nonpolarized:
                spin_moments_orbitals = self._raw_local_moment.spin_moments[-1, -1]
        spin_moment_total_min = None
        spin_moment_total_max = None
        if spin_moments_orbitals is not None:
            spin_moments_total = np.sum(spin_moments_orbitals, axis=-1)
            spin_moment_total_min = float(np.min(spin_moments_total))
            spin_moment_total_max = float(np.max(spin_moments_total))
        return {
            "local_moment": LocalMoment_DB(
                has_orbital_moments=self._has_orbital_moments,
                final_spin_moment_total_min=spin_moment_total_min,
                final_spin_moment_total_max=spin_moment_total_max,
            )
        }

    def to_view(self, selection="total", supercell=None):
        structure = StructureHandler.from_data(self._raw_local_moment.structure, steps=self._steps)
        viewer = structure.to_view(supercell)
        if not self._is_nonpolarized:
            viewer.ion_arrows = list(self._prepare_magnetic_moments_for_plotting(selection))
        return viewer

    def projected_charge(self):
        self._raise_error_if_steps_out_of_bounds()
        return self._raw_local_moment.spin_moments[self._steps_or_last, 0]

    def projected_magnetic(self, selection="total"):
        self._raise_error_if_steps_out_of_bounds()
        self._raise_error_if_no_magnetic_moments()
        tree = select.Tree.from_selection(selection)
        moments = [self._magnetic_moments(sel) for sel in tree.selections()]
        return np.squeeze(moments)

    def charge(self):
        return _sum_over_orbitals(self.projected_charge())

    def magnetic(self, selection="total"):
        return _sum_over_orbitals(
            self.projected_magnetic(selection), is_vector=self._is_noncollinear
        )

    def selections(self):
        result = {}
        if self._raw_local_moment.spin_moments.shape[-1] == 4:
            result[_ORBITAL_PROJECTION] = ["s", "p", "d", "f"]
        else:
            result[_ORBITAL_PROJECTION] = ["s", "p", "d"]
        if self._is_nonpolarized:
            result["component"] = ["charge"]
        elif self._has_orbital_moments:
            result["component"] = ["charge", "total", "spin", "orbital"]
        else:
            result["component"] = ["charge", "total", "spin"]
        return result

    def number_steps(self) -> int:
        n = len(np.array(self._raw_local_moment.spin_moments))
        return len(range(n)[self._to_slice])

    @property
    def _is_nonpolarized(self):
        return self._raw_local_moment.spin_moments.shape[1] == 1

    @property
    def _is_collinear(self):
        return self._raw_local_moment.spin_moments.shape[1] == 2

    @property
    def _is_noncollinear(self):
        return self._raw_local_moment.spin_moments.shape[1] == 4

    @property
    def _has_orbital_moments(self):
        return not check.is_none(self._raw_local_moment.orbital_moments)

    @property
    def _steps_or_last(self):
        if self._steps is None or self._steps == -1:
            return -1
        return self._steps

    @property
    def _to_slice(self):
        if self._steps is None or self._steps == -1:
            return slice(-1, None)
        if isinstance(self._steps, slice):
            return self._steps
        return slice(self._steps, self._steps + 1)

    def _magnetic_moments(self, selection):
        self._raise_error_if_selection_not_available(selection)
        if self._is_collinear:
            return self._spin_moments()
        else:
            return self._noncollinear_moments(selection[0])

    def _noncollinear_moments(self, selection):
        spin_moments = self._spin_moments()
        orbital_moments = self._orbital_moments(spin_moments)
        if selection == "orbital":
            moments = orbital_moments
        elif selection == "spin":
            moments = spin_moments
        else:
            moments = spin_moments + orbital_moments
        direction_axis = 1 if moments.ndim == 4 else 0
        return np.moveaxis(moments, direction_axis, -1)

    def _spin_moments(self):
        return self._raw_local_moment.spin_moments[self._steps_or_last, 1:]

    def _orbital_moments(self, spin_moments):
        if not self._has_orbital_moments:
            return np.zeros_like(spin_moments)
        zero_s_moments = np.zeros((*spin_moments.shape[:-1], 1))
        orbital_moments = self._raw_local_moment.orbital_moments[self._steps_or_last]
        return np.concatenate((zero_s_moments, orbital_moments), axis=-1)

    def _add_total_magnetic_moment(self):
        if self._is_nonpolarized:
            return {}
        return {"total": self.projected_magnetic()}

    def _add_spin_and_orbital_moments(self):
        if not self._has_orbital_moments:
            return {}
        spin_moments = self._spin_moments()
        orbital_moments = self._orbital_moments(spin_moments)
        direction_axis = 1 if spin_moments.ndim == 4 else 0
        return {
            "spin": np.moveaxis(spin_moments, direction_axis, -1),
            "orbital": np.moveaxis(orbital_moments, direction_axis, -1),
        }

    def _prepare_magnetic_moments_for_plotting(self, selection):
        tree = select.Tree.from_selection(selection)
        for (sel, *_) in tree.selections():
            moments = self.magnetic(sel)
            moments = self._make_sure_moments_have_timestep_dimension(moments)
            moments = _convert_moment_to_3d_vector(moments)
            max_length_moments = _max_length_moments(moments)
            if max_length_moments > 1e-15:
                rescale_moments = LocalMomentHandler.length_moments / max_length_moments
                yield view.IonArrow(
                    quantity=rescale_moments * moments,
                    label=f"{sel} moments",
                    color=_color(sel),
                    radius=0.2,
                )

    def _make_sure_moments_have_timestep_dimension(self, moments):
        is_slice = isinstance(self._steps, slice)
        if not is_slice and moments is not None:
            moments = moments[np.newaxis]
        return moments

    def _raise_error_if_steps_out_of_bounds(self):
        try:
            np.zeros(self._raw_local_moment.spin_moments.shape[0])[self._steps_or_last]
        except IndexError as error:
            raise exception.IncorrectUsage(
                f"Error reading the magnetic moments. Please check if the steps "
                f"`{self._steps}` are properly formatted and within the boundaries."
            ) from error

    def _raise_error_if_no_magnetic_moments(self):
        if self._is_nonpolarized:
            raise exception.NoData(
                "There are no magnetic moments in the data. Please make sure that you "
                "either set ISPIN = 2 or LNONCOLLINEAR = T or LSORBIT = T."
            )

    def _raise_error_if_selection_not_available(self, selection):
        if len(selection) != 1:
            raise exception.IncorrectUsage()
        selection = selection[0]
        if selection not in ("spin", "orbital", "total"):
            raise exception.IncorrectUsage(
                f"The selection {selection} is incorrect. Please check if it is spelled "
                "correctly. Possible choices are total, spin, or orbital."
            )
        if selection != "orbital" or self._has_orbital_moments:
            return
        raise exception.NoData(
            "There are no orbital moments in the VASP output. Please make sure that you "
            "run the calculation with LORBMOM = T and LSORBIT = T."
        )


@quantity("local_moment")
class LocalMoment(view.Mixin):
    """The local moments describe the charge and magnetization near an atom."""

    length_moments = LocalMomentHandler.length_moments

    def __init__(self, source, quantity_name: str = "local_moment", steps=None):
        self._source = source
        self._quantity_name = quantity_name
        self._steps = steps

    @classmethod
    def from_data(cls, raw_local_moment: raw.LocalMoment) -> "LocalMoment":
        return cls(source=DataSource(raw_local_moment))

    @classmethod
    def from_path(cls, path=".") -> "LocalMoment":
        return cls(source=FileSource(path))

    @classmethod
    def from_file(cls, file_name) -> "LocalMoment":
        resolved = pathlib.Path(file_name).expanduser().resolve()
        return cls(source=FileSource(resolved.parent, file=file_name))

    @property
    def _path(self):
        return self._source.path

    def __getitem__(self, steps) -> "LocalMoment":
        new = copy.copy(self)
        new._steps = steps
        return new

    def _handler_factory(self, raw):
        return LocalMomentHandler.from_data(raw, steps=self._steps)

    def __str__(self, selection=None) -> str:
        return merge_strings(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            LocalMomentHandler.__str__,
        )

    def _repr_pretty_(self, p, cycle):
        p.text(str(self) if not cycle else "...")

    def read(self, selection=None) -> dict:
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            LocalMomentHandler.read,
        )

    def to_dict(self, selection=None) -> dict:
        return self.read(selection=selection)

    def to_view(self, selection="total", supercell=None):
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            LocalMomentHandler.to_view,
            selection,
            supercell,
        )

    def projected_charge(self, selection=None):
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            LocalMomentHandler.projected_charge,
        )

    def projected_magnetic(self, selection="total"):
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            LocalMomentHandler.projected_magnetic,
            selection,
        )

    def charge(self, selection=None):
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            LocalMomentHandler.charge,
        )

    def magnetic(self, selection="total"):
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            LocalMomentHandler.magnetic,
            selection,
        )

    def selections(self, selection=None) -> dict:
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            LocalMomentHandler.selections,
        )

    def number_steps(self, selection=None) -> int:
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            LocalMomentHandler.number_steps,
        )


def _sum_over_orbitals(quantity, is_vector=False):
    if quantity is None:
        return None
    if is_vector:
        return np.sum(quantity, axis=-2)
    return np.sum(quantity, axis=-1)


def _convert_moment_to_3d_vector(moments):
    if moments is not None and moments.ndim == 2:
        moments = moments.reshape((*moments.shape, 1))
        no_new_moments = (0, 0)
        add_zero_for_xy_axis = (2, 0)
        padding = (no_new_moments, no_new_moments, add_zero_for_xy_axis)
        moments = np.pad(moments, padding)
    return moments


def _max_length_moments(moments):
    if moments is not None:
        return np.max(np.linalg.norm(moments, axis=2))
    else:
        return 0.0


def _color(selection):
    if selection == "total":
        return _config.VASP_COLORS["blue"]
    if selection == "spin":
        return _config.VASP_COLORS["purple"]
    if selection == "orbital":
        return _config.VASP_COLORS["red"]
    raise exception.IncorrectUsage(f"Unknown component {selection} selected.")
