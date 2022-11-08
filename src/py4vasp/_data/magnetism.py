# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import exception
from py4vasp._data import base, slice_, structure
from py4vasp._util import documentation, reader

_index_note = """Notes
-----
The index order is different compared to the raw data when noncollinear calculations
are used. This routine returns the magnetic moments as (steps, atoms, orbitals,
directions)."""


_magnetism_docs = f"""
The magnetic moments and localized charges for selected ionic steps.

This class gives access to the magnetic moments and charges projected on the
different orbitals on every atom.

{slice_.examples("magnetism")}
""".strip()


@documentation.add(_magnetism_docs)
class Magnetism(slice_.Mixin, base.Refinery, structure.Mixin):
    _missing_data_message = "Atom resolved magnetic information not present, please verify LORBIT tag is set."

    length_moments = 1.5
    "Length in Å how a magnetic moment is displayed relative to the largest moment."

    @base.data_access
    def __str__(self):
        magmom = "MAGMOM = "
        moments_last_step = self.total_moments()
        moments_to_string = lambda vec: " ".join(f"{moment:.2f}" for moment in vec)
        if moments_last_step is None:
            return "not spin polarized"
        elif moments_last_step.ndim == 1:
            return magmom + moments_to_string(moments_last_step)
        else:
            separator = " \\\n         "
            generator = (moments_to_string(vec) for vec in moments_last_step)
            return magmom + separator.join(generator)

    @base.data_access
    @documentation.add(
        f"""Read the charges and magnetization data into a dictionary.

Returns
-------
dict
    Contains the charges and magnetic moments generated by VASP projected
    on atoms and orbitals.

{_index_note}

{slice_.examples("magnetism", "read")}"""
    )
    def to_dict(self):
        return {
            "charges": self.charges(),
            "moments": self.moments(),
        }

    @base.data_access
    @documentation.add(
        f"""Visualize the magnetic moments as arrows inside the structure.

Returns
-------
Viewer3d
    Contains the atoms and the unit cell as well as an arrow indicating the
    strength of the magnetic moment. If noncollinear magnetism is used
    the moment points in the actual direction; for collinear magnetism
    the moments are aligned along the z axis by convention.

{slice_.examples("magnetism", "plot")}"""
    )
    def plot(self):
        if self._is_slice:
            message = (
                "Visualizing magnetic moments for more than one step is not implemented"
            )
            raise exception.NotImplemented(message)
        viewer = self._structure[self._steps].plot()
        moments = self._prepare_magnetic_moments_for_plotting()
        if moments is not None:
            viewer.show_arrows_at_atoms(moments)
        return viewer

    @base.data_access
    @documentation.add(
        f"""Read the charges of the selected steps.

Returns
-------
np.ndarray
    Contains the charges for the selected steps projected on atoms and orbitals.

{slice_.examples("magnetism", "charges")}"""
    )
    def charges(self):
        moments = _Moments(self._raw_data.moments)
        return moments[self._steps, 0, :, :]

    @base.data_access
    @documentation.add(
        f"""Read the magnetic moments of the selected steps.

Returns
-------
np.ndarray
    Contains the magnetic moments for the selected steps projected on atoms and
    orbitals.

{_index_note}

{slice_.examples("magnetism", "moments")}"""
    )
    def moments(self):
        moments = _Moments(self._raw_data.moments)
        _fail_if_steps_out_of_bounds(moments, self._steps)
        if moments.shape[1] == 1:
            return None
        elif moments.shape[1] == 2:
            return moments[self._steps, 1, :, :]
        else:
            moments = moments[self._steps, 1:, :, :]
            direction_axis = 1 if moments.ndim == 4 else 0
            return np.moveaxis(moments, direction_axis, -1)

    @base.data_access
    @documentation.add(
        f"""Read the total charges of the selected steps.

Returns
-------
np.ndarray
    Contains the total charges for the selected steps projected on atoms. This
    corresponds to the charges summed over the orbitals.

{slice_.examples("magnetism", "total_charges")}"""
    )
    def total_charges(self):
        return _sum_over_orbitals(self.charges())

    @base.data_access
    @documentation.add(
        f"""Read the total magnetic moments of the selected steps.

Returns
-------
np.ndarray
    Contains the total magnetic moments for the selected steps projected on atoms.
    This corresponds to the magnetic moments summed over the orbitals.

{_index_note}

{slice_.examples("magnetism", "total_moments")}"""
    )
    def total_moments(self):
        moments = _Moments(self._raw_data.moments)
        _fail_if_steps_out_of_bounds(moments, self._steps)
        if moments.shape[1] == 1:
            return None
        elif moments.shape[1] == 2:
            return _sum_over_orbitals(self.moments())
        else:
            total_moments = _sum_over_orbitals(moments[self._steps, 1:, :, :])
            direction_axis = 1 if total_moments.ndim == 3 else 0
            return np.moveaxis(total_moments, direction_axis, -1)

    def _prepare_magnetic_moments_for_plotting(self):
        moments = self.total_moments()
        moments = _convert_moment_to_3d_vector(moments)
        max_length_moments = _max_length_moments(moments)
        if max_length_moments > 1e-15:
            rescale_moments = Magnetism.length_moments / max_length_moments
            return rescale_moments * moments
        else:
            return None


class _Moments(reader.Reader):
    def error_message(self, key, err):
        key = np.array(key)
        steps = key if key.ndim == 0 else key[0]
        return (
            f"Error reading the magnetic moments. Please check if the steps "
            f"`{steps}` are properly formatted and within the boundaries. "
            "Additionally, you may consider the original error message:\n" + err.args[0]
        )


def _fail_if_steps_out_of_bounds(moments, steps):
    moments[steps]  # try to access requested step raising an error if out of bounds


def _sum_over_orbitals(quantity):
    return np.sum(quantity, axis=-1)


def _convert_moment_to_3d_vector(moments):
    if moments is not None and moments.ndim == 1:
        moments = moments.reshape((len(moments), 1))
        no_new_moments = (0, 0)
        add_zero_for_xy_axis = (2, 0)
        moments = np.pad(moments, (no_new_moments, add_zero_for_xy_axis))
    return moments


def _max_length_moments(moments):
    if moments is not None:
        return np.max(np.linalg.norm(moments, axis=1))
    else:
        return 0.0
