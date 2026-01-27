# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from dataclasses import dataclass
from types import EllipsisType

import numpy as np
from numpy.typing import ArrayLike

from py4vasp import exception
from py4vasp._calculation import base, cell
from py4vasp._third_party import graph, numeric
from py4vasp._util import check, convert, index, select


@dataclass
class _CoulombPotential:
    component: str
    selector_label: str
    strength: ArrayLike

    def to_omega_series(self, omega):
        label = f"{self.component} {self.selector_label}"
        strength = self.strength if self.strength.ndim == 1 else self.strength[:, 0]
        return graph.Series(omega, strength.real, label=label)

    def to_radial_series(self, radius, marker):
        label = f"{self.component} {self.selector_label}"
        strength = self.strength if self.strength.ndim == 1 else self.strength[0]
        return graph.Series(radius, strength.real, label=label, marker=marker)


class EffectiveCoulomb(base.Refinery, graph.Mixin):
    """Effective Coulomb interaction U obtained with the constrained random phase approximation (cRPA).

    This class provides post-processing routines to read and visualize first-principles
    results from constrained Random Phase Approximation (cRPA) calculations. After you
    have performed a cRPA calculation using VASP this class can visualize the effective
    Coulomb interaction *U* along the radial or frequency axis. Youy can use this *U*
    mean-field theories like DFT+*U* and Dynamical Mean Field Theory (DMFT).

    The cRPA method is essential for strongly correlated materials, where standard Density
    Functional Theory (DFT) often incorrectly predicts a metallic ground state or fails to
    capture magnetic order. You can activate the cRPA calculation in VASP by setting
    :tag:`ALGO` = `CRPAR` in the INCAR file. The method computes the effective Coulomb
    interaction *U* in real space by excluding screening processes within a predefined
    correlated subspace, typically associated with localized orbitals such as *d* or *f*
    states.

    While different flavors of cRPA exist, we recommend using the spectral cRPA (s-cRPA)
    method that you activate by setting :tag:`LSCRPA` = `.TRUE.`. in the INCAR file. This
    approach overcomes significant limitations of earlier cRPA formulations [1]_, in
    particular numerical instabilities for highly occupied correlated shells or unphysical
    results like negative *U* values.

    References
    ----------
    .. [1] Kaltak, M., *et al.*, Constrained Random Phase Approximation: the spectral
        method, Phys. Rev. B 112, 245102 (2025), https://doi.org/10.1103/m3gh-g6r6
    """

    @base.data_access
    def __str__(self):
        data = self._to_database()["effective_coulomb"]
        return f"""\
averaged bare interaction
bare Hubbard U = {data["bare_V"].real:8.4f} {data["bare_V"].imag:8.4f}
bare Hubbard u = {data["bare_v"].real:8.4f} {data["bare_v"].imag:8.4f}
bare Hubbard J = {data["bare_J"].real:8.4f} {data["bare_J"].imag:8.4f}

averaged interaction parameter
screened Hubbard U = {data["screened_U"].real:8.4f} {data["screened_U"].imag:8.4f}
screened Hubbard u = {data["screened_u"].real:8.4f} {data["screened_u"].imag:8.4f}
screened Hubbard J = {data["screened_J"].real:8.4f} {data["screened_J"].imag:8.4f}
"""

    def _to_database(self):
        wannier_iiii = self._wannier_indices_iiii()
        wannier_ijji = self._wannier_indices_ijji()
        wannier_ijij = self._wannier_indices_ijij()
        spin_diagonal = slice(None, 2)
        omega_0 = origin = 0
        complex_ = slice(None)
        if self._has_positions and self._has_frequencies:
            access_U = (omega_0, spin_diagonal, wannier_iiii, origin, complex_)
            access_u = (omega_0, spin_diagonal, wannier_ijji, origin, complex_)
            access_J = (omega_0, spin_diagonal, wannier_ijij, origin, complex_)
            access_V = (spin_diagonal, wannier_iiii, origin, complex_)
            access_v = (spin_diagonal, wannier_ijji, origin, complex_)
            access_Vj = (spin_diagonal, wannier_ijij, origin, complex_)
        elif self._has_frequencies:
            access_U = (omega_0, spin_diagonal, wannier_iiii, complex_)
            access_u = (omega_0, spin_diagonal, wannier_ijji, complex_)
            access_J = (omega_0, spin_diagonal, wannier_ijij, complex_)
            access_V = (spin_diagonal, wannier_iiii, complex_)
            access_v = (spin_diagonal, wannier_ijji, complex_)
            access_Vj = (spin_diagonal, wannier_ijij, complex_)
        elif self._has_positions:
            access_U = access_V = (spin_diagonal, wannier_iiii, origin, complex_)
            access_u = access_v = (spin_diagonal, wannier_ijji, origin, complex_)
            access_J = access_Vj = (spin_diagonal, wannier_ijij, origin, complex_)
        else:
            access_U = access_V = (spin_diagonal, wannier_iiii, complex_)
            access_u = access_v = (spin_diagonal, wannier_ijji, complex_)
            access_J = access_Vj = (spin_diagonal, wannier_ijij, complex_)
        U = convert.to_complex(self._raw_data.screened_potential[access_U])
        u = convert.to_complex(self._raw_data.screened_potential[access_u])
        J = convert.to_complex(self._raw_data.screened_potential[access_J])
        V = convert.to_complex(self._raw_data.bare_potential_high_cutoff[access_V])
        v = convert.to_complex(self._raw_data.bare_potential_high_cutoff[access_v])
        Vj = convert.to_complex(self._raw_data.bare_potential_high_cutoff[access_Vj])
        overview = {
            "screened_U": complex(np.average(U)),
            "screened_u": complex(np.average(u)),
            "screened_J": complex(np.average(J)),
            "bare_V": complex(np.average(V)),
            "bare_v": complex(np.average(v)),
            "bare_J": complex(np.average(Vj)),
        }
        return {"effective_coulomb": overview}

    def _wannier_indices_iiii(self):
        """Return the indices that trace over diagonal of the 4 Wannier states. This
        should be equivalent to `np.einsum('iiii->', data)` if data is a reshaped array
        of 4 Wannier indices."""
        n = self._raw_data.number_wannier_states
        step = n**3 + n**2 + n + 1
        stop = n**4
        return slice(0, stop, step)

    def _wannier_indices_ijij(self):
        """Return the indices that run over pairs of Wannier states. This should be
        equivalent to `np.einsum('ijij->', data` if data is a reshaped array of 4
        Wannier indices and the diagonal is set to 0."""
        n = self._raw_data.number_wannier_states
        stop = n**4
        slice_included = slice(0, stop, n**2 + 1)
        slice_excluded = slice(0, stop, n**3 + n**2 + n + 1)
        indices = np.arange(stop)
        return np.setdiff1d(indices[slice_included], indices[slice_excluded])

    def _wannier_indices_ijji(self):
        """Return the indices that run over pairs of Wannier states. This should be
        equivalent to `np.einsum('ijji->', data` if data is a reshaped array of 4
        Wannier indices and the diagonal is set to 0."""
        n = self._raw_data.number_wannier_states
        stop = n**4
        indices_included = np.concatenate(
            [i * (n**3 + 1) + np.arange(0, n**3, n**2 + n) for i in range(n)]
        )
        slice_excluded = slice(0, stop, n**3 + n**2 + n + 1)
        indices = np.arange(stop)
        return np.setdiff1d(indices_included, indices[slice_excluded])

    @base.data_access
    def to_dict(self) -> dict[str, np.ndarray]:
        """Convert the effective Coulomb object to a dictionary representation.

        The integrals are evaluated over 4 Wannier functions. For the bare Coulomb
        interaction, these integrals can be computed with either a high :tag:`ENCUT`
        or low cutoff :tag:`ENCUTGW` that you set in the INCAR file. The screened Coulomb
        interaction is evaluated with the dielectric function and will have smaller
        values than the bare Coulomb potential. If you set :tag:`TWO_CENTER` = `.TRUE.`
        in the INCAR file, the Coulomb interactions are evaluated also at neighboring
        cells.

        Returns
        -------
        -
            A dictionary containing the effective Coulomb interaction data. In particular,
            it includes the bare Coulomb interaction with high and low cutoffs, the screened
            Coulomb interaction, and optionally the frequencies and positions at which the
            interactions are evaluated.
        """
        return {
            "bare_high_cutoff": self._read_high_cutoff(),
            "bare_low_cutoff": self._read_low_cutoff(),
            "screened": self._read_screened(),
            **self._read_frequencies(),
            **self._read_positions(),
        }

    @property
    def _has_frequencies(self):
        return len(self._raw_data.frequencies) > 1

    @property
    def _has_positions(self):
        return not check.is_none(self._raw_data.positions)

    @property
    def _is_collinear(self):
        return len(self._raw_data.bare_potential_low_cutoff) == 3

    def _read_high_cutoff(self):
        V = convert.to_complex(self._raw_data.bare_potential_high_cutoff[:])
        if self._has_positions:
            V = np.moveaxis(V, -1, 0)
        V = self._unpack_wannier_indices(V)
        if self._has_frequencies:
            V = V[..., np.newaxis]
        return V

    def _read_low_cutoff(self):
        C = convert.to_complex(self._raw_data.bare_potential_low_cutoff[:])
        C = self._unpack_wannier_indices(C)
        if self._has_frequencies:
            C = C[..., np.newaxis]
        return C

    def _read_screened(self):
        U = convert.to_complex(self._raw_data.screened_potential[:])
        if self._has_positions:
            U = np.moveaxis(U, -1, 0)
        U = self._unpack_wannier_indices(U)
        if self._has_frequencies:
            U = np.moveaxis(U, 1 if self._has_positions else 0, -1)
        return U

    def _unpack_wannier_indices(self, data):
        num_wannier = self._raw_data.number_wannier_states
        new_shape = data.shape[:-1] + 4 * (num_wannier,)
        return data.reshape(new_shape)

    def _read_frequencies(self):
        if not self._has_frequencies:
            return {}
        return {"frequencies": convert.to_complex(self._raw_data.frequencies[:])}

    def _read_positions(self):
        if not self._has_positions:
            return {}
        return {
            "lattice_vectors": self._cell().lattice_vectors(),
            "positions": self._raw_data.positions[:],
        }

    def _cell(self):
        return cell.Cell.from_data(self._raw_data.cell)

    @base.data_access
    def to_graph(
        self,
        selection: str = "U J V",
        omega: None | EllipsisType | np.ndarray = None,
        radius: None | EllipsisType | np.ndarray = None,
    ) -> graph.Graph:
        """Generate a graph representation of the effective Coulomb interaction.

        The method automatically determines the plot type based on which parameters
        are provided:

        - If only omega is given: creates a frequency-dependent plot
        - If only radius is given: creates a radial-dependent plot
        - If both omega and radius are given: creates a frequency plot for all radii

        Parameters
        ----------
        selection
            Specifies which data to plot. Default is "total". For collinear calculations,
            you can select a specific spin coupling like "up~up" or "up~down".

        omega
            Frequency values for frequency-dependent plots. If not set, or set to
            ellipsis (...), the frequency points along the imaginary axis are used.
            You can also provide specific frequencies on the real axis, then the data
            will be analytically continued and plotted for the selected frequencies.

        radius
            Radial distance values for radial-dependent plots. If not set, the plot
            will be for r=0. If set to ellipsis (...), the radial points used in VASP
            are used. You can also provide specific radii, then the data  will be
            interpolated to the selected radii.

        Returns
        -------
        -
            A graph object containing the visualization of the effective Coulomb
            interaction data.
        """
        selected_dimension = self._select_dimension(omega, radius)
        tree = select.Tree.from_selection(selection)
        if selected_dimension == "frequency":
            return self._plot_frequency(tree, omega)
        elif selected_dimension == "radial":
            return self._plot_radial(tree, radius)
        elif selected_dimension == "both":
            return self._plot_both(tree, omega, radius)
        else:
            raise exception.NotImplemented(
                f"Plotting for the selected dimension {selected_dimension} is not implemented."
            )

    def _select_dimension(self, omega, radius):
        if omega is not None and radius is not None:
            return "both"
        elif omega is not None:
            return "frequency"
        elif radius is not None:
            return "radial"
        else:
            return "frequency"

    def _plot_frequency(self, tree, omega):
        omega_in = self._read_frequencies().get("frequencies")
        omega_set = omega is None or omega is ...
        if omega_in is None:
            raise exception.DataMismatch("The output does not contain frequency data.")
        omega_out = omega_in if omega_set else omega
        selected_potentials = self._get_potentials_omega(tree, omega_in, omega_out)
        xlabel = "Im(ω) (eV)" if omega_set else "ω (eV)"
        series = list(self._generate_series_omega(omega_out, selected_potentials))
        return graph.Graph(series, xlabel=xlabel, ylabel="Coulomb potential (eV)")

    def _get_potentials_omega(self, tree, omega_in, omega_out):
        for selection in tree.selections():
            if self._bare_potential_selected(selection):
                yield self._get_bare_potential(selection, len(omega_out))
            else:
                yield self._get_screened_potential(selection, omega_in, omega_out)

    def _bare_potential_selected(self, selection):
        if select.contains(selection, "bare"):
            return True
        if select.contains(selection, "screened"):
            return False
        return select.contains(selection, "V") or select.contains(selection, "v")

    def _get_bare_potential(self, selection, num_omega):
        selection = self._filter_component_from_selection(selection)
        maps = self._create_map("bare")
        potential = self._raw_data.bare_potential_high_cutoff
        selector = index.Selector(maps, potential, reduction=np.average)
        V = convert.to_complex(selector[selection])
        V = np.broadcast_to(V, (num_omega,) + V.shape)
        return _CoulombPotential("bare", selector.label(selection), V)

    def _get_screened_potential(self, selection, omega_in, omega_out):
        selection = self._filter_component_from_selection(selection)
        maps = self._create_map("screened")
        potential = self._raw_data.screened_potential
        selector = index.Selector(maps, potential, reduction=np.average)
        U = convert.to_complex(selector[selection])
        needs_interpolation = omega_in is not omega_out
        if needs_interpolation:
            U = numeric.analytic_continuation(omega_in, U.T, omega_out).T
        return _CoulombPotential("screened", selector.label(selection), U)

    def _filter_component_from_selection(self, selection):
        return tuple(part for part in selection if part not in {"bare", "screened"})

    def _create_map(self, component):
        if self._is_collinear:
            spin_map = {
                convert.text_to_string(label): slice(i, i + 1)
                for i, label in enumerate(self._raw_data.spin_labels[:])
            }
            spin_map[None] = spin_map["total"] = slice(0, 2)
        else:
            spin_map = {None: slice(None), "total": slice(None)}
        wannier_iiii = self._wannier_indices_iiii()
        wannier_ijij = self._wannier_indices_ijij()
        wannier_ijji = self._wannier_indices_ijji()
        component_map = {
            None: wannier_iiii,
            "U": wannier_iiii,
            "u": wannier_ijji,
            "J": wannier_ijij,
            "V": wannier_iiii,
            "v": wannier_ijji,
        }
        if component == "bare" or not self._has_frequencies:
            return {0: spin_map, 1: component_map}
        else:
            return {1: spin_map, 2: component_map}

    def _generate_series_omega(self, omega, selected_potentials):
        if np.isclose(omega.real, omega).all():
            omega = omega.real
        else:
            omega = omega.imag
        for coulomb_potential in selected_potentials:
            yield coulomb_potential.to_omega_series(omega)

    def _plot_radial(self, tree, radius):
        positions = self._read_positions()
        if not positions:
            raise exception.DataMismatch("The output does not contain position data.")
        radius_in = self._transform_positions_to_radial(positions)
        radius_out = radius_in if radius is ... else radius
        marker = "*" if radius is ... else None
        potentials = self._get_effective_potentials_radial(tree, radius_in, radius_out)
        series = list(self._generate_series_radial(radius_out, potentials, marker))
        return graph.Graph(series, xlabel="Radius (Å)", ylabel="Coulomb potential (eV)")

    def _transform_positions_to_radial(self, positions):
        return np.linalg.norm(
            positions["lattice_vectors"] @ positions["positions"].T, axis=0
        )

    def _get_effective_potentials_radial(self, tree, radius_in, radius_out):
        for selection in tree.selections():
            if self._bare_potential_selected(selection):
                potential = self._get_bare_potential2(selection)
            else:
                potential = self._get_screened_potential2(selection)
            needs_interpolation = radius_in is not radius_out
            if needs_interpolation:
                potential.strength = self._ohno_interpolation(
                    radius_in, potential.strength.real, radius_out
                )
            yield potential

    def _get_bare_potential2(self, selection):
        selection = self._filter_component_from_selection(selection)
        maps = self._create_map("bare")
        potential = self._raw_data.bare_potential_high_cutoff
        selector = index.Selector(maps, potential, reduction=np.average)
        V = convert.to_complex(selector[selection])
        return _CoulombPotential("bare", selector.label(selection), V)

    def _get_screened_potential2(self, selection):
        selection = self._filter_component_from_selection(selection)
        maps = self._create_map("screened")
        potential = self._raw_data.screened_potential
        selector = index.Selector(maps, potential, reduction=np.average)
        U = convert.to_complex(selector[selection])
        return _CoulombPotential("screened", selector.label(selection), U)

    def _ohno_interpolation(self, radius_in, potential, radius_out):
        if potential.ndim == 2:
            # if multiple frequencies are present, take only omega = 0
            potential = potential[0]
        interpolation = numeric.interpolate_with_function(
            self.ohno_potential, radius_in, potential / potential[0], radius_out
        )
        return potential[0] * interpolation

    @staticmethod
    def ohno_potential(radius: ArrayLike, delta: float) -> np.ndarray:
        """Ohno potential for given radius/radii and delta.

        This is used to interpolate the Coulomb potential to other radii.

        Parameters
        ----------
        radius
            The radial distance(s) at which to evaluate the potential.
        delta
            The delta parameter for the Ohno potential.

        Returns
        -------
        -
            The Ohno potential evaluated at the given radius/radii.
        """
        delta = np.abs(delta)
        return np.sqrt(delta / (radius + delta))

    def _generate_series_radial(self, radius, selected_potentials, marker):
        for coulomb_potential in selected_potentials:
            yield coulomb_potential.to_radial_series(radius, marker)

    def _plot_both(self, tree, omega, radius):
        omega_in = self._read_frequencies().get("frequencies")
        omega_set = omega is ...
        if omega_in is None:
            raise exception.DataMismatch("The output does not contain frequency data.")
        omega_out = omega_in if omega_set else omega
        positions = self._read_positions()
        if not positions:
            raise exception.DataMismatch("The output does not contain position data.")
        if radius is not ...:
            raise exception.NotImplemented(
                "Interpolating radial data for frequency plots is not implemented."
            )
        potentials = {}
        for i, position in enumerate(positions["positions"]):
            data = self._get_effective_potentials_omega(omega_in, omega_out, position=i)
            potentials[f"U @ {position}"] = data["screened U"]
        series = list(self._generate_series_omega(tree, omega_out, potentials))
        return graph.Graph(series, xlabel="ω (eV)", ylabel="Coulomb potential (eV)")
