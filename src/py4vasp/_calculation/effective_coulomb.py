# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import exception
from py4vasp._calculation import base, cell
from py4vasp._raw import data
from py4vasp._third_party import graph, numeric
from py4vasp._util import check, convert, index, select


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
    def to_graph(self, selection="total", omega=None, radius=None) -> graph.Graph:
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
        potentials = self._get_effective_potentials_omega(omega_in, omega_out)
        series = list(self._generate_series_omega(tree, omega_out, potentials))
        xlabel = "Im(ω) (eV)" if omega_set else "ω (eV)"
        return graph.Graph(series, xlabel=xlabel, ylabel="Coulomb potential (eV)")

    def _get_effective_potentials_omega(self, omega_in, omega_out, position=0):
        wannier_iiii = self._wannier_indices_iiii()
        wannier_ijij = self._wannier_indices_ijij()
        all_omega = all_spin = complex_ = slice(None)
        real_part = 0
        if self._has_positions:
            access_U = (all_omega, all_spin, wannier_iiii, position, complex_)
            access_J = (all_omega, all_spin, wannier_ijij, position, complex_)
            access_V = (all_spin, wannier_iiii, position, real_part)
        else:
            access_U = (all_omega, all_spin, wannier_iiii, complex_)
            access_J = (all_omega, all_spin, wannier_ijij, complex_)
            access_V = (all_spin, wannier_iiii, real_part)

        U_in = convert.to_complex(self._raw_data.screened_potential[access_U])
        U_in = np.average(U_in, axis=-1)
        J_in = convert.to_complex(self._raw_data.screened_potential[access_J])
        J_in = np.average(J_in, axis=-1)
        V_in = np.average(self._raw_data.bare_potential_high_cutoff[access_V], axis=-1)
        V_out = np.repeat(V_in, len(omega_out)).reshape(-1, len(omega_out))

        needs_interpolation = omega_in is not omega_out
        if needs_interpolation:
            U_out = numeric.analytic_continuation(omega_in, U_in.T, omega_out).real
            J_out = numeric.analytic_continuation(omega_in, J_in.T, omega_out).real
        else:
            U_out = U_in.T.real
            J_out = J_in.T.real
        return {"screened U": U_out, "screened J": J_out, "bare V": V_out}

    def _wannier_indices_iiii(self):
        """Return the indices that trace over diagonal of the 4 Wannier states. This
        should be equivalent to `np.einsum('iiii->', data[..., 0])`
        if there are no other indices."""
        n = self._raw_data.number_wannier_states
        step = n**3 + n**2 + n + 1
        stop = n**4
        return slice(0, stop, step)

    def _wannier_indices_ijij(self):
        """Return the indices that trace over diagonal of the 4 Wannier states. This
        should be equivalent to `np.einsum('ijij->', data[..., 0])`
        if there are no other indices."""
        n = self._raw_data.number_wannier_states
        stop = n**4
        slice_included = slice(0, stop, n**2 + 1)
        slice_excluded = slice(0, stop, n**3 + n**2 + n + 1)
        indices = np.arange(stop)
        return np.setdiff1d(indices[slice_included], indices[slice_excluded])

    def _generate_series_omega(self, tree, omega, potentials):
        if np.isclose(omega.real, omega).all():
            omega = omega.real
        else:
            omega = omega.imag
        maps = self._create_map()
        for label, potential in potentials.items():
            selector = index.Selector(maps, potential, reduction=np.average)
            for selection in tree.selections():
                selector_label = selector.label(selection)
                suffix = f"_{selector_label}" if selector_label != "total" else ""
                yield graph.Series(omega, selector[selection], label=f"{label}{suffix}")

    def _create_map(self):
        if self._is_collinear:
            spin_map = {
                convert.text_to_string(label): i
                for i, label in enumerate(self._raw_data.spin_labels[:])
            }
            return {0: {"total": slice(0, 2), **spin_map}}
        else:
            return {0: {"total": 0}}

    def _plot_radial(self, tree, radius):
        positions = self._read_positions()
        if not positions:
            raise exception.DataMismatch("The output does not contain position data.")
        radius_in = self._transform_positions_to_radial(positions)
        radius_out = radius_in if radius is ... else radius
        potentials = self._get_effective_potentials_radial(radius_in, radius_out)
        series = list(self._generate_series_radial(tree, radius_out, potentials))
        return graph.Graph(series, xlabel="Radius (Å)", ylabel="Coulomb potential (eV)")

    def _transform_positions_to_radial(self, positions):
        return np.linalg.norm(
            positions["lattice_vectors"] @ positions["positions"].T, axis=0
        )

    def _get_effective_potentials_radial(self, radius_in, radius_out):
        wannier_iiii = self._wannier_indices_iiii()
        wannier_ijij = self._wannier_indices_ijij()
        all_positions = all_spin = slice(None)
        omega_zero = real_part = 0
        if self._has_frequencies:
            access_U = (omega_zero, all_spin, wannier_iiii, all_positions, real_part)
            access_J = (omega_zero, all_spin, wannier_ijij, all_positions, real_part)
        else:
            access_U = (all_spin, wannier_iiii, all_positions, real_part)
            access_J = (all_spin, wannier_ijij, all_positions, real_part)
        access_V = (all_spin, wannier_iiii, all_positions, real_part)
        U_in = np.average(self._raw_data.screened_potential[access_U], axis=1)
        J_in = np.average(self._raw_data.screened_potential[access_J], axis=0)
        V_in = np.average(self._raw_data.bare_potential_high_cutoff[access_V], axis=1)
        needs_interpolation = radius_in is not radius_out
        if needs_interpolation:
            U_out = self._ohno_interpolation(radius_in, U_in, radius_out)
            V_out = self._ohno_interpolation(radius_in, V_in, radius_out)
            return {"screened U": U_out, "bare V": V_out}
        else:
            return {"screened U": U_in, "screened J": J_in, "bare V": V_in}

    def _ohno_interpolation(self, radius_in, spin_potential, radius_out):
        potential = np.average(spin_potential[:2], axis=0)
        interpolation = numeric.interpolate_with_function(
            self.ohno_potential, radius_in, potential / potential[0], radius_out
        )
        return np.multiply.outer(spin_potential[:, 0], interpolation)

    @staticmethod
    def ohno_potential(radius, delta):
        delta = np.abs(delta)
        return np.sqrt(delta / (radius + delta))

    def _generate_series_radial(self, tree, radius, potentials):
        maps = self._create_map()
        for label, potential in potentials.items():
            selector = index.Selector(maps, potential, reduction=np.average)
            for selection in tree.selections():
                selector_label = selector.label(selection)
                suffix = f"_{selector_label}" if selector_label != "total" else ""
                yield graph.Series(
                    radius, selector[selection], label=f"{label}{suffix}", marker="*"
                )

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
