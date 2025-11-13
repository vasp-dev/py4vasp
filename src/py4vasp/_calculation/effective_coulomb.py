# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import exception
from py4vasp._calculation import base, cell
from py4vasp._raw import data
from py4vasp._third_party import graph
from py4vasp._util import check, convert


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
        method, https://arxiv.org/abs/2508.15368, 2025.
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

    @base.data_access
    def to_graph(self):
        if not self._has_frequencies:
            raise exception.DataMismatch("The output does not contain frequency data.")
        _trace_indices = self._trace_wannier_indices()
        omega = self._raw_data.frequencies[:, 1]
        if self._has_positions:
            access_U = (slice(None), 0, 0, _trace_indices, 0, 0)
            access_V = (0, 0, _trace_indices, 0, 0)
        else:
            access_U = (slice(None), 0, 0, _trace_indices, 0)
            access_V = (0, 0, _trace_indices, 0)
        U = np.average(self._raw_data.screened_potential[access_U], axis=1)
        V = np.average(self._raw_data.bare_potential_high_cutoff[access_V])
        screened_potential = graph.Series(omega, U, label="U")
        bare_potential = graph.Series(omega, np.full_like(U, fill_value=V), label="V")
        return graph.Graph([screened_potential, bare_potential])

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

    @property
    def _has_frequencies(self):
        return len(self._raw_data.frequencies) > 1

    @property
    def _has_positions(self):
        return not check.is_none(self._raw_data.positions)

    def _trace_wannier_indices(self):
        """Return the indices that trace over diagonal of the 4 Wannier states. This
        should be equivalent to `np.einsum('iiii->', data[..., 0])`
        if there are no other indices."""
        n = self._raw_data.number_wannier_states
        step = n**3 + n**2 + n + 1
        stop = n**4
        return range(0, stop, step)
