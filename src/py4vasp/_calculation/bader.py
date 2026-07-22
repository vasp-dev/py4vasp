# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Expose the Bader charge analysis on quantities defined on a real-space grid.

The heavy lifting lives in :mod:`py4vasp._util.bader`; this module only wires it
into the dispatch architecture. A quantity opts in by adding :class:`BaderMixin`
to its public class and a ``_bader_grid`` method to its handler. The two
module-level functions act as the dispatched "handler methods" (their first
parameter is named ``self`` so the dispatcher forwards the selection correctly).
"""
from py4vasp._calculation.dispatch import merge_default
from py4vasp._util import bader as _bader


def _bader_analysis(self, selection=None, *, snap_to_atoms=True):
    return _bader.analysis_from_selection(
        self._structure(), self._bader_grid, selection, snap_to_atoms
    )


def _bader_charge(self, selection=None, *, bader_analysis=None, snap_to_atoms=True):
    return _bader.charges_from_selection(
        self._structure(), self._bader_grid, selection, snap_to_atoms, bader_analysis
    )


class BaderMixin:
    """Add a Bader charge analysis to a quantity defined on a real-space grid."""

    def bader_analysis(self, selection=None, *, snap_to_atoms=True):
        """Partition the selected density into atomic Bader basins.

        Parameters
        ----------
        selection : str
            Select a single density component to partition. Defaults to the
            quantity's default density.
        snap_to_atoms : bool
            Snap every density maximum to its nearest atom (default). Disable to
            label basins by the raw maxima instead.

        Returns
        -------
        BaderAnalysis
            An analysis object whose basins can be visualized with ``plot`` or
            reused to integrate another density via ``bader_charge``.

        Examples
        --------
        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)
        >>> analysis = calculation.density.bader_analysis()
        >>> analysis.charges()
        {...}
        """
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            _bader_analysis,
            snap_to_atoms=snap_to_atoms,
        )

    def bader_charge(self, selection=None, *, bader_analysis=None, snap_to_atoms=True):
        """Integrate the selected density within Bader basins.

        Parameters
        ----------
        selection : str
            Select the density to integrate. You may define the basins from a
            different density of the same quantity with the ``basins=`` syntax,
            e.g. ``"m(basins=scalar)"`` integrates the magnetization within basins
            built from the scalar charge.
        bader_analysis : BaderAnalysis
            Reuse basins from a previous :meth:`bader_analysis` call, possibly of a
            different quantity, e.g.
            ``potential.bader_charge(bader_analysis=density.bader_analysis())``.
        snap_to_atoms : bool
            Snap every density maximum to its nearest atom (default).

        Returns
        -------
        dict
            ``{atom: charge}`` for a single selection, or
            ``{selection: {atom: charge}}`` for several.

        Examples
        --------
        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)
        >>> calculation.density.bader_charge()
        {...}
        """
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            _bader_charge,
            bader_analysis=bader_analysis,
            snap_to_atoms=snap_to_atoms,
        )
