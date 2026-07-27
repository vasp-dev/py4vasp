# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Expose the Bader charge analysis on quantities defined on a real-space grid.

The heavy lifting lives in :mod:`py4vasp._util.bader`; this module only wires it
into the dispatch architecture. Bader basins are defined by the topology of the
*charge density*, so only the charge density can construct them: it opts in with
:class:`BaderAnalysisMixin`. Every other grid quantity opts in with the lighter
:class:`BaderMixin`, which only integrates a quantity within basins that were
supplied from the density. A quantity contributes a ``_bader_grid`` method to its
handler. The two module-level functions act as the dispatched "handler methods"
(their first parameter is named ``self`` so the dispatcher forwards the selection
correctly).
"""

from py4vasp._calculation.dispatch import merge_default
from py4vasp._util import bader as _bader


def _bader_analysis(self, selection=None, *, snap_to_atoms=True):
    return _bader.analysis_from_selection(
        self._structure(),
        self._bader_grid,
        selection,
        snap_to_atoms,
        reference_for_selection=getattr(self, "_bader_reference", None),
    )


def _bader_charge(self, selection=None, *, bader_analysis=None):
    return _bader.charges_from_selection(self._bader_grid, selection, bader_analysis)


class BaderMixin:
    """Integrate a grid quantity within Bader basins supplied by the density."""

    def _bader_selection(self, selection):
        # Honor a source chosen via item access (e.g. density["all_electron"]) the
        # same way read and plot do, combining it with the method selection.
        source = getattr(self, "_selection_name", None)
        if source is None:
            return selection
        if selection is None:
            return source
        return f"{source}({selection})"

    def bader_charge(self, selection=None, *, bader_analysis=None):
        """Integrate the selected quantity within Bader basins.

        Bader basins are defined by the topology of the charge density, so they
        cannot be constructed from this quantity itself (a potential, for instance,
        is minimal rather than maximal at the nuclei). You therefore always supply
        the basins through ``bader_analysis``, obtained from
        :meth:`Density.bader_analysis`.

        Parameters
        ----------
        selection : str
            Select which grid(s) of this quantity to integrate. Defaults to the
            quantity's default grid.
        bader_analysis : BaderAnalysis
            The basins to integrate in, obtained from a :meth:`Density.bader_analysis`
            call, e.g.
            ``potential.bader_charge(bader_analysis=density.bader_analysis())``.

        Returns
        -------
        dict
            ``{atom: charge}`` for a single selection, or
            ``{selection: {atom: charge}}`` for several.

        Examples
        --------
        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)
        >>> basins = calculation.density.bader_analysis()
        >>> calculation.potential.bader_charge(bader_analysis=basins)
        {...}
        """
        return merge_default(
            self._source,
            self._quantity_name,
            self._bader_selection(selection),
            self._handler_factory,
            _bader_charge,
            bader_analysis=bader_analysis,
        )


class BaderAnalysisMixin(BaderMixin):
    """Add the full Bader charge analysis to the charge density.

    Only the charge density can *define* Bader basins, so it alone exposes
    :meth:`bader_analysis` in addition to the :meth:`bader_charge` integration
    inherited from :class:`BaderMixin`.
    """

    def bader_analysis(self, selection=None, *, snap_to_atoms=True):
        """Partition the selected density into atomic Bader basins.

        Parameters
        ----------
        selection : str
            Select a single density component to partition. Defaults to the
            default charge density.
        snap_to_atoms : bool
            Snap every density maximum to its nearest atom (default). Disable to
            label basins by the raw maxima instead.

        Returns
        -------
        BaderAnalysis
            An analysis object whose basins can be visualized with ``plot``, whose
            Bader charges are available via ``charges``, or which can be passed to
            any quantity's ``bader_charge`` to integrate it in these basins.

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
            self._bader_selection(selection),
            self._handler_factory,
            _bader_analysis,
            snap_to_atoms=snap_to_atoms,
        )
