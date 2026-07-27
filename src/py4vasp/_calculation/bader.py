# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Wire the Bader charge analysis into the grid quantities via composition.

The heavy lifting lives in :mod:`py4vasp._util.bader`; this module only bridges it
to the dispatch architecture. Each grid quantity composes the analysis by defining
its own ``bader_charge`` (and the charge density additionally ``bader_analysis``)
as a thin method that forwards to :func:`charge` / :func:`analysis` here -- there
is no shared base class. Bader basins are defined by the topology of the charge
density, so only the charge density can construct them; every other quantity
integrates within basins that were supplied from the density. A quantity
contributes a ``_bader_grid`` method (and optionally ``_bader_reference``) to its
handler; the private dispatch functions below receive that handler as ``self``.
"""

from py4vasp._calculation.dispatch import merge_default
from py4vasp._util import bader as _bader


def analysis(quantity, selection=None, *, snap_to_atoms=True):
    """Build a :class:`~py4vasp._util.bader.BaderAnalysis` from a grid quantity.

    Forwarding target for the charge density's ``bader_analysis`` method.
    """
    return merge_default(
        quantity._source,
        quantity._quantity_name,
        _combine_source(quantity, selection),
        quantity._handler_factory,
        _dispatch_analysis,
        snap_to_atoms=snap_to_atoms,
    )


def charge(quantity, selection=None, *, bader_analysis=None):
    """Integrate a grid quantity within the basins of ``bader_analysis``.

    Forwarding target for every quantity's ``bader_charge`` method.
    """
    return merge_default(
        quantity._source,
        quantity._quantity_name,
        _combine_source(quantity, selection),
        quantity._handler_factory,
        _dispatch_charge,
        bader_analysis=bader_analysis,
    )


def _combine_source(quantity, selection):
    # Honor a source chosen via item access (e.g. density["all_electron"]) the same
    # way read and plot do, combining it with the method selection.
    source = getattr(quantity, "_selection_name", None)
    if source is None:
        return selection
    if selection is None:
        return source
    return f"{source}({selection})"


def _dispatch_analysis(self, selection=None, *, snap_to_atoms=True):
    # self is the quantity handler; its first parameter name lets the dispatcher
    # forward the remaining selection here.
    return _bader.analysis_from_selection(
        self._structure(),
        self._bader_grid,
        selection,
        snap_to_atoms,
        reference_for_selection=getattr(self, "_bader_reference", None),
    )


def _dispatch_charge(self, selection=None, *, bader_analysis=None):
    return _bader.charges_from_selection(self._bader_grid, selection, bader_analysis)
