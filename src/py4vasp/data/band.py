import functools
import itertools
import numpy as np
import plotly.graph_objects as go
from IPython.lib.pretty import pretty
from .projectors import _projectors_or_dummy, _selection_doc
from .kpoints import Kpoints
from py4vasp.data import _util
from py4vasp.data._base import DataBase, RefinementDescriptor


class Band(DataBase):
    """The electronic band structure.

    The most common use case of this class is to produce the electronic band
    structure along a path in the Brillouin zone used in a non self consistent
    Vasp calculation. In some cases you may want to use the `to_dict` function
    just to obtain the eigenvalue and projection data though in that case the
    **k**-point distances that are calculated are meaningless.

    Parameters
    ----------
    raw_band : RawBand
        Dataclass containing the raw data necessary to produce a band structure
        (eigenvalues, kpoints, ...).
    """

    read = RefinementDescriptor("_to_dict")
    to_dict = RefinementDescriptor("_to_dict")
    plot = RefinementDescriptor("_to_plotly")
    to_plotly = RefinementDescriptor("_to_plotly")
    __str__ = RefinementDescriptor("_to_string")


def _to_string(raw_band):
    path = _create_path_if_available(raw_band)
    return f"""
{"spin polarized" if _spin_polarized(raw_band) else ""} band structure{path}:
    {raw_band.eigenvalues.shape[1]} k-points
    {raw_band.eigenvalues.shape[2]} bands
{pretty(_projectors(raw_band))}
    """.strip()


@_util.add_doc(
    f"""Read the data into a dictionary.

Parameters
----------
{_selection_doc}

Returns
-------
dict
    Contains the **k**-point path for plotting band structures with the
    eigenvalues shifted to bring the Fermi energy to 0. If available
    and a selection is passed, the projections of these bands on the
    selected projectors are included."""
)
def _to_dict(raw_band, selection=None):
    kpoints = _kpoints(raw_band)
    return {
        "kpoint_distances": kpoints.distances(),
        "kpoint_labels": kpoints.labels(),
        "fermi_energy": raw_band.fermi_energy,
        **_shift_bands_by_fermi_energy(raw_band),
        "projections": _projectors(raw_band).read(selection, raw_band.projections),
    }


@_util.add_doc(
    f"""Read the data and generate a plotly figure.

Parameters
----------
{_selection_doc}
width : float
    Specifies the width of the flatbands if a selection of projections is specified.
Returns
-------
plotly.graph_objects.Figure
    plotly figure containing the spin-up and spin-down bands. If a selection
    is provided the width of the bands represents the projections of the
    bands onto the specified projectors."""
)
def _to_plotly(raw_band, selection=None, width=0.5):
    ticks, labels = _ticks_and_labels(raw_band)
    data = _band_structure(raw_band, selection, width)
    default = {
        "xaxis": {"tickmode": "array", "tickvals": ticks, "ticktext": labels},
        "yaxis": {"title": {"text": "Energy (eV)"}},
    }
    return go.Figure(data=data, layout=default)


def _spin_polarized(raw_band):
    return len(raw_band.eigenvalues) == 2


def _kpoints(raw_band):
    return Kpoints(raw_band.kpoints)


def _projectors(raw_band):
    return _projectors_or_dummy(raw_band.projectors)


def _shift_bands_by_fermi_energy(raw_band):
    if _spin_polarized(raw_band):
        return {
            "up": raw_band.eigenvalues[0] - raw_band.fermi_energy,
            "down": raw_band.eigenvalues[1] - raw_band.fermi_energy,
        }
    else:
        return {"bands": raw_band.eigenvalues[0] - raw_band.fermi_energy}


def _band_structure(raw_band, selection, width):
    bands = _shift_bands_by_fermi_energy(raw_band)
    projections = _projectors(raw_band).read(selection, raw_band.projections)
    if len(projections) == 0:
        return _regular_band_structure(raw_band, bands)
    else:
        return _fat_band_structure(raw_band, bands, projections, width)


def _regular_band_structure(raw_band, bands):
    kdists = _kpoints(raw_band).distances()
    return [_scatter(name, kdists, lines) for name, lines in bands.items()]


def _fat_band_structure(raw_band, bands, projections, width):
    error_message = "Width of fat band structure must be a number."
    _util.raise_error_if_not_number(width, error_message)
    data = (
        _fat_band(raw_band, args, width)
        for args in itertools.product(bands.items(), projections.items())
    )
    return list(filter(None, data))


def _fat_band(raw_band, args, width):
    (key, lines), (name, projection) = args
    if _spin_polarized(raw_band) and not key in name:
        return None
    kdists = _kpoints(raw_band).distances()
    fatband_kdists = np.concatenate((kdists, np.flip(kdists)))
    upper = lines + width * projection
    lower = lines - width * projection
    fatband_lines = np.concatenate((lower, np.flip(upper, axis=0)), axis=0)
    plot = _scatter(name, fatband_kdists, fatband_lines)
    plot.fill = "toself"
    plot.mode = "none"
    return plot


def _scatter(name, kdists, lines):
    # insert NaN to split separate lines
    num_bands = lines.shape[-1]
    kdists = np.tile([*kdists, np.NaN], num_bands)
    lines = np.append(lines, [np.repeat(np.NaN, num_bands)], axis=0)
    return go.Scatter(x=kdists, y=lines.flatten(order="F"), name=name)


def _create_path_if_available(raw_band):
    _, labels = _ticks_and_labels(raw_band)
    if any(len(label.strip()) > 0 for label in labels):
        return " (" + " - ".join(labels) + ")"
    else:
        return ""


def _ticks_and_labels(raw_band):
    def filter_unique(current, item):
        tick, label = item
        previous_tick = current[-2]
        if tick == previous_tick:
            previous_label = current[-1]
            label = previous_label + "|" + label if previous_label else label
            return current[:-1] + (label,)
        else:
            return current + item

    ticks_and_labels = _degenerate_ticks_and_labels(raw_band)
    ticks_and_labels = functools.reduce(filter_unique, ticks_and_labels)
    return _split_and_replace_empty_labels(ticks_and_labels)


def _split_and_replace_empty_labels(ticks_and_labels):
    ticks = [tick for tick in ticks_and_labels[::2]]
    labels = [label or " " for label in ticks_and_labels[1::2]]
    # plotly replaces empty labels with tick position, so we replace them
    return ticks, labels


def _degenerate_ticks_and_labels(raw_band):
    labels = _kpoint_labels(raw_band)
    mask = np.logical_or(_edge_of_line(raw_band), labels != "")
    return zip(_kpoints(raw_band).distances()[mask], labels[mask])


def _kpoint_labels(raw_band):
    labels = _kpoints(raw_band).labels()
    if labels is None:
        labels = [""] * len(raw_band.kpoints.coordinates)
    return np.array(labels)


def _edge_of_line(raw_band):
    indices = np.arange(len(raw_band.kpoints.coordinates))
    edge_of_line = (indices + 1) % _kpoints(raw_band).line_length() == 0
    edge_of_line[0] = True
    return edge_of_line
