# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Every quantity declares which tokens the selection of each routine combines.

The declaration is what a tool offers a user, so it is checked against the
routine itself: a routine that declares no axes takes only a source, and every token a
routine declares must be accepted by it.
"""

import inspect
import warnings

import pytest

from py4vasp import demo, exception
from py4vasp._calculation import _public_quantities, _quantity_object
from py4vasp._calculation.dispatch import _availability_quantity_of
from py4vasp._calculation.effective_coulomb import EffectiveCoulomb
from py4vasp._calculation.electron_phonon_bandgap import ElectronPhononBandgap
from py4vasp._calculation.electron_phonon_chemical_potential import (
    ElectronPhononChemicalPotential,
)
from py4vasp._calculation.electron_phonon_self_energy import ElectronPhononSelfEnergy
from py4vasp._calculation.electron_phonon_transport import ElectronPhononTransport
from py4vasp._calculation.electronic_minimization import ElectronicMinimization
from py4vasp._calculation.exciton_eigenvector import ExcitonEigenvector
from py4vasp._calculation.internal_strain import InternalStrain
from py4vasp._calculation.piezoelectric_tensor import PiezoelectricTensor
from py4vasp._calculation.polarization import Polarization
from py4vasp._util import form

from .test_public_routines import ROUTINES

_KINDS_OF_DATA = (
    None,
    "collinear",
    "noncollinear",
    "metal",
    "surface",
    "spin_texture",
    "perovskite",
)
# Routines that take only a source by design or offer no selection to a user.
_NOT_PROJECTED = ("print", "__str__", "selections", "_to_database", "plot")
# Arguments a routine cannot be called without.
_REQUIRED = {
    "to_contour": {"a": 0.1},
    "to_quiver": {"a": 0.1},
    "NeighborList.read": {"cutoff": 4.0},
    "NeighborList.to_dict": {"cutoff": 4.0},
}
# Routines that need a selection to run at all.
_DEFAULT_SELECTIONS = {"Band.to_quiver": "x~y(band=1)"}
# Grammars in which a token is only valid as part of a larger selection.
_TEMPLATES = {"Band.to_quiver": lambda token: _spin_texture(token)}


def _spin_texture(token):
    if "~" in token:
        return f"{token}(band=1)"
    if token.startswith("band="):
        return f"x~y({token})"
    return f"x~y(band=1({token}))"


@pytest.fixture(scope="module")
def calculations(tmp_path_factory):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return [
            demo.calculation(tmp_path_factory.mktemp("demo") / "data", selection=kind)
            for kind in _KINDS_OF_DATA
        ]


def _cases():
    calculation = demo.calculation()
    for call_name, _ in _public_quantities():
        quantity = _quantity_object(calculation, call_name)
        for method in ROUTINES:
            routine = getattr(quantity, method, None)
            if method in _NOT_PROJECTED or routine is None:
                continue
            if "selection" in inspect.signature(routine).parameters:
                yield pytest.param(call_name, method, id=f"{call_name}.{method}")


@pytest.mark.parametrize("call_name, method", _cases())
def test_declared_axes_match_the_routine(call_name, method, calculations):
    quantity, required = _quantity_with_data(calculations, call_name, method)
    axes = quantity._selection_axes(method)
    name = f"{type(quantity).__name__}.{method}"
    routine = getattr(quantity, method)
    if not axes:
        with pytest.raises(exception.IncorrectUsage, match="takes no further"):
            routine(selection="not_a_token", **required)
        return
    template = _TEMPLATES.get(name, lambda token: token)
    for axis in axes:
        for example in _examples(axis):
            try:
                routine(selection=template(example), **required)
            except exception.NoData:
                pass  # a token may describe data this calculation lacks


def _quantity_with_data(calculations, call_name, method):
    """The quantity of the demo data with which the routine runs by default.

    Some routines need a selection or a particular structure, e.g. a vacuum for STM;
    without demo data they run with, the first one containing the quantity is used.
    """
    fallback = None
    for calculation in calculations:
        quantity = _quantity_object(calculation, call_name)
        routine = getattr(quantity, method)
        required = _required_arguments(routine, f"{type(quantity).__name__}.{method}")
        if not quantity.is_available(method=method):
            continue
        fallback = fallback or (quantity, required)
        name = f"{type(quantity).__name__}.{method}"
        selection = _DEFAULT_SELECTIONS.get(name, {})
        try:
            routine(**required, **({"selection": selection} if selection else {}))
        except exception.ModuleNotInstalled as error:
            pytest.skip(str(error))  # the core installation lacks an optional package
        except exception.Py4VaspError:
            continue
        return quantity, required
    if fallback is None:
        pytest.skip(f"the demo data contains no {call_name}")
    return fallback


def _required_arguments(routine, name):
    parameters = inspect.signature(routine).parameters
    required = {**_REQUIRED.get(routine.__name__, {}), **_REQUIRED.get(name, {})}
    return {key: value for key, value in required.items() if key in parameters}


def _examples(axis):
    if axis["kind"] == "atom":
        first, last = axis["indices"]
        return [*axis["elements"], str(first), f"{first}:{last}"]
    if axis["kind"] == "choice":
        return [name for token in axis["tokens"] for name in _names(token)]
    if axis["kind"] == "pair":
        members = _distinct_members(axis["members"])
        pair = form.OPERATORS["pair"]
        examples = [f"{members[0]}{pair}{members[-1]}"]
        if not axis["distinct"]:
            examples.append(f"{members[0]}{pair}{members[0]}")
        return examples
    if axis["kind"] == "index":
        return [str(axis["indices"][0]), str(axis["indices"][1])]
    if axis["kind"] == "assignment":
        return [f"{key}={_examples(values)[0]}" for key, values in axis["keys"].items()]
    raise AssertionError(f"unknown kind of axis {axis['kind']}")


def _distinct_members(axis):
    if axis["kind"] == "choice":
        return [token["name"] for token in axis["tokens"]][:2]
    if axis["kind"] == "atom" and axis["elements"]:
        return axis["elements"][:2]
    first, last = axis["indices"]
    return [str(first), str(min(first + 1, last))]


def _names(token):
    return [token["name"], *token["aliases"]]


# Quantities the demo data lacks are built from the raw data of the tests instead.
_FROM_RAW_DATA = {
    "effective_coulomb": lambda raw_data: EffectiveCoulomb.from_data(
        raw_data.effective_coulomb("crpar")
    ),
    "electronic_minimization": lambda raw_data: ElectronicMinimization.from_data(
        raw_data.electronic_minimization()
    ),
    "electron_phonon.transport": lambda raw_data: ElectronPhononTransport.from_data(
        raw_data.electron_phonon_transport("default")
    ),
    "electron_phonon.bandgap": lambda raw_data: ElectronPhononBandgap.from_data(
        raw_data.electron_phonon_band_gap("default")
    ),
    "electron_phonon.self_energy": lambda raw_data: ElectronPhononSelfEnergy.from_data(
        raw_data.electron_phonon_self_energy("default")
    ),
    "electron_phonon.chemical_potential": lambda raw_data: (
        ElectronPhononChemicalPotential.from_data(
            raw_data.electron_phonon_chemical_potential("carrier_den")
        )
    ),
    "exciton.eigenvector": lambda raw_data: ExcitonEigenvector.from_data(
        raw_data.exciton_eigenvector("default")
    ),
    "internal_strain": lambda raw_data: InternalStrain.from_data(
        raw_data.internal_strain("Sr2TiO4")
    ),
    "piezoelectric_tensor": lambda raw_data: PiezoelectricTensor.from_data(
        raw_data.piezoelectric_tensor("default")
    ),
    "polarization": lambda raw_data: Polarization.from_data(
        raw_data.polarization("default")
    ),
}
_RAW_CASES = [
    pytest.param(name, method, id=f"{name}.{method}")
    for name, methods in {
        "effective_coulomb": ("read", "to_dict", "to_graph"),
        "electronic_minimization": ("read", "to_dict", "to_graph"),
        "electron_phonon.transport": ("read", "to_dict", "to_graph"),
        **dict.fromkeys(
            (
                "electron_phonon.bandgap",
                "electron_phonon.self_energy",
                "electron_phonon.chemical_potential",
                "exciton.eigenvector",
                "internal_strain",
                "piezoelectric_tensor",
                "polarization",
            ),
            ("read", "to_dict"),
        ),
    }.items()
    for method in methods
]
# A quantity built from raw data has no source to select, so only the projection
# is checked here; the transport plot always needs a transport quantity.
_RAW_TEMPLATES = {"ElectronPhononTransport.to_graph": lambda token: token}
_RAW_DEFAULTS = {"ElectronPhononTransport.to_graph": "mobility"}


@pytest.mark.parametrize("name, method", _RAW_CASES)
def test_declared_axes_match_the_routine_for_raw_data(name, method, raw_data):
    quantity = _FROM_RAW_DATA[name](raw_data)
    routine = getattr(quantity, method)
    axes = quantity._selection_axes(method)
    if not axes:
        with pytest.raises(exception.IncorrectUsage, match="takes no further"):
            routine(selection="not_a_token")
        return
    qualified = f"{type(quantity).__name__}.{method}"
    for axis in axes:
        for example in _examples(axis):
            selection = example
            if qualified in _RAW_DEFAULTS and axis["name"] != "transport":
                selection = f"{_RAW_DEFAULTS[qualified]}({example})"
            routine(selection=selection)


# Tokens selections() lists for routines a form does not offer.
_LISTED_FOR_OTHER_ROUTINES = {
    "projector": "project() projects other data on the atoms, orbitals, and spins",
    "electron_phonon.bandgap": "select() filters the instances",
    "electron_phonon.self_energy": "select() filters the instances",
}
_ONLY_FOR_OTHER_ROUTINES = {"raman": {"depolarization"}}  # activity() reads the ratio


@pytest.mark.parametrize(
    "call_name", sorted({name for name, _ in _public_quantities()})
)
def test_selections_agree_with_the_routines(call_name, calculations, raw_data):
    quantity = _quantity_offering_selections(calculations, raw_data, call_name)
    listed = quantity.selections()
    if call_name in _LISTED_FOR_OTHER_ROUTINES:
        pytest.skip(_LISTED_FOR_OTHER_ROUTINES[call_name])
    if isinstance(listed, list):  # pairs are listed as such, without their sources
        axes = _axes_of_all_routines(quantity)
        accepted = {t for ts in form.selections_from_axes(axes).values() for t in ts}
        assert set(listed) <= accepted, "selections() lists pairs no routine takes"
        return
    own = {quantity._quantity_name, _availability_quantity_of(quantity)}
    listed = {str(t) for key, ts in listed.items() if key not in own for t in ts}
    axes = _axes_of_all_routines(quantity)
    accepted = {t for ts in form.selections_from_axes(axes).values() for t in ts}
    extra = _ONLY_FOR_OTHER_ROUTINES.get(call_name, set())
    assert listed <= accepted | extra, "selections() lists tokens no routine takes"
    names = {
        t for ts in form.selections_from_axes(_names_only(axes)).values() for t in ts
    }
    assert names <= listed, "selections() misses tokens a routine takes"


def _quantity_offering_selections(calculations, raw_data, call_name):
    for calculation in calculations:
        quantity = _quantity_object(calculation, call_name)
        try:
            quantity.selections()
        except exception.Py4VaspError:
            continue
        if quantity.is_available():
            return quantity
    if call_name in _FROM_RAW_DATA:
        return _FROM_RAW_DATA[call_name](raw_data)
    pytest.skip(f"no data for {call_name}")


def _axes_of_all_routines(quantity):
    "The axes of every routine the data of this calculation can be used with."
    axes = {}
    for method in ROUTINES + ("to_numpy", "to_frame"):
        declared = quantity._selection_axes(method)
        if declared and quantity.is_available(method=method):
            for axis in declared:
                axes.setdefault(axis["name"], axis)
    return list(axes.values())


def _names_only(axes):
    "The axes with the aliases of their tokens left out."
    result = []
    for axis in axes:
        if axis["kind"] == "choice":
            tokens = [{**token, "aliases": []} for token in axis["tokens"]]
            axis = {**axis, "tokens": tokens}
        elif axis["kind"] == "assignment":
            axis = {
                **axis,
                "keys": dict(zip(axis["keys"], _names_only(axis["keys"].values()))),
            }
        result.append(axis)
    return result
