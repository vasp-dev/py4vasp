# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Test that all dispatcher methods follow the selection parameter convention.

The dispatch system uses `merge_X(source, quantity_name, selection, handler_factory,
method, *args, **kwargs)` where:
- The 3rd argument (`selection`) is used for source routing AND is automatically
  forwarded (as `remaining_selection`) to the handler method when it accepts a
  `selection` parameter.
- `*args` are extra arguments forwarded to the handler method AFTER the automatic
  selection injection. Selection must NEVER appear in *args.

Rules:
1. If the dispatcher method has a `selection` parameter:
   - The merge call's 3rd argument MUST be `selection`.
2. If the dispatcher method does NOT have a `selection` parameter:
   - The merge call's 3rd argument MUST be `None`.
3. In ALL cases: `selection` must NOT appear in *args (the dispatch system
   handles forwarding automatically).
"""

import ast
import importlib
import inspect
import pathlib
import sys

import pytest

from py4vasp import exception
from py4vasp._calculation.dispatch import (
    _REGISTRY,
    _availability_quantity_of,
    merge_default,
)
from py4vasp._raw import definition
from py4vasp._raw.definition import schema
from py4vasp._raw.schema import DEFAULT_SELECTION

# Force-import all dispatcher modules so the registry is populated.
_CALCULATION_DIR = (
    pathlib.Path(__file__).resolve().parent.parent.parent
    / "src"
    / "py4vasp"
    / "_calculation"
)
for _f in sorted(_CALCULATION_DIR.glob("*.py")):
    if _f.name.startswith("_") and _f.name != "_CONTCAR.py":
        continue
    _module_name = _f.stem
    try:
        importlib.import_module(f"py4vasp._calculation.{_module_name}")
    except (ImportError, Exception):
        pass

MERGE_FUNCS = frozenset({"merge_default", "merge_graphs", "merge_strings"})

# Decorator quantity names that correspond to multi-source schema entries.
# Built from the schema at import time.
MULTI_SOURCE_NAMES = frozenset(
    qty for qty, srcs in schema._sources.items() if len(srcs) > 1
)

# Some @quantity decorators use names that differ from the schema key.
# Map decorator name -> schema name for multi-source lookup.
_DECORATOR_TO_SCHEMA = {
    "transport": "electron_phonon_transport",
}

# Classes that use a non-standard selection pattern (e.g. self._selection_name)
# and should be excluded from the standard convention check.
_EXCLUDED_CLASSES = frozenset({"Density"})


def _get_all_dispatcher_classes():
    """Yield (quantity_name, cls) for every registered dispatcher class."""
    for key, value in _REGISTRY.items():
        if isinstance(value, dict):
            # Group (e.g. phonon -> {band: cls, dos: cls})
            for sub_name, cls in value.items():
                yield sub_name, cls
        else:
            yield key, value


def _get_source_file(cls):
    """Return the Path to the source file of a class."""
    return pathlib.Path(inspect.getfile(cls))


def _parse_class_ast(cls):
    """Return the AST ClassDef node for the given class."""
    source_file = _get_source_file(cls)
    tree = ast.parse(source_file.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls.__name__:
            return node, tree
    raise ValueError(f"Could not find class {cls.__name__} in {source_file}")


def _get_handler_classes_from_file(tree):
    """Return a dict of {ClassName: ClassDef} for non-dispatcher classes."""
    handlers = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        is_dispatcher = any(
            isinstance(d, ast.Call)
            and isinstance(d.func, ast.Name)
            and d.func.id == "quantity"
            for d in node.decorator_list
        )
        if not is_dispatcher:
            handlers[node.name] = node
    return handlers


def _get_handler_method_params(handler_classes, handler_class_name, method_name):
    """Return the list of parameter names for a handler method."""
    cls_node = handler_classes.get(handler_class_name)
    if cls_node is None:
        return []
    for item in cls_node.body:
        if isinstance(item, ast.FunctionDef) and item.name == method_name:
            return [arg.arg for arg in item.args.args]
    return []


def _get_ast_repr(node):
    """Get a string representation of an AST node for comparison."""
    if isinstance(node, ast.Constant) and node.value is None:
        return "None"
    elif isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        parts = []
        n = node
        while isinstance(n, ast.Attribute):
            parts.append(n.attr)
            n = n.value
        if isinstance(n, ast.Name):
            parts.append(n.id)
        return ".".join(reversed(parts))
    return "other"


def _find_merge_calls(method_node):
    """Find all merge_X calls within a method and return analysis info."""
    calls = []
    for node in ast.walk(method_node):
        if not isinstance(node, ast.Call):
            continue
        func_name = getattr(node.func, "id", None)
        if func_name not in MERGE_FUNCS:
            continue
        # Extract 3rd argument (selection for source routing)
        third_arg = _get_ast_repr(node.args[2]) if len(node.args) >= 3 else "missing"
        # Extract 5th argument (handler method reference)
        handler_method_ref = None
        if len(node.args) >= 5:
            ref = node.args[4]
            if isinstance(ref, ast.Attribute) and isinstance(ref.value, ast.Name):
                handler_method_ref = (ref.value.id, ref.attr)
        # Extract extra args (after handler method ref, i.e. args[5:])
        extra_args = [_get_ast_repr(a) for a in node.args[5:]]
        calls.append(
            {
                "func": func_name,
                "third_arg": third_arg,
                "handler_ref": handler_method_ref,
                "extra_args": extra_args,
            }
        )
    return calls


def _is_public_method(method_node):
    """Check if a method is public (not starting with _) and not a dunder helper."""
    name = method_node.name
    if name.startswith("_") and not name.startswith("__"):
        return False
    # Skip non-dispatch helpers
    if name in ("__init__", "__getitem__", "__copy__", "_repr_pretty_"):
        return False
    return True


def _get_quantity_name_from_decorator(cls_node):
    """Extract the quantity name string from @quantity('name') decorator."""
    for d in cls_node.decorator_list:
        if (
            isinstance(d, ast.Call)
            and isinstance(d.func, ast.Name)
            and d.func.id == "quantity"
        ):
            if d.args and isinstance(d.args[0], ast.Constant):
                return d.args[0].value
    return None


def _collect_test_cases():
    """Collect all (quantity_key, class, method_name, merge_call) test cases."""
    cases = []
    for key, value in _REGISTRY.items():
        if isinstance(value, dict):
            for sub_key, cls in value.items():
                cases.extend(_cases_for_class(sub_key, cls))
        else:
            cases.extend(_cases_for_class(key, value))
    return cases


def _cases_for_class(registry_key, cls):
    """Generate test cases for a single dispatcher class."""
    cases = []
    if cls.__name__ in _EXCLUDED_CLASSES:
        return cases
    try:
        cls_node, tree = _parse_class_ast(cls)
    except (ValueError, OSError):
        return cases

    qty_name = _get_quantity_name_from_decorator(cls_node)
    if qty_name is None:
        return cases

    # Map decorator name to schema name for multi-source check
    schema_name = _DECORATOR_TO_SCHEMA.get(qty_name, qty_name)
    is_multi = schema_name in MULTI_SOURCE_NAMES
    handler_classes = _get_handler_classes_from_file(tree)

    for item in cls_node.body:
        if not isinstance(item, ast.FunctionDef):
            continue
        if not _is_public_method(item):
            continue

        merge_calls = _find_merge_calls(item)
        if not merge_calls:
            continue

        method_params = [arg.arg for arg in item.args.args]
        has_selection_param = "selection" in method_params

        for call_info in merge_calls:
            handler_ref = call_info["handler_ref"]
            handler_has_selection = False
            if handler_ref:
                handler_class_name, handler_method_name = handler_ref
                handler_params = _get_handler_method_params(
                    handler_classes, handler_class_name, handler_method_name
                )
                handler_has_selection = "selection" in handler_params

            cases.append(
                (
                    registry_key,
                    cls.__name__,
                    item.name,
                    is_multi,
                    handler_has_selection,
                    has_selection_param,
                    call_info,
                )
            )
    return cases


_ALL_CASES = _collect_test_cases()


def _case_id(case):
    registry_key, cls_name, method_name, *_ = case
    return f"{cls_name}.{method_name}"


@pytest.mark.parametrize("case", _ALL_CASES, ids=_case_id)
def test_selection_convention(case):
    (
        registry_key,
        cls_name,
        method_name,
        is_multi,
        handler_has_selection,
        has_selection_param,
        call_info,
    ) = case

    third_arg = call_info["third_arg"]
    extra_args = call_info["extra_args"]

    # Rule 1: If the dispatcher has a `selection` parameter, the 3rd merge arg
    # MUST be `selection` (enables source routing AND auto-forwarding).
    # Rule 2: If it does NOT have `selection`, the 3rd arg MUST be `None`.
    if has_selection_param:
        assert third_arg == "selection", (
            f"{cls_name}.{method_name}: dispatcher has `selection` parameter, "
            f"so 3rd merge argument must be `selection`, got `{third_arg}`"
        )
    else:
        assert third_arg == "None", (
            f"{cls_name}.{method_name}: dispatcher has no `selection` parameter, "
            f"so 3rd merge argument must be `None`, got `{third_arg}`"
        )

    # Rule 3: `selection` must NEVER appear in *args — the dispatch system
    # handles forwarding automatically via introspection.
    assert "selection" not in extra_args, (
        f"{cls_name}.{method_name}: `selection` must not be passed in *args "
        f"(dispatch auto-forwards it). Extra args: {extra_args}"
    )


# ---------------------------------------------------------------------------
# Which public methods take a `selection`. Three rules decide it:
#
# 1. Legacy: in py4vasp 0.11.3 every method decorated with @base.data_access accepted
#    a selection, so these methods keep it for backwards compatibility.
# 2. Effect: a method takes a selection if its quantity has a non-default source in
#    the schema or if it reaches a handler method that takes a selection.
# 3. No effect: any other method takes no selection.
#
# Rules 2 and 3 are derived from the code; the sources are read from the schema at
# collection time, so a new source puts its quantity under rule 2 without changing
# this test.
# ---------------------------------------------------------------------------

# `selections` lists the sources, `is_available` is shared by all quantities, and the
# abc.Sequence protocol fixes the signatures of `count` and `index`.
_NOT_CHECKED = frozenset({"selections", "is_available", "count", "index"})

# Public methods decorated with @base.data_access in py4vasp 0.11.3, including those
# inherited from base.Refinery (print) and `read`, which forwarded to `to_dict`.
_LEGACY_SELECTION = {
    "Band": {"print", "read", "to_dict", "to_frame", "to_graph", "to_quiver"},
    "Bandgap": {
        "conduction_band_minimum",
        "direct",
        "fundamental",
        "print",
        "read",
        "to_dict",
        "to_graph",
        "valence_band_maximum",
    },
    "BornEffectiveCharge": {"print", "read", "to_dict"},
    "CurrentDensity": {"print", "read", "to_contour", "to_dict", "to_quiver"},
    "Density": {
        "is_collinear",
        "is_noncollinear",
        "is_nonpolarized",
        "print",
        "read",
        "to_contour",
        "to_dict",
        "to_numpy",
        "to_quiver",
        "to_view",
    },
    "DielectricFunction": {"print", "read", "to_dict", "to_graph"},
    "DielectricTensor": {"print", "read", "to_dict"},
    "Dos": {"print", "read", "to_dict", "to_frame", "to_graph"},
    "EffectiveCoulomb": {"print", "read", "to_dict", "to_graph"},
    "ElasticModulus": {"print", "read", "to_dict"},
    "ElectronPhononBandgap": {
        "chemical_potential_mu_tag",
        "print",
        "read",
        "select",
        "to_dict",
    },
    "ElectronPhononChemicalPotential": {"label", "mu_tag", "print", "read", "to_dict"},
    "ElectronPhononSelfEnergy": {
        "chemical_potential_mu_tag",
        "eigenvalues",
        "print",
        "read",
        "select",
        "to_dict",
    },
    "ElectronPhononTransport": {
        "chemical_potential_mu_tag",
        "print",
        "read",
        "select",
        "to_dict",
        "to_graph",
    },
    "ElectronicMinimization": {"is_converged", "print", "read", "to_dict"},
    "Energy": {"print", "read", "to_dict", "to_graph", "to_numpy"},
    "ExcitonDensity": {"print", "read", "to_dict", "to_numpy", "to_view"},
    "ExcitonEigenvector": {"print", "read", "to_dict"},
    "Force": {"number_steps", "print", "read", "to_dict", "to_view"},
    "ForceConstant": {"eigenvectors", "print", "read", "to_dict", "to_molden"},
    "InternalStrain": {"print", "read", "to_dict"},
    "Kpoint": {
        "distances",
        "labels",
        "line_length",
        "mode",
        "number_kpoints",
        "number_lines",
        "path_indices",
        "print",
        "read",
        "to_dict",
    },
    "LocalMoment": {
        "charge",
        "magnetic",
        "number_steps",
        "print",
        "projected_charge",
        "projected_magnetic",
        "read",
        "to_dict",
        "to_view",
    },
    "Nics": {"print", "read", "to_contour", "to_dict", "to_numpy", "to_view"},
    "PairCorrelation": {"labels", "print", "read", "to_dict", "to_graph"},
    "PartialDensity": {
        "bands",
        "grid",
        "kpoints",
        "print",
        "read",
        "to_dict",
        "to_numpy",
        "to_stm",
        "to_view",
    },
    "PhononBand": {"print", "read", "to_dict", "to_graph"},
    "PhononDos": {"print", "read", "to_dict", "to_graph"},
    "PhononMode": {"frequencies", "print", "read", "to_dict"},
    "PiezoelectricTensor": {"print", "read", "to_dict"},
    "Polarization": {"print", "read", "to_dict"},
    "Potential": {"print", "read", "to_contour", "to_dict", "to_quiver", "to_view"},
    "Projector": {"print", "project", "read", "to_dict"},
    "RunInfo": {"print", "read", "to_dict"},
    "Stress": {"number_steps", "print", "read", "to_dict"},
    "Structure": {
        "cartesian_positions",
        "lattice_vectors",
        "number_atoms",
        "number_steps",
        "positions",
        "print",
        "read",
        "to_POSCAR",
        "to_ase",
        "to_dict",
        "to_lammps",
        "to_mdtraj",
        "to_view",
        "volume",
    },
    "System": {"print", "read", "to_dict"},
    "Velocity": {"number_steps", "print", "read", "to_dict", "to_numpy", "to_view"},
    "Workfunction": {"print", "read", "to_dict", "to_graph"},
}


def _sources(cls):
    quantity_name = _availability_quantity_of(cls)
    try:
        return list(definition.unique_selections(quantity_name))
    except exception.FileAccessError:
        return []


def _has_non_default_source(cls):
    return any(source != DEFAULT_SELECTION for source in _sources(cls))


def _public_methods(cls):
    for name, member in inspect.getmembers(cls):
        if name.startswith("_") or name in _NOT_CHECKED:
            continue
        static_member = inspect.getattr_static(cls, name)
        if isinstance(static_member, (classmethod, staticmethod, property)):
            continue
        if callable(member) and not inspect.isclass(member):
            yield name


def _public_quantities():
    # private quantities (leading underscore) are not reachable by the user
    for quantity_name, cls in _get_all_dispatcher_classes():
        if not quantity_name.startswith("_"):
            yield cls


def _is_legacy(cls, method_name):
    return method_name in _LEGACY_SELECTION.get(cls.__name__, ())


def _collect_methods(include):
    for cls in _public_quantities():
        for method_name in _public_methods(cls):
            if not include(cls, method_name):
                continue
            id_ = f"{cls.__name__}.{method_name}"
            yield pytest.param(cls, method_name, id=id_)


def _is_inherited_from_mixin(cls, method_name):
    owner = next(klass for klass in cls.__mro__ if method_name in vars(klass))
    return owner.__module__.startswith("py4vasp._third_party")


def _accepts_selection(cls, method_name):
    """The method takes `selection` by keyword, or it is a plotting mixin method
    forwarding its arguments to a method of the quantity."""
    parameters = inspect.signature(getattr(cls, method_name)).parameters.values()
    forwards = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    if any(parameter.kind in forwards for parameter in parameters):
        if _is_inherited_from_mixin(cls, method_name):
            return True
    return any(
        parameter.name == "selection"
        and parameter.kind != inspect.Parameter.POSITIONAL_ONLY
        for parameter in parameters
    )


def _method_node(cls, method_name):
    for klass in cls.__mro__:
        try:
            cls_node, _ = _parse_class_ast(klass)
        except (ValueError, OSError, TypeError):
            continue
        for item in cls_node.body:
            if isinstance(item, ast.FunctionDef) and item.name == method_name:
                return item
    return None


def _handler_takes_selection(cls, merge_call):
    """Whether the handler method passed to a merge_* call takes `selection`. A
    reference that cannot be resolved counts as taking it."""
    if len(merge_call.args) < 5:
        return True
    reference = merge_call.args[4]
    if not (
        isinstance(reference, ast.Attribute) and isinstance(reference.value, ast.Name)
    ):
        return True
    module = sys.modules[cls.__module__]
    handler_class = getattr(module, reference.value.id, None)
    handler_method = getattr(handler_class, reference.attr, None)
    if handler_method is None:
        return True
    return "selection" in inspect.signature(handler_method).parameters


def _is_call_of_own_method(call):
    function = call.func
    return (
        isinstance(function, ast.Attribute)
        and isinstance(function.value, ast.Name)
        and function.value.id == "self"
    )


def _call_arguments(call):
    return [*call.args, *(keyword.value for keyword in call.keywords)]


def _is_selection(node):
    return isinstance(node, ast.Name) and node.id == "selection"


def _reaches_handler_with_selection(cls, method_name, seen=None):
    """Whether the method, or a method of the same class it calls, passes data to a
    handler method that takes a selection. Any use of `selection` the trace cannot
    follow, e.g. passing it to a helper function, counts as reaching one."""
    seen = set() if seen is None else seen
    if method_name in seen:
        return False
    seen.add(method_name)
    method_node = _method_node(cls, method_name)
    if method_node is None:
        return True
    traced = set()
    for node in ast.walk(method_node):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in MERGE_FUNCS:
            traced.update(id(argument) for argument in _call_arguments(node)[2:3])
            if _handler_takes_selection(cls, node):
                return True
        elif _is_call_of_own_method(node):
            traced.update(id(argument) for argument in _call_arguments(node))
            if _reaches_handler_with_selection(cls, node.func.attr, seen):
                return True
    return any(
        _is_selection(node)
        and isinstance(node.ctx, ast.Load)
        and id(node) not in traced
        for node in ast.walk(method_node)
    )


def _selection_has_effect(cls, method_name):
    if _has_non_default_source(cls):
        return True
    return _reaches_handler_with_selection(cls, method_name)


def _takes_selection(cls, method_name):
    return _accepts_selection(cls, method_name)


@pytest.mark.parametrize("cls, method_name", list(_collect_methods(_is_legacy)))
def test_legacy_methods_take_selection(cls, method_name):
    assert _takes_selection(cls, method_name), (
        f"{cls.__name__}.{method_name} accepted a selection in py4vasp 0.11.3, so it "
        "must keep taking `selection` for backwards compatibility."
    )


@pytest.mark.parametrize(
    "cls, method_name",
    list(_collect_methods(_selection_has_effect)),
)
def test_methods_where_selection_has_effect_take_selection(cls, method_name):
    assert _takes_selection(cls, method_name), (
        f"{cls.__name__}.{method_name} must take `selection`, because its quantity "
        "has a non-default source or it calls a handler method taking a selection."
    )


def _selection_has_no_effect(cls, method_name):
    return not _is_legacy(cls, method_name) and not _selection_has_effect(
        cls, method_name
    )


@pytest.mark.parametrize(
    "cls, method_name",
    list(_collect_methods(_selection_has_no_effect)),
)
def test_methods_where_selection_has_no_effect_take_no_selection(cls, method_name):
    parameters = inspect.signature(getattr(cls, method_name)).parameters
    assert "selection" not in parameters, (
        f"{cls.__name__}.{method_name} takes `selection`, but its quantity has only "
        "the default source and no handler method it calls takes a selection."
    )


def _declares_selection(cls, method_name):
    method_node = _method_node(cls, method_name)
    if method_node is None:
        return False
    arguments = method_node.args
    return any(
        argument.arg == "selection"
        for argument in [*arguments.args, *arguments.kwonlyargs]
    )


@pytest.mark.parametrize(
    "cls, method_name", list(_collect_methods(_declares_selection))
)
def test_methods_taking_selection_use_it(cls, method_name):
    # a selection that is accepted but never reaches the data silently returns the
    # default data
    assert _selection_flows_to_data(
        cls, method_name
    ), f"{cls.__name__}.{method_name} takes `selection` but does not pass it on."


# functions that parse a selection into its sources for dispatch
_SELECTION_PARSERS = frozenset({"_parse_selections"})


def _mentions_selection(node):
    return any(_is_selection(child) for child in ast.walk(node))


def _is_call_on_handler(call):
    # e.g. self._handler_factory(raw).select(selection)
    function = call.func
    return (
        isinstance(function, ast.Attribute)
        and isinstance(function.value, ast.Call)
        and _is_call_of_own_method(function.value)
        and function.value.func.attr == "_handler_factory"
    )


def _selection_flows_to_data(cls, method_name, seen=None):
    """Whether `selection` reaches the source argument of a merge_* call, another
    method of the class passing it on, a helper that also receives `self`, or a
    method of the handler."""
    seen = set() if seen is None else seen
    if method_name in seen:
        return False
    seen.add(method_name)
    method_node = _method_node(cls, method_name)
    if method_node is None:
        return False
    for node in ast.walk(method_node):
        if not isinstance(node, ast.Call):
            continue
        arguments = _call_arguments(node)
        if not any(_mentions_selection(argument) for argument in arguments):
            continue
        function_name = getattr(node.func, "id", None) or getattr(
            node.func, "attr", None
        )
        if function_name in MERGE_FUNCS:
            if len(node.args) >= 3 and _mentions_selection(node.args[2]):
                return True
        elif function_name in _SELECTION_PARSERS or _is_call_on_handler(node):
            return True
        elif _is_call_of_own_method(node):
            if _selection_flows_to_data(cls, node.func.attr, seen):
                return True
        elif any(
            isinstance(argument, ast.Name) and argument.id == "self"
            for argument in arguments
        ):
            return True
    return False


class _FakeQuantity:
    """Patterns the AST checks must tell apart; never executed."""

    def logs_selection(self, selection=None):
        print(selection)
        return merge_default(self._source, "fake", None, self._factory, len)

    def forwards_selection(self, selection=None):
        return self.routes_selection(selection)

    def routes_selection(self, selection=None):
        return merge_default(self._source, "fake", selection, self._factory, len)

    def to_view(self, *args, **kwargs):
        return merge_default(self._source, "fake", None, self._factory, len)


def test_selection_must_reach_the_data():
    assert not _selection_flows_to_data(_FakeQuantity, "logs_selection")
    assert _selection_flows_to_data(_FakeQuantity, "forwards_selection")
    # **kwargs on the quantity itself does not stand in for a selection parameter
    assert not _accepts_selection(_FakeQuantity, "to_view")


def test_legacy_methods_exist():
    # a legacy method that is renamed or removed would silently escape rule 1
    collected = {param.id for param in _collect_methods(_is_legacy)}
    legacy = {
        f"{cls.__name__}.{method_name}"
        for cls in _public_quantities()
        for method_name in _LEGACY_SELECTION.get(cls.__name__, ())
    }
    assert legacy == collected


def test_selection_is_traced_to_the_handler():
    local_moment = _REGISTRY["local_moment"]
    force = _REGISTRY["force"]
    assert _reaches_handler_with_selection(local_moment, "to_view")  # spin component
    assert not _reaches_handler_with_selection(force, "to_dict")  # via read()
    assert not _reaches_handler_with_selection(force, "print")  # via __str__()
    potential = _REGISTRY["potential"]
    assert _reaches_handler_with_selection(potential, "bader_charge")  # via helper


def test_quantity_with_new_source_must_take_selection(monkeypatch):
    unique_selections = definition.unique_selections

    def add_source_to_symmetry(quantity_name):
        sources = list(unique_selections(quantity_name))
        return sources + ["new_source"] if quantity_name == "symmetry" else sources

    assert not _selection_has_effect(_REGISTRY["symmetry"], "space_group")
    monkeypatch.setattr(definition, "unique_selections", add_source_to_symmetry)
    assert _selection_has_effect(_REGISTRY["symmetry"], "space_group")
