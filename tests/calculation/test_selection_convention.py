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

import pytest

from py4vasp._calculation.dispatch import _REGISTRY
from py4vasp._raw.definition import schema

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
    "_CONTCAR": "CONTCAR",
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
            yield key, cls


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
