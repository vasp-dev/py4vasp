---
name: port-quantity
description: "Port a py4vasp quantity from the inheritance-based Refinery architecture to the new Dispatcher/Impl architecture. USE WHEN: migrating an existing quantity class, porting tests, or adding a new quantity. Triggers: 'port quantity', 'migrate to new architecture', 'convert to composition', 'new architecture', 'refactor quantity'."
---

# Port a py4vasp Quantity to the Dispatcher/Handler Architecture

Port an existing `Refinery`-based quantity to the new architecture described in `docs/architecture/calculation.rst`.

## Core Design Contract

Each quantity is split into two classes:

- **Dispatcher** (public, e.g. ``Band``) — user-facing, attached to ``Calculation``. Owns the ``Source``, calls standalone dispatch functions that parse selections, open data, construct the Handler, call methods, and merge results.
- **Handler** (private, e.g. ``BandHandler``) — constructed via ``from_data(raw)``. Works with exactly one raw data object. Contains all transform logic. Primary unit-testing target.

The dispatcher does **not** have ``from_data``. That lives exclusively on the Handler.

### Selection dispatch

Selection dispatch is handled by standalone functions named after their merge
strategy. The dispatcher just calls the appropriate one:

```python
def merge_default(source, quantity_name, selection, handler_factory, method, *args, **kwargs):
    """Use this if no specific merge strategy is required. This will merge the results
    into a dict by selection name if more than one result is returned."""
    ...

def merge_graphs(source, quantity_name, selection, handler_factory, method, *args, **kwargs):
    """Overlay Graph results into a single figure."""
    ...

def merge_strings(source, quantity_name, selection, handler_factory, method, *args, **kwargs):
    """Return a single string by concatenating results for all selections."""
    ...
```

**Choosing the merge strategy:**

- `merge_default` — **the default** for every method. Use this almost always.
- `merge_graphs` — for methods that return a `Graph` (typically `plot`).
- `merge_strings` — for methods that return a string (e.g. `__str__`).

All call the inner `_dispatch` which:
1. Parses `selection` via `_parse_selections` → list of `SelectionContext(selection_name, remaining_selection)`.
2. For each context, calls `source.access(quantity_name, selection=ctx.selection_name)`.
3. Inside the context, calls `handler_factory(raw)` then `method(handler, *args, **kwargs)`.
4. Collects results as `{selection_name: result}`.

```python
class SelectionContext(typing.NamedTuple):
    selection_name: str | None
    remaining_selection: str | None
```

### Handler pattern

Type hints on `_raw_band` and in `from_data` are **mandatory** — never omit them.
They document which raw dataclass the Handler works with and are essential for
readability and static analysis.

```python
from py4vasp import raw

class BandHandler:
    def __init__(self, raw_band: raw.Band):
        self._raw_band = raw_band

    @classmethod
    def from_data(cls, raw_band: raw.Band) -> "BandHandler":
        return cls(raw_band)

    def read(self) -> dict:
        return {
            "eigenvalues": np.array(self._raw_band.eigenvalues) - self._raw_band.fermi_energy,
            ...
        }

    def plot(self) -> Graph:
        ...
```

### Dispatcher pattern

```python
@quantity("band")
class Band:
    def __init__(self, source: Source, quantity_name: str = "band"):
        self._source = source
        self._quantity_name = quantity_name

    def read(self, selection: str | None = None) -> dict:
        return merge_default(
            self._source, self._quantity_name, selection,
            BandHandler.from_data, BandHandler.read,
        )

    def to_dict(self, selection: str | None = None) -> dict:
        """Public alias for read(). Part of the public interface — never deprecate."""
        return self.read(selection=selection)

    def plot(self, selection: str | None = None) -> Graph:
        return merge_graphs(
            self._source, self._quantity_name, selection,
            BandHandler.from_data, BandHandler.plot,
        )
```

**Important:** Every dispatcher method that has `@base.data_access` in the old
Refinery code has `selection` injected into it at runtime. After porting, **all**
dispatcher methods must declare `selection: str | None = None` in their signature,
even if the Refinery method had no explicit `selection` parameter.

**`to_dict` is part of the public interface and must never be deprecated or
removed.** In the new architecture, `to_dict` is a thin alias that delegates to
`read()`. Tests for `to_dict` should verify it returns the same result as `read()`:

```python
# Tests for to_dict
def test_to_dict_matches_read(raw_bandgap):
    handler = BandgapHandler.from_data(raw_bandgap)
    assert handler.to_dict() == handler.read()  # or via dispatcher:

def test_dispatcher_to_dict_matches_read(raw_bandgap):
    source = DataSource(raw_bandgap)
    dispatcher = Bandgap(source=source, quantity_name="bandgap")
    assert dispatcher.to_dict() == dispatcher.read()
```

Extra arguments from the dispatcher method are forwarded:

```python
def plot(self, selection=None, fermi_energy=None):
    return merge_graphs(
        self._source, self._quantity_name, selection,
        BandHandler.from_data, BandHandler.plot,
        fermi_energy=fermi_energy,
    )
```

---

## Migration Procedure

### 1 — Identify the raw dataclass

Open `src/py4vasp/_raw/data.py`. Find the dataclass matching this quantity (CamelCase of the quantity name). This becomes the type for the Handler's `_raw_<quantity>` attribute. Note all fields and their types.

Example for `bandgap`:
```python
@dataclasses.dataclass
class Bandgap:
    labels: VaspData
    values: VaspData
```

### 2 — Create the Handler class

Create a private `<Quantity>Handler` class. It takes raw data in its constructor and has a `from_data` classmethod. Move all transform logic here.

Type hints on `_raw_<quantity>` and in `from_data` are **mandatory**. Always use the exact
raw dataclass type from `py4vasp.raw` defined in `py4vasp/_raw/data.py`.

```python
# Before (on the Refinery)
class Bandgap(slice_.Mixin, base.Refinery, graph.Mixin):
    _raw_data: raw_data.Bandgap

    @base.data_access
    def to_dict(self):
        return {
            **self._gap_dict("fundamental"),
            ...
            "fermi_energy": self._get("Fermi energy", component=0),
        }

# After (Handler)
class BandgapHandler:
    def __init__(self, raw_bandgap: raw.Bandgap, steps=None):
        self._raw_bandgap = raw_bandgap
        self._steps = steps

    @classmethod
    def from_data(cls, raw_bandgap: raw.Bandgap, steps=None) -> "BandgapHandler":
        return cls(raw_bandgap, steps=steps)

    def read(self) -> dict:
        return {
            **self._gap_dict("fundamental"),
            ...
            "fermi_energy": self._get("Fermi energy", component=0),
        }
```

Key changes:
- Replace `self._raw_data` with `self._raw_<quantity>` everywhere in the Handler.
- Remove `@base.data_access` decorators — Handler methods are plain methods.
- The Handler never touches `Source` or selection dispatch.

### 3 — Create the Dispatcher class

Create the public class with the `@quantity()` decorator. It owns the `Source` and delegates to dispatch functions.

```python
@quantity("bandgap")
class Bandgap(graph.Mixin):
    def __init__(self, source: Source, quantity_name: str = "bandgap", steps=None):
        self._source = source
        self._quantity_name = quantity_name
        self._steps = steps

    def __getitem__(self, steps) -> Bandgap:
        return Bandgap(self._source, self._quantity_name, steps=steps)

    def _handler_factory(self, raw):
        return BandgapHandler.from_data(raw, steps=self._steps)

    def read(self, selection: str | None = None) -> dict:
        return merge_default(
            self._source, self._quantity_name, selection,
            self._handler_factory, BandgapHandler.read,
        )

    def to_dict(self, selection: str | None = None) -> dict:
        return self.read(selection=selection)

    def plot(self, selection: str | None = None) -> Graph:
        return merge_graphs(
            self._source, self._quantity_name, selection,
            self._handler_factory, BandgapHandler.plot,
        )
```

For step-indexed quantities, use `self._handler_factory` as a bound method that captures `self._steps` via the partial pattern shown above.

### 4 — Move private helpers to the Handler

Private helpers (`_gap`, `_get`, `_kpoint`, `_spin_polarized`, etc.) that read `self._raw_data` move to the Handler and read `self._raw_<quantity>` instead.

```python
# Before (Refinery)
def _spin_polarized(self):
    return self._raw_data.values.shape[1] == 3

# After (Handler)
def _spin_polarized(self):
    return self._raw_bandgap.values.shape[1] == 3
```

### 5 — Handle selection forwarding

`_dispatch` does **not** automatically forward `remaining_selection` to the
Handler. `SelectionContext.remaining_selection` is used only internally to
identify which source to open.

If a Handler method needs to parse a selection string further (e.g. filtering
spin channels or orbital projections), pass the full `selection` string
explicitly as a keyword argument from the dispatcher:

```python
# Dispatcher — forward selection explicitly when the handler needs it
def plot(self, selection=None):
    return merge_graphs(
        self._source, self._quantity_name, selection,
        self._handler_factory, BandgapHandler.plot,
        selection=selection,
    )

# Handler — receives the full selection string and parses it itself
class BandgapHandler:
    def plot(self, selection=None) -> Graph:
        tree = select.Tree.from_selection(selection)
        ...
```

For quantities whose Handler methods do not need further selection parsing,
omit the `selection=selection` kwarg entirely.

### 6 — Handle composition with other quantities

Use the other Impl's `from_data` directly — no Source needed:

```python
class DensityHandler:
    def read(self) -> dict:
        structure = _StructureHandler.from_data(self._raw_density.structure)
        return {
            "charge": np.array(self._raw_density.charge),
            "structure": structure.read(),
        }
```

### 7 — Step-indexed quantities

Steps live on the dispatcher and are passed to the Handler via the factory:

```python
# Dispatcher
def __getitem__(self, steps) -> Structure:
    return Structure(self._source, self._quantity_name, steps=steps)

def _handler_factory(self, raw):
    return StructureHandler.from_data(raw, steps=self._steps)
```

The Handler applies `slice_steps` explicitly:

```python
from py4vasp._calculation.dispatch import slice_steps

class StructureHandler:
    def __init__(self, raw_structure: raw.Structure, steps=None):
        self._raw_structure = raw_structure
        self._steps = steps

    def read(self) -> dict:
        return {
            "lattice_vectors": slice_steps(
                np.array(self._raw_structure.lattice_vectors), self._steps, default_ndim=2
            ),
        }
```

`slice_steps(data, steps, default_ndim)` rules:
- `steps=None` → last step (default)
- `steps=3` → single step
- `steps=slice(1, 8)` → range
- `data.ndim <= default_ndim` → no step axis, return unchanged

### 8 — Port `__str__` and display methods

Move the string formatting logic to the Handler. The dispatcher calls it through dispatch:

```python
# Handler
class BandgapHandler:
    def __str__(self):
        template = """..."""
        return template.format(...)

# Dispatcher
class Bandgap:
    def __str__(self):
        return merge_strings(
            self._source, self._quantity_name, None,
            self._handler_factory, BandgapHandler.__str__,
        )
```

### 9 — Port `_to_database`

Same dispatch pattern. The Handler has the `_to_database` method:

```python
# Handler
class BandgapHandler:
    def _to_database(self) -> dict:
        bandgap_dict = {...}
        return {"bandgap": Bandgap_DB(**final_dict)}

# Dispatcher
class Bandgap:
    def _read_to_database(self, *args, **kwargs):
        return merge_default(
            self._source, self._quantity_name, None,
            self._handler_factory, BandgapHandler._to_database,
        )
```

### 10 — Port the tests

**Never remove an existing test.** If a test cannot work yet (e.g. because the
dispatcher infrastructure isn't fully wired or factory methods changed), mark it
with `@pytest.mark.skip(reason="...")` so it remains visible and will be
re-enabled later.

**`to_dict` tests:** `to_dict` is a public method — do not skip or delete its
tests. Instead, restructure them so they verify that `to_dict` and `read`
return the same result:

```python
@pytest.mark.skip(reason="Dispatcher not yet wired to Calculation")
def test_factory_methods(raw_data, check_factory_methods):
    ...
```

```python
# to_dict test: verify it equals read()
def test_to_dict_matches_read(raw_bandgap):
    handler = BandgapHandler.from_data(raw_bandgap)
    assert handler.to_dict() == handler.read()
```

Tests split into two categories:

**Unit tests (Handler directly, no I/O):**

```python
def test_bandgap_read():
    raw = raw_data.Bandgap(labels=..., values=...)
    handler = BandgapHandler.from_data(raw)
    result = handler.read()
    assert ...

def test_bandgap_step_selection():
    raw = raw_data.Bandgap(labels=..., values=...)
    handler = BandgapHandler.from_data(raw, steps=3)
    result = handler.read()
    assert ...
```

**Integration tests (full pipeline via DictSource):**

```python
def test_bandgap_via_calculation():
    source = DictSource({"bandgap": raw.Bandgap(...)})
    calc = Calculation(source=source)
    result = calc.bandgap.read()
    assert ...
```

The existing `from_data` pattern in tests like:
```python
bandgap = Bandgap.from_data(raw_bandgap)
```
should be migrated to:
```python
handler = BandgapHandler.from_data(raw_bandgap)
```

Or for testing the dispatcher with the full dispatch pipeline:
```python
source = DataSource(raw_bandgap)
bandgap = Bandgap(source=source, quantity_name="bandgap")
```

### 11 — Remove from `QUANTITIES` / `GROUPS`

In `src/py4vasp/_calculation/__init__.py`, remove the quantity's string entry from `QUANTITIES` (or `GROUPS`). The `@quantity()` decorator handles registration automatically.

### 12 — Verify

```bash
pytest tests/calculation/test_{name}.py -v   # quantity-level tests
pytest tests/ -x                             # full suite
```

---

## Checklist

For each quantity being ported:

- [ ] Raw dataclass type identified in `_raw/data.py`
- [ ] Impl class created: `<Name>Handler` with `from_data(raw, steps=None)`
- [ ] All transform logic moved from Refinery to Handler
- [ ] `self._raw_data.x` → `self._raw_<name>.x` in Handler
- [ ] `@base.data_access` decorators removed
- [ ] Dispatcher class created with `@quantity("name")` decorator
- [ ] All dispatcher methods have `selection: str | None = None` parameter
- [ ] Dispatcher calls `merge_default` (default) or `merge_graphs` (for Graph) — no lambdas; `merge_strings` (for strings)
- [ ] `to_dict` kept in both Handler and dispatcher; dispatcher delegates to `read()`; not deprecated
- [ ] Type hints on `_raw_<name>` attribute and `from_data` classmethod (mandatory, never omit)
- [ ] Handler method passed as unbound reference: `BandHandler.read`
- [ ] Extra args forwarded via `*args, **kwargs`
- [ ] Step indexing: `__getitem__` on dispatcher, `_handler_factory` captures steps
- [ ] Composition: other Handler's `from_data` called directly
- [ ] `__str__` / `_repr_pretty_` ported through dispatch
- [ ] `_to_database` ported through dispatch
- [ ] Small mixins kept (e.g. `graph.Mixin`)
- [ ] `base.Refinery` and `slice_.Mixin` removed from inheritance
- [ ] Tests split: Handler unit tests + dispatcher integration tests
- [ ] Tests for `to_dict` restructured to verify it matches `read()` — not skipped
- [ ] Tests never removed — non-working tests marked `@pytest.mark.skip(reason="...")`
- [ ] Removed from `QUANTITIES`/`GROUPS` in `__init__.py`
- [ ] All non-skipped tests pass

---

## Reference Files

| File | Purpose |
|------|---------|
| `docs/architecture/calculation.rst` | Full architecture description |
| `src/py4vasp/_raw/data.py` | Raw dataclass definitions |
| `src/py4vasp/_raw/definition.py` | Schema (sources per quantity) |
| `src/py4vasp/_calculation/bandgap.py` | Reference: existing Refinery quantity |
| `tests/calculation/test_bandgap.py` | Reference: existing test structure |
