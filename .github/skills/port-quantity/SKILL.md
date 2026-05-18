---
name: port-quantity
description: "Port a py4vasp quantity from the inheritance-based Refinery architecture to the new Dispatcher/Impl architecture. USE WHEN: migrating an existing quantity class, porting tests, or adding a new quantity. Triggers: 'port quantity', 'migrate to new architecture', 'convert to composition', 'new architecture', 'refactor quantity'."
---

# Port a py4vasp Quantity to the Dispatcher/Impl Architecture

Port an existing `Refinery`-based quantity to the new architecture described in `docs/architecture/calculation.rst`.

## Core Design Contract

Each quantity is split into two classes:

- **Dispatcher** (public, e.g. ``Band``) — user-facing, attached to ``Calculation``. Owns the ``Source``, calls standalone dispatch functions that parse selections, open data, construct the Impl, call methods, and merge results.
- **Impl** (private, e.g. ``_BandImpl``) — constructed via ``from_data(raw)``. Works with exactly one raw data object. Contains all transform logic. Primary unit-testing target.

The dispatcher does **not** have ``from_data``. That lives exclusively on the Impl.

### Selection dispatch

Selection dispatch is handled by standalone functions named after their merge
strategy. The dispatcher just calls the appropriate one:

```python
def merge_graphs(source, quantity_name, selection, impl_factory, method, *args, **kwargs):
    """Overlay Graph results into a single figure."""
    ...

def merge_dicts(source, quantity_name, selection, impl_factory, method, *args, **kwargs):
    """Combine dict results with selection-prefixed keys."""
    ...
```

All call the inner `_dispatch` which:
1. Parses `selection` via `_parse_selections` → list of `SelectionContext(selection_name, remaining_selection)`.
2. For each context, calls `source.access(quantity_name, selection=ctx.selection_name)`.
3. Inside the context, calls `impl_factory(raw)` then `method(impl, *args, **kwargs)`.
4. Collects results as `{selection_name: result}`.

```python
class SelectionContext(typing.NamedTuple):
    selection_name: str | None
    remaining_selection: str | None
```

### Impl pattern

```python
class _BandImpl:
    def __init__(self, raw: RawBand):
        self._raw = raw

    @classmethod
    def from_data(cls, raw: RawBand) -> _BandImpl:
        return cls(raw)

    def read(self) -> dict:
        return {
            "eigenvalues": np.array(self._raw.eigenvalues) - self._raw.fermi_energy,
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
        return merge_dicts(
            self._source, self._quantity_name, selection,
            _BandImpl.from_data, _BandImpl.read,
        )

    def plot(self, selection: str | None = None) -> Graph:
        return merge_graphs(
            self._source, self._quantity_name, selection,
            _BandImpl.from_data, _BandImpl.plot,
        )
```

Extra arguments from the dispatcher method are forwarded:

```python
def plot(self, selection=None, fermi_energy=None):
    return merge_graphs(
        self._source, self._quantity_name, selection,
        _BandImpl.from_data, _BandImpl.plot,
        fermi_energy=fermi_energy,
    )
```

---

## Migration Procedure

### 1 — Identify the raw dataclass

Open `src/py4vasp/_raw/data.py`. Find the dataclass matching this quantity (CamelCase of the quantity name). This becomes the type for the Impl's `_raw` attribute. Note all fields and their types.

Example for `bandgap`:
```python
@dataclasses.dataclass
class Bandgap:
    labels: VaspData
    values: VaspData
```

### 2 — Create the Impl class

Create a private `_<Name>Impl` class. It takes raw data in its constructor and has a `from_data` classmethod. Move all transform logic here.

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

# After (Impl)
class _BandgapImpl:
    def __init__(self, raw: raw_data.Bandgap, steps=None):
        self._raw = raw
        self._steps = steps

    @classmethod
    def from_data(cls, raw: raw_data.Bandgap, steps=None) -> _BandgapImpl:
        return cls(raw, steps=steps)

    def read(self) -> dict:
        return {
            **self._gap_dict("fundamental"),
            ...
            "fermi_energy": self._get("Fermi energy", component=0),
        }
```

Key changes:
- Replace `self._raw_data` with `self._raw` everywhere in the Impl.
- Remove `@base.data_access` decorators — Impl methods are plain methods.
- The Impl never touches `Source` or selection dispatch.

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

    def _impl_factory(self, raw):
        return _BandgapImpl.from_data(raw, steps=self._steps)

    def read(self, selection: str | None = None) -> dict:
        return merge_dicts(
            self._source, self._quantity_name, selection,
            self._impl_factory, _BandgapImpl.read,
        )

    def plot(self, selection: str | None = None) -> Graph:
        return merge_graphs(
            self._source, self._quantity_name, selection,
            self._impl_factory, _BandgapImpl.plot,
            selection=selection,  # if the Impl's plot method needs the selection
        )
```

For step-indexed quantities, use `self._impl_factory` as a bound method that captures `self._steps` via the partial pattern shown above.

### 4 — Move private helpers to the Impl

Private helpers (`_gap`, `_get`, `_kpoint`, `_spin_polarized`, etc.) that read `self._raw_data` move to the Impl and read `self._raw` instead.

```python
# Before (Refinery)
def _spin_polarized(self):
    return self._raw_data.values.shape[1] == 3

# After (Impl)
def _spin_polarized(self):
    return self._raw.values.shape[1] == 3
```

### 5 — Handle selection forwarding

When the Impl method needs the remaining selection (e.g. for projection parsing), it accepts it as a parameter. The dispatch function forwards it via `**kwargs`:

```python
# Dispatcher
def plot(self, selection=None):
    return merge_graphs(
        self._source, self._quantity_name, selection,
        self._impl_factory, _BandgapImpl.plot,
    )

# The remaining_selection from SelectionContext is available inside
# _dispatch and forwarded to the Impl method.
```

For quantities where the Impl's `plot` method handles its own selection parsing internally (like Bandgap's `_parse`), the `selection` argument is forwarded directly as a kwarg.

### 6 — Handle composition with other quantities

Use the other Impl's `from_data` directly — no Source needed:

```python
class _DensityImpl:
    def read(self) -> dict:
        structure = _StructureImpl.from_data(self._raw.structure)
        return {
            "charge": np.array(self._raw.charge),
            "structure": structure.read(),
        }
```

### 7 — Step-indexed quantities

Steps live on the dispatcher and are passed to the Impl via the factory:

```python
# Dispatcher
def __getitem__(self, steps) -> Structure:
    return Structure(self._source, self._quantity_name, steps=steps)

def _impl_factory(self, raw):
    return _StructureImpl.from_data(raw, steps=self._steps)
```

The Impl applies `slice_steps` explicitly:

```python
from py4vasp._core import slice_steps

class _StructureImpl:
    def __init__(self, raw, steps=None):
        self._raw = raw
        self._steps = steps

    def read(self) -> dict:
        return {
            "lattice_vectors": slice_steps(
                np.array(self._raw.lattice_vectors), self._steps, default_ndim=2
            ),
        }
```

`slice_steps(data, steps, default_ndim)` rules:
- `steps=None` → last step (default)
- `steps=3` → single step
- `steps=slice(1, 8)` → range
- `data.ndim <= default_ndim` → no step axis, return unchanged

### 8 — Port `__str__` and display methods

Move the string formatting logic to the Impl. The dispatcher calls it through dispatch:

```python
# Impl
class _BandgapImpl:
    def __str__(self):
        template = """..."""
        return template.format(...)

# Dispatcher
class Bandgap:
    def __str__(self):
        return merge_strings(
            self._source, self._quantity_name, None,
            self._impl_factory, _BandgapImpl.__str__,
        )
```

### 9 — Port `_to_database`

Same dispatch pattern. The Impl has the `_to_database` method:

```python
# Impl
class _BandgapImpl:
    def _to_database(self) -> dict:
        bandgap_dict = {...}
        return {"bandgap": Bandgap_DB(**final_dict)}

# Dispatcher
class Bandgap:
    def _read_to_database(self, *args, **kwargs):
        return merge_dicts(
            self._source, self._quantity_name, None,
            self._impl_factory, _BandgapImpl._to_database,
        )
```

### 10 — Port the tests

**Never remove an existing test.** If a test cannot work yet (e.g. because the
dispatcher infrastructure isn't fully wired or factory methods changed), mark it
with `@pytest.mark.skip(reason="...")` so it remains visible and will be
re-enabled later.

```python
@pytest.mark.skip(reason="Dispatcher not yet wired to Calculation")
def test_factory_methods(raw_data, check_factory_methods):
    ...
```

Tests split into two categories:

**Unit tests (Impl directly, no I/O):**

```python
def test_bandgap_read():
    raw = raw_data.Bandgap(labels=..., values=...)
    impl = _BandgapImpl.from_data(raw)
    result = impl.read()
    assert ...

def test_bandgap_step_selection():
    raw = raw_data.Bandgap(labels=..., values=...)
    impl = _BandgapImpl.from_data(raw, steps=3)
    result = impl.read()
    assert ...
```

**Integration tests (full pipeline via DictSource):**

```python
def test_bandgap_via_calculation():
    source = DictSource({"bandgap": raw_data.Bandgap(...)})
    calc = Calculation(source=source)
    result = calc.bandgap.read()
    assert ...
```

The existing `from_data` pattern in tests like:
```python
bandgap = Bandgap.from_data(raw_gap)
```
should be migrated to:
```python
impl = _BandgapImpl.from_data(raw_gap)
```

Or for testing the dispatcher with the full dispatch pipeline:
```python
source = DataSource(raw_gap)
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
- [ ] Impl class created: `_<Name>Impl` with `from_data(raw, steps=None)`
- [ ] All transform logic moved from Refinery to Impl
- [ ] `self._raw_data.x` → `self._raw.x` in Impl
- [ ] `@base.data_access` decorators removed
- [ ] Dispatcher class created with `@quantity("name")` decorator
- [ ] Dispatcher calls `merge_graphs` / `merge_dicts` — no lambdas
- [ ] Impl method passed as unbound reference: `_BandImpl.read`
- [ ] Extra args forwarded via `*args, **kwargs`
- [ ] Step indexing: `__getitem__` on dispatcher, `_impl_factory` captures steps
- [ ] Composition: other Impl's `from_data` called directly
- [ ] `__str__` / `_repr_pretty_` ported through dispatch
- [ ] `_to_database` ported through dispatch
- [ ] Small mixins kept (e.g. `graph.Mixin`)
- [ ] `base.Refinery` and `slice_.Mixin` removed from inheritance
- [ ] Tests split: Impl unit tests + dispatcher integration tests
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
