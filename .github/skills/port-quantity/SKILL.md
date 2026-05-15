---
name: port-quantity
description: "Port a py4vasp quantity from the inheritance-based Refinery architecture to the new composition-based DataAccess[T] architecture. USE WHEN: migrating an existing quantity class, porting tests, or adding a new quantity. Triggers: 'port quantity', 'migrate to new architecture', 'convert to composition', 'new architecture', 'refactor quantity'."
---

# Port a py4vasp Quantity to the Composition Architecture

Port an existing `Refinery`-based quantity to the new architecture described in `docs/architecture/calculation.rst`. The prototype is in `notebook/ArchitectureVariant.ipynb`.

## Core Design Contract

**Quantities are plain classes** that own a `DataAccess[T]` and call it as an **iterable**.

### What `DataAccess.__call__` returns

`DataAccess.__call__(selection)` returns an **iterable of `DataContext[T]`** — one per matched source. Each `DataContext` carries raw data and selection metadata:

```python
class DataContext(Generic[T]):
    selection_name: str | None       # resolved source name (e.g. "kpoints_opt", or None = default)
    remaining_selection: str | None  # selection string after the source part is removed

    def access_data(self) -> ContextManager[T]: ...  # explicit access
    def __iter__(self): ...  # supports tuple unpacking as (raw, self)
```

This replaces the hidden state that `_FunctionWrapper` previously managed via `self._data_context.selection` and the rewritten `selection=` kwarg passed to the inner function.

**Usage pattern — most methods (context not needed):**

```python
def read(self) -> dict:
    for raw, _ in self._data():
        return {"fermi_energy": raw.fermi_energy, ...}
```

**Usage pattern — selection forwarding (context needed):**

```python
def to_dict(self, selection: str | None = None, ...) -> dict:
    for raw, ctx in self._data(selection=selection):
        projections = self._projector(raw).project(ctx.remaining_selection, raw.projections)
        return {...}
```

**Usage pattern — multi-source (loop runs multiple times):**

```python
def to_dict(self, selection=None) -> dict:
    results = {}
    for raw, ctx in self._data(selection=selection):
        results[ctx.selection_name or "default"] = self._process(raw)
    if len(results) == 1:
        return next(iter(results.values()))
    return results
```

`DataAccess.__call__` internally reproduces the `_FunctionWrapper` source-resolution logic:
1. Parse `selection` via `select.Tree`.
2. Look up matching source names against the schema.
3. Remove the matched source from the selection; reassemble remainder as a string.
4. Call `source.access(quantity_name, source_name)` to enter the HDF5 context.
5. Yield `DataContext(raw=data, selection_name=source_name, remaining_selection=remainder)`.

For single-source selections (the common case), the loop body runs once. For multiple sources (e.g. `"kpoints_opt, default"`), it runs once per source.

### `from_data` constructor (unchanged public API)

```python
@classmethod
def from_data(cls, raw: RawBand) -> Band:
    return cls(data=DataAccess.from_data(raw))
```

`DataAccess.from_data(raw)` wraps the object in a `DataSource` that yields it unchanged. The `DataContext` will have `selection_name=None` and `remaining_selection` equal to the original `selection` argument — because there is no source to strip when data is injected directly. The loop always runs exactly once.

---

## Migration Procedure

### 1 — Identify the raw dataclass

Open `src/py4vasp/_raw/data.py`. Find the dataclass matching this quantity (CamelCase of the quantity name). This becomes `T` in `DataAccess[T]`. Note all fields and their types — they will be used via `raw.<field>` inside the `for` loop.

Example for `band`:
```python
@dataclasses.dataclass
class Band:
    dispersion: Dispersion
    fermi_energy: float
    occupations: VaspData
    projectors: Projector
    projections: VaspData = NONE()
```

### 2 — Rewrite the class header

Remove `base.Refinery` from the inheritance list. Keep small mixins (e.g. `graph.Mixin`). Add the `@quantity()` decorator.

```python
# Before
class Band(base.Refinery, graph.Mixin):
    _raw_data: raw_data.Band

# After
from py4vasp._core import DataAccess, quantity

@quantity("band")                         # or @quantity("dos", group="phonon")
class Band(graph.Mixin):
    def __init__(self, data: DataAccess[raw_data.Band]):
        self._data = data

    @classmethod
    def from_data(cls, raw: raw_data.Band) -> Band:
        return cls(data=DataAccess.from_data(raw))
```

For step-indexed quantities (structure, energy, force, …) also add:
```python
    def __init__(self, data: DataAccess[raw_data.Structure], steps=None):
        self._data = data
        self._steps = steps

    def __getitem__(self, steps) -> Structure:
        return Structure(data=self._data, steps=steps)
```

### 3 — Replace every `@base.data_access` method

For each method decorated with `@base.data_access`:

1. **Remove** the decorator.
2. **Iterate** with `for raw, _ in self._data(selection=selection):` (use `ctx` instead of `_` when selection info is needed).
3. **Replace** `self._raw_data.<field>` with `raw.<field>`.
4. **Replace** the remaining selection (old transparent kwarg injection) with `ctx.remaining_selection`.
5. **Use** `ctx.selection_name` anywhere the old code used `self._selection`.

```python
# Before
@base.data_access
def to_dict(self, selection: Optional[str] = None, fermi_energy=None) -> dict:
    dispersion = self._dispersion().read()
    fermi_e = self._raw_data.fermi_energy
    projections = self._read_projections(selection)   # selection = remaining part
    return {...}

# After — context needed (selection forwarding)
def to_dict(self, selection: str | None = None, fermi_energy=None) -> dict:
    for raw, ctx in self._data(selection=selection):
        dispersion = self._dispersion(raw).read()
        fermi_e = raw.fermi_energy
        projections = self._read_projections(ctx.remaining_selection, raw)
        return {...}

# After — context not needed
def to_graph(self, fermi_energy=None) -> graph.Graph:
    for raw, _ in self._data():
        return self._dispersion(raw).plot(fermi_energy=fermi_energy)
```

### 4 — Update internal helpers

Private helpers (`_dispersion`, `_projector`, `_kpoint`, `_read_projections`, etc.) currently read `self._raw_data` directly. Pass raw data as an explicit argument instead, since the context is only open inside the calling public method.

```python
# Before
def _dispersion(self):
    return _dispersion.Dispersion.from_data(self._raw_data.dispersion)

# After
def _dispersion(self, raw: raw_data.Band):
    return _dispersion.Dispersion.from_data(raw.dispersion)
```

Call from public methods:
```python
for raw, _ in self._data(selection=selection):
    graph = self._dispersion(raw).plot(...)
```

### 5 — Port `_to_database`

Same pattern as public methods — open the context, use `ctx.raw`:

```python
# Before
@base.data_access
def _to_database(self, selection=None, **kwargs) -> dict:
    dispersion = self._dispersion()._read_to_database(**kwargs)
    fermi_e = self._raw_data.fermi_energy
    return database.combine_db_dicts({"band": Band_DB(fermi_energy=fermi_e, ...)}, dispersion)

# After
def _to_database(self, selection=None, **kwargs) -> dict:
    for raw, _ in self._data(selection=selection):
        dispersion = self._dispersion(raw)._read_to_database(**kwargs)
        return database.combine_db_dicts(
            {"band": Band_DB(fermi_energy=raw.fermi_energy, ...)},
            dispersion,
        )
```

### 6 — Handle composition with other quantities

Use `from_data` with the relevant raw sub-field — same as before:

```python
for raw, _ in self._data():
    structure = Structure.from_data(raw.structure)
    lattice = structure.to_dict()
```

### 7 — Step-indexed quantities

Apply `slice_steps` explicitly in each method. Import from `py4vasp._core`:

```python
from py4vasp._core import slice_steps

def to_dict(self) -> dict:
    for raw, _ in self._data():
        return {
            "lattice_vectors": slice_steps(
                np.array(raw.lattice_vectors), self._steps, single_step_ndim=2
            ),
            "positions": slice_steps(
                np.array(raw.positions), self._steps, single_step_ndim=2
            ),
            "elements": raw.elements,
        }
```

`slice_steps(data, steps, single_step_ndim)` rules:
- `steps=None` → last step (default)
- `steps=3` → single step
- `steps=slice(1, 8)` → range
- `data.ndim <= single_step_ndim` → no step axis, return unchanged

### 8 — Port the tests

Tests using `QuantityClass.from_data(raw)` are unchanged. Only remove references to `_data_context` or `_raw_data` internals.

To test selection forwarding, use a `SpySource`:

```python
from contextlib import contextmanager

class SpySource:
    def __init__(self, raw):
        self._raw, self.calls = raw, []

    @contextmanager
    def access(self, quantity, selection=None):
        self.calls.append({"quantity": quantity, "selection": selection})
        yield self._raw

spy = SpySource(raw_band)
band = Band(data=DataAccess(spy, "band"))
band.read(selection="kpoints_opt")
assert spy.calls[-1]["selection"] == "kpoints_opt"
```

### 9 — Remove from `QUANTITIES` / `GROUPS`

In `src/py4vasp/_calculation/__init__.py`, remove the quantity's string entry from `QUANTITIES` (or `GROUPS`). The `@quantity()` decorator handles registration automatically.

### 10 — Verify

```bash
pytest tests/calculation/test_{name}.py -v   # quantity-level tests
pytest tests/ -x                             # full suite
```

Confirm that `ctx.raw.<field>` autocompletes in the IDE inside `with self._data(...) as ctx:`.

---

## Checklist

For each quantity being ported:

- [ ] Raw dataclass type `T` identified in `_raw/data.py`
- [ ] `base.Refinery` removed; `@quantity("name")` (or `group=`) decorator added
- [ ] `__init__(self, data: DataAccess[T])` added; small mixins kept
- [ ] `from_data(cls, raw: T)` class method added
- [ ] All `@base.data_access` decorators removed
- [ ] `for raw, _ in self._data(...):` — or `raw, ctx` when selection info needed
- [ ] `self._raw_data.x` → `raw.x` inside the `for` loop
- [ ] Remaining `selection` arg → `ctx.remaining_selection`
- [ ] `self._selection` (source name) → `ctx.selection_name`
- [ ] Private helpers accept `raw` as explicit argument
- [ ] `_to_database` migrated with `for raw, _ in self._data(...):` pattern
- [ ] Step indexing added if applicable (`__getitem__`, `self._steps`, `slice_steps()`)
- [ ] Composition via `OtherQuantity.from_data(ctx.raw.subfield)`
- [ ] Tests pass; no references to `_data_context` or `_raw_data` internals
- [ ] Removed from `QUANTITIES`/`GROUPS` in `__init__.py`
- [ ] IDE autocomplete works on `raw.<field>` inside the `for` loop

---

## Reference Files

| File | Purpose |
|------|---------|
| `docs/architecture/calculation.rst` | Full architecture description |
| `notebook/ArchitectureVariant.ipynb` | Runnable prototype |
| `src/py4vasp/_raw/data.py` | Raw dataclass definitions |
| `src/py4vasp/_raw/definition.py` | Schema (sources per quantity) |
| `src/py4vasp/_calculation/data_access.py` | Production DataAccess implementation |
| `tests/calculation/test_data_access.py` | DataAccess test suite |
| `src/py4vasp/_calculation/band.py` | Reference: complex existing quantity |
| `tests/calculation/test_band.py` | Reference: test structure to preserve |
