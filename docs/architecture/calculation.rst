Calculation Architecture
========================

This document describes the internal architecture of the ``Calculation`` class
and how quantities access raw data. The design uses **composition over
inheritance**: quantities are split into a public dispatcher class and a private
implementation class. The dispatcher handles multi-selection parsing and
iteration; the implementation works with a single raw data object.

Overview
--------

.. code-block:: text

   Calculation ──→ Source ──→ Band (dispatcher)
                    ↑              │
       FileSource | DataSource    │ for each selection:
                                  │   source.access(name, sel)
                                  ▼
                              _BandImpl (single raw data)
                                  │
                                  ▼
                              transform → result

   Multiple selections → {selection: result} dict (or custom merge)

Components:

1. **Source** — where data comes from (file vs. in-memory)
2. **Dispatcher (public)** — handles selection parsing, iteration, merging
3. **Impl (private)** — constructed via ``from_data(raw)``, works on one raw dataset
4. **Groups** — thin namespaces for nested quantities (one level deep)
5. **Calculation** — top-level entry point, resolves attributes from a registry
6. **Registry** — ``@quantity()`` decorator for declarative registration


Source
------

A ``Source`` provides a context manager that yields **one** raw data object for
a given quantity name and selection. No iteration — one call, one dataset.

``FileSource(path)``
   Production: opens HDF5 file at ``path``, yields lazy dataset references for
   the requested quantity/selection.

``DataSource(raw_data)``
   Wraps a single raw data object directly. Used for unit testing an Impl class
   and for composition (passing a data subset to another quantity).

``DictSource(data_dict)``
   Maps ``(quantity_name, selection)`` pairs to raw data objects. Used for
   integration-testing a full ``Calculation`` without file I/O.

.. code-block:: python

   class Source(Protocol):
       @contextmanager
       def access(self, quantity: str, selection: str | None = None) -> Iterator[T]:
           """Yield one raw data object for the given quantity and selection."""
           ...

   class DataSource:
       """Wraps a single raw data object. Ignores quantity/selection."""
       def __init__(self, raw_data):
           self._raw_data = raw_data

       @contextmanager
       def access(self, quantity: str, selection: str | None = None):
           yield self._raw_data

   class DictSource:
       """Maps quantity names (with optional selection) to raw data."""
       def __init__(self, data: dict):
           self._data = data  # e.g. {"band": raw_band, ("band", "up"): raw_band_up}

       @contextmanager
       def access(self, quantity: str, selection: str | None = None):
           key = (quantity, selection) if selection else quantity
           if key not in self._data:
               key = quantity  # fall back to unselected
           yield self._data[key]


Raw Data Definitions
--------------------

Each quantity has a corresponding dataclass. Fields hold either lazy HDF5
dataset references (production) or numpy arrays (testing).

.. code-block:: python

   @dataclass
   class RawBand:
       kpoint_distances: np.ndarray
       eigenvalues: np.ndarray
       fermi_energy: float
       kpoint_labels: list[str] | None = None

For composition, a raw dataclass can embed another:

.. code-block:: python

   @dataclass
   class RawDensity:
       charge: np.ndarray
       structure: RawStructure  # subset passed to _StructureImpl.from_data()


Quantities — The Dispatcher/Impl Split
---------------------------------------

Each quantity is split into two classes:

- **Dispatcher** (public, e.g. ``Band``) — the user-facing class attached to
  ``Calculation``. It owns the ``Source``, parses multi-selections, calls
  ``source.access()`` for each individual selection, constructs the Impl, and
  merges results. The dispatcher does **not** have ``from_data``; that lives
  exclusively on the Impl.

- **Impl** (private, e.g. ``_BandImpl``) — constructed via ``from_data(raw)``.
  Works with exactly one raw data object. Contains all transform logic. This is
  the primary unit-testing target.


Selection Context
~~~~~~~~~~~~~~~~~

``_parse_selections`` returns a list of ``SelectionContext`` named tuples, each
carrying the resolved source name and the remaining (unconsumed) selection
string:

.. code-block:: python

   class SelectionContext(typing.NamedTuple):
       selection_name: str | None       # resolved source (e.g. "kpoints_opt")
       remaining_selection: str | None  # leftover after source is stripped

   def _parse_selections(selection: str | None) -> list[SelectionContext]:
       """Parse a user selection into individual (source, remainder) pairs.

       Uses the schema to identify which part of the selection refers to a data
       source and which part is forwarded to the Impl method.
       """
       if selection is None:
           return [SelectionContext(None, None)]
       tree = select.Tree.from_selection(selection)
       result = []
       for sel in tree.selections():
           source_name, remaining = _match_source(sel)
           result.append(SelectionContext(source_name, remaining))
       return result


Standalone Dispatch Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The dispatch logic is a set of standalone functions — not a method on any
class. Each quantity's public methods call one of these. All extra arguments
from the dispatcher method are forwarded to the Impl method.

The core building block is ``_dispatch``, which iterates over parsed
selections and calls the Impl method for each one:

.. code-block:: python

   def _dispatch(source, quantity_name, selection, impl_factory, method, *args, **kwargs):
       """Core dispatch: parse selections, call method for each, collect results.

       Parameters
       ----------
       source : Source
           The data source (FileSource, DictSource, etc.).
       quantity_name : str
           Name used to look up data in the source.
       selection : str | None
           User-provided selection string (may contain multiple comma-separated items).
       impl_factory : callable(raw, ...) -> Impl
           Typically ``_BandImpl.from_data``. Called with the raw data object.
       method : unbound method reference
           The Impl method to call, e.g. ``_BandImpl.read``.
       *args, **kwargs
           Extra arguments forwarded to ``method(impl, *args, **kwargs)``.

       Returns
       -------
       dict[str, result]
           Maps selection_name (or "default") to each result.
       """
       contexts = _parse_selections(selection)
       results = {}
       for ctx in contexts:
           with source.access(quantity_name, selection=ctx.selection_name) as raw:
               impl = impl_factory(raw)
               result = method(impl, *args, **kwargs)
               key = ctx.selection_name or "default"
               results[key] = result
       return results

Higher-level helpers differ only in how they merge the results. Each calls
``_dispatch`` internally and is named after its merge strategy:

.. code-block:: python

   def merge_dicts(source, quantity_name, selection, impl_factory, method, *args, **kwargs):
       """Dispatch and merge dict results into a single combined dict.

       **Default for every method that returns a dict.** Use this almost always.
       Keys are prefixed with the selection name when multiple selections are present.
       """
       results = _dispatch(source, quantity_name, selection, impl_factory, method, *args, **kwargs)
       if len(results) == 1:
           return next(iter(results.values()))
       combined = {}
       for sel, d in results.items():
           for k, v in d.items():
               combined[f"{k}_{sel}"] = v
       return combined

   def merge_graphs(source, quantity_name, selection, impl_factory, method, *args, **kwargs):
       """Dispatch and merge Graph results into a single overlay Graph.

       Use for methods that return a ``Graph`` (typically ``plot``).
       """
       results = _dispatch(source, quantity_name, selection, impl_factory, method, *args, **kwargs)
       all_series = []
       for sel, graph in results.items():
           for series in graph.series:
               series.label = sel
               all_series.append(series)
       return Graph(series=all_series)

   def merge_single(source, quantity_name, selection, impl_factory, method, *args, **kwargs):
       """Dispatch and unwrap — EXCEPTION only.

       Use only when the method must return exactly one element and returning a
       dict for multiple selections would be wrong. This is rare.
       """
       results = _dispatch(source, quantity_name, selection, impl_factory, method, *args, **kwargs)
       if len(results) == 1:
           return next(iter(results.values()))
       return results

Choose the merge helper as follows:

- ``merge_dicts`` — **the default**. Use for every method that returns a ``dict``.
- ``merge_graphs`` — for methods that return a ``Graph`` (typically ``plot``).
- ``merge_single`` — **exception only**. Use when exactly one return element is
  required and a dict-of-results would not make sense.


Example: Band Quantity
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # --- Impl: single raw data, pure transform logic ---

   class _BandImpl:
       """Processes a single band dataset. Testable without file I/O."""

       def __init__(self, raw: RawBand):
           self._raw = raw

       @classmethod
       def from_data(cls, raw: RawBand) -> _BandImpl:
           return cls(raw)

       def read(self) -> dict:
           return {
               "kpoint_distances": np.array(self._raw.kpoint_distances),
               "eigenvalues": np.array(self._raw.eigenvalues) - self._raw.fermi_energy,
           }

       def to_dict(self) -> dict:
           return self.read()

       def plot(self) -> Graph:
           data = self.read()
           return Graph(series=[Series(x=data["kpoint_distances"], y=data["eigenvalues"])])

   # --- Dispatcher ---

   @quantity("band")
   class Band:
       """Public band structure quantity. Handles selection dispatch."""

       def __init__(self, source: Source, quantity_name: str = "band"):
           self._source = source
           self._quantity_name = quantity_name

       def read(self, selection: str | None = None) -> dict | dict[str, dict]:
           return merge_dicts(
               self._source, self._quantity_name, selection,
               _BandImpl.from_data, _BandImpl.read,
           )

       def to_dict(self, selection: str | None = None) -> dict | dict[str, dict]:
           """``to_dict`` is part of the public interface. Always keep it.
           In the new architecture, it delegates to ``read()``."""
           return self.read(selection=selection)

       def plot(self, selection: str | None = None) -> Graph:
           return merge_graphs(
               self._source, self._quantity_name, selection,
               _BandImpl.from_data, _BandImpl.plot,
           )

Note: ``_BandImpl.read`` is passed as an unbound method reference — the
dispatch function calls it as ``method(impl)``. Extra ``*args`` and
``**kwargs`` from the dispatcher are forwarded, e.g.:

.. code-block:: python

   # Dispatcher method with extra arguments
   def plot(self, selection=None, fermi_energy=None):
       return merge_graphs(
           self._source, self._quantity_name, selection,
           _BandImpl.from_data, _BandImpl.plot,
           fermi_energy=fermi_energy,
       )

   # Impl method receives them
   class _BandImpl:
       def plot(self, fermi_energy=None) -> Graph:
           ...

.. note::

   ``@base.data_access`` in the old Refinery architecture injects a
   ``selection`` argument into every decorated method at runtime. After porting,
   **every** dispatcher method must declare ``selection: str | None = None`` in
   its signature — even if the original Refinery method had no ``selection``
   parameter.

   Type hints on ``_raw`` and in ``from_data`` are **mandatory**. Always use
   the exact raw dataclass type from ``_raw/data.py``. Example::

       def __init__(self, raw: raw_data.Band):
           self._raw = raw

       @classmethod
       def from_data(cls, raw: raw_data.Band) -> _BandImpl:
           return cls(raw)


Separation of Concerns
~~~~~~~~~~~~~~~~~~~~~~

=============================  ============================  ==========================
Responsibility                 Dispatcher (``Band``)         Impl (``_BandImpl``)
=============================  ============================  ==========================
Owns the Source                ✓
Calls dispatch functions       ✓
Parses multi-selection         (done by ``_parse_selections``)
Opens data (context manager)   (done by ``_dispatch``)
Constructs Impl from raw       (done by ``_dispatch``)
Transform logic                                              ✓
Testable without I/O                                         ✓ (via ``from_data``)
=============================  ============================  ==========================


Step Indexing
~~~~~~~~~~~~~

Step indexing lives on the **dispatcher**. ``__getitem__`` returns a copy of
the dispatcher with the step selection stored. The dispatcher passes steps to
the Impl via a partial ``impl_factory``:

.. code-block:: python

   class _StructureImpl:
       def __init__(self, raw: RawStructure, steps=None):
           self._raw = raw
           self._steps = steps

       @classmethod
       def from_data(cls, raw: RawStructure, steps=None) -> _StructureImpl:
           return cls(raw, steps=steps)

       def read(self) -> dict:
           return {
               "lattice_vectors": slice_steps(self._raw.lattice_vectors, self._steps, default_ndim=2),
               "positions": slice_steps(self._raw.positions, self._steps, default_ndim=2),
           }

   @quantity("structure")
   class Structure:
       def __init__(self, source: Source, quantity_name: str = "structure", steps=None):
           self._source = source
           self._quantity_name = quantity_name
           self._steps = steps

       def __getitem__(self, steps) -> Structure:
           return Structure(self._source, self._quantity_name, steps=steps)

       def _impl_factory(self, raw):
           return _StructureImpl.from_data(raw, steps=self._steps)

       def read(self, selection: str | None = None) -> dict:
           return merge_dicts(
               self._source, self._quantity_name, selection,
               self._impl_factory, _StructureImpl.read,
           )

The ``slice_steps`` helper handles:

- ``steps=None`` → return last step (default)
- ``steps=3`` → return single step
- ``steps=slice(1, 8)`` → return range of steps
- Data has no step dimension (``ndim <= default_ndim``) → pass through unchanged


Composition Between Quantities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a quantity needs another quantity's logic, it uses the other Impl's
``from_data`` directly — no Source needed:

.. code-block:: python

   class _DensityImpl:
       def __init__(self, raw: RawDensity):
           self._raw = raw

       def read(self) -> dict:
           structure = _StructureImpl.from_data(self._raw.structure)
           return {
               "charge": np.array(self._raw.charge),
               "structure": structure.read(),
           }

This keeps composition simple: one Impl calls another Impl's ``from_data``.


Selection Parsing
~~~~~~~~~~~~~~~~~

Selection parsing is described in the `Selection Context`_ section above.
``_parse_selections`` uses the schema to separate the source part from the
remaining selection. The ``SelectionContext.remaining_selection`` is forwarded
to the Impl method via ``**kwargs`` when the Impl method accepts a
``selection`` parameter.

Merging is handled by choosing the appropriate dispatch helper:

- ``merge_dicts`` — **the default**. Use for every method that returns a ``dict``.
- ``merge_graphs`` — overlays Graph results into one figure (for ``plot``).
- ``merge_single`` — **exception only**, when exactly one return element is required.

``to_dict`` is part of the public interface and must never be deprecated.
In the new architecture, the Impl's ``to_dict`` delegates to ``read()``, and
the dispatcher's ``to_dict`` delegates to ``read(selection=selection)``. Tests
should verify that ``to_dict`` and ``read`` return the same result.


Registry & Decorator
--------------------

The ``@quantity()`` decorator registers the **dispatcher** class:

.. code-block:: python

   @quantity("band")                   # → Calculation.band
   @quantity("dos", group="phonon")    # → Calculation.phonon.dos

The registry maps names to dispatcher classes (top-level) or to dicts (groups):

.. code-block:: python

   _REGISTRY = {
       "band": Band,
       "structure": Structure,
       "phonon": {"dos": PhononDos, "band": PhononBand},
   }


Group
-----

A ``Group`` is a thin namespace. It receives the source and a dict of
dispatcher classes. On attribute access it instantiates the dispatcher.

.. code-block:: python

   class Group:
       def __init__(self, source: Source, quantities: dict[str, type]):
           ...

       def __getattr__(self, name: str):
           cls = self._quantities[name]
           return cls(source=self._source, quantity_name=cls._quantity_name)


Calculation
-----------

``Calculation`` is the public entry point. Immutable after construction.
Resolves attributes from the registry, instantiating dispatchers or groups on
first access.

.. code-block:: python

   class Calculation:
       @classmethod
       def from_path(cls, path: str = ".") -> Calculation:
           return cls(source=FileSource(path))

       @classmethod
       def from_data(cls, data: dict) -> Calculation:
           return cls(source=DictSource(data))

       def __getattr__(self, name: str):
           entry = _REGISTRY[name]
           if isinstance(entry, dict):
               return Group(self._source, entry)
           return entry(source=self._source, quantity_name=entry._quantity_name)


Testing Patterns
----------------

**Unit test an Impl (Approach A — no I/O, no Source):**

.. code-block:: python

   def test_band_read():
       raw = RawBand(
           kpoint_distances=np.array([0, 0.5, 1.0]),
           eigenvalues=np.array([[0.0, 1.0, 2.0]]),
           fermi_energy=0.5,
       )
       impl = _BandImpl.from_data(raw)
       result = impl.read()
       assert np.allclose(result["eigenvalues"], [[-0.5, 0.5, 1.5]])

**Integration test via Calculation (Approach B — full pipeline, no files):**

.. code-block:: python

   def test_band_plot_multiselection():
       source = DictSource({
           ("band", "up"): RawBand(...),
           ("band", "down"): RawBand(...),
       })
       calc = Calculation(source=source)
       result = calc.band.plot(selection="up, down")
       assert len(result.series) == 2


Public API
----------

The architecture preserves the existing user-facing API:

.. code-block:: python

   calc = Calculation.from_path("path/to/calculation")
   calc.band.plot()
   calc.phonon.dos.read()
   calc.structure[3].read()
   calc.energy[1:8].plot()
   calc.band.plot(selection="custom")


Resolved Decisions
------------------

The following were resolved during the design process:

1. **Dispatcher does not have ``from_data``.**
   Composition uses the Impl directly. External testing uses ``DictSource``.

2. **Steps are passed to the Impl** (not applied after).
   The Impl's ``from_data`` signature is ``from_data(raw, steps=None)``.

3. **DictSource uses tuple keys** — ``(quantity, selection)``, falling back to
   plain ``quantity`` string when selection is None.

4. **Dispatch is a set of standalone functions** — ``_dispatch``,
   ``merge_single``, ``merge_dicts``, ``merge_graphs``. No mixin, no base
   class. All extra arguments from the dispatcher method are forwarded via
   ``*args, **kwargs``.

5. **Merge helpers are named after their strategy** — ``merge_dicts``, ``merge_graphs``.
   Each calls ``_dispatch`` internally. No generic ``dispatch_merge`` with a merge 
   parameter — the function name *is* the strategy.

6. **``_parse_selections`` returns ``SelectionContext`` tuples** carrying
   ``(selection_name, remaining_selection)`` — matching the previous
   ``DataContext`` semantics but without iteration over a generic.

7. **Impl methods are passed as unbound references** — ``_BandImpl.read``
   instead of ``lambda impl: impl.read()``. The dispatch function calls
   ``method(impl, *args, **kwargs)``.
