Calculation Architecture
========================

This document describes the internal architecture of the ``Calculation`` class
and how quantities access raw data. The design uses **composition over
inheritance**: quantities are plain classes that own a generic ``DataAccess[T]``
object rather than inheriting from a base class.

Overview
--------

.. code-block:: text

   Calculation ──→ Source ──→ DataAccess[T]() ──→ iterable of DataContext[T]
                    ↑              ↑
       FileSource | DictSource    Generic: carries the raw dataclass type

Components:

1. **Source** — where data comes from (file vs. in-memory)
2. **DataAccess[T]** — typed, callable iterable for raw data
3. **Quantities** — plain classes owning a ``DataAccess[T]``
4. **Groups** — thin namespaces for nested quantities (one level deep)
5. **Calculation** — top-level entry point, resolves attributes from a registry
6. **Registry** — ``@quantity()`` decorator for declarative registration


Source
------

A ``Source`` provides a context manager that yields raw data for a given
quantity name. Three implementations cover all use cases:

``FileSource(path)``
   Production: opens HDF5 file, yields lazy dataset references.

``DataSource(raw_data)``
   Wraps a single raw data object. Used for unit testing one quantity and for
   composition (passing a data subset to another quantity).

``DictSource(data_dict)``
   Maps quantity names to raw data objects. Used for integration-testing a full
   ``Calculation`` without file I/O.

.. code-block:: python

   class Source(Protocol):
       def access(self, quantity: str, selection: str | None = None):
           ...

   class DataSource:
       def __init__(self, raw_data):
           self._raw_data = raw_data

       @contextmanager
       def access(self, quantity: str, selection: str | None = None):
           yield self._raw_data


DataAccess[T]
-------------

The central generic. Quantities own a ``DataAccess[T]`` and call it as an
**iterable**. The type parameter ``T`` is the raw dataclass type, giving
full autocomplete and type checking on the ``raw`` variable.

Each call returns an iterable of ``DataContext[T]`` objects — one per matched
source. Each ``DataContext`` supports tuple unpacking as ``(raw, context)``.

.. code-block:: python

   T = TypeVar("T")

   class DataContext(Generic[T]):
       selection_name: str | None
       remaining_selection: str | None

       def access_data(self) -> ContextManager[T]: ...
       def __iter__(self): return iter((self._raw, self))

   class DataAccess(Generic[T]):
       def __init__(self, source: Source, quantity_name: str):
           self._source = source
           self._quantity_name = quantity_name

       @classmethod
       def from_data(cls, raw_data: T) -> DataAccess[T]:
           """Wrap raw data directly (testing / composition)."""
           return cls(DataSource(raw_data), quantity_name="")

       def __call__(self, selection: str | None = None) -> Iterator[DataContext[T]]:
           ...

Usage inside a quantity:

.. code-block:: python

   for raw, _ in self._data(selection=selection):
       raw.eigenvalues  # ← typed as RawBand, autocomplete works


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
       structure: RawStructure  # subset passed to Structure.from_data()


Quantities
----------

Quantities are **plain classes** — no base class. They receive a
``DataAccess[T]`` via their constructor and provide a ``from_data`` class
method for direct construction from raw data.

.. code-block:: python

   @quantity("band")
   class Band:
       def __init__(self, data: DataAccess[RawBand]):
           self._data = data

       @classmethod
       def from_data(cls, raw: RawBand) -> Band:
           return cls(data=DataAccess.from_data(raw))

       def read(self, selection: str | None = None) -> dict:
           for raw, _ in self._data(selection=selection):
               return {
                   "eigenvalues": np.array(raw.eigenvalues) - raw.fermi_energy,
                   ...
               }

       def plot(self, selection: str | None = None) -> dict:
           ...

``from_data`` serves two purposes:

- **Testing**: construct a quantity with fake numpy data, no file I/O.
- **Composition**: one quantity passes a raw data subset to another quantity's
  ``from_data``.


Step Indexing
~~~~~~~~~~~~~

Quantities defined over multiple ionic steps support ``__getitem__``. It
returns a new instance sharing the same ``DataAccess`` but storing the step
selection. Data is read lazily — only when ``read()`` or ``plot()`` is called.

.. code-block:: python

   @quantity("structure")
   class Structure:
       def __init__(self, data: DataAccess[RawStructure], steps=None):
           self._data = data
           self._steps = steps

       def __getitem__(self, steps) -> Structure:
           return Structure(data=self._data, steps=steps)

       def read(self) -> dict:
           for raw, _ in self._data():
               return {
                   "lattice_vectors": slice_steps(raw.lattice_vectors, self._steps, single_step_ndim=2),
                   ...
               }

The ``slice_steps`` helper handles three cases:

- ``steps=None`` → return last step (default)
- ``steps=3`` → return single step
- ``steps=slice(1, 8)`` → return range of steps
- Data has no step dimension (``ndim <= single_step_ndim``) → pass through unchanged


Composition Between Quantities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a quantity needs another quantity's logic, it calls ``from_data`` with the
relevant subset of its raw data:

.. code-block:: python

   @quantity("density")
   class Density:
       def read(self) -> dict:
           for raw, _ in self._data():
               structure = Structure.from_data(raw.structure)
               return {
                   "charge": np.array(raw.charge),
                   "structure": structure.read(),
               }


Selection Forwarding
~~~~~~~~~~~~~~~~~~~~

Users pass a ``selection`` argument to methods. It propagates through
``DataAccess.__call__`` to ``Source.access``, which uses it to select the
appropriate dataset:

.. code-block:: python

   calc.band.plot(selection="custom_kpath")
   # → DataAccess.__call__(selection="custom_kpath")
   # → source.access("band", selection="custom_kpath")


Registry & Decorator
--------------------

The ``@quantity()`` decorator registers a class and sets its
``_quantity_name``:

.. code-block:: python

   @quantity("band")                   # → Calculation.band
   @quantity("dos", group="phonon")    # → Calculation.phonon.dos

The registry maps names to classes (top-level) or to dicts of classes (groups):

.. code-block:: python

   _REGISTRY = {
       "band": Band,
       "structure": Structure,
       "phonon": {"dos": PhononDos, "band": PhononBand},
   }


Group
-----

A ``Group`` is a thin namespace. It receives the source and a dict of quantity
classes. On attribute access it creates the quantity with
``DataAccess(source, quantity_name)``.

.. code-block:: python

   class Group:
       def __init__(self, source: Source, quantities: dict[str, type]):
           ...

       def __getattr__(self, name: str):
           cls = self._quantities[name]
           return cls(data=DataAccess(self._source, cls._quantity_name))


Calculation
-----------

``Calculation`` is the public entry point. It resolves attributes from the
registry, instantiating quantities or groups on first access.

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
           return entry(data=DataAccess(self._source, entry._quantity_name))


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
