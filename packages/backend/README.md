# vasp-backend

Internal py4vasp interface for the tools delivered by VASP.

py4vasp keeps its database interface private because it is not meant to be called by
users. The tools that VASP builds on top of py4vasp do need it, so this package exposes a
small, curated and documented surface over those internals:

~~~python
from py4vasp import Calculation
from vasp import backend

data = backend.to_database(Calculation.from_path("."))
~~~

This is not part of py4vasp's public API. It is released in lockstep with py4vasp, pins
`py4vasp-core` exactly, and carries no stability guarantees for anyone outside VASP. If
you are looking for py4vasp itself, see https://vasp.at/py4vasp/latest.
