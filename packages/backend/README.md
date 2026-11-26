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

Tools that offer py4vasp's routines in a form describe a routine with `backend.inspect`
(its sources, steps, fields, and selection tokens, in JSON types) and call it with the
values a user entered through `backend.call`, which checks them against that description.
The data a routine reads comes back as dictionaries of numpy arrays and scalars, so a tool
can send the arrays on as raw bytes; `backend.to_tree` brings other data, such as the
records of `backend.to_database`, into the same form.

This is not part of py4vasp's public API. It is released in lockstep with py4vasp, pins
`py4vasp-core` exactly, and carries no stability guarantees for anyone outside VASP. If
you are looking for py4vasp itself, see https://vasp.at/py4vasp/latest.
