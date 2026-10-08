# `read()` answers a selection with a bare TypeError where `print()` explains

Several quantities have a `print(selection=None)` and a `read()` without any
parameter, so the same argument gets two different answers:

- `calculation.born_effective_charge.read("default")` raises
  `TypeError: BornEffectiveCharge.read() takes 1 positional argument but 2 were given`,
  while `print` and `to_INCAR` take a selection (`born_effective_charge.py:52`, `:135`).
- `calculation.elastic_modulus.read("relaxed_ion")` raises the same `TypeError`, while
  `print("relaxed_ion")` raises `IncorrectUsage` naming the sources
  (`elastic_modulus.py:256`, `:311`). The user wanted the relaxed-ion tensor, which
  `read()` returns as one of its keys; neither message says so.

A `TypeError` from a method the user called correctly by analogy is the kind of error
py4vasp otherwise converts into `py4vasp.exception`. Either give every `read` the same
`selection` parameter its `print` has, or make the dispatch layer turn a surplus
positional argument into `IncorrectUsage`. The second would cover every quantity at
once. Worth checking which other quantities have the same split.

Found alongside, on `born_effective_charge`:

- the header says "cumulative output" where OUTCAR says "cummulative output"
  (`linear_response.F:967`, `pead.F:2356,2368`), so grepping one for the other fails;
- `print()` and `read()` have no doctest examples, although `to_INCAR` has.

Collected from the reviewer notes of #345 and #352.
