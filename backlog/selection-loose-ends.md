# Loose ends from giving every quantity its selection back

`tests/calculation/test_selection_convention.py` now decides which methods take a
`selection`. The review and the simulated user of that change left two points open.

- **The command line cannot pick a source.** `py4vasp convert structure` always uses
  the default structure; there is no option for the `final`, `exciton`, `phonon` or
  `poscar` sources the Python interface now reaches.
- **`structure[...]` and `density[...]` mean different things.** Indexing a structure
  selects steps, indexing a density selects a source. Both are documented, but a user
  who learned one expects the other to work the same way.

Open question found earlier: `born_effective_charge.print()` says "cumulative output"
where OUTCAR says "cummulative output" (`linear_response.F:967`, `pead.F:2356,2368`), so
grepping one for the other fails. Copy VASP's spelling, or keep the correct one?
