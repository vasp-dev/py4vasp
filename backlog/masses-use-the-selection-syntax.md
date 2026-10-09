# `masses=` should take a mapping keyed like a DOS selection, not a list

`force_constant.frequencies` / `displacements` / `to_molden` and `phonon.mode.displace`
/ `displacements` / `to_view` accept `masses` as either a mapping from element to mass
or a sequence with one mass per atom (`_util/masses.py::resolve`). The sequence is the
only way to change a single site, and it forces the user to repeat the standard weight
of every other atom (the `Mass` docstring builds such a list from
`STANDARD_ATOMIC_WEIGHTS`).

Preferred: one form only, a mapping whose keys follow the selection the DOS and band
projections already use, so `{"O": 17.999}` changes every oxygen and `{"4": 17.999}`
only the fourth atom. `Stoichiometry._specific_selection` (`_stoichiometry.py:152`)
already produces exactly those keys (element and 1-based atom index) with their
indices, so the mapping can be resolved against it instead of against the list of
elements. To decide:

- precedence when both an element and one of its atoms are given (the more specific
  key should win, independent of the order in the dict);
- whether ranges such as `"1:3"` are accepted, as in `dos.plot("1:3")`;
- the error for a key that is neither, which should reuse `suggest.did_you_mean`.

Dropping the sequence needs no deprecation: `masses=` arrived with #330 and #344 after
v0.11.3, so it has never been released. Touches the six methods' docstrings, the `Mass`
example, `tests/util/test_masses.py` and the parametrized mass tests of
`test_force_constant.py` and `test_phonon_mode.py`.
