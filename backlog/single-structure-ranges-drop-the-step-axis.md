# A range of steps adds no step axis to a single structure

A single structure, e.g. read from a POSCAR file or the `phonon`, `final`, or `poscar`
source of a run, takes `[0]`, `[-1]`, and ranges that contain its one step. A range then
returns the structure itself: `structure[:].volume()` is a number and
`lattice_vectors()` a 3x3 matrix. A trajectory keeps the step axis for a range, even
of one step (`structure[-1:]` gives `(1, 3, 3)`), so a script that loops over
`structure[:].volume()` works on a relaxation and fails on a POSCAR with
`TypeError: 'numpy.float64' object is not iterable`. The class docstring of `Structure`
now says so, and `test_phonon_source_is_a_single_structure` pins the current shapes.

Related: `structure[:].to_POSCAR()` on a single structure refuses with "Converting
multiple structures to a POSCAR is currently not implemented", although the range holds
one structure.

Open decision: should a range on a single structure behave exactly like a trajectory of
one step (a step axis, `to_POSCAR` refused like `traj[-1:]`), or keep returning the
structure itself (then `to_POSCAR` should accept it)?
