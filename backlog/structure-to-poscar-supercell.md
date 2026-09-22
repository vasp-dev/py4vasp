# `to_POSCAR` has no `supercell` argument, so building a supercell needs pymatgen

`structure.to_view(supercell=...)` (`structure.py:153`) and `structure.to_ase(supercell=...)`
(`structure.py:197`) both take a supercell. `structure.to_POSCAR(ion_types=None)`
(`structure.py:145`, and the slice version at `:1177`) does not. So py4vasp can *show* you
a supercell and can hand one to ASE, but it cannot write one to the file VASP reads.

Every phonon calculation starts by writing a supercell POSCAR. In one workshop tutorial
that gap pulled `pymatgen.core.Structure` and `pymatgen.io.vasp.Poscar` into nine cells
across two notebooks — a heavyweight dependency in a VASP tutorial, for a three-line
operation — purely to call `make_supercell` and `write_file`.

Adding `supercell=` to `to_POSCAR` (and a `write_POSCAR(path, supercell=...)`
convenience) collapses all of it to
`calc.structure.to_POSCAR(supercell=[n, n, 1])`. The repetition logic exists already in
`to_ase`; this is mostly wiring it to the POSCAR writer and agreeing what a scalar, a
length-3 list and a 3×3 matrix each mean, consistently with `to_view`/`to_ase`.

Of everything a phonon tutorial reaches outside py4vasp for, this is the clearest case of
something that belongs inside it.
