# Displacing a structure along a phonon mode is missing, and users get the mass weighting wrong

`phonon.mode` eigenvectors are mass-weighted. Nothing in the API says so, and the
`to_dict` key is simply `"eigenvectors"`, so the natural reading — "displacement pattern
of the mode" — is wrong by a factor 1/√m per ion.

Measured on a real BaTiO₃ run: for the Γ acoustic mode, `e/√m` is constant to 0.0013 while
`e` itself varies by 0.4708, and the Ba:O amplitude ratio 2.96 matches √(m_Ba/m_O) = 2.93.
A workshop tutorial built its frozen-phonon POSCARs straight from `e`, which displaced Ti
by +43.5 % and O by −17.1 % against the true soft mode and carried a spurious rigid
translation 44× too large. Everything downstream — the double-well scan, the relaxation,
the IRC starting point — sat on a distorted coordinate, and it all still *looked* right
because the symmetry was correct.

`phonon.mode.displace(index, amplitude) -> Structure` (or `to_structure(mode, amplitude)`)
would make that error impossible: the un-weighting by √m, the normalization and the
Cartesian→direct conversion happen once, inside. Every piece is already here —
`structure.to_POSCAR()`, `Structure.from_POSCAR()`, `structure.to_ase()`, and the
stoichiometry that carries the masses.

It also removes pymatgen from the frozen-phonon workflow entirely: the tutorial pulled in
that dependency for this one operation. Frozen-phonon, ferroelectric-polarization,
mode-decomposition and anharmonic-sampling studies all want the same call.

Whether or not this lands, `phonon.mode`'s docstring should state that the eigenvectors
are mass-weighted.
