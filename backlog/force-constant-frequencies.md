# `force_constant` cannot give you a frequency, and `to_molden` puts eigenvalues where cm⁻¹ belong

`eigenvectors()` (`force_constant.py:70`) is the only way into the diagonalization, and
nothing returns what the vectors are eigenvectors *of*. To get phonon frequencies — the
reason one computes force constants — a user has to take `read()["force_constants"]`, call
`np.linalg.eigvalsh` themselves, supply the masses, and supply the unit conversion. The
eigenvalues are computed anyway in `_diagonalize` (`force_constant.py:87`) and then thrown
away everywhere except `to_molden`.

Worse, `to_molden` (`force_constant.py:113`) writes those eigenvalues straight into the
`[FREQ]` block, a slot the molden format defines as wavenumbers in cm⁻¹. On a graphene
6×6×1 supercell the block reads `119.604146` where the answer is ≈1646 cm⁻¹; the top
optical branch is labelled "119.6 cm⁻¹" by any viewer that opens the file. Because the
stored number is λ rather than √(λ/m), no constant rescaling repairs the file — a user who
distrusts the numbers cannot fix them, and one who trusts them gets a plausible-looking
wrong answer. A simulated user who found this judged that an ordinary user would *not*
notice, because people open molden to watch the animation and read the label off the side.
The docstring warns that the eigenvectors ignore the masses, which is about the vectors and
reads as reassurance about the frequencies.

Suggested approach: add `eigenvalues()` returning the eigenvalues in eV/Å², and derive the
`[FREQ]` block from the mass-weighted dynamical matrix instead, converted to cm⁻¹. The
masses are already reachable through the structure, so both are small; the decision worth
making deliberately is whether `eigenvectors()` should gain a mass-weighting option at the
same time, since the mass-weighted eigenvectors are the actual normal modes and the current
ones are not.

Related to [frequency-unit-keyword], which proposes the `unit=` keyword this would use, and
to [public-unit-constants] for the conversion factor itself.
