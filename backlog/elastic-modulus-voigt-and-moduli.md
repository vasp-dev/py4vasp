# `elastic_modulus.read()` gives only the rank-4 tensor

`ElasticModulus.read()` returns the `(3, 3, 3, 3)` tensor in kBar. What a user usually
wants is the 6×6 Voigt matrix in GPa and the bulk, shear and Young's moduli. py4vasp
computes all of them already for `to_database`, but keeps them private, so the
Common-tasks recipe for "which elastic constants does my crystal have, in GPa?" has the
user reshape and convert by hand. The printed table, in turn, is a 6×6 matrix with C_66
rather than C_44 in the fourth row and column, which the docstring has to warn about.

Expose the Voigt matrix (in the standard 11, 22, 33, 23, 13, 12 order) and the moduli,
e.g. as `voigt()` and `moduli()` returning GPa, sharing the code `to_database` uses.

From the reviewer notes of #345.
