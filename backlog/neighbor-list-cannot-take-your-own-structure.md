# `neighbor_list` makes a promise about tilted cells that a user cannot check

`NeighborList`'s docstring states that it measures the perpendicular width of the cell
rather than the length of the lattice vectors, "so it stays correct for the tilted cells
where ``d - np.rint(d)`` quietly reports the wrong atom". That claim is the reason to
use it, and a user with a hexagonal cell has no supported way to test it:

- `demo.calculation` offers `collinear`, `noncollinear`, `metal`, `surface`,
  `spin_texture` and `perovskite`. **None of them is hexagonal**, so the claim cannot be
  exercised on the data py4vasp ships.
- `Structure.from_POSCAR` gives a `Structure`, but nothing connects it to a neighbor
  list. `NeighborList.from_data(structure)` fails with
  `AttributeError: 'function' object has no attribute 'ndim'` -- an internal error, not
  a `py4vasp.exception`, and `from_data`'s one-line docstring ("Create a NeighborList
  dispatcher from raw structure data") never says what a raw structure is or where to
  get one. Building `raw.Structure` by hand then fails on `scale=1.0` with
  `TypeError: 'float' object is not subscriptable`, because it silently requires
  `np.array(1.0)`.

A user simulation reached the hexagonal check only by going through `py4vasp.raw` and
constructing the dataclasses by hand, and rated that a ten-minute detour most users
would abandon. It did confirm the claim -- 832 of 832 pairs agreeing with a brute-force
supercell sum to 1.8e-15 on a γ = 120° graphene cell, where `d - np.rint(d)` finds two
of the four nearest bonds -- but a promise only the authors can verify is not much of a
promise.

Two separable pieces of work:

1. A hexagonal selection for `demo.calculation`, which would let the docstring carry its
   own proof the way `dielectric_function.read` does.
2. A documented route from a structure the user supplies to a neighbor list. Either
   give `from_data` a `Parameters` section and a worked example, or raise
   `IncorrectUsage` naming `from_path` instead of letting an `AttributeError` escape.
   `from_data` appears in `help()` and in tab completion on every quantity, so it reads
   as public whatever its intent.

Related: [quantities-without-showcase-data], [existing-api-users-do-not-find].
