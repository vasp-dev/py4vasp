# MD trajectories of consecutive runs cannot be combined into one

A long molecular-dynamics simulation is usually split into several VASP runs, each
continuing from the CONTCAR of the previous one. py4vasp reads every run as its own
calculation, so to watch the whole trajectory, compute a property along all of it,
or export it, a user has to concatenate the arrays by hand. The same happens for the
two branches of an IRC calculation, which both start at the transition state.

`calculation.reaction_path` already solves this for the distances between pairs of
atoms: `Path.join(other, max_gap=None)` (also `first + second`) appends one path to
another, `path[::-1]` reverses it, and the join raises `IncorrectUsage` if the second
path does not start where the first one ends, which catches a branch that was not
reversed. Offer the same for whole structures and for the viewer:

- **Structure:** join the trajectories of several calculations into one structure
  trajectory, e.g. `structure.join(other)` or a function taking several
  calculations. It should keep `to_POSCAR`, `to_ase`, `to_mdtraj`, `to_lammps` and
  `plot` working on the combined steps. Check that the runs describe the same atoms
  (stoichiometry and order) and that each run starts close to where the previous one
  ended, with an overridable threshold rather than a warning. Atoms that crossed the
  cell boundary between runs must not appear to jump.
- **View:** play the combined trajectory as one animation. Note that `View + View`
  already exists and means something else: it overlays isosurfaces and arrows on the
  same structure (`_merge_view_fields` in `src/py4vasp/_third_party/view/view.py`).
  Appending steps therefore needs its own name, e.g. `View.join`, or it falls out of
  joining the structures first.
- **Reversing:** a reversed trajectory (`[::-1]`) is needed for the IRC case; check
  whether the step selection of `Structure` already supports negative steps.

Decide whether energies, forces, and velocities of the runs should be joined the same
way. That would allow the energy profile along both IRC branches, for which the
transition-state tutorial currently reverses and concatenates `energy[:]` by hand.
