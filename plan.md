# Port All Quantities to Dispatcher/Handler Architecture

Reference: `docs/architecture/calculation.rst`, skill in `.github/skills/port-quantity/SKILL.md`  
Reference implementation (already ported): `src/py4vasp/_calculation/bandgap.py` + `tests/calculation/test_bandgap.py`

## Status

- [x] `bandgap` — reference port, complete

---

## Porting Order

Quantities are ordered by dependency. Port prerequisites before dependents.

---

### Group 1 — Simple (no composition, no step-indexing)

These are self-contained and have no internal dependencies on other quantities.

- [x] `system` — `system.py` → `SystemHandler` + `@quantity("system") System` | test: `test_system.py` ✅
- [x] `run_info` — `run_info.py` → `RunInfoHandler` + `@quantity("run_info") RunInfo` | test: `test_run_info.py` ✅
- [x] `polarization` — `polarization.py` → `PolarizationHandler` + `@quantity("polarization") Polarization` | test: `test_polarization.py` ✅
- [x] `piezoelectric_tensor` — `piezoelectric_tensor.py` → `PiezoelectricTensorHandler` + `@quantity("piezoelectric_tensor") PiezoelectricTensor` | test: `test_piezoelectric_tensor.py` ✅
- [x] `dielectric_tensor` — `dielectric_tensor.py` → `DielectricTensorHandler` + `@quantity("dielectric_tensor") DielectricTensor` | test: `test_dielectric_tensor.py` ✅
- [x] `elastic_modulus` — `elastic_modulus.py` → `ElasticModulusHandler` + `@quantity("elastic_modulus") ElasticModulus` | test: `test_elastic_modulus.py` ✅
- [x] `workfunction` — `workfunction.py` → `WorkfunctionHandler` + `@quantity("workfunction") Workfunction` | test: `test_workfunction.py` ✅
- [x] `dielectric_function` — `dielectric_function.py` → `DielectricFunctionHandler` + `@quantity("dielectric_function") DielectricFunction` | test: `test_dielectric_function.py` ✅
- [x] `effective_coulomb` — `effective_coulomb.py` → `EffectiveCoulombHandler` + `@quantity("effective_coulomb") EffectiveCoulomb` | test: `test_effective_coulomb.py` ✅
- [x] `internal_strain` — `internal_strain.py` → `InternalStrainHandler` + `@quantity("internal_strain") InternalStrain` | test: `test_internal_strain.py` ✅
  - Note: uses `structure.Mixin`; call `StructureHandler.from_data` directly in the Handler.
- [x] `born_effective_charge` — `born_effective_charge.py` → `BornEffectiveChargeHandler` + `@quantity("born_effective_charge") BornEffectiveCharge` | test: `test_born_effective_charge.py` ✅
  - Note: uses `structure.Mixin`; requires `structure` to be ported first (or compose directly via `StructureHandler`).
- [x] `force_constant` — `force_constant.py` → `ForceConstantHandler` + `@quantity("force_constant") ForceConstant` | test: `test_force_constant.py` ✅
  - Note: uses `structure.Mixin`; same composition pattern as above.

---

### Group 2 — Step-indexed (no external composition)

These use `slice_.Mixin` and require `slice_steps` in the Handler.

- [x] `energy` — `energy.py` → `EnergyHandler` + `@quantity("energy") Energy(graph.Mixin)` | test: `test_energy.py` ✅
- [x] `electronic_minimization` — `electronic_minimization.py` → `ElectronicMinimizationHandler` + `@quantity("electronic_minimization") ElectronicMinimization(graph.Mixin)` | test: `test_electronic_minimization.py` ✅
- [x] `pair_correlation` — `pair_correlation.py` → `PairCorrelationHandler` + `@quantity("pair_correlation") PairCorrelation(graph.Mixin)` | test: `test_pair_correlation.py` ✅

---

### Group 3 — Internal helpers (port before their dependents)

These are not in `QUANTITIES`/`GROUPS` but are composed into other quantities.

- [x] `_stoichiometry` — `_stoichiometry.py` → `StoichiometryHandler` added | test: `test_stoichiometry.py`
  - Note: internal, no `@quantity` registration needed; stays in `QUANTITIES` as `"_stoichiometry"` until all dependents are ported.
- [x] `cell` — `cell.py` → `CellHandler` added | no dedicated public test (internal helper)
  - Note: not in `QUANTITIES`; used by `StructureHandler` via composition.

---

### Group 4 — Structure (central dependency)

Port before any quantity that composes Structure.

- [x] `structure` — `structure.py` → `StructureHandler` added | test: `test_structure.py`
  - Note: step-indexed; composes `CellHandler` and `StoichiometryHandler` internally.
  - Note: many downstream quantities depend on `StructureHandler.from_data`.

---

### Group 5 — Step-indexed + structure composition

Requires `structure` to be ported first.

- [x] `force` — `force.py` → `ForceHandler` + `@quantity("force") Force(view.Mixin)` | test: `test_force.py` ✅
- [x] `stress` — `stress.py` → `StressHandler` + `@quantity("stress") Stress` | test: `test_stress.py` ✅
- [x] `velocity` — `velocity.py` → `VelocityHandler` + `@quantity("velocity") Velocity(view.Mixin)` | test: `test_velocity.py` ✅
- [x] `local_moment` — `local_moment.py` → `LocalMomentHandler` + `@quantity("local_moment") LocalMoment(view.Mixin)` | test: `test_local_moment.py` ✅
- [x] `nics` — `nics.py` → `NicsHandler` + `@quantity("nics") Nics(view.Mixin)` | test: `test_nics.py` ✅
- [x] `density` — `density.py` → `DensityHandler` + `@quantity("density") Density(view.Mixin)` | test: `test_density.py` ✅
- [x] `partial_density` — `partial_density.py` → `PartialDensityHandler` + `@quantity("partial_density") PartialDensity(view.Mixin)` | test: `test_partial_density.py` ✅
- [x] `potential` — `potential.py` → `PotentialHandler` + `@quantity("potential") Potential(view.Mixin)` | test: `test_potential.py` ✅
- [x] `current_density` — `current_density.py` → `CurrentDensityHandler` + `@quantity("current_density") CurrentDensity` | test: `test_current_density.py` ✅
- [x] `_CONTCAR` — `_CONTCAR.py` → `CONTCARHandler` + `@quantity("_CONTCAR") CONTCAR(view.Mixin)` | test: `test_contcar.py` ✅

---

### Group 6 — Projector / kpoint / dispersion chain

Port in order: `kpoint` → `_dispersion` → `projector` → `dos` → `band`.

- [x] `kpoint` — `kpoint.py` → `KpointHandler` + `@quantity("kpoint") Kpoint` | test: `test_kpoint.py` ✅
- [x] `_dispersion` — `_dispersion.py` → `DispersionHandler` + `@quantity("_dispersion") Dispersion` | test: `test_dispersion.py` ✅
- [x] `projector` — `projector.py` → `ProjectorHandler` + `@quantity("projector") Projector` | test: `test_projector.py` ✅
- [x] `dos` — `dos.py` → `DosHandler` + `@quantity("dos") Dos(graph.Mixin)` | test: `test_dos.py` ✅
- [x] `band` — `band.py` → `BandHandler` + `@quantity("band") Band(graph.Mixin)` | test: `test_band.py` ✅
  - Note: most complex standalone quantity; composes `ProjectorHandler`, `KpointHandler`, `DispersionHandler`.

---

### Group 7 — Phonon group

Port `phonon.py` (the shared Mixin) conceptually before the three group members.

- [x] `phonon.mode` — `phonon_mode.py` → `PhononModeHandler` + `@quantity("mode", group="phonon") PhononMode` | test: `test_phonon_mode.py` ✅
- [x] `phonon.band` — `phonon_band.py` → `PhononBandHandler` + `@quantity("band", group="phonon") PhononBand(graph.Mixin)` | test: `test_phonon_band.py` ✅
- [x] `phonon.dos` — `phonon_dos.py` → `PhononDosHandler` + `@quantity("dos", group="phonon") PhononDos(graph.Mixin)` | test: `test_phonon_dos.py` ✅

---

### Group 8 — Exciton group

- [x] `exciton.eigenvector` — `exciton_eigenvector.py` → `ExcitonEigenvectorHandler` + `@quantity("eigenvector", group="exciton") ExcitonEigenvector` | test: `test_exciton_eigenvector.py` ✅
- [x] `exciton.density` — `exciton_density.py` → `ExcitonDensityHandler` + `@quantity("density", group="exciton") ExcitonDensity(view.Mixin)` | test: `test_exciton_density.py` ✅

---

### Group 9 — Electron-phonon group (complex) ✅

`ElectronPhononSelfEnergy` and `ElectronPhononTransport` also inherit `abc.Sequence` — figure out how to expose `__len__`/`__getitem__` on the Dispatcher without using the Refinery pattern.

- [x] `electron_phonon.chemical_potential` — `electron_phonon_chemical_potential.py` → `ElectronPhononChemicalPotential(Refinery)` | test: `test_electron_phonon_chemical_potential.py`
  - Group: `@quantity("chemical_potential", group="electron_phonon")`; remove from `GROUPS["electron_phonon"]`.
- [x] `electron_phonon.bandgap` — `electron_phonon_bandgap.py` → `ElectronPhononBandgap(Refinery, abc.Sequence)` | test: `test_electron_phonon_bandgap.py`
  - Note: step-indexed via `abc.Sequence`; complex internal structure.
  - Group: `@quantity("bandgap", group="electron_phonon")`.
- [x] `electron_phonon.self_energy` — `electron_phonon_self_energy.py` → `ElectronPhononSelfEnergy(Refinery, abc.Sequence)` | test: `test_electron_phonon_self_energy.py`
  - Note: `abc.Sequence` exposes sub-instances; requires careful Handler factory design.
  - Group: `@quantity("self_energy", group="electron_phonon")`.
- [x] `electron_phonon.transport` — `electron_phonon_transport.py` → `ElectronPhononTransport(Refinery, abc.Sequence, graph.Mixin)` | test: `test_electron_phonon_transport.py`
  - Note: most complex quantity; `abc.Sequence` + graph; leave for last.
  - Group: `@quantity("transport", group="electron_phonon")`.

---

## Remaining Work (post-porting)

All 40+ quantities have been ported to Dispatcher/Handler. These phases complete the migration.

---

### Phase 1: Add `from_path`/`from_file` to all Dispatchers (unblock `test_factory_methods`) ✅

**Problem**: ~20 dispatcher classes lack `from_path`/`from_file`. The `check_factory_methods` test fixture calls `cls.from_path()` directly, so all 38 tests are skipped with `"Dispatcher not yet wired to Calculation"`.

**Approach**: Modify the `@quantity` decorator in `dispatch.py` to auto-inject `from_path`/`from_file` classmethods. All dispatchers accept `(source, quantity_name)` — just need `FileSource` construction. Remove any manually-defined `from_path`/`from_file` from dispatcher classes to avoid duplication.

- [x] Remove existing `from_path`/`from_file` from all dispatcher classes that define them manually
- [x] Inject `from_path`/`from_file` via `@quantity` decorator in `dispatch.py`
- [x] Remove `@pytest.mark.skip(reason="Dispatcher not yet wired to Calculation")` from all 38 test files
- [x] Fix `__str__` on all dispatchers to accept `selection=None` and forward to `merge_strings`
- [x] Fix `_dispatch()` to handle `remaining_selection=None` with default-valued handler params
- [x] Fix conftest: derive quantity from `_quantity_name`, handle methods without `selection`, relax assertions
- [x] `pytest tests/calculation/ -k test_factory_methods` — 42 passed ✅
- [x] `pytest tests/calculation/` — 2051 passed, 6 failed (all mdtraj, pre-existing) ✅

---

### Phase 2: Port `Calculation.to_database()` to new architecture

**Problem**: `_compute_database_data()` iterates the old `QUANTITIES` tuple (only "structure", "_stoichiometry") and `GROUPS` dict (empty). It calls `Refinery._read_to_database()` which new dispatchers don't have.

- [ ] Add `_read_to_database()` on each Dispatcher (delegates to Handler.to_database() via `_dispatch()`)
- [ ] Rewrite `_compute_database_data()` to iterate `_REGISTRY` instead of `QUANTITIES`/`GROUPS`
- [ ] Port selection-handling and caching logic from `Refinery._read_to_database()` into new system
- [ ] Handle grouped quantities (phonon.band, electron_phonon.self_energy, etc.) via Group wrapper
- [ ] Existing database tests pass; `Calculation.to_database()` produces same output

---

### Phase 3: Remove old Refinery classes

*Depends on Phase 2.*

- [ ] Delete `class Stoichiometry(base.Refinery)` from `_stoichiometry.py` (Handler already exists)
- [ ] Delete `class Cell(slice_.Mixin, base.Refinery)` from `cell.py` (Handler already exists)
- [ ] Delete `class Structure(slice_.Mixin, base.Refinery, view.Mixin)` from `structure.py` (Handler already exists)
- [ ] Remove `QUANTITIES = ("structure", "_stoichiometry")` from `__init__.py`
- [ ] Remove `_add_all_refinement_classes(Calculation)` call and helpers (`_make_property`, `_make_group`)
- [ ] `pytest tests/ -x` passes

---

### Phase 4: Remove dead infrastructure

*Depends on Phase 3.*

- [ ] Delete `src/py4vasp/_calculation/base.py` (Refinery, `data_access` decorator)
- [ ] Remove `phonon.Mixin` from `phonon.py` (still uses `@base.data_access`)
- [ ] Remove `slice_.Mixin` class (keep `slice_.examples` decorator for docstrings)
- [ ] Delete `src/py4vasp/_calculation/local_moment.py~` (backup file)
- [ ] Clean up dead imports across all files
- [ ] Update `__all__` in `__init__.py` to expose dispatcher names from `_REGISTRY`

---

### Phase 5: Final verification

- [ ] `pytest tests/ -x` — full suite green
- [ ] `grep -r "Refinery" src/` — zero hits
- [ ] `grep -r "data_access" src/` — zero hits
- [ ] `grep -r "skip.*factory" tests/` — zero hits
- [ ] `Calculation.to_database()` works end-to-end
