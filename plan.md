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

- [ ] `force` — `force.py` → `Force(slice_.Mixin, Refinery, structure.Mixin, view.Mixin)` | test: `test_force.py`
- [ ] `stress` — `stress.py` → `Stress(slice_.Mixin, Refinery, structure.Mixin)` | test: `test_stress.py`
- [ ] `velocity` — `velocity.py` → `Velocity(slice_.Mixin, Refinery, structure.Mixin, view.Mixin)` | test: `test_velocity.py`
- [ ] `local_moment` — `local_moment.py` → `LocalMoment(slice_.Mixin, Refinery, structure.Mixin, view.Mixin)` | test: `test_local_moment.py`
  - Note: step-indexed; stale backup `local_moment.py~` can be deleted once ported.
- [ ] `nics` — `nics.py` → `Nics(Refinery, structure.Mixin, view.Mixin)` | test: `test_nics.py`
- [ ] `density` — `density.py` → `Density(Refinery, structure.Mixin, view.Mixin)` | test: `test_density.py`
- [ ] `partial_density` — `partial_density.py` → `PartialDensity(Refinery, structure.Mixin, view.Mixin)` | test: `test_partial_density.py`
- [ ] `potential` — `potential.py` → `Potential(Refinery, structure.Mixin, view.Mixin)` | test: `test_potential.py`
- [ ] `current_density` — `current_density.py` → `CurrentDensity(Refinery, structure.Mixin)` | test: `test_current_density.py`
- [ ] `_CONTCAR` — `_CONTCAR.py` → `CONTCAR(Refinery, view.Mixin, structure.Mixin)` | test: `test_contcar.py`
  - Note: internal; stays in `QUANTITIES` as `"_CONTCAR"` until deprecated/removed.

---

### Group 6 — Projector / kpoint / dispersion chain

Port in order: `kpoint` → `_dispersion` → `projector` → `dos` → `band`.

- [ ] `kpoint` — `kpoint.py` → `Kpoint(Refinery)` | test: `test_kpoint.py`
  - Note: used by Band and Dispersion.
- [ ] `_dispersion` — `_dispersion.py` → `Dispersion(Refinery)` | test: `test_dispersion.py`
  - Note: internal helper used by Band.
- [ ] `projector` — `projector.py` → `Projector(Refinery)` | test: `test_projector.py`
  - Note: used by Band and Dos.
- [ ] `dos` — `dos.py` → `Dos(Refinery, graph.Mixin)` | test: `test_dos.py`
  - Note: composes `ProjectorHandler`; selection forwarding for spin/orbital projections.
- [ ] `band` — `band.py` → `Band(Refinery, graph.Mixin)` | test: `test_band.py`
  - Note: most complex standalone quantity; composes `ProjectorHandler`, `KpointHandler`, `DispersionHandler`.

---

### Group 7 — Phonon group

Port `phonon.py` (the shared Mixin) conceptually before the three group members.

- [ ] `phonon.mode` — `phonon_mode.py` → `PhononMode(Refinery, structure.Mixin)` | test: `test_phonon_mode.py`
  - Group: `@quantity("mode", group="phonon")`; remove from `GROUPS["phonon"]`.
- [ ] `phonon.band` — `phonon_band.py` → `PhononBand(phonon.Mixin, Refinery, graph.Mixin)` | test: `test_phonon_band.py`
  - Group: `@quantity("band", group="phonon")`.
- [ ] `phonon.dos` — `phonon_dos.py` → `PhononDos(phonon.Mixin, Refinery, graph.Mixin)` | test: `test_phonon_dos.py`
  - Group: `@quantity("dos", group="phonon")`.

---

### Group 8 — Exciton group

- [ ] `exciton.eigenvector` — `exciton_eigenvector.py` → `ExcitonEigenvector(Refinery)` | test: `test_exciton_eigenvector.py`
  - Group: `@quantity("eigenvector", group="exciton")`; remove from `GROUPS["exciton"]`.
- [ ] `exciton.density` — `exciton_density.py` → `ExcitonDensity(Refinery, structure.Mixin, view.Mixin)` | test: `test_exciton_density.py`
  - Group: `@quantity("density", group="exciton")`.

---

### Group 9 — Electron-phonon group (complex)

`ElectronPhononSelfEnergy` and `ElectronPhononTransport` also inherit `abc.Sequence` — figure out how to expose `__len__`/`__getitem__` on the Dispatcher without using the Refinery pattern.

- [ ] `electron_phonon.chemical_potential` — `electron_phonon_chemical_potential.py` → `ElectronPhononChemicalPotential(Refinery)` | test: `test_electron_phonon_chemical_potential.py`
  - Group: `@quantity("chemical_potential", group="electron_phonon")`; remove from `GROUPS["electron_phonon"]`.
- [ ] `electron_phonon.bandgap` — `electron_phonon_bandgap.py` → `ElectronPhononBandgap(Refinery, abc.Sequence)` | test: `test_electron_phonon_bandgap.py`
  - Note: step-indexed via `abc.Sequence`; complex internal structure.
  - Group: `@quantity("bandgap", group="electron_phonon")`.
- [ ] `electron_phonon.self_energy` — `electron_phonon_self_energy.py` → `ElectronPhononSelfEnergy(Refinery, abc.Sequence)` | test: `test_electron_phonon_self_energy.py`
  - Note: `abc.Sequence` exposes sub-instances; requires careful Handler factory design.
  - Group: `@quantity("self_energy", group="electron_phonon")`.
- [ ] `electron_phonon.transport` — `electron_phonon_transport.py` → `ElectronPhononTransport(Refinery, abc.Sequence, graph.Mixin)` | test: `test_electron_phonon_transport.py`
  - Note: most complex quantity; `abc.Sequence` + graph; leave for last.
  - Group: `@quantity("transport", group="electron_phonon")`.

---

## Final Steps (after all quantities ported)

- [ ] Remove all remaining entries from `QUANTITIES` in `__init__.py` (auto-registered via `@quantity`)
- [ ] Remove all remaining entries from `GROUPS` in `__init__.py` (auto-registered via `@quantity(..., group=...)`)
- [ ] Remove `base.Refinery` import from `__init__.py` if no longer needed
- [ ] Delete `base.py` (or keep minimal for backward compat) once no Refineries remain
- [ ] Delete stale backup `local_moment.py~`
- [ ] Run full test suite: `pytest tests/ -x`
- [ ] Verify no `base.Refinery` references remain: `grep -r "Refinery" src/`

---

## Per-Quantity Checklist (apply for each item above)

- [ ] Raw dataclass type identified in `_raw/data.py`
- [ ] `<Name>Handler` created with `from_data(cls, raw_<name>: raw.Name, steps=None)` and type hints
- [ ] All transform logic moved from Refinery to Handler; `self._raw_data` → `self._raw_<name>`
- [ ] `@base.data_access` decorators removed from Handler methods
- [ ] Dispatcher `@quantity("name")` created; inherits small mixins (e.g. `graph.Mixin`) if needed
- [ ] All dispatcher methods have `selection: str | None = None`
- [ ] Dispatcher uses `merge_default` / `merge_graphs` / `merge_strings` appropriately
- [ ] `to_dict` kept as alias for `read()` on both Handler and Dispatcher
- [ ] Docstrings + `@documentation.format` / `slice_.examples` ported to Dispatcher
- [ ] Step indexing: `__getitem__` on Dispatcher, `_handler_factory` passes `steps` to Handler
- [ ] Composition: other Handler's `from_data` called directly (no Source)
- [ ] `__str__` / `_repr_pretty_` ported via `merge_strings`
- [ ] `Handler.to_database()` implemented (public); Dispatcher has no database method
- [ ] Tests split into Handler unit tests + Dispatcher integration tests via `DictSource`
- [ ] `to_dict` tests verify it matches `read()`
- [ ] Non-working tests marked `@pytest.mark.skip(reason="...")` — never deleted
- [ ] Removed from `QUANTITIES` / `GROUPS` in `__init__.py`
- [ ] `pytest tests/calculation/test_<name>.py -v` passes
