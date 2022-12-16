# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import py4vasp._raw.data as raw
from py4vasp._raw.schema import Length, Link, Schema

DEFAULT_FILE = "vaspout.h5"
DEFAULT_SOURCE = "default"
VERSION_DATA = raw.Version("version/major", "version/minor", "version/patch")

schema = Schema(VERSION_DATA)
#
schema.add(
    raw.Band,
    dispersion=Link("dispersion", DEFAULT_SOURCE),
    fermi_energy="results/electron_dos/efermi",
    occupations="results/electron_eigenvalues/fermiweights",
    projectors=Link("projector", DEFAULT_SOURCE),
    projections="results/projectors/par",
)
group = "results/electron_eigenvalues_kpoints_opt"
schema.add(
    raw.Band,
    name="kpoints_opt",
    dispersion=Link("dispersion", "kpoints_opt"),
    fermi_energy="results/electron_dos_kpoints_opt/efermi",
    occupations="results/electron_eigenvalues_kpoints_opt/fermiweights",
    projectors=Link("projector", "kpoints_opt"),
    projections="results/projectors_kpoints_opt/par",
)
schema.add(
    raw.Band,
    name="kpoints_wan",
    dispersion=Link("dispersion", "kpoints_wan"),
    fermi_energy="results/electron_dos_kpoints_wan/efermi",
    occupations="results/electron_eigenvalues_kpoints_wan/fermiweights",
    projectors=Link("projector", "kpoints_wan"),
    projections="results/projectors_kpoints_wan/par",
)
#
schema.add(
    raw.BornEffectiveCharge,
    required=raw.Version(6, 3),
    structure=Link("structure", DEFAULT_SOURCE),
    charge_tensors="results/linear_response/born_charges",
)
#
schema.add(
    raw.Cell,
    scale="results/positions/scale",
    lattice_vectors="intermediate/ion_dynamics/lattice_vectors",
)
schema.add(
    raw.Cell,
    name="phonon",
    required=raw.Version(6, 4),
    scale="results/phonons/primitive/scale",
    lattice_vectors="results/phonons/primitive/lattice_vectors",
)
#
schema.add(
    raw.Density,
    structure=Link("structure", DEFAULT_SOURCE),
    charge="charge/charge",
)
#
group = "results/linear_response"
energies = "energies_dielectric_function"
values = "_dielectric_function"
schema.add(
    raw.DielectricFunction,
    required=raw.Version(6, 3),
    energies=f"{group}/{energies}",
    dielectric_function=f"{group}/density_density{values}",
    current_current=f"{group}/current_current{values}",
)
schema.add(
    raw.DielectricFunction,
    name="ion",
    required=raw.Version(6, 4),
    energies=f"{group}/ion_{energies}",
    dielectric_function=f"{group}/ion{values}",
    current_current=None,
)
schema.add(
    raw.DielectricFunction,
    name="bse",
    required=raw.Version(6, 4),
    energies=f"{group}/bse_{energies}",
    dielectric_function=f"{group}/bse{values}",
    current_current=None,
)
schema.add(
    raw.DielectricFunction,
    name="tdhf",
    required=raw.Version(6, 4),
    energies=f"{group}/tdhf_{energies}",
    dielectric_function=f"{group}/tdhf{values}",
    current_current=None,
)
schema.add(
    raw.DielectricFunction,
    name="ipa",
    required=raw.Version(6, 4),
    energies=f"{group}/ipa_{energies}",
    dielectric_function=f"{group}/ipa{values}",
    current_current=None,
)
schema.add(
    raw.DielectricFunction,
    name="rpa",
    required=raw.Version(6, 4),
    energies=f"{group}/rpa_{energies}",
    dielectric_function=f"{group}/rpa{values}",
    current_current=None,
)
schema.add(
    raw.DielectricFunction,
    name="resonant",
    required=raw.Version(6, 4),
    energies=f"{group}/resonant_{energies}",
    dielectric_function=f"{group}/resonant{values}",
    current_current=None,
)
schema.add(
    raw.DielectricFunction,
    name="dft",
    required=raw.Version(6, 4),
    energies=f"{group}/dft_{energies}",
    dielectric_function=f"{group}/dft{values}",
    current_current=None,
)
schema.add(
    raw.DielectricFunction,
    name="tcte",
    required=raw.Version(6, 4),
    energies=f"{group}/tcte_{energies}",
    dielectric_function=f"{group}/tcte{values}",
    current_current=None,
)
schema.add(
    raw.DielectricFunction,
    name="tctc",
    required=raw.Version(6, 4),
    energies=f"{group}/tctc_{energies}",
    dielectric_function=f"{group}/tctc{values}",
    current_current=None,
)
group = "results/linear_response_kpoints_opt"
schema.add(
    raw.DielectricFunction,
    name="kpoints_opt",
    required=raw.Version(6, 4),
    energies=f"{group}/{energies}",
    dielectric_function=f"{group}/density_density{values}",
    current_current=f"{group}/current_current{values}",
)
group = "results/linear_response_kpoints_wan"
schema.add(
    raw.DielectricFunction,
    name="kpoints_wan",
    required=raw.Version(6, 4),
    energies=f"{group}/{energies}",
    dielectric_function=f"{group}/density_density{values}",
    current_current=f"{group}/current_current{values}",
)
#
group = "results/linear_response"
schema.add(
    raw.DielectricTensor,
    required=raw.Version(6, 3),
    electron=f"{group}/electron_dielectric_tensor",
    ion=f"{group}/ion_dielectric_tensor",
    independent_particle=f"{group}/independent_particle_dielectric_tensor",
    method=f"{group}/method_dielectric_tensor",
)
#
schema.add(
    raw.Dispersion,
    kpoints=Link("kpoint", DEFAULT_SOURCE),
    eigenvalues="results/electron_eigenvalues/eigenvalues",
)
schema.add(
    raw.Dispersion,
    name="kpoints_opt",
    kpoints=Link("kpoint", "kpoints_opt"),
    eigenvalues="results/electron_eigenvalues_kpoints_opt/eigenvalues",
)
schema.add(
    raw.Dispersion,
    name="kpoints_wan",
    kpoints=Link("kpoint", "kpoints_wan"),
    eigenvalues="results/electron_eigenvalues_kpoints_wan/eigenvalues",
)
schema.add(
    raw.Dispersion,
    name="phonon",
    kpoints=Link("kpoint", "phonon"),
    eigenvalues="results/phonons/frequencies",
)
#
group = "results/electron_dos"
schema.add(
    raw.Dos,
    fermi_energy=f"{group}/efermi",
    energies=f"{group}/energies",
    dos=f"{group}/dos",
    projectors=Link("projector", DEFAULT_SOURCE),
    projections=f"{group}/dospar",
)
group = "results/electron_dos_kpoints_opt"
schema.add(
    raw.Dos,
    name="kpoints_opt",
    fermi_energy=f"{group}/efermi",
    energies=f"{group}/energies",
    dos=f"{group}/dos",
    projectors=Link("projector", "kpoints_opt"),
    projections=f"{group}/dospar",
)
#
schema.add(
    raw.Energy,
    labels="intermediate/ion_dynamics/energies_tags",
    values="intermediate/ion_dynamics/energies",
)
#
group = "results/linear_response"
schema.add(
    raw.ElasticModulus,
    required=raw.Version(6, 3),
    clamped_ion=f"{group}/clamped_ion_elastic_modulus",
    relaxed_ion=f"{group}/relaxed_ion_elastic_modulus",
)
#
group = "results/linear_response"
schema.add(
    raw.Fatband,
    required=raw.Version(6, 4),
    dispersion=Link("dispersion", DEFAULT_SOURCE),
    fermi_energy=f"{group}/efermi",
    bse_index=f"{group}/bse_index",
    fatbands=f"{group}/bse_fatbands",
    first_valence_band=f"{group}/bse_vbmin",
    first_conduction_band=f"{group}/bse_cbmin",
)
#
schema.add(
    raw.Force,
    structure=Link("structure", DEFAULT_SOURCE),
    forces="intermediate/ion_dynamics/forces",
)
#
schema.add(
    raw.ForceConstant,
    required=raw.Version(6, 3),
    structure=Link("structure", DEFAULT_SOURCE),
    force_constants="results/linear_response/force_constants",
)
#
schema.add(
    raw.InternalStrain,
    required=raw.Version(6, 3),
    structure=Link("structure", DEFAULT_SOURCE),
    internal_strain="results/linear_response/internal_strain",
)
#
input = "input/kpoints"
result = "results/electron_eigenvalues"
schema.add(
    raw.Kpoint,
    mode=f"{input}/mode",
    number=f"{input}/number_kpoints",
    coordinates=f"{result}/kpoint_coords",
    weights=f"{result}/kpoints_symmetry_weight",
    labels=f"{input}/labels_kpoints",
    label_indices=f"{input}/positions_labels_kpoints",
    cell=Link("cell", DEFAULT_SOURCE),
)
input = "input/kpoints_opt"
result = "results/electron_eigenvalues_kpoints_opt"
schema.add(
    raw.Kpoint,
    name="kpoints_opt",
    mode=f"{input}/mode",
    number=f"{input}/number_kpoints",
    coordinates=f"{result}/kpoint_coords",
    weights=f"{result}/kpoints_symmetry_weight",
    labels=f"{input}/labels_kpoints",
    label_indices=f"{input}/positions_labels_kpoints",
    cell=Link("cell", DEFAULT_SOURCE),
)
input = "input/kpoints_wan"
result = "results/electron_eigenvalues_kpoints_wan"
schema.add(
    raw.Kpoint,
    name="kpoints_wan",
    mode=f"{input}/mode",
    number=f"{input}/number_kpoints",
    coordinates=f"{result}/kpoint_coords",
    weights=f"{result}/kpoints_symmetry_weight",
    labels=f"{input}/labels_kpoints",
    label_indices=f"{input}/positions_labels_kpoints",
    cell=Link("cell", DEFAULT_SOURCE),
)
input = "input/qpoints"
result = "results/phonons"
schema.add(
    raw.Kpoint,
    name="phonon",
    required=raw.Version(6, 4),
    mode=f"{input}/mode",
    number=f"{input}/number_kpoints",
    coordinates=f"{result}/qpoint_coords",
    weights=f"{result}/qpoints_symmetry_weight",
    labels=f"{input}/labels_kpoints",
    label_indices=f"{input}/positions_labels_kpoints",
    cell=Link("cell", "phonon"),
)
#
schema.add(
    raw.Magnetism,
    structure=Link("structure", DEFAULT_SOURCE),
    moments="intermediate/ion_dynamics/magnetism/moments",
)
#
group = "intermediate/pair_correlation"
schema.add(
    raw.PairCorrelation,
    required=raw.Version(6, 4),
    distances=f"{group}/distances",
    function=f"{group}/function",
    labels=f"{group}/labels",
)
#
group = "results/phonons"
schema.add(
    raw.PhononBand,
    required=raw.Version(6, 4),
    dispersion=Link("dispersion", "phonon"),
    topology=Link("topology", "phonon"),
    eigenvectors=f"{group}/eigenvectors",
)
schema.add(
    raw.PhononDos,
    required=raw.Version(6, 4),
    energies=f"{group}/dos_mesh",
    dos=f"{group}/dos",
    topology=Link("topology", "phonon"),
    projections=f"{group}/dospar",
)
#
group = "results/linear_response"
schema.add(
    raw.PiezoelectricTensor,
    required=raw.Version(6, 3),
    electron=f"{group}/electron_piezoelectric_tensor",
    ion=f"{group}/ion_piezoelectric_tensor",
)
#
group = "results/linear_response"
schema.add(
    raw.Polarization,
    required=raw.Version(6, 3),
    electron=f"{group}/electron_dipole_moment",
    ion=f"{group}/ion_dipole_moment",
)
#
schema.add(
    raw.Projector,
    topology=Link("topology", DEFAULT_SOURCE),
    orbital_types="results/projectors/lchar",
    number_spins=Length("results/electron_eigenvalues/eigenvalues"),
)
schema.add(
    raw.Projector,
    name="kpoints_opt",
    topology=Link("topology", DEFAULT_SOURCE),
    orbital_types="results/projectors_kpoints_opt/lchar",
    number_spins=Length("results/electron_eigenvalues/eigenvalues"),
)
schema.add(
    raw.Projector,
    name="kpoints_wan",
    topology=Link("topology", DEFAULT_SOURCE),
    orbital_types="results/projectors_kpoints_wan/lchar",
    number_spins=Length("results/electron_eigenvalues/eigenvalues"),
)
#
schema.add(
    raw.Stress,
    structure=Link("structure", DEFAULT_SOURCE),
    stress="intermediate/ion_dynamics/stress",
)
#
schema.add(
    raw.Structure,
    topology=Link("topology", DEFAULT_SOURCE),
    cell=Link("cell", DEFAULT_SOURCE),
    positions="intermediate/ion_dynamics/position_ions",
)
#
schema.add(raw.System, system="input/incar/SYSTEM")
#
schema.add(
    raw.Topology,
    ion_types="results/positions/ion_types",
    number_ion_types="results/positions/number_ion_types",
)
schema.add(
    raw.Topology,
    name="phonon",
    required=raw.Version(6, 4),
    ion_types="results/phonons/primitive/ion_types",
    number_ion_types="results/phonons/primitive/number_ion_types",
)
#
schema.add(
    raw.Velocity,
    required=raw.Version(6, 4),
    structure=Link("structure", DEFAULT_SOURCE),
    velocities="intermediate/ion_dynamics/ion_velocities",
)
