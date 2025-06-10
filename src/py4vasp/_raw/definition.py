# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import py4vasp._raw.data as raw
from py4vasp._raw import read
from py4vasp._raw.schema import Length, Link, Schema

DEFAULT_FILE = "vaspout.h5"
DEFAULT_SOURCE = "default"
VERSION_DATA = raw.Version("version/major", "version/minor", "version/patch")

schema = Schema(VERSION_DATA)


def get_schema():
    "Return a YAML representation of the schema."
    return str(schema)


def selections(quantity):
    "Return all possible selections for a particular quantity."
    return schema.selections(quantity)


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
    raw.Bandgap,
    required=raw.Version(6, 5),
    labels="intermediate/electron/band/labels",
    values="intermediate/electron/band/gap_from_weight",
)
schema.add(
    raw.Bandgap,
    name="kpoint",
    required=raw.Version(6, 5),
    labels="intermediate/electron/band/labels",
    values="intermediate/electron/band/gap_from_kpoint",
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
    scale="intermediate/ion_dynamics/scale",
    lattice_vectors="intermediate/ion_dynamics/lattice_vectors",
)
schema.add(
    raw.Cell,
    name="final",
    required=raw.Version(6, 5),
    scale="results/positions/scale",
    lattice_vectors="results/positions/lattice_vectors",
)
schema.add(
    raw.Cell,
    name="phonon",
    required=raw.Version(6, 4),
    scale="results/phonons/primitive/scale",
    lattice_vectors="results/phonons/primitive/lattice_vectors",
)
schema.add(
    raw.Cell,
    name="exciton",
    required=raw.Version(6, 5),
    scale="results/supercell/scale",
    lattice_vectors="results/supercell/lattice_vectors",
)
#
schema.add(
    raw.CONTCAR,
    required=raw.Version(6, 5),
    structure=Link("structure", "final"),
    system="results/positions/system",
    selective_dynamics="/results/positions/selective_dynamics_ions",
    lattice_velocities="/results/positions/lattice_velocities",
    ion_velocities="/results/positions/ion_velocities",
    _predictor_corrector="NotImplemented",
)
schema.add(raw.CONTCAR, name="CONTCAR", file="CONTCAR", data_factory=read.CONTCAR)
#
schema.add(
    raw.CurrentDensity,
    name="nmr",
    required=raw.Version(6, 5, 1),
    valid_indices="/results/nmrcur_meta/perturbations",
    structure=Link("structure", DEFAULT_SOURCE),
    current_density="/results/nmrcur{}/values",
)
#
schema.add(
    raw.Density,
    alias=["charge", "n", "charge_density", "electronic_charge_density"],
    file="vaspwave.h5",
    structure=Link("structure", DEFAULT_SOURCE),
    charge="charge/charge",
)
schema.add(
    raw.Density,
    name="tau",
    required=raw.Version(6, 5),
    alias=["kinetic_energy", "kinetic_energy_density"],
    file="vaspwave.h5",
    structure=Link("structure", DEFAULT_SOURCE),
    charge="kinetic_energy_density/values",
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
schema.add(
    raw.DielectricFunction,
    name="xas",
    required=raw.Version(6, 5, 1),
    energies=f"{group}/XAS_{energies}",
    dielectric_function=f"{group}/XAS{values}",
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
schema.add(
    raw.Energy,
    name="afqmc",
    file="vaspwave.h5",
    required=raw.Version(6, 5, 1),
    labels="afqmc/ensemble/energy_labels",
    values="afqmc/ensemble/ensemble_sampling",
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
schema.add(
    raw.ElectronicMinimization,
    required=raw.Version(6, 5),
    label="intermediate/ion_dynamics/oszicar_label",
    convergence_data="intermediate/ion_dynamics/oszicar",
    is_elmin_converged="/intermediate/ion_dynamics/electronic_step_converged",
)
#
group = "/results/electron_phonon/electrons"
schema.add(
    raw.ElectronPhononChemicalPotential,
    required=raw.Version(6, 5),
    fermi_energy=f"{group}/chemical_potential/efermi",
    chemical_potential=f"{group}/chemical_potential/muij",
    carrier_density=f"{group}/chemical_potential/nij",
    temperatures=f"{group}/chemical_potential/temperatures",
    # would like to read these variables from the incar but because of a bug only the first element is ever written
    # carrier_per_cell="input/incar/elph_selfen_carrier_per_cell",
    # carrier_den="input/incar/elph_selfen_carrier_den",
    # mu="input/incar/elph_selfen_carrier_mu",
    carrier_per_cell=f"{group}/chemical_potential/selfen_carrier_per_cell",
    carrier_den=f"{group}/chemical_potential/selfen_carrier_den",
    mu=f"{group}/chemical_potential/selfen_mu",
)
schema.add(
    raw.ElectronPhononSelfEnergy,
    required=raw.Version(6, 5),
    id_name=f"{group}/self_energy_meta/id_name",
    id_size=f"{group}/self_energy_meta/id_size",
    valid_indices=f"{group}/self_energy_meta/ncalculators",
    chemical_potential=Link("electron_phonon_chemical_potential", DEFAULT_SOURCE),
    eigenvalues=f"{group}/eigenvalues/eigenvalues",
    id_index=f"{group}/self_energy_{{}}/id_idx",
    debye_waller=f"{group}/self_energy_{{}}/selfen_dw",
    band_kpoint_spin_index=f"{group}/self_energy_{{}}/bks_idx",
    fan=f"{group}/self_energy_{{}}/selfen_fan",
    band_start=f"{group}/self_energy_{{}}/band_start",
    nbands_sum=f"{group}/self_energy_{{}}/nbands_sum",
    delta=f"{group}/self_energy_{{}}/delta",
    scattering_approximation=f"{group}/self_energy_{{}}/scattering_approximation",
)
schema.add(
    raw.ElectronPhononBandgap,
    required=raw.Version(6, 5),
    id_name=f"{group}/self_energy_meta/id_name",
    id_size=f"{group}/self_energy_meta/id_size",
    id_index=f"{group}/self_energy_{{}}/id_idx",
    chemical_potential=Link("electron_phonon_chemical_potential", DEFAULT_SOURCE),
    self_energy=Link("electron_phonon_self_energy", DEFAULT_SOURCE),
    valid_indices=f"{group}/self_energy_meta/ncalculators",
    fundamental_renorm=f"{group}/self_energy_{{}}/fundamental_gap_renorm",
    fundamental=f"{group}/self_energy_{{}}/fundamental_gap",
    direct_renorm=f"{group}/self_energy_{{}}/direct_gap_renorm",
    direct=f"{group}/self_energy_{{}}/direct_gap",
    temperatures=f"{group}/self_energy_{{}}/temps",
    nbands_sum=f"{group}/self_energy_{{}}/nbands_sum",
    delta=f"{group}/self_energy_{{}}/delta",
    scattering_approximation=f"{group}/self_energy_{{}}/scattering_approximation",
)
schema.add(
    raw.ElectronPhononTransport,
    required=raw.Version(6, 5),
    id_name=f"{group}/transport_meta/id_name",
    id_size=f"{group}/transport_meta/id_size",
    chemical_potential=Link("electron_phonon_chemical_potential", DEFAULT_SOURCE),
    self_energy=Link("electron_phonon_self_energy", DEFAULT_SOURCE),
    valid_indices=f"{group}/transport_meta/ncalculators",
    id_index=f"{group}/transport_{{}}/id_idx",
    temperatures=f"{group}/transport_{{}}/temps",
    electronic_conductivity=f"{group}/transport_{{}}/e_conductivity",
    mobility=f"{group}/transport_{{}}/mobility",
    seebeck=f"{group}/transport_{{}}/seebeck",
    peltier=f"{group}/transport_{{}}/peltier",
    nbands_sum=f"{group}/self_energy_{{}}/nbands_sum",
    delta=f"{group}/self_energy_{{}}/delta",
    electronic_thermal_conductivity=f"{group}/transport_{{}}/e_t_conductivity",
    transport_function=f"{group}/transport_{{}}/transport_function",
    scattering_approximation=f"{group}/transport_{{}}/scattering_approximation",
)
#
group = "results/linear_response"
schema.add(
    raw.ExcitonDensity,
    required=raw.Version(6, 5),
    structure=Link("structure", "exciton"),
    exciton_charge=f"{group}/exciton_charge",
)
schema.add(
    raw.ExcitonEigenvector,
    required=raw.Version(6, 4),
    dispersion=Link("dispersion", DEFAULT_SOURCE),
    fermi_energy=f"{group}/efermi",
    bse_index=f"{group}/bse_index",
    eigenvectors=f"{group}/bse_fatbands",
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
    selective_dynamics="/results/positions/selective_dynamics_ions",
)
#
schema.add(
    raw.InternalStrain,
    required=raw.Version(6, 3),
    structure=Link("structure", DEFAULT_SOURCE),
    internal_strain="results/linear_response/internal_strain",
)
#
input_ = "input/kpoints"
result = "results/electron_eigenvalues"
schema.add(
    raw.Kpoint,
    mode=f"{input_}/mode",
    number=f"{input_}/number_kpoints",
    coordinates=f"{result}/kpoint_coords",
    number_x=f"{input_}/nkpx",
    number_y=f"{input_}/nkpy",
    number_z=f"{input_}/nkpz",
    weights=f"{result}/kpoints_symmetry_weight",
    labels=f"{input_}/labels_kpoints",
    label_indices=f"{input_}/positions_labels_kpoints",
    cell=Link("cell", DEFAULT_SOURCE),
)
input_ = "input/kpoints_opt"
result = "results/electron_eigenvalues_kpoints_opt"
schema.add(
    raw.Kpoint,
    name="kpoints_opt",
    mode=f"{input_}/mode",
    number=f"{input_}/number_kpoints",
    number_x=f"{input_}/nkpx",
    number_y=f"{input_}/nkpy",
    number_z=f"{input_}/nkpz",
    coordinates=f"{result}/kpoint_coords",
    weights=f"{result}/kpoints_symmetry_weight",
    labels=f"{input_}/labels_kpoints",
    label_indices=f"{input_}/positions_labels_kpoints",
    cell=Link("cell", DEFAULT_SOURCE),
)
input_ = "input/kpoints_wan"
result = "results/electron_eigenvalues_kpoints_wan"
schema.add(
    raw.Kpoint,
    name="kpoints_wan",
    mode=f"{input_}/mode",
    number=f"{input_}/number_kpoints",
    number_x=f"{input_}/nkpx",
    number_y=f"{input_}/nkpy",
    number_z=f"{input_}/nkpz",
    coordinates=f"{result}/kpoint_coords",
    weights=f"{result}/kpoints_symmetry_weight",
    labels=f"{input_}/labels_kpoints",
    label_indices=f"{input_}/positions_labels_kpoints",
    cell=Link("cell", DEFAULT_SOURCE),
)
input_ = "input/qpoints"
result = "results/phonons"
schema.add(
    raw.Kpoint,
    name="phonon",
    required=raw.Version(6, 4),
    mode=f"{input_}/mode",
    number=f"{input_}/number_kpoints",
    number_x=f"{input_}/nkpx",
    number_y=f"{input_}/nkpy",
    number_z=f"{input_}/nkpz",
    coordinates=f"{result}/qpoint_coords",
    weights=f"{result}/qpoints_symmetry_weight",
    labels=f"{input_}/labels_kpoints",
    label_indices=f"{input_}/positions_labels_kpoints",
    cell=Link("cell", "phonon"),
)
#
schema.add(
    raw.LocalMoment,
    required=raw.Version(6, 5),
    structure=Link("structure", DEFAULT_SOURCE),
    spin_moments="intermediate/ion_dynamics/magnetism/spin_moments/values",
    orbital_moments="intermediate/ion_dynamics/magnetism/orbital_moments/values",
)
#
schema.add(
    raw.Nics,
    required=raw.Version(6, 5),
    structure=Link("structure", DEFAULT_SOURCE),
    nics_grid="results/nics/values",
    nics_points="results/posnics/values",
    positions="results/posnics/positions",
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
schema.add(
    raw.PartialDensity,
    required=raw.Version(6, 5),
    structure=Link("structure", DEFAULT_SOURCE),
    partial_charge="results/partial_charges/parchg",
    bands="results/partial_charges/bands",
    kpoints="results/partial_charges/kpoints",
    grid="results/partial_charges/grid",
)
#
group = "results/phonons"
schema.add(
    raw.PhononBand,
    required=raw.Version(6, 4),
    dispersion=Link("dispersion", "phonon"),
    stoichiometry=Link("stoichiometry", "phonon"),
    eigenvectors=f"{group}/eigenvectors",
)
schema.add(
    raw.PhononDos,
    required=raw.Version(6, 4),
    energies=f"{group}/dos_mesh",
    dos=f"{group}/dos",
    stoichiometry=Link("stoichiometry", "phonon"),
    projections=f"{group}/dospar",
)
#
group = "results/linear_response/dynmat"
schema.add(
    raw.PhononMode,
    required=raw.Version(6, 5, 2),
    structure=Link("structure", "default"),
    frequencies=f"{group}/eigenvalues",
    eigenvectors=f"{group}/eigenvectors",
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
    raw.Potential,
    required=raw.Version(6, 5),
    structure=Link("structure", DEFAULT_SOURCE),
    hartree_potential="results/potential/hartree",
    ionic_potential="results/potential/ionic",
    xc_potential="results/potential/xc",
    total_potential="results/potential/total",
)
#
schema.add(
    raw.Projector,
    stoichiometry=Link("stoichiometry", DEFAULT_SOURCE),
    orbital_types="results/projectors/lchar",
    number_spin_projections=Length("results/projectors/par"),
)
schema.add(
    raw.Projector,
    name="kpoints_opt",
    stoichiometry=Link("stoichiometry", DEFAULT_SOURCE),
    orbital_types="results/projectors_kpoints_opt/lchar",
    number_spin_projections=Length("results/projectors_kpoints_opt/par"),
)
schema.add(
    raw.Projector,
    name="kpoints_wan",
    stoichiometry=Link("stoichiometry", DEFAULT_SOURCE),
    orbital_types="results/projectors_kpoints_wan/lchar",
    number_spin_projections=Length("results/projectors_kpoints_wan/par"),
)
#
schema.add(
    raw.Stoichiometry,
    ion_types="results/positions/ion_types",
    number_ion_types="results/positions/number_ion_types",
)
schema.add(
    raw.Stoichiometry,
    name="phonon",
    required=raw.Version(6, 4),
    ion_types="results/phonons/primitive/ion_types",
    number_ion_types="results/phonons/primitive/number_ion_types",
)
schema.add(
    raw.Stoichiometry,
    name="exciton",
    required=raw.Version(6, 5),
    ion_types="results/supercell/ion_types",
    number_ion_types="results/supercell/number_ion_types",
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
    stoichiometry=Link("stoichiometry", DEFAULT_SOURCE),
    cell=Link("cell", DEFAULT_SOURCE),
    positions="intermediate/ion_dynamics/position_ions",
)
schema.add(
    raw.Structure,
    name="final",
    required=raw.Version(6, 5),
    stoichiometry=Link("stoichiometry", DEFAULT_SOURCE),
    cell=Link("cell", "final"),
    positions="results/positions/position_ions",
)
schema.add(
    raw.Structure,
    name="exciton",
    required=raw.Version(6, 5),
    cell=Link("cell", "exciton"),
    stoichiometry=Link("stoichiometry", "exciton"),
    positions="results/supercell/position_ions",
)
schema.add(raw.Structure, name="poscar", file="POSCAR", data_factory=read.structure)
#
schema.add(raw.System, system="input/incar/SYSTEM")
#
schema.add(
    raw.Velocity,
    required=raw.Version(6, 4),
    structure=Link("structure", DEFAULT_SOURCE),
    velocities="intermediate/ion_dynamics/ion_velocities",
)
#
schema.add(
    raw.Workfunction,
    required=raw.Version(6, 5),
    idipol="input/incar/IDIPOL",
    distance="results/potential/distance_along_IDIPOL",
    average_potential="results/potential/average_potential_along_IDIPOL",
    vacuum_potential="results/potential/vacuum_potential",
    reference_potential=Link("bandgap", DEFAULT_SOURCE),
    fermi_energy="results/electron_dos/efermi",
)
