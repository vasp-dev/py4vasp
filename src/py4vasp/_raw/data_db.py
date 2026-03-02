from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__SCHEMA_VERSION__ = "0.1.0"

@dataclass
class _DBDataMixin:
    """Mixin for dataclasses that will be stored in the database."""

    __schema_version__: str = field(init=False, default_factory=lambda: __SCHEMA_VERSION__)
    """The version of the database data schema. This can be used to track changes in the data structure and ensure compatibility when reading from the database."""

@dataclass
class CONTCAR_DB(_DBDataMixin):
    """Data class for storing CONTCAR data in the database."""

    system: Optional[str] = None
    """The description of the system as given in the CONTCAR file (first line)."""

@dataclass
class Dispersion_DB(_DBDataMixin):
    """Data class for storing dispersion data in the database."""

    eigenvalue_min: Optional[float] = None
    """The minimum eigenvalue across all bands and k-points. This can be used to quickly determine the energy range of the dispersion without having to load the full eigenvalue array."""
    eigenvalue_max: Optional[float] = None
    """The maximum eigenvalue across all bands and k-points. This can be used to quickly determine the energy range of the dispersion without having to load the full eigenvalue array."""
    eigenvalue_min_up: Optional[float] = None
    """The minimum eigenvalue for spin-up electrons across all bands and k-points."""
    eigenvalue_max_up: Optional[float] = None
    """The maximum eigenvalue for spin-up electrons across all bands and k-points."""
    eigenvalue_min_down: Optional[float] = None
    """The minimum eigenvalue for spin-down electrons across all bands and k-points."""
    eigenvalue_max_down: Optional[float] = None
    """The maximum eigenvalue for spin-down electrons across all bands and k-points."""

@dataclass
class Stoichiometry_DB(_DBDataMixin):
    """Data class for storing stoichiometry data in the database."""

    ion_types: Optional[List[str]] = field(default_factory=lambda: None)
    """The distinct types of ions in the system."""
    num_ion_types: Optional[List[int]] = field(default_factory=lambda: None)
    """The number of ions of each type in the system."""
    num_ion_types_primitive: Optional[List[int]] = field(default_factory=lambda: None)
    """The number of ions of each type in the primitive cell."""
    formula: Optional[str] = None
    """The chemical formula of the system, in the format {element}{count if count > 1 else ''}, e.g. A3B2CD4."""
    compound: Optional[str] = None
    """The name of the compound, in the format {element1}-{element2}-..., e.g. A-B-C."""

@dataclass
class Band_DB(_DBDataMixin):
    """Data class for storing band structure data in the database."""

    num_considered_bands: Optional[int] = None
    """The total number of bands that were considered in the band structure calculation."""
    num_occupied_bands: Optional[int] = None
    """The total number of occupied bands across all k-points."""
    num_occupied_bands_up: Optional[int] = None
    """The number of occupied spin-up bands across all k-points."""
    num_occupied_bands_down: Optional[int] = None
    """The number of occupied spin-down bands across all k-points."""
    fermi_energy_raw: Optional[float] = None
    """The raw Fermi energy as read from the OUTCAR file."""
    fermi_energy: Optional[float] = None
    """The Fermi energy used for plotting the band structure, which may be different from the raw Fermi energy based on user input."""

@dataclass
class Bandgap_DB(_DBDataMixin):
    """Data class for storing band gap data in the database."""

    fundamental_bandgap_spin_independent: Optional[float] = None
    """The value of the fundamental band gap, in eV. This can be used to quickly determine if the system is metallic or insulating."""
    fundamental_bandgap_spin_up: Optional[float] = None
    """The value of the fundamental band gap for spin-up electrons, in eV."""
    fundamental_bandgap_spin_down: Optional[float] = None
    """The value of the fundamental band gap for spin-down electrons, in eV."""
    valence_band_maximum_spin_independent: Optional[float] = None
    """The energy of the valence band maximum, in eV."""
    valence_band_maximum_spin_up: Optional[float] = None
    """The energy of the valence band maximum for spin-up electrons, in eV."""
    valence_band_maximum_spin_down: Optional[float] = None
    """The energy of the valence band maximum for spin-down electrons, in eV."""
    conduction_band_minimum_spin_independent: Optional[float] = None
    """The energy of the conduction band minimum, in eV."""
    conduction_band_minimum_spin_up: Optional[float] = None
    """The energy of the conduction band minimum for spin-up electrons, in eV."""
    conduction_band_minimum_spin_down: Optional[float] = None
    """The energy of the conduction band minimum for spin-down electrons, in eV."""
    kpoint_vbm_spin_independent: Optional[List[float]] = field(default_factory=lambda: None)
    """The k-point where the valence band maximum occurs, in fractional coordinates."""
    kpoint_vbm_spin_up: Optional[List[float]] = field(default_factory=lambda: None)
    """The k-point where the valence band maximum for spin-up electrons occurs, in fractional coordinates."""
    kpoint_vbm_spin_down: Optional[List[float]] = field(default_factory=lambda: None)
    """The k-point where the valence band maximum for spin-down electrons occurs, in fractional coordinates."""
    kpoint_cbm_spin_independent: Optional[List[float]] = field(default_factory=lambda: None)
    """The k-point where the conduction band minimum occurs, in fractional coordinates."""
    kpoint_cbm_spin_up: Optional[List[float]] = field(default_factory=lambda: None)
    """The k-point where the conduction band minimum for spin-up electrons occurs, in fractional coordinates."""
    kpoint_cbm_spin_down: Optional[List[float]] = field(default_factory=lambda: None)
    """The k-point where the conduction band minimum for spin-down electrons occurs, in fractional coordinates."""

    direct_bandgap_spin_independent: Optional[float] = None
    """The value of the direct band gap, in eV. This can be used to quickly determine if the system has a direct or indirect band gap."""
    direct_bandgap_spin_up: Optional[float] = None
    """The value of the direct band gap for spin-up electrons, in eV."""
    direct_bandgap_spin_down: Optional[float] = None
    """The value of the direct band gap for spin-down electrons, in eV."""
    lower_band_direct_spin_independent: Optional[float] = None
    """The energy of the highest occupied band at the k-point where the direct band gap occurs, in eV."""
    lower_band_direct_spin_up: Optional[float] = None
    """The energy of the highest occupied spin-up band at the k-point where the direct band gap occurs, in eV."""
    lower_band_direct_spin_down: Optional[float] = None
    """The energy of the highest occupied spin-down band at the k-point where the direct band gap occurs, in eV."""
    upper_band_direct_spin_independent: Optional[float] = None
    """The energy of the lowest unoccupied band at the k-point where the direct band gap occurs, in eV."""
    upper_band_direct_spin_up: Optional[float] = None
    """The energy of the lowest unoccupied spin-up band at the k-point where the direct band gap occurs, in eV."""
    upper_band_direct_spin_down: Optional[float] = None
    """The energy of the lowest unoccupied spin-down band at the k-point where the direct band gap occurs, in eV."""
    kpoint_direct_bandgap_spin_independent: Optional[List[float]] = field(default_factory=lambda: None)
    """The k-point where the direct band gap occurs, in fractional coordinates."""
    kpoint_direct_bandgap_spin_up: Optional[List[float]] = field(default_factory=lambda: None)
    """The k-point where the direct band gap for spin-up electrons occurs, in fractional coordinates."""
    kpoint_direct_bandgap_spin_down: Optional[List[float]] = field(default_factory=lambda: None)
    """The k-point where the direct band gap for spin-down electrons occurs, in fractional coordinates."""

@dataclass
class BornEffectiveCharge_DB(_DBDataMixin):
    """Data class for storing Born effective charge data in the database."""

    eigenvalue_min: Optional[float] = None
    """The minimum eigenvalue of the Born effective charge tensors across all ions. This can be used to quickly determine the range of the Born effective charges without having to load the full array."""
    eigenvalue_min_index: Optional[int] = None
    """The index of the ion with the minimum eigenvalue of the Born effective charge tensor."""
    eigenvalue_max: Optional[float] = None
    """The maximum eigenvalue of the Born effective charge tensors across all ions. This can be used to quickly determine the range of the Born effective charges without having to load the full array."""
    eigenvalue_max_index: Optional[int] = None
    """The index of the ion with the maximum eigenvalue of the Born effective charge tensor."""

@dataclass 
class DielectricFunction_DB(_DBDataMixin):
    """Data class for storing dielectric function data in the database."""

    energy_min: Optional[float] = None
    """The minimum energy at which the dielectric function was evaluated, in eV."""
    energy_max: Optional[float] = None
    """The maximum energy at which the dielectric function was evaluated, in eV."""

@dataclass 
class DielectricTensor_DB(_DBDataMixin):
    """Data class for storing dielectric tensor data in the database."""

    method: Optional[str] = None
    """The method used to calculate the dielectric tensor."""
    
    total_3d_tensor: Optional[List[List[float]]] = field(default_factory=lambda: None)
    """The full 3D dielectric tensor for the total response, including both ionic and electronic contributions. Because of symmetry, the tensor is shown in its compact form, in the order (xx, yy, zz, xy, yz, zx)."""
    total_3d_isotropic_dielectric_constant: Optional[float] = None
    """The isotropic dielectric constant for the total response, calculated as the average of the diagonal elements of the total 3D dielectric tensor."""
    total_2d_polarizability: Optional[float] = None
    """The 2D polarizability for the total response, calculated from the total 3D dielectric tensor and the cell geometry."""
    
    ionic_3d_tensor: Optional[List[List[float]]] = field(default_factory=lambda: None)
    """The full 3D dielectric tensor for the ionic contribution. Because of symmetry, the tensor is shown in its compact form, in the order (xx, yy, zz, xy, yz, zx)."""
    ionic_3d_isotropic_dielectric_constant: Optional[float] = None
    """The isotropic dielectric constant for the ionic contribution, calculated as the average of the diagonal elements of the ionic 3D dielectric tensor."""
    ionic_2d_polarizability: Optional[float] = None
    """The 2D polarizability for the ionic contribution, calculated from the ionic 3D dielectric tensor and the cell geometry."""
    
    electronic_3d_tensor: Optional[List[List[float]]] = field(default_factory=lambda: None)
    """The full 3D dielectric tensor for the electronic contribution. Because of symmetry, the tensor is shown in its compact form, in the order (xx, yy, zz, xy, yz, zx)."""
    electronic_3d_isotropic_dielectric_constant: Optional[float] = None
    """The isotropic dielectric constant for the electronic contribution, calculated as the average of the diagonal elements of the electronic 3D dielectric tensor."""
    electronic_2d_polarizability: Optional[float] = None
    """The 2D polarizability for the electronic contribution, calculated from the electronic 3D dielectric tensor and the cell geometry."""

@dataclass
class Dos_DB(_DBDataMixin):
    """Data class for storing density of states data in the database."""

    dos_at_fermi_total: Optional[float] = None
    """The total density of states at the Fermi energy, in states/eV."""
    dos_at_fermi_up: Optional[float] = None
    """The density of states at the Fermi energy for spin-up electrons, in states/eV."""
    dos_at_fermi_down: Optional[float] = None
    """The density of states at the Fermi energy for spin-down electrons, in states/eV."""

    dos_at_raw_fermi_total: Optional[float] = None
    """The total density of states at the raw Fermi energy as read from the OUTCAR file, in states/eV."""
    dos_at_raw_fermi_up: Optional[float] = None
    """The density of states at the raw Fermi energy as read from the OUTCAR file for spin-up electrons, in states/eV."""
    dos_at_raw_fermi_down: Optional[float] = None
    """The density of states at the raw Fermi energy as read from the OUTCAR file for spin-down electrons, in states/eV.""" 

    energy_min: Optional[float] = None
    """The minimum energy at which the density of states was evaluated, in eV."""
    energy_max: Optional[float] = None
    """The maximum energy at which the density of states was evaluated, in eV."""

@dataclass
class EffectiveCoulomb_DB(_DBDataMixin):
    """Data class for storing effective Coulomb interaction data in the database."""

    screened_U_uppercase: Optional[float] = None
    """The value of the screened effective Coulomb interaction U, in eV."""
    screened_u_lowercase: Optional[float] = None
    """The value of the screened effective Coulomb interaction u, in eV."""
    screened_J_uppercase: Optional[float] = None
    """The value of the screened effective Coulomb interaction J, in eV."""

    bare_V_uppercase: Optional[float] = None
    """The value of the bare effective Coulomb interaction V, in eV."""
    bare_v_lowercase: Optional[float] = None
    """The value of the bare effective Coulomb interaction v, in eV."""
    bare_J_uppercase: Optional[float] = None
    """The value of the bare effective Coulomb interaction J, in eV."""

@dataclass
class ElasticModulus_DB(_DBDataMixin):
    """Data class for storing elastic modulus data in the database."""

    total_3d_tensor: Optional[List[List[float]]] = field(default_factory=lambda: None)
    """The full 3D elastic modulus tensor, including both ionic and electronic contributions. Because of symmetry, the tensor is shown in its compact form, in the order (xx, yy, zz, xy, yz, zx)."""
    total_bulk_modulus: Optional[float] = None
    """The bulk modulus calculated from the total 3D elastic modulus tensor, in GPa."""
    total_shear_modulus: Optional[float] = None
    """The shear modulus calculated from the total 3D elastic modulus tensor, in GPa."""
    total_pugh_ratio: Optional[float] = None
    """The Pugh ratio calculated from the total bulk and shear moduli."""
    total_vickers_hardness: Optional[float] = None
    """The Vickers hardness calculated from the total bulk and shear moduli, in GPa."""
    total_fracture_toughness: Optional[float] = None
    """The fracture toughness calculated from the total bulk and shear moduli, in MPa*m^0.5."""
    ionic_3d_tensor: Optional[List[List[float]]] = field(default_factory=lambda: None)
    """The full 3D elastic modulus tensor for the ionic contribution. Because of symmetry, the tensor is shown in its compact form, in the order (xx, yy, zz, xy, yz, zx)."""
    ionic_bulk_modulus: Optional[float] = None
    """The bulk modulus calculated from the ionic contribution to the elastic modulus tensor, in GPa."""
    ionic_shear_modulus: Optional[float] = None
    """The shear modulus calculated from the ionic contribution to the elastic modulus tensor, in GPa."""
    ionic_pugh_ratio: Optional[float] = None
    """The Pugh ratio calculated from the ionic contribution to the bulk and shear moduli."""
    ionic_vickers_hardness: Optional[float] = None
    """The Vickers hardness calculated from the ionic contribution to the bulk and shear moduli, in GPa."""
    ionic_fracture_toughness: Optional[float] = None
    """The fracture toughness calculated from the ionic contribution to the bulk and shear moduli, in MPa*m^0.5."""
    electronic_3d_tensor: Optional[List[List[float]]] = field(default_factory=lambda: None)
    """The full 3D elastic modulus tensor for the electronic contribution. Because of symmetry, the tensor is shown in its compact form, in the order (xx, yy, zz, xy, yz, zx)."""
    electronic_bulk_modulus: Optional[float] = None
    """The bulk modulus calculated from the electronic contribution to the elastic modulus tensor, in GPa."""
    electronic_shear_modulus: Optional[float] = None
    """The shear modulus calculated from the electronic contribution to the elastic modulus tensor, in GPa."""
    electronic_pugh_ratio: Optional[float] = None
    """The Pugh ratio calculated from the electronic contribution to the bulk and shear moduli."""
    electronic_vickers_hardness: Optional[float] = None
    """The Vickers hardness calculated from the electronic contribution to the bulk and shear moduli, in GPa."""
    electronic_fracture_toughness: Optional[float] = None
    """The fracture toughness calculated from the electronic contribution to the bulk and shear moduli, in MPa*m^0.5."""

@dataclass
class ElectronicMinimization_DB(_DBDataMixin):
    """Data class for storing electronic minimization data in the database."""

    num_electronic_steps: Optional[int] = None
    """The total number of electronic steps taken across all ionic steps."""
    num_max_electronic_steps_per_ionic_step: Optional[int] = None
    """The maximum number of electronic steps taken for any ionic step."""
    num_min_electronic_steps_per_ionic_step: Optional[int] = None
    """The minimum number of electronic steps taken for any ionic step."""
    elmin_is_converged_all: Optional[bool] = None
    """Whether the electronic minimization converged for all ionic steps."""
    elmin_is_converged_final: Optional[bool] = None
    """Whether the electronic minimization converged for the final ionic step."""

@dataclass
class Energy_DB(_DBDataMixin):
    """Data class for storing energy data in the database."""

    ion_electron_initial: Optional[float] = None
    """The initial ion-electron energy, in eV."""
    ion_electron_min: Optional[float] = None
    """The minimum ion-electron energy during the calculation, in eV."""
    ion_electron_step_min: Optional[float] = None
    """The ion-electron energy at the step where the minimum ion-electron energy occurs, in eV."""
    ion_electron_final: Optional[float] = None
    """The final ion-electron energy, in eV."""

    kinetic_energy_initial: Optional[float] = None
    """The initial kinetic energy, in eV."""
    kinetic_energy_min: Optional[float] = None
    """The minimum kinetic energy during the calculation, in eV."""
    kinetic_energy_step_min: Optional[float] = None
    """The kinetic energy at the step where the minimum kinetic energy occurs, in eV."""
    kinetic_energy_final: Optional[float] = None
    """The final kinetic energy, in eV."""

    kinetic_energy_lattice_initial: Optional[float] = None
    """The initial kinetic energy of the lattice, in eV."""
    kinetic_energy_lattice_min: Optional[float] = None
    """The minimum kinetic energy of the lattice during the calculation, in eV."""
    kinetic_energy_lattice_step_min: Optional[float] = None
    """The kinetic energy of the lattice at the step where the minimum kinetic energy of the lattice occurs, in eV."""
    kinetic_energy_lattice_final: Optional[float] = None
    """The final kinetic energy of the lattice, in eV."""   

    temperature_initial: Optional[float] = None
    """The initial temperature, in K."""
    temperature_min: Optional[float] = None
    """The minimum temperature during the calculation, in K."""
    temperature_step_min: Optional[float] = None
    """The temperature at the step where the minimum temperature occurs, in K."""
    temperature_final: Optional[float] = None
    """The final temperature, in K."""

    nose_potential_initial: Optional[float] = None
    """The initial Nose potential energy, in eV."""
    nose_potential_min: Optional[float] = None
    """The minimum Nose potential energy during the calculation, in eV."""
    nose_potential_step_min: Optional[float] = None
    """The Nose potential energy at the step where the minimum Nose potential energy occurs, in eV."""
    nose_potential_final: Optional[float] = None
    """The final Nose potential energy, in eV."""

    nose_kinetic_initial: Optional[float] = None
    """The initial Nose kinetic energy, in eV."""
    nose_kinetic_min: Optional[float] = None
    """The minimum Nose kinetic energy during the calculation, in eV."""
    nose_kinetic_step_min: Optional[float] = None
    """The Nose kinetic energy at the step where the minimum Nose kinetic energy occurs, in eV."""
    nose_kinetic_final: Optional[float] = None
    """The final Nose kinetic energy, in eV."""

    total_energy_initial: Optional[float] = None
    """The initial total energy, in eV."""
    total_energy_min: Optional[float] = None
    """The minimum total energy during the calculation, in eV."""
    total_energy_step_min: Optional[float] = None
    """The total energy at the step where the minimum total energy occurs, in eV."""
    total_energy_final: Optional[float] = None
    """The final total energy, in eV."""

    free_energy_initial: Optional[float] = None
    """The initial free energy, in eV."""
    free_energy_min: Optional[float] = None
    """The minimum free energy during the calculation, in eV."""
    free_energy_step_min: Optional[float] = None
    """The free energy at the step where the minimum free energy occurs, in eV."""
    free_energy_final: Optional[float] = None
    """The final free energy, in eV."""

    energy_without_entropy_initial: Optional[float] = None
    """The initial energy without entropy, in eV."""
    energy_without_entropy_min: Optional[float] = None
    """The minimum energy without entropy during the calculation, in eV."""
    energy_without_entropy_step_min: Optional[float] = None
    """The energy without entropy at the step where the minimum energy without entropy occurs, in eV."""
    energy_without_entropy_final: Optional[float] = None
    """The final energy without entropy, in eV."""

    energy_sigma_0_initial: Optional[float] = None
    """The initial energy at sigma->0, in eV."""
    energy_sigma_0_min: Optional[float] = None
    """The minimum energy at sigma->0 during the calculation, in eV."""
    energy_sigma_0_step_min: Optional[float] = None
    """The energy at sigma->0 at the step where the minimum energy at sigma->0 occurs, in eV."""
    energy_sigma_0_final: Optional[float] = None
    """The final energy at sigma->0, in eV."""

    step_initial: Optional[float] = None
    """The initial step, for which energies are evaluated."""
    step_final: Optional[float] = None
    """The final step, for which energies are evaluated."""

    one_electron_energy_initial: Optional[float] = None
    """The initial one-electron energy, in eV."""
    one_electron_energy_min: Optional[float] = None
    """The minimum one-electron energy during the calculation, in eV."""
    one_electron_energy_step_min: Optional[float] = None
    """The one-electron energy at the step where the minimum one-electron energy occurs, in eV."""
    one_electron_energy_final: Optional[float] = None
    """The final one-electron energy, in eV."""

    hartree_energy_initial: Optional[float] = None
    """The initial Hartree energy, in eV."""
    hartree_energy_min: Optional[float] = None
    """The minimum Hartree energy during the calculation, in eV."""
    hartree_energy_step_min: Optional[float] = None
    """The Hartree energy at the step where the minimum Hartree energy occurs, in eV."""
    hartree_energy_final: Optional[float] = None
    """The final Hartree energy, in eV."""

    exchange_energy_initial: Optional[float] = None
    """The initial exchange energy, in eV."""
    exchange_energy_min: Optional[float] = None
    """The minimum exchange energy during the calculation, in eV."""
    exchange_energy_step_min: Optional[float] = None
    """The exchange energy at the step where the minimum exchange energy occurs, in eV."""
    exchange_energy_final: Optional[float] = None
    """The final exchange energy, in eV."""

    free_energy_initial: Optional[float] = None
    """The initial free energy, in eV."""
    free_energy_min: Optional[float] = None
    """The minimum free energy during the calculation, in eV."""
    free_energy_step_min: Optional[float] = None
    """The free energy at the step where the minimum free energy occurs, in eV."""
    free_energy_final: Optional[float] = None
    """The final free energy, in eV."""

    # TODO what is it?
    free_energy_cap_initial: Optional[float] = None
    free_energy_cap_min: Optional[float] = None
    free_energy_cap_step_min: Optional[float] = None
    free_energy_cap_final: Optional[float] = None

    # TODO what is it?
    weight_initial: Optional[float] = None
    weight_min: Optional[float] = None
    weight_step_min: Optional[float] = None
    weight_final: Optional[float] = None

    other_energy_data: Optional[Dict[str, float]] = field(default_factory=lambda: None)
    """A dictionary to store any additional energy data that may be relevant for the calculation, where the keys are descriptive names of the energy terms and the values are the corresponding energy values in eV."""

@dataclass
class ExcitonEigenvector_DB(_DBDataMixin):
    """Data class for storing exciton eigenvector data in the database."""

    num_kpoints: Optional[int] = None
    """The number of k-points included in the BSE calculation."""
    num_valence_bands: Optional[int] = None
    """The number of valence bands included in the BSE calculation."""
    num_conduction_bands: Optional[int] = None
    """The number of conduction bands included in the BSE calculation."""

@dataclass
class Force_DB(_DBDataMixin):
    """Data class for storing force data in the database."""

    final_force_min: Optional[float] = None
    """The minimum force across all atoms on the final step, in eV/Å."""
    final_force_median: Optional[float] = None
    """The median force across all atoms on the final step, in eV/Å."""
    final_force_mean: Optional[float] = None
    """The mean force across all atoms on the final step, in eV/Å."""
    final_force_max: Optional[float] = None
    """The maximum force across all atoms on the final step, in eV/Å."""
    final_index_force_max: Optional[int] = None
    """The index of the atom with the maximum force on the final step."""

    initial_force_min: Optional[float] = None
    """The minimum force across all atoms on the initial step, in eV/Å."""
    initial_force_max: Optional[float] = None
    """The maximum force across all atoms on the initial step, in eV/Å."""
    initial_index_force_max: Optional[int] = None
    """The index of the atom with the maximum force on the initial step."""

@dataclass
class Kpoint_DB(_DBDataMixin):
    """Data class for storing k-point data in the database."""

    mode: Optional[str] = None
    """The mode of the k-point sampling, e.g. automatic, explicit, gamma, monkhorst etc."""
    line_length: Optional[float] = None
    """The number of points used to sample a single line in the Brillouin zone."""
    num_kpoints_total: Optional[int] = None
    """The total number of k-points in the calculation."""
    num_kpoints_grid: Optional[List[float, float, float]] = field(default_factory=lambda: None)
    """The number of k-points along each reciprocal lattice vector for grid sampling, in the order (kx, ky, kz)."""
    num_lines: Optional[int] = None
    """The number of lines in the Brillouin zone along which the band structure is sampled."""
    labels: Optional[List[str]] = field(default_factory=lambda: None)
    """The labels of the high-symmetry k-points corresponding to each line, in the order (line1_start, line1_end, line2_start, line2_end, ...)."""
    labels_unique: Optional[List[str]] = field(default_factory=lambda: None)
    """The unique labels of the high-symmetry k-points, in alphabetical order."""

@dataclass
class LocalMoment_DB(_DBDataMixin):
    """Data class for storing local magnetic moment data in the database."""

    has_orbital_moments: Optional[bool] = None
    """Whether the calculation includes orbital magnetic moments."""
    final_spin_moment_total_min: Optional[float] = None
    """The minimum total spin magnetic moment across all atoms on the final step, in μB."""
    final_spin_moment_total_max: Optional[float] = None
    """The maximum total spin magnetic moment across all atoms on the final step, in μB."""

@dataclass
class Nics_DB(_DBDataMixin):
    """Data class for storing NICS data in the database."""

    method: Optional[str] = None
    """The method used to calculate the NICS, e.g. grid, positions, etc."""

@dataclass
class PairCorrelation_DB(_DBDataMixin):
    """Data class for storing pair correlation function data in the database."""

    distance_min: Optional[float] = None
    """The minimum distance at which the pair correlation function was evaluated, in Å."""
    distance_max: Optional[float] = None
    """The maximum distance at which the pair correlation function was evaluated, in Å."""

@dataclass
class PhononDos_DB(_DBDataMixin):
    """Data class for storing phonon density of states data in the database."""

    energy_min: Optional[float] = None
    """The minimum energy at which the phonon density of states was evaluated, in THz."""
    energy_max: Optional[float] = None
    """The maximum energy at which the phonon density of states was evaluated, in THz."""

@dataclass
class PhononMode_DB(_DBDataMixin):
    """Data class for storing phonon mode data in the database."""

    frequencies_real_max: Optional[float] = None
    """The maximum real frequency across all phonon modes and q-points, in THz."""
    frequencies_imaginary_max: Optional[float] = None
    """The maximum imaginary frequency across all phonon modes and q-points, in THz."""

@dataclass
class PiezoelectricTensor_DB(_DBDataMixin):
    """Data class for storing piezoelectric tensor data in the database."""

    total_3d_tensor_x: Optional[List[List[float]]] = field(default_factory=lambda: None)
    """The full 3D piezoelectric tensor for the total response in Voigt notation in order (xx, yy, zz, xy, yz, zx) along x direction, in C/m^2."""
    total_3d_tensor_y: Optional[List[List[float]]] = field(default_factory=lambda: None)
    """The full 3D piezoelectric tensor for the total response in Voigt notation in order (xx, yy, zz, xy, yz, zx) along y direction, in C/m^2."""
    total_3d_tensor_z: Optional[List[List[float]]] = field(default_factory=lambda: None)
    """The full 3D piezoelectric tensor for the total response in Voigt notation in order (xx, yy, zz, xy, yz, zx) along z direction."""
    total_3d_piezoelectric_stress_coefficient_x: Optional[float] = field(default_factory=lambda: None)
    """The piezoelectric stress coefficient for the total response along x direction, in C/m^2."""
    total_3d_piezoelectric_stress_coefficient_y: Optional[float] = field(default_factory=lambda: None)
    """The piezoelectric stress coefficient for the total response along y direction, in C/m^2."""
    total_3d_piezoelectric_stress_coefficient_z: Optional[float] = field(default_factory=lambda: None)
    """The piezoelectric stress coefficient for the total response along z direction, in C/m^2."""
    total_3d_mean_absolute: Optional[float] = None
    """The mean absolute value of the elements of the total 3D piezoelectric tensor, in C/m^2."""
    total_3d_rms: Optional[float] = None
    """The root mean square value of the elements of the total 3D piezoelectric tensor, in C/m^2."""
    total_3d_frobenius_norm: Optional[float] = None
    """The Frobenius norm of the total 3D piezoelectric tensor, in C/m^2."""
    total_2d_tensor_x: Optional[List[List[float]]] = field(default_factory=lambda: None)
    """The full 2D piezoelectric tensor for the total response in Voigt notation in order (xx, yy, zz, xy, yz, zx) along x direction, in C/m."""
    total_2d_tensor_y: Optional[List[List[float]]] = field(default_factory=lambda: None)
    """The full 2D piezoelectric tensor for the total response in Voigt notation in order (xx, yy, zz, xy, yz, zx) along y direction, in C/m."""
    total_2d_tensor_z: Optional[List[List[float]]] = field(default_factory=lambda: None)
    """The full 2D piezoelectric tensor for the total response in Voigt notation in order (xx, yy, zz, xy, yz, zx) along z direction, in C/m."""

    ionic_3d_tensor_x: Optional[List[List[float]]] = field(default_factory=lambda: None)
    """The full 3D piezoelectric tensor for the ionic contribution in Voigt notation in order (xx, yy, zz, xy, yz, zx) along x direction, in C/m^2."""
    ionic_3d_tensor_y: Optional[List[List[float]]] = field(default_factory=lambda: None)
    """The full 3D piezoelectric tensor for the ionic contribution in Voigt notation in order (xx, yy, zz, xy, yz, zx) along y direction, in C/m^2."""
    ionic_3d_tensor_z: Optional[List[List[float]]] = field(default_factory=lambda: None)
    """The full 3D piezoelectric tensor for the ionic contribution in Voigt notation in order (xx, yy, zz, xy, yz, zx) along z direction, in C/m^2."""
    ionic_3d_piezoelectric_stress_coefficient_x: Optional[float] = field(default_factory=lambda: None)
    """The piezoelectric stress coefficient for the ionic contribution along x direction, in C/m^2."""
    ionic_3d_piezoelectric_stress_coefficient_y: Optional[float] = field(default_factory=lambda: None)
    """The piezoelectric stress coefficient for the ionic contribution along y direction, in C/m^2."""
    ionic_3d_piezoelectric_stress_coefficient_z: Optional[float] = field(default_factory=lambda: None)
    """The piezoelectric stress coefficient for the ionic contribution along z direction, in C/m^2."""
    ionic_3d_mean_absolute: Optional[float] = None
    """The mean absolute value of the elements of the ionic contribution to the 3D piezoelectric tensor, in C/m^2."""
    ionic_3d_rms: Optional[float] = None
    """The root mean square value of the elements of the ionic contribution to the 3D piezoelectric tensor, in C/m^2."""
    ionic_3d_frobenius_norm: Optional[float] = None
    """The Frobenius norm of the ionic contribution to the 3D piezoelectric tensor, in C/m^2."""
    ionic_2d_tensor_x: Optional[List[List[float]]] = field(default_factory=lambda: None)
    """The full 2D piezoelectric tensor for the ionic contribution in Voigt notation in order (xx, yy, zz, xy, yz, zx) along x direction, in C/m."""
    ionic_2d_tensor_y: Optional[List[List[float]]] = field(default_factory=lambda: None)
    """The full 2D piezoelectric tensor for the ionic contribution in Voigt notation in order (xx, yy, zz, xy, yz, zx) along y direction, in C/m."""
    ionic_2d_tensor_z: Optional[List[List[float]]] = field(default_factory=lambda: None)
    """The full 2D piezoelectric tensor for the ionic contribution in Voigt notation in order (xx, yy, zz, xy, yz, zx) along z direction, in C/m."""    

    electronic_3d_tensor_x: Optional[List[List[float]]] = field(default_factory=lambda: None)
    """The full 3D piezoelectric tensor for the electronic contribution in Voigt notation in order (xx, yy, zz, xy, yz, zx) along x direction, in C/m^2."""
    electronic_3d_tensor_y: Optional[List[List[float]]] = field(default_factory=lambda: None)
    """The full 3D piezoelectric tensor for the electronic contribution in Voigt notation in order (xx, yy, zz, xy, yz, zx) along y direction, in C/m^2."""
    electronic_3d_tensor_z: Optional[List[List[float]]] = field(default_factory=lambda: None)
    """The full 3D piezoelectric tensor for the electronic contribution in Voigt notation in order (xx, yy, zz, xy, yz, zx) along z direction, in C/m^2."""
    electronic_3d_piezoelectric_stress_coefficient_x: Optional[float] = field(default_factory=lambda: None)
    """The piezoelectric stress coefficient for the electronic contribution along x direction, in C/m^2."""
    electronic_3d_piezoelectric_stress_coefficient_y: Optional[float] = field(default_factory=lambda: None)
    """The piezoelectric stress coefficient for the electronic contribution along y direction, in C/m^2."""
    electronic_3d_piezoelectric_stress_coefficient_z: Optional[float] = field(default_factory=lambda: None)
    """The piezoelectric stress coefficient for the electronic contribution along z direction, in C/m^2."""
    electronic_3d_mean_absolute: Optional[float] = None
    """The mean absolute value of the elements of the electronic contribution to the 3D piezoelectric tensor, in C/m^2."""
    electronic_3d_rms: Optional[float] = None
    """The root mean square value of the elements of the electronic contribution to the 3D piezoelectric tensor, in C/m^2."""
    electronic_3d_frobenius_norm: Optional[float] = None
    """The Frobenius norm of the electronic contribution to the 3D piezoelectric tensor, in C/m^2."""
    electronic_2d_tensor_x: Optional[List[List[float]]] = field(default_factory=lambda: None)
    """The full 2D piezoelectric tensor for the electronic contribution in Voigt notation in order (xx, yy, zz, xy, yz, zx) along x direction, in C/m."""
    electronic_2d_tensor_y: Optional[List[List[float]]] = field(default_factory=lambda: None)
    """The full 2D piezoelectric tensor for the electronic contribution in Voigt notation in order (xx, yy, zz, xy, yz, zx) along y direction, in C/m."""
    electronic_2d_tensor_z: Optional[List[List[float]]] = field(default_factory=lambda: None)
    """The full 2D piezoelectric tensor for the electronic contribution in Voigt notation in order (xx, yy, zz, xy, yz, zx) along z direction, in C/m."""

@dataclass
class Polarization_DB(_DBDataMixin):
    """Data class for storing polarization data in the database."""

    total_dipole_norm: Optional[float] = None
    """The norm of the total dipole moment vector, including both ionic and electronic contributions."""
    total_dipole_moment: Optional[List[float, float, float]] = None
    """The total dipole moment vector, including both ionic and electronic contributions."""
    ionic_dipole_norm: Optional[float] = None
    """The norm of the ionic dipole moment vector."""
    ionic_dipole_moment: Optional[List[float, float, float]] = None
    """The ionic dipole moment vector."""
    electronic_dipole_norm: Optional[float] = None
    """The norm of the electronic dipole moment vector."""
    electronic_dipole_moment: Optional[List[float, float, float]] = None
    """The electronic dipole moment vector."""

@dataclass
class Potential_DB(_DBDataMixin):
    """Data class for storing potential data in the database."""

    has_total_potential: bool = False
    """Whether the total local potential data is available."""
    has_xc_potential: bool = False
    """Whether the exchange-correlation potential data is available."""
    has_hartree_potential: bool = False
    """Whether the Hartree potential data is available."""
    has_ionic_potential: bool = False
    """Whether the ionic potential data is available."""

    total_potential_mean: Optional[float] = None
    """The mean value of the total local potential, including all considered contributions, in eV."""
    total_potential_mean_up: Optional[float] = None
    """The mean value of the total local potential for spin-up electrons, including all considered contributions, in eV."""
    total_potential_mean_down: Optional[float] = None
    """The mean value of the total local potential for spin-down electrons, including all considered contributions, in eV."""
    total_potential_mean_magnetization: Optional[float] = None
    """The mean value of the total local potential magnetization, calculated as the mean of the norm of the magnetization potential vector, in eV."""

@dataclass
class Projector_DB(_DBDataMixin):
    """Data class for storing projector data in the database."""

    orbital_types: Optional[List[str]] = field(default_factory=lambda: None)
    """The types of orbitals used for the projectors, e.g. s, p, d, f."""

@dataclass
class RunInfo_DB(_DBDataMixin):
    """Data class for storing general run information in the database."""

    vasp_version: Optional[str] = None
    """The version of VASP used for the calculation."""
    
    grid_coarse_shape: Optional[List[int]] = field(default_factory=lambda: None)
    """The shape of the coarse grid used for the calculation, in the order (nx, ny, nz)."""
    grid_fine_shape: Optional[List[int]] = field(default_factory=lambda: None)
    """The shape of the fine grid used for the calculation, in the order (nx, ny, nz)."""
    is_success: Optional[bool] = None
    """Whether the calculation completed successfully."""
    fermi_energy: Optional[float] = None
    """The Fermi energy for the calculation according to the OUTCAR, in eV."""
    is_collinear: Optional[bool] = None
    """Whether the calculation was performed with collinear magnetism."""
    is_noncollinear: Optional[bool] = None
    """Whether the calculation was performed with non-collinear magnetism."""
    is_metallic: Optional[bool] = None
    """Whether the system is metallic based on the presence of states at the Fermi energy."""
    is_magnetic: Optional[bool] = None
    """Whether the system is magnetic based on the presence of a non-zero magnetic moment."""
    magnetization_order: Optional[str] = None
    """The order of the magnetization, e.g. ferromagnetic, antiferromagnetic, ferrimagnetic, etc."""

    system_tag: Optional[str] = None
    """The system tag from the INCAR file, if set, which may provide additional information about the system or calculation."""

    num_ionic_steps: Optional[int] = None
    """The number of ionic steps taken during the calculation."""

    has_selective_dynamics: Optional[bool] = None
    """Whether the calculation used selective dynamics."""
    has_lattice_velocities: Optional[bool] = None
    """Whether the calculation included lattice velocities."""
    has_ion_velocities: Optional[bool] = None
    """Whether the calculation stored ion velocities."""

    phonon_num_qpoints: Optional[int] = None
    """The number of q-points used in the phonon calculation."""
    phonon_num_modes: Optional[int] = None
    """The number of phonon modes calculated."""

@dataclass
class Stress_DB(_DBDataMixin):
    """Data class for storing stress data in the database."""

    initial_stress_mean: Optional[float] = None
    """The mean trace of the stress tensor across all atoms on the initial step, in GPa."""
    final_stress_mean: Optional[float] = None
    """The mean trace of the stress tensor across all atoms on the final step, in GPa."""
    final_stress_tensor: Optional[List[List[float]]] = field(default_factory=lambda: None)
    """The full 3D stress tensor on the final step, in GPa, in the order (xx, yy, zz, xy, yz, zx)."""

@dataclass
class Structure_DB(_DBDataMixin):
    """Data class for storing structure data in the database."""

    num_ions: Optional[int] = None
    """The number of ions in the structure."""
    dimensionality: Optional[int] = None
    """The dimensionality of the structure, as determined by the presence of vacuum along different lattice vectors.
    
    - 3 = bulk structure with no vacuum along any lattice vector
    - 2 = slab structure with vacuum along one lattice vector
    - 1 = multi-atom molecule or wire structure with vacuum along two lattice vectors
    - 0 = single-atom structure with vacuum along all three lattice vectors"""
    
    final_cell_volume: Optional[float] = None
    """The volume of the unit cell on the final step, in Å^3."""
    final_cell_area_2d: Optional[float] = None
    """The area of the unit cell in 2D materials, calculated as the product of the two lattice vectors that are not along the vacuum direction, in Å^2."""
    final_cell_area_2d_span: Optional[str] = None
    """The two lattice vectors that are used to calculate the area of the unit cell in 2D materials, in the format '12', '13', or '23'."""
    final_lattice_vector_1: Optional[List[float]] = field(default_factory=lambda: None)
    """The first lattice vector on the final step, in Å."""
    final_lattice_vector_2: Optional[List[float]] = field(default_factory=lambda: None)
    """The second lattice vector on the final step, in Å."""
    final_lattice_vector_3: Optional[List[float]] = field(default_factory=lambda: None)
    """The third lattice vector on the final step, in Å."""
    final_lattice_vector_1_length: Optional[float] = None
    """The length of the first lattice vector on the final step, in Å."""
    final_lattice_vector_2_length: Optional[float] = None
    """The length of the second lattice vector on the final step, in Å."""
    final_lattice_vector_3_length: Optional[float] = None
    """The length of the third lattice vector on the final step, in Å."""
    final_angle_alpha: Optional[float] = None
    """The angle between the second and third lattice vectors on the final step, in degrees."""
    final_angle_beta: Optional[float] = None
    """The angle between the first and third lattice vectors on the final step, in degrees."""
    final_angle_gamma: Optional[float] = None
    """The angle between the first and second lattice vectors on the final step, in degrees."""
    
    initial_cell_volume: Optional[float] = None
    """The volume of the unit cell on the initial step, in Å^3."""
    initial_cell_area_2d: Optional[float] = None
    """The area of the unit cell in 2D materials on the initial step, calculated as the product of the two lattice vectors that are not along the vacuum direction, in Å^2."""
    initial_cell_area_2d_span: Optional[str] = None
    """The two lattice vectors that are used to calculate the area of the unit cell in 2D materials on the initial step, in the format '12', '13', or '23'."""
    initial_lattice_vector_1: Optional[List[float]] = field(default_factory=lambda: None)
    """The first lattice vector on the initial step, in Å."""
    initial_lattice_vector_2: Optional[List[float]] = field(default_factory=lambda: None)
    """The second lattice vector on the initial step, in Å."""
    initial_lattice_vector_3: Optional[List[float]] = field(default_factory=lambda: None)
    """The third lattice vector on the initial step, in Å."""
    initial_lattice_vector_1_length: Optional[float] = None
    """The length of the first lattice vector on the initial step, in Å."""
    initial_lattice_vector_2_length: Optional[float] = None
    """The length of the second lattice vector on the initial step, in Å."""
    initial_lattice_vector_3_length: Optional[float] = None
    """The length of the third lattice vector on the initial step, in Å."""
    initial_angle_alpha: Optional[float] = None
    """The angle between the second and third lattice vectors on the initial step, in degrees."""
    initial_angle_beta: Optional[float] = None
    """The angle between the first and third lattice vectors on the initial step, in degrees."""
    initial_angle_gamma: Optional[float] = None
    """The angle between the first and second lattice vectors on the initial step, in degrees."""

@dataclass
class Velocity_DB(_DBDataMixin):
    """Data class for storing velocity data in the database."""

    final_velocity_min: Optional[float] = None
    """The minimum velocity across all atoms on the final step, in Å/fs."""
    final_velocity_max: Optional[float] = None
    """The maximum velocity across all atoms on the final step, in Å/fs."""
    final_velocity_median: Optional[float] = None
    """The median velocity across all atoms on the final step, in Å/fs."""
    final_velocity_mean: Optional[float] = None
    """The mean velocity across all atoms on the final step, in Å/fs."""
    final_velocity_std: Optional[float] = None
    """The standard deviation of the velocity across all atoms on the final step, in Å/fs."""
    final_index_velocity_max: Optional[int] = None
    """The index of the atom with the maximum velocity on the final step."""

    initial_velocity_min: Optional[float] = None
    """The minimum velocity across all atoms on the initial step, in Å/fs."""
    initial_velocity_max: Optional[float] = None
    """The maximum velocity across all atoms on the initial step, in Å/fs."""
    initial_index_velocity_max: Optional[int] = None
    """The index of the atom with the maximum velocity on the initial step."""

@dataclass
class Workfunction_DB(_DBDataMixin):
    """Data class for storing work function data in the database."""

    direction: Optional[int] = None
    """The direction along which the work function is calculated, in the order (x, y, z) corresponding to (1, 2, 3)."""
    workfunction_value: Optional[float] = None
    """The value of the work function along the specified direction, in eV."""
