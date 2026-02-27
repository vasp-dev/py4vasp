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

    fundamental_spin_independent: Optional[float] = None
    """The value of the fundamental band gap, in eV. This can be used to quickly determine if the system is metallic or insulating."""
    fundamental_spin_up: Optional[float] = None
    """The value of the fundamental band gap for spin-up electrons, in eV."""
    fundamental_spin_down: Optional[float] = None
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

    direct_spin_independent: Optional[float] = None
    """The value of the direct band gap, in eV. This can be used to quickly determine if the system has a direct or indirect band gap."""
    direct_spin_up: Optional[float] = None
    """The value of the direct band gap for spin-up electrons, in eV."""
    direct_spin_down: Optional[float] = None
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
    kpoint_direct_spin_independent: Optional[List[float]] = field(default_factory=lambda: None)
    """The k-point where the direct band gap occurs, in fractional coordinates."""
    kpoint_direct_spin_up: Optional[List[float]] = field(default_factory=lambda: None)
    """The k-point where the direct band gap for spin-up electrons occurs, in fractional coordinates."""
    kpoint_direct_spin_down: Optional[List[float]] = field(default_factory=lambda: None)
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