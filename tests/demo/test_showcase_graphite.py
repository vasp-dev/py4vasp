# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
import pytest

from py4vasp._calculation.bandgap import BandgapHandler
from py4vasp._calculation.density import Density
from py4vasp._demo.showcase import (
    cell,
    density,
    partial_density,
    structure,
    workfunction,
)

# Literature values for graphite: a = 2.4612 A, c = 6.7079 A, so the layers sit
# 3.35395 A apart and the carbon atoms 1.42104 A from each other in a layer.
LATTICE_CONSTANT = 2.4612
INTERLAYER = 3.35395
# The work function of highly oriented pyrolytic graphite, as the literature reports it.
WORK_FUNCTION = 4.6
BOND_LENGTH = LATTICE_CONSTANT / np.sqrt(3)
NUMBER_LAYERS = 4
NUMBER_ATOMS = 2 * NUMBER_LAYERS


@pytest.fixture
def raw_structure():
    return structure.Graphite()


@pytest.fixture
def cartesian(raw_structure):
    positions = np.array(raw_structure.positions)
    lattice_vectors = np.array(raw_structure.cell.lattice_vectors)
    return positions @ lattice_vectors


def test_slab_holds_four_layers_of_carbon(raw_structure):
    assert np.array(raw_structure.positions).shape == (NUMBER_ATOMS, 3)
    stoichiometry = raw_structure.stoichiometry
    assert list(np.array(stoichiometry.number_ion_types)) == [NUMBER_ATOMS]
    ion_types = [name.decode().strip() for name in np.array(stoichiometry.ion_types)]
    assert ion_types == ["C"]


def test_in_plane_cell_is_the_literature_one(raw_structure, Assert):
    lattice_vectors = np.array(raw_structure.cell.lattice_vectors)
    lengths = np.linalg.norm(lattice_vectors[:2], axis=1)
    Assert.allclose(lengths, np.full(2, LATTICE_CONSTANT))
    angle = np.degrees(np.arccos(np.dot(*lattice_vectors[:2]) / np.prod(lengths)))
    Assert.allclose(angle, 120.0)


def test_layers_sit_at_the_literature_distance(cartesian, Assert):
    heights = np.unique(np.round(cartesian[:, 2], 8))
    assert len(heights) == NUMBER_LAYERS
    Assert.allclose(np.diff(heights), np.full(NUMBER_LAYERS - 1, INTERLAYER))


def test_carbon_atoms_are_a_honeycomb(cartesian, Assert):
    lattice_vectors = np.array(structure.Graphite().cell.lattice_vectors)
    lowest = cartesian[np.isclose(cartesian[:, 2], np.min(cartesian[:, 2]))]
    shifts = np.array([[i, j, 0] for i in (-1, 0, 1) for j in (-1, 0, 1)])
    images = lowest[:, np.newaxis, :] + shifts @ lattice_vectors
    distances = np.linalg.norm(images[0] - images[1][:, np.newaxis], axis=-1)
    Assert.allclose(np.min(distances), BOND_LENGTH)


def test_layers_are_stacked_the_way_graphite_is(raw_structure, Assert):
    # Bernal stacking: half the atoms of a layer sit above an atom of the layer below
    # and half above the centre of a hexagon, which is what distinguishes graphite
    # from a stack of aligned graphene sheets
    positions = np.array(raw_structure.positions)
    layers = positions.reshape(NUMBER_LAYERS, 2, 3)[:, :, :2]
    for lower, upper in zip(layers, layers[1:]):
        eclipsed = np.isclose(lower[:, np.newaxis], upper).all(axis=-1)
        assert np.count_nonzero(eclipsed) == 1
    Assert.allclose(layers[0], layers[2])
    Assert.allclose(layers[1], layers[3])


def test_vacuum_is_wide_enough_for_a_scanning_tip(cartesian, raw_structure):
    height = np.array(raw_structure.cell.lattice_vectors)[2, 2]
    thickness = np.ptp(cartesian[:, 2])
    # py4vasp refuses to simulate a scanning tunneling microscope with less than this
    assert height - thickness > 5.0


def test_vacuum_lies_along_the_third_lattice_vector(raw_structure):
    # py4vasp checks that the atoms span less of the third direction than of the other
    # two before it places a tip above the surface
    positions = np.array(raw_structure.positions)
    span = np.ptp(positions, axis=0)
    assert span[2] < span[0] and span[2] < span[1]


def test_slab_is_centred_in_the_cell(cartesian, raw_structure, Assert):
    # an equal amount of vacuum on either side keeps the two surfaces equivalent, which
    # is what makes a single work function meaningful
    height = np.array(raw_structure.cell.lattice_vectors)[2, 2]
    below = np.min(cartesian[:, 2])
    above = height - np.max(cartesian[:, 2])
    Assert.allclose(below, above)


def test_cell_has_no_extra_scale(raw_structure):
    assert np.array(raw_structure.cell.scale) == 1.0


@pytest.fixture
def raw_partial_density():
    return partial_density.Graphite()


@pytest.fixture
def surface_state(raw_partial_density):
    """The partial charge with the grid axes in the order the field was evaluated in."""
    return np.array(raw_partial_density.partial_charge)[0, 0, 0].T


def test_partial_charge_is_summed_over_bands_and_kpoints(raw_partial_density):
    # py4vasp refuses to simulate a microscope from a separated calculation, and zero
    # is how VASP marks a summed band or k point
    assert list(np.array(raw_partial_density.bands)) == [0]
    assert list(np.array(raw_partial_density.kpoints)) == [0]
    assert np.array(raw_partial_density.partial_charge).shape[:3] == (1, 1, 1)


def test_partial_charge_shares_the_grid_with_the_density(raw_partial_density, Assert):
    # the Bader basins a partial charge is integrated in are built from the density, and
    # py4vasp refuses to combine two grids of different shape
    charge = np.array(density.Graphite().charge)[0]
    Assert.allclose(np.array(raw_partial_density.grid), np.array(charge.shape[::-1]))
    assert np.array(raw_partial_density.partial_charge)[0, 0, 0].shape == charge.shape


def test_partial_charge_peaks_at_one(surface_state, Assert):
    # py4vasp draws its isosurface at an absolute level, so the largest value has to be
    # a known one for the default level to mean anything
    assert np.all(surface_state >= 0.0)
    Assert.allclose(np.max(surface_state), 1.0)


def test_default_isolevel_encloses_the_slab(surface_state):
    # a level that encloses almost nothing shows no surface, and one that encloses
    # almost everything shows the box; the surface has to hug the atoms
    enclosed = np.mean(surface_state > 0.2)
    assert 0.15 < enclosed < 0.35


def test_state_decays_into_the_vacuum(surface_state, raw_partial_density):
    heights = _height_of_every_plane(raw_partial_density)
    top = np.max(np.array(structure.Graphite().positions)[:, 2]) * cell.GRAPHITE_HEIGHT
    far_above = surface_state[:, :, heights > top + 3.0]
    assert np.max(far_above) < 1e-3
    # but not so fast that a tip held above the surface has nothing to tunnel into
    just_above = surface_state[:, :, np.abs(heights - top - 1.0) < 0.1]
    assert np.max(just_above) > 1e-3


def _height_of_every_plane(raw_partial_density):
    number_planes = np.array(raw_partial_density.grid)[2]
    return np.arange(number_planes) / number_planes * cell.GRAPHITE_HEIGHT


def test_only_one_sublattice_shows_at_the_surface(surface_state, raw_partial_density):
    # Bernal stacking makes the two atoms of a layer inequivalent, so a microscope
    # image of graphite shows a triangular lattice of one maximum per cell rather than
    # the honeycomb the atoms form
    heights = _height_of_every_plane(raw_partial_density)
    positions = np.array(structure.Graphite().positions)
    top = np.max(positions[:, 2]) * cell.GRAPHITE_HEIGHT
    plane = surface_state[:, :, np.argmin(np.abs(heights - top - 1.0))]
    counts = np.array(raw_partial_density.grid)[:2]
    exposed = plane[tuple(np.rint(np.array([2 / 3, 1 / 3]) * counts).astype(int))]
    eclipsed = plane[tuple(np.rint(np.array([1 / 3, 2 / 3]) * counts).astype(int))]
    assert exposed > 2 * eclipsed
    assert np.isclose(np.max(plane), exposed)


def test_exposed_sublattice_carries_more_of_the_state():
    positions = np.array(structure.Graphite().positions)
    weights = partial_density.sublattice_weights(positions)
    # every layer has exactly one atom eclipsed by the layer next to it
    assert np.count_nonzero(weights == partial_density.EXPOSED_WEIGHT) == 4
    assert np.count_nonzero(weights == partial_density.ECLIPSED_WEIGHT) == 4
    assert partial_density.ECLIPSED_WEIGHT < partial_density.EXPOSED_WEIGHT


def test_density_holds_the_valence_electrons_of_every_carbon(Assert):
    # VASP multiplies a density by the volume of the cell, so the mean over the grid is
    # the number of electrons rather than a density
    charge = np.array(density.Graphite().charge)
    electrons = NUMBER_ATOMS * density.VALENCE_ELECTRONS["C"]
    assert abs(np.mean(charge) / electrons - 1) < 1e-9


def test_bader_analysis_finds_the_electrons_of_every_atom(Assert):
    quantity = Density.from_data(density.Graphite())
    charges = quantity.bader_charge(bader_analysis=quantity.bader_analysis())
    values = np.array(list(charges.values()))
    assert len(values) == NUMBER_ATOMS
    electrons = NUMBER_ATOMS * density.VALENCE_ELECTRONS["C"]
    # the basins partition the whole cell, so they account for every electron but the
    # Gaussian tails the images leave out
    assert abs(np.sum(values) / electrons - 1) < 1e-9
    assert np.all(np.abs(values - density.VALENCE_ELECTRONS["C"]) < 0.5)


@pytest.fixture
def raw_workfunction():
    return workfunction.Graphite()


@pytest.fixture
def profile(raw_workfunction):
    return (
        np.array(raw_workfunction.distance),
        np.array(raw_workfunction.average_potential),
    )


def test_potential_is_averaged_over_the_planes_normal_to_the_vacuum(raw_workfunction):
    assert int(raw_workfunction.idipol) == 3


def test_distance_spans_the_cell(profile, raw_partial_density):
    distance, _ = profile
    assert distance[0] == 0.0
    assert np.all(np.diff(distance) > 0)
    assert distance[-1] < cell.GRAPHITE_HEIGHT
    # the same points the grid quantities are sampled on, so the profiles line up
    assert len(distance) == np.array(raw_partial_density.grid)[2]


def test_potential_is_flat_in_the_vacuum(profile):
    distance, potential = profile
    heights = np.array(structure.Graphite().positions)[:, 2] * cell.GRAPHITE_HEIGHT
    # the slab is centred, so the vacuum wraps around the boundary of the cell
    in_vacuum = (distance < np.min(heights) - 4.5) | (distance > np.max(heights) + 4.5)
    assert np.count_nonzero(in_vacuum) > 20
    assert np.ptp(potential[in_vacuum]) < 0.05


def test_potential_is_deep_inside_the_slab(profile):
    distance, potential = profile
    heights = np.array(structure.Graphite().positions)[:, 2] * cell.GRAPHITE_HEIGHT
    inside = (distance > np.min(heights)) & (distance < np.max(heights))
    # deep and without a gap all the way through: the potential of a slab does not
    # return to the vacuum level between its layers, because it is the density
    # convolved with a Coulomb interaction rather than the density itself
    assert np.all(potential[inside] < -10.0)


def test_vacuum_potential_is_the_level_the_potential_settles_at(
    raw_workfunction, profile, Assert
):
    _, potential = profile
    vacuum = np.array(raw_workfunction.vacuum_potential)
    assert vacuum.shape == (2,)
    # a slab centred in its cell exposes the same surface on both sides
    Assert.allclose(vacuum[0], vacuum[1])
    assert abs(vacuum[0] - np.max(potential)) < 1e-3


def test_work_function_is_the_literature_value(raw_workfunction, Assert):
    # the literature value is spelled out here rather than taken from the producer, so
    # that changing the constant there does not silently change what is being claimed
    vacuum = np.array(raw_workfunction.vacuum_potential)[0]
    Assert.allclose(vacuum - raw_workfunction.fermi_energy, WORK_FUNCTION)


def test_fermi_energy_lies_between_the_vacuum_and_the_bottom_of_the_potential(
    raw_workfunction, profile
):
    _, potential = profile
    assert np.min(potential) < raw_workfunction.fermi_energy < np.max(potential)


def test_surface_has_no_band_gap(raw_workfunction, Assert):
    # graphite is a semimetal, so the bands touch and every gap the work function
    # reports alongside the vacuum level is zero
    gap = BandgapHandler.from_data(raw_workfunction.reference_potential)
    Assert.allclose(gap.fundamental(), 0.0)
    Assert.allclose(gap.direct(), 0.0)
    Assert.allclose(gap.valence_band_maximum(), raw_workfunction.fermi_energy)
