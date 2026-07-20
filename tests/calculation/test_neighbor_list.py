# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
import pytest

from py4vasp import raw
from py4vasp._calculation.neighbor_list import NeighborList, _replica_counts


def test_replica_counts_cubic():
    # A cube of edge a has perpendicular width a along every direction, so the
    # number of replicas is ceil(cutoff / a) in each direction.
    lattice = 4.0 * np.eye(3)
    np.testing.assert_array_equal(_replica_counts(lattice, 4.0), [1, 1, 1])
    np.testing.assert_array_equal(_replica_counts(lattice, 5.0), [2, 2, 2])
    np.testing.assert_array_equal(_replica_counts(lattice, 9.0), [3, 3, 3])


def test_replica_counts_tilted_cell():
    # a0 and a1 enclose a small angle (~5.7°). Shearing a1 keeps the volume at 1
    # but shrinks the perpendicular width along direction 0 to 1/sqrt(1+k^2)=0.1
    # while the other two directions keep width 1.
    k = np.sqrt(99.0)
    lattice = np.array([[1.0, 0.0, 0.0], [k, 1.0, 0.0], [0.0, 0.0, 1.0]])
    counts = _replica_counts(lattice, 0.95)
    # ceil(0.95 / 0.1) = 10 replicas are needed along the tilted direction
    assert counts[0] == 10
    # the other two directions have width 1, so a single replica suffices
    assert counts[1] == 1
    assert counts[2] == 1
    # a naive |a_i|-based count would use ceil(0.95 / |a0|=1) = 1 replica along
    # direction 0 and miss neighbors; the perpendicular-width criterion must not.
    naive = int(np.ceil(0.95 / np.linalg.norm(lattice[0])))
    assert counts[0] > naive
