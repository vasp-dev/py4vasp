from numpy.testing import assert_array_almost_equal_nulp
import pytest


class _Assert:
    @staticmethod
    def allclose(actual, desired):
        assert_array_almost_equal_nulp(actual, desired, 10)


@pytest.fixture
def Assert():
    return _Assert
