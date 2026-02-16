"""Smoke tests for src/diffu/random_walk.py."""

import numpy as np


def test_random_walk1D_shape_and_start():
    """1D walk returns correct shape and starts at x0."""
    from src.diffu.random_walk import random_walk1D

    np.random.seed(42)
    pos = random_walk1D(x0=0, N=10, p=0.5, random=np.random)
    assert pos.shape == (11,)
    assert pos[0] == 0


def test_random_walk1D_steps_unit():
    """Each step should be exactly +1 or -1."""
    from src.diffu.random_walk import random_walk1D

    np.random.seed(42)
    pos = random_walk1D(x0=0, N=100, p=0.5, random=np.random)
    diffs = np.abs(np.diff(pos))
    assert np.all(diffs == 1)


def test_random_walk1D_vec_matches_scalar():
    """Vectorized walk matches scalar walk for the same seed."""
    from src.diffu.random_walk import random_walk1D, random_walk1D_vec

    x0, N, p = 2, 20, 0.6
    np.random.seed(10)
    scalar = random_walk1D(x0, N, p, random=np.random)
    np.random.seed(10)
    vec = random_walk1D_vec(x0, N, p)
    np.testing.assert_array_equal(scalar, vec)
