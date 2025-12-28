"""Tests for combinatorics utilities in cross_document_utils."""

import random
from math import comb
from itertools import combinations

import pytest

from yourbench.utils.cross_document_utils import (
    _unrank_comb,
    _floyd_sample_indices,
    _sample_exact_combinations,
)


class TestUnrankComb:
    """Tests for combination unranking."""

    def test_basic_unranking(self):
        """Basic unranking produces expected combinations."""
        assert _unrank_comb(5, 3, 0) == [0, 1, 2]
        assert _unrank_comb(5, 3, 9) == [2, 3, 4]  # Last
        assert _unrank_comb(4, 4, 0) == [0, 1, 2, 3]  # k == n
        assert _unrank_comb(5, 0, 0) == []  # k == 0

    def test_invalid_inputs(self):
        """Invalid inputs raise ValueError."""
        with pytest.raises(ValueError):
            _unrank_comb(4, 5, 0)  # k > n
        with pytest.raises(ValueError):
            _unrank_comb(5, 2, -1)  # negative rank
        with pytest.raises(ValueError):
            _unrank_comb(5, 2, comb(5, 2))  # rank out of bounds

    def test_matches_colex_order(self):
        """All ranks match expected colex ordering."""
        for n in range(1, 8):
            for k in range(0, n + 1):
                expected = sorted(combinations(range(n), k), key=lambda x: x[::-1])
                for rank, exp in enumerate(expected):
                    assert _unrank_comb(n, k, rank) == list(exp)


class TestFloydSample:
    """Tests for Floyd sampling."""

    def test_basic_sampling(self):
        """Samples have correct size and range."""
        result = _floyd_sample_indices(10, 5)
        assert len(result) == 5
        assert all(0 <= x < 10 for x in result)

        assert _floyd_sample_indices(10, 10) == set(range(10))  # full
        assert _floyd_sample_indices(10, 0) == set()  # empty

    def test_invalid_sample_size(self):
        with pytest.raises(ValueError):
            _floyd_sample_indices(5, 6)

    def test_deterministic_with_seed(self):
        """Same seed produces same result."""
        r1, r2 = random.Random(42), random.Random(42)
        assert _floyd_sample_indices(100, 10, rng=r1) == _floyd_sample_indices(100, 10, rng=r2)


class TestSampleExactCombinations:
    """Tests for sampling exact combinations."""

    def test_basic_sampling(self):
        """Samples have correct structure."""
        objects = ["a", "b", "c", "d"]
        result = _sample_exact_combinations(objects, k=2, N=3)
        assert len(result) == 3
        for combo in result:
            assert len(combo) == 2
            assert len(set(combo)) == 2  # unique within
            assert all(x in objects for x in combo)

    def test_all_unique_combinations(self):
        """All sampled combinations are unique."""
        samples = _sample_exact_combinations(list(range(6)), k=3, N=10)
        assert len({tuple(sorted(s)) for s in samples}) == 10

    def test_exhaustive_sampling(self):
        """Can sample all possible combinations."""
        objects = ["a", "b", "c", "d"]
        samples = _sample_exact_combinations(objects, k=2, N=comb(4, 2))
        assert {tuple(sorted(s)) for s in samples} == set(combinations(objects, 2))

    def test_invalid_request(self):
        with pytest.raises(ValueError):
            _sample_exact_combinations([1, 2, 3], k=2, N=4)  # C(3,2)=3 < 4

    def test_deterministic_with_seed(self):
        r1, r2 = random.Random(123), random.Random(123)
        assert _sample_exact_combinations(list(range(20)), k=4, N=5, rng=r1) == _sample_exact_combinations(
            list(range(20)), k=4, N=5, rng=r2
        )
