import numpy as np

from rsfabc_portfolio.analysis import (
    exact_paired_wilcoxon,
    holm_adjust,
    moving_block_indices,
)


def test_exact_wilcoxon_all_positive_three_pairs():
    statistic, p_value, effect = exact_paired_wilcoxon([1.0, 2.0, 3.0])
    assert statistic == 6.0
    assert p_value == 0.25
    assert effect == 1.0


def test_exact_wilcoxon_zero_differences_are_neutral():
    statistic, p_value, effect = exact_paired_wilcoxon([0.0, 0.0])
    assert statistic == 0.0
    assert p_value == 1.0
    assert effect == 0.0


def test_holm_adjustment_is_monotone_in_sorted_order():
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert np.allclose(adjusted, [0.03, 0.06, 0.06])


def test_moving_block_indices_have_requested_length_and_valid_blocks():
    rng = np.random.default_rng(123)
    indices = moving_block_indices(rng, observations=25, block_length=6)
    assert len(indices) == 25
    assert indices.min() >= 0 and indices.max() < 25
    for start in range(0, 24, 6):
        block = indices[start : min(start + 6, 25)]
        assert np.all(np.diff(block) == 1)
