"""Tests for cmpdist.

The reference values are computed straight from the CMP definition with plain
Python floats, independently of the numba kernels under test.
"""

import math

import numpy as np
import pytest
from scipy import stats

from cmpdist import CMP, cmp


@pytest.fixture
def d():
    return CMP(seed=12345)


def reference_log_terms(rate, nu, max_x):
    """``log(rate**k / k!**nu)`` for k = 0..max_x, in plain Python.

    Done in log space because ``k!**nu`` overflows a float well before k = 50.
    """
    return [k * math.log(rate) - nu * math.lgamma(k + 1) for k in range(max_x + 1)]


def reference_pmf(rate, nu, max_x):
    """CMP pmf on 0..max_x, straight from the definition."""
    if rate <= 0:
        return np.eye(1, max_x + 1)[0]
    logs = reference_log_terms(rate, nu, max_x)
    m = max(logs)
    terms = [math.exp(t - m) for t in logs]
    z = sum(terms)
    return np.array([t / z for t in terms])


def reference_log_norm(rate, nu, max_x):
    """log Z(rate, nu) truncated at max_x, in plain Python."""
    logs = reference_log_terms(rate, nu, max_x)
    m = max(logs)
    return m + math.log(sum(math.exp(t - m) for t in logs))


# --------------------------------------------------------------------------
# pmf
# --------------------------------------------------------------------------

@pytest.mark.parametrize("nu", [0.5, 0.6, 1.0, 1.5, 2.5, 4.0])
@pytest.mark.parametrize("lam", [0.3, 1.0, 4.0])
def test_pmf_matches_definition(d, lam, nu):
    max_x = d.max_count(lam)
    got = d.pmf(np.arange(max_x + 1), lam, nu)
    want = reference_pmf(d.rate(lam, nu), nu, max_x)
    assert np.allclose(got, want, rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("nu", [0.6, 1.0, 1.9, 4.0])
def test_pmf_sums_to_one(d, nu):
    for lam in [0.05, 1.0, 10.0, 200.0]:
        total = d.pmf(np.arange(d.max_count(lam) + 1), lam, nu).sum()
        assert total == pytest.approx(1.0, rel=1e-12)


def test_pmf_poisson_limit(d):
    k = np.arange(40)
    for lam in [0.5, 3.0, 12.0]:
        assert np.allclose(d.pmf(k, lam, 1.0), stats.poisson.pmf(k, lam), atol=1e-12)


def test_pmf_shape_and_out_of_range(d):
    assert d.pmf(3, 2.0, 1.0).shape == ()      # scalar in, 0-d out
    assert np.shape(d.pmf([[0, 1], [2, 3]], 2.0, 1.0)) == (2, 2)
    assert d.pmf([-1, 10 ** 6], 2.0, 1.0).tolist() == [0.0, 0.0]


# --------------------------------------------------------------------------
# sample: shapes, types, reproducibility
# --------------------------------------------------------------------------

def test_sample_scalar_returns_int(d):
    out = d.sample(5.0, 1.0)
    assert isinstance(out, int)
    assert out >= 0


def test_sample_size_and_shapes(d):
    assert d.sample(5.0, 1.0, size=7).shape == (7,)
    assert d.sample(np.full(9, 5.0), 1.0).shape == (9,)
    assert d.sample(np.full((3, 4), 5.0), 1.0).shape == (3, 4)
    assert d.sample(np.zeros(0), 1.0).shape == (0,)


def test_sample_dtype_is_int64(d):
    assert d.sample(np.full(5, 3.0), 1.0).dtype == np.int64


def test_reproducible_from_seed():
    a = CMP(seed=7).sample(np.full(500, 4.0), 0.8)
    b = CMP(seed=7).sample(np.full(500, 4.0), 0.8)
    assert np.array_equal(a, b)
    c = CMP(seed=8).sample(np.full(500, 4.0), 0.8)
    assert not np.array_equal(a, c)


def test_accepts_shared_generator():
    rng = np.random.default_rng(3)
    a = CMP(seed=rng).sample(np.full(50, 4.0), 1.0)
    rng2 = np.random.default_rng(3)
    b = CMP(seed=rng2).sample(np.full(50, 4.0), 1.0)
    assert np.array_equal(a, b)


def test_generator_advances_between_calls(d):
    a = d.sample(np.full(200, 5.0), 1.0)
    b = d.sample(np.full(200, 5.0), 1.0)
    assert not np.array_equal(a, b)


# --------------------------------------------------------------------------
# sample: statistics
# --------------------------------------------------------------------------

@pytest.mark.parametrize("lam", [0.05, 0.2, 2.0, 5.0, 20.0, 100.0])
def test_poisson_reduction(lam):
    """nu = 1 must reproduce Poisson, mean and variance both."""
    s = CMP(seed=1).sample(np.full(200_000, lam), 1.0)
    assert s.mean() == pytest.approx(lam, rel=0.02)
    assert s.var() == pytest.approx(lam, rel=0.05)


def test_poisson_reduction_distribution():
    """nu = 1 draws must follow the Poisson pmf, not just match its moments."""
    lam = 4.0
    s = CMP(seed=2).sample(np.full(200_000, lam), 1.0)
    k = np.arange(0, 20)
    observed = np.bincount(s, minlength=20)[:20]
    expected = stats.poisson.pmf(k, lam) * s.size
    keep = expected > 30                       # keep well-populated bins
    chi2 = (((observed - expected) ** 2) / expected)[keep].sum()
    assert chi2 / keep.sum() < 3.0


@pytest.mark.parametrize("lam", [2.0, 20.0])
def test_mean_is_recovered_across_nu(lam):
    for nu in [0.6, 0.8, 1.0, 1.5, 1.9]:
        s = CMP(seed=4).sample(np.full(200_000, lam), nu)
        assert s.mean() == pytest.approx(lam, rel=0.05), f"lam={lam} nu={nu}"


def test_dispersion_is_monotonic_in_nu():
    """var/mean must fall as nu rises, crossing 1 at the Poisson point."""
    ratios = []
    for nu in [0.6, 0.8, 1.0, 1.5, 1.9]:
        s = CMP(seed=5).sample(np.full(200_000, 10.0), nu)
        ratios.append(s.var() / s.mean())
    assert all(a > b for a, b in zip(ratios, ratios[1:])), ratios
    assert ratios[2] == pytest.approx(1.0, rel=0.05)     # nu = 1
    assert ratios[0] > 1.1 and ratios[-1] < 0.9


def test_sampled_histogram_matches_pmf():
    lam, nu = 6.0, 1.4
    d = CMP(seed=6)
    s = d.sample(np.full(300_000, lam), nu)
    k = np.arange(d.max_count(lam) + 1)
    observed = np.bincount(s, minlength=k.size)[:k.size] / s.size
    assert np.abs(observed - d.pmf(k, lam, nu)).max() < 0.005


def test_zero_mean_gives_zero_counts(d):
    assert np.all(d.sample(np.zeros(1000), 1.0) == 0)


# --------------------------------------------------------------------------
# per-element nu
# --------------------------------------------------------------------------

def test_array_nu_matches_scalar_nu_on_same_deviates():
    """A constant nu array must be bit-for-bit identical to the scalar path."""
    lam = np.abs(np.random.default_rng(1).lognormal(0, 1, 5000))
    for nu in [0.6, 1.0, 1.8]:
        a = CMP(seed=10).sample(lam, nu)
        b = CMP(seed=10).sample(lam, np.full(lam.size, nu))
        assert np.array_equal(a, b), f"nu={nu}"


def test_per_element_nu_actually_varies():
    """Blocks drawn with different nu must show the matching dispersion."""
    block = 60_000
    nus = np.array([0.6, 1.0, 1.9])
    lam = np.full(block * nus.size, 10.0)
    s = CMP(seed=11).sample(lam, np.repeat(nus, block))

    ratios = [s[j * block:(j + 1) * block].var() / s[j * block:(j + 1) * block].mean()
              for j in range(nus.size)]
    assert all(a > b for a, b in zip(ratios, ratios[1:])), ratios
    assert ratios[1] == pytest.approx(1.0, rel=0.05)


def test_nu_length_mismatch_raises(d):
    with pytest.raises(ValueError, match="scalar or match"):
        d.sample(np.zeros(10), np.ones(7))


def test_nu_matches_size_argument(d):
    assert d.sample(3.0, np.full(5, 1.0), size=5).shape == (5,)
    with pytest.raises(ValueError):
        d.sample(3.0, np.full(4, 1.0), size=5)


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("nu", [0.55, 0.49, 0.3, 0.0, 4.0, 4.01, 5.0, -1.0])
def test_nu_outside_valid_range_raises(d, nu):
    """Default policy rejects nu outside the open interval (0.55, 4)."""
    with pytest.raises(ValueError, match=r"0\.55 < nu < 4"):
        d.sample(np.full(10, 1.0), nu)


@pytest.mark.parametrize("nu", [0.6, 0.75, 1.0, 1.5, 2.0, 3.0])
def test_nu_inside_valid_range_accepted(d, nu):
    """Values strictly between 0.55 and 4 are accepted."""
    assert d.sample(np.full(10, 1.0), nu).shape == (10,)


def test_array_nu_validated_elementwise(d):
    nu = np.full(10, 1.0)
    nu[4] = 5.0
    with pytest.raises(ValueError, match=r"0\.55 < nu < 4"):
        d.sample(np.full(10, 1.0), nu)


def test_invalid_nu_reported_in_message(d):
    with pytest.raises(ValueError, match="0.3"):
        d.sample(np.full(10, 1.0), 0.3)


def test_on_invalid_nu_warn_mode():
    lenient = CMP(seed=0, on_invalid_nu="warn")
    with pytest.warns(RuntimeWarning, match=r"0\.5 <= nu <= 4"):
        out = lenient.sample(np.full(10, 1.0), 0.3)
    assert out.shape == (10,)


def test_on_invalid_nu_ignore_mode():
    quiet = CMP(seed=0, on_invalid_nu="ignore")
    with warnings_as_errors():
        quiet.sample(np.full(10, 1.0), 0.3)


def test_nu_range_none_disables_check():
    quiet = CMP(seed=0, nu_range=None)
    with warnings_as_errors():
        quiet.sample(np.full(10, 1.0), 0.3)


def test_custom_nu_range():
    wide = CMP(seed=0, nu_range=(0.1, 5.0))
    assert wide.sample(np.full(10, 1.0), 0.3).shape == (10,)
    with pytest.raises(ValueError, match=r"0\.1 < nu < 5"):
        wide.sample(np.full(10, 1.0), 0.1)


def test_bad_policy_rejected():
    with pytest.raises(ValueError, match="on_invalid_nu must be one of"):
        CMP(on_invalid_nu="explode")


def test_validation_applies_to_pmf_and_rate(d):
    with pytest.raises(ValueError, match=r"0\.55 < nu < 4"):
        d.pmf([0, 1], 1.0, 0.3)
    with pytest.raises(ValueError, match=r"0\.55 < nu < 4"):
        d.rate(1.0, 4.01)


def test_negative_or_nonfinite_lam_raises(d):
    with pytest.raises(ValueError, match="non-negative"):
        d.sample(np.array([1.0, -1.0]), 1.0)
    with pytest.raises(ValueError, match="finite"):
        d.sample(np.array([1.0, np.nan]), 1.0)


def test_size_mismatch_raises(d):
    with pytest.raises(ValueError, match="does not match"):
        d.sample(np.zeros(4), 1.0, size=9)


def test_pmf_rejects_array_lam(d):
    with pytest.raises(TypeError, match="scalar"):
        d.pmf([0, 1], np.array([1.0, 2.0]), 1.0)


# --------------------------------------------------------------------------
# rate / normalisation
# --------------------------------------------------------------------------

def test_rate_reproduces_requested_mean(d):
    """The rate returned must give a distribution with the requested mean."""
    for lam in [1.0, 5.0, 50.0]:
        for nu in [0.7, 1.0, 1.6]:
            max_x = d.max_count(lam)
            p = reference_pmf(d.rate(lam, nu), nu, max_x)
            mean = (np.arange(max_x + 1) * p).sum()
            assert mean == pytest.approx(lam, rel=0.05), f"lam={lam} nu={nu}"


def test_rate_is_poisson_lambda_at_nu_one(d):
    """At nu = 1 both branches collapse to mu, so the rate must be exact."""
    for lam in [0.0, 0.2, 0.5, 0.51, 0.7, 0.8, 3.0, 40.0, 5000.0]:
        assert d.rate(lam, 1.0) == pytest.approx(lam, rel=1e-12, abs=1e-15)


def test_rate_is_continuous_across_the_blend(d):
    """No jump in rate(mu) around the small/large crossover."""
    for nu in [0.6, 1.0, 1.5, 1.9]:
        mu = np.linspace(0.01, 3.0, 4000)
        rates = np.array([d.rate(m, nu) for m in mu])
        assert np.all(np.diff(rates) > 0), f"rate(mu) not increasing at nu={nu}"
        jumps = np.abs(np.diff(rates)) / np.diff(mu)
        assert jumps.max() < 20 * np.median(jumps), f"kink at nu={nu}"


def test_log_norm_matches_direct_sum(d):
    for rate, nu in [(0.5, 1.0), (3.0, 1.4), (10.0, 0.8)]:
        direct = reference_log_norm(rate, nu, 200)
        assert d.log_norm(rate, nu, max_x=200) == pytest.approx(direct, rel=1e-12)


def test_log_norm_poisson_limit(d):
    """At nu = 1, Z is just exp(rate)."""
    for rate in [0.5, 3.0, 20.0]:
        assert d.log_norm(rate, 1.0, max_x=400) == pytest.approx(rate, rel=1e-12)


@pytest.mark.parametrize("nu", [0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0])
def test_mean_recovered_within_5_percent_across_nu_range(nu):
    """The accuracy claim for nu_range, pinned so it cannot drift silently.

    Analytic (noise-free): the mean of the resulting distribution must be
    within 5% of the requested one, everywhere from lam = 1e-3 to 1e3.
    """
    d = CMP(nu_range=None)
    for lam in np.geomspace(1e-3, 1e3, 60):
        k = np.arange(d.max_count(lam) + 1)
        mean = (k * d.pmf(k, lam, nu)).sum()
        assert abs(mean / lam - 1) <= 0.05, f"nu={nu} lam={lam:g} -> {mean / lam - 1:+.4f}"


def test_nu_lower_endpoint_is_the_documented_exception():
    """nu = 0.5 is inside nu_range but is the one point exceeding 5%.

    Documented as ~9%, confined to lam near the branch crossover. If this
    tightens or worsens, the docs and README table need updating with it.
    """
    d = CMP(nu_range=None)
    lam = np.geomspace(1e-3, 1e3, 200)
    bias = np.array([
        (np.arange(d.max_count(l) + 1) * d.pmf(np.arange(d.max_count(l) + 1), l, 0.5)).sum() / l - 1
        for l in lam
    ])
    assert 0.05 < np.abs(bias).max() < 0.10
    offending = lam[np.abs(bias) > 0.05]
    assert 0.5 < offending.min() and offending.max() < 1.5


def test_beyond_upper_bound_degrades_gradually_not_catastrophically():
    """Unlike small nu, large nu has no negative-base branch, so it stays finite."""
    d = CMP(nu_range=None)
    for nu in [5.0, 10.0]:
        s = d.sample(np.full(20_000, 5.0), nu)
        assert s.mean() == pytest.approx(5.0, rel=0.15)   # drifting, but not zero
        assert np.isfinite(d.rate(5.0, nu))


def test_max_count_is_wide_enough(d):
    """Truncation must leave a negligible tail across the valid nu range."""
    for lam in [1.0, 10.0, 50.0, 300.0]:
        for nu in [0.5, 1.0, 2.0, 4.0]:
            max_x = d.max_count(lam)
            tail = d.pmf(np.arange(max_x - 4, max_x + 1), lam, nu).sum()
            assert tail < 1e-8, f"lam={lam} nu={nu} tail={tail}"


# --------------------------------------------------------------------------
# table growth
# --------------------------------------------------------------------------

def test_log_factorial_table_grows_on_demand(d):
    small = d._logfac.size
    d.sample(np.full(3, 5000.0), 1.0)
    assert d._logfac.size > small
    # and the result is still correct after growing
    s = d.sample(np.full(20_000, 5000.0), 1.0)
    assert s.mean() == pytest.approx(5000.0, rel=0.01)


def test_module_level_instance_works():
    assert isinstance(cmp, CMP)
    assert cmp.sample(3.0, 1.0) >= 0


# --------------------------------------------------------------------------

import contextlib
import warnings as _warnings


@contextlib.contextmanager
def warnings_as_errors():
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        yield
