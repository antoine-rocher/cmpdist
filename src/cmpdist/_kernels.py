"""numba kernels behind :class:`cmpdist.CMP`.

Every kernel here is a free function that takes its state explicitly — the
log-factorial table, the scratch buffer, the uniform deviates — rather than
reading a module global. That keeps them cacheable, keeps the sampling
deterministic given ``u``, and lets :class:`~cmpdist.core.CMP` own all the
mutable state.

Two parameterisations appear throughout and must not be confused:

``mu``    the mean of the distribution — the Poisson expectation you want it to
          have. This is what users pass in.
``rate``  the CMP rate λ in ``P(k) ∝ rate**k / k!**nu``, which is *not* the
          mean unless nu = 1. `rate_from_mean` maps one to the other.
"""

import math

import numpy as np
from numba import njit, types
from numba.extending import overload
from scipy.special import gammaln

_JIT = dict(fastmath=True, cache=True)


def build_log_factorial(n_max):
    """Table of ``ln(k!)`` for ``k = 0 .. n_max`` (length ``n_max + 1``).

    ``gammaln(k + 1) = ln(k!)``, evaluated on the whole range at once — the
    original per-element loop over ``range(1_000_000)`` cost seconds of import
    time and 8 MB, for a table that is only ever indexed up to ``max_count``.
    """
    return gammaln(np.arange(1, n_max + 2))


# --------------------------------------------------------------------------
# scalar/array nu dispatch
# --------------------------------------------------------------------------
# numba compiles one specialisation per argument type, so "is nu a scalar or an
# array?" has to be answered at compile time rather than with a runtime `if`.
# The overload below does that: each specialisation of `sample_into` inlines
# only the branch matching the nu it was called with, so a shared nu costs
# nothing at all.

def nu_at(nu, i):
    """Dispersion for element ``i``: ``nu`` itself if scalar, ``nu[i]`` if per-element."""
    return nu[i] if np.ndim(nu) else nu


@overload(nu_at)
def _nu_at_overload(nu, i):
    if isinstance(nu, types.Array):
        return lambda nu, i: nu[i]
    return lambda nu, i: nu


# --------------------------------------------------------------------------
# truncation
# --------------------------------------------------------------------------

@njit(**_JIT)
def max_count(mu):
    """Count at which the PMF/CDF can be truncated for a mean ``mu``.

    Piecewise ladder chosen so the neglected tail is negligible while keeping
    the arrays short for the small means that dominate most count data.
    """
    if mu < 15:
        return 50
    elif mu < 100:
        return 200
    elif mu < 700:
        return 1000
    elif mu < 4000:
        return 5000
    else:
        return int(mu * 1.1)


@njit(**_JIT)
def max_count_over(mu):
    """Widest truncation needed by any entry of ``mu``.

    Scans rather than taking ``max_count(mu.max())``: `max_count` is not
    monotonic, dropping from 5000 to ~1.1*mu at mu = 4000.
    """
    m = 0
    for i in range(mu.size):
        c = max_count(mu[i])
        if c > m:
            m = c
    return m


# --------------------------------------------------------------------------
# mean -> rate
# --------------------------------------------------------------------------

@njit(**_JIT)
def rate_small_mu(mu, nu):
    """Small-rate branch: series reversion of mu(rate), truncated after k = 4.

    Only accurate for mu well below 1.
    """
    a2 = 2.0 ** (-nu)
    a3 = 6.0 ** (-nu)
    a4 = 24.0 ** (-nu)

    A = 2 * a2 - 1
    B = 3 * a3 - 3 * a2 + 1
    C = 4 * a4 - 4 * a3 - 2 * a2 * a2 + 4 * a2 - 1

    return (
        mu
        - A * mu ** 2
        + (2 * A * A - B) * mu ** 3
        + (-5 * A ** 3 + 5 * A * B - C) * mu ** 4
    )


@njit(**_JIT)
def rate_large_mu(mu, nu):
    """Large-rate branch: inverse of ``mu ≈ rate**(1/nu) - (nu - 1) / (2 nu)``.

    Degrades as mu drops towards zero, where `rate_small_mu` takes over.
    """
    return (mu + (nu - 1) / (2 * nu)) ** nu


@njit(**_JIT)
def rate_from_mean(mu, nu, sig):
    """CMP rate λ reproducing a mean of ``mu``, valid at any scale.

    The two branches are blended with an error-function ramp centred on
    mu = 0.8 and of width ``sig``, which keeps λ(mu) smooth across the
    crossover.

    Valid for 0.5 <= nu <= 4, which is what :class:`~cmpdist.core.CMP` enforces.
    The two ends fail differently. Below nu = 1 the offset ``(nu - 1) / (2 nu)``
    is negative, so for small enough mu the large branch would take a negative
    base to a fractional power; it is switched off there, and by nu < 0.5 that
    dead zone overlaps the blend badly enough to wreck the mapping (NaN rates
    and all-zero counts). Above nu = 1 the offset is positive and the base is
    never negative, so large nu only degrades gradually — the mean drifts about
    5% off at nu = 4 and roughly 8% by nu = 10.
    """
    threshold = 0.8

    w = 0.5 * (math.erf((mu - threshold) / sig) + 1)
    small = rate_small_mu(mu, nu)

    # The large branch is only defined where its base is positive; below that
    # it would be a negative number to a fractional power (NaN, and numba has
    # no np.nan_to_num). Guarding on the base itself rather than on a fixed
    # mu > 0.5 matters: at nu = 1 the base is just mu, so the large branch —
    # which is then exact — stays in play all the way down.
    base = mu + (nu - 1) / (2 * nu)
    large = base ** nu if base > 0.0 else 0.0

    return (1 - w) * small + w * large


# --------------------------------------------------------------------------
# pmf / normalisation
# --------------------------------------------------------------------------

@njit(**_JIT)
def log_norm(rate, nu, max_x, logfac):
    """``log Z(rate, nu)``, the series truncated at ``max_x``.

    Summed with the log-sum-exp trick so large rates do not overflow.
    """
    if rate <= 0.0:
        return 0.0      # only the k = 0 term survives, Z = 1

    log_rate = np.log(rate)

    m = -np.inf
    for k in range(max_x + 1):
        t = k * log_rate - nu * logfac[k]
        if t > m:
            m = t

    s = 0.0
    for k in range(max_x + 1):
        s += np.exp(k * log_rate - nu * logfac[k] - m)
    return m + np.log(s)


@njit(**_JIT)
def pmf_into(out, k, rate, nu, log_z, logfac):
    """Fill ``out`` with the CMP mass at each count in ``k``."""
    if rate <= 0.0:
        for i in range(k.size):
            out[i] = 1.0 if k[i] == 0 else 0.0
        return

    log_rate = np.log(rate)
    for i in range(k.size):
        kk = k[i]
        if kk < 0 or kk >= logfac.size:
            out[i] = 0.0
        else:
            out[i] = np.exp(kk * log_rate - nu * logfac[kk] - log_z)


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------

@njit(**_JIT)
def icdf(rate, nu, max_x, u, work, logfac):
    """Invert the CMP CDF truncated at ``max_x`` for the uniform deviate ``u``.

    ``work`` is scratch of length at least ``max_x + 1``, holding first the
    log-terms and then the running CDF. Z is deliberately *not* used: the CDF
    is normalised by its own last entry, so any constant factor cancels. The
    terms are offset by their maximum instead, which plays the same
    overflow-guarding role.
    """
    if rate <= 0.0:     # mu = 0, or an expansion that underflowed
        return 0

    log_rate = np.log(rate)

    m = -np.inf
    for k in range(max_x + 1):
        t = k * log_rate - nu * logfac[k]
        work[k] = t
        if t > m:
            m = t

    tot = 0.0
    for k in range(max_x + 1):
        tot += np.exp(work[k] - m)
        work[k] = tot           # unnormalised CDF

    target = u * tot            # equivalent to comparing u against cdf / cdf[-1]
    for k in range(max_x + 1):
        if work[k] >= target:
            return k
    return max_x


@njit(**_JIT)
def sample_into(out, mu, nu, u, work, logfac, sig):
    """Fill ``out`` with one CMP draw per entry of ``mu``.

    Each element needs its own rate and therefore its own CDF, so the loop is
    unavoidable; what is shared is the single scratch buffer and the single
    block of uniform deviates. ``u`` is supplied by the caller, which keeps the
    draw reproducible through numpy's Generator instead of numba's separate
    (and separately seeded) RNG state.
    """
    for i in range(mu.size):
        nu_i = nu_at(nu, i)
        rate = rate_from_mean(mu[i], nu_i, sig)
        out[i] = icdf(rate, nu_i, max_count(mu[i]), u[i], work, logfac)
