"""The :class:`CMP` distribution object."""

import warnings

import numpy as np

from . import _kernels as _k

__all__ = ["CMP"]

# Covers max_count up to 1000, i.e. means below 700. Grows on demand from there,
# so the common small-mean case never pays for a large table.
_INITIAL_TABLE = 1024

#: Accepted values of ``CMP.on_invalid_nu``.
_NU_POLICIES = frozenset({"raise", "warn", "ignore"})


class CMP:
    """Conway-Maxwell-Poisson counts, parameterised by their mean.

    The CMP law generalises the Poisson distribution with a dispersion
    parameter ``nu``::

        P(k | rate, nu) = rate**k / (k!**nu * Z(rate, nu))

    with nu = 1 recovering Poisson, nu > 1 giving sub-Poisson (narrower) counts
    and nu < 1 super-Poisson (wider) counts at fixed mean.

    The rate is not the mean, so every method here takes the **mean** ``lam``
    you actually want — the Poisson expectation to reproduce — and converts
    it internally. :meth:`rate` exposes that conversion if you need the CMP rate
    itself.

    Parameters
    ----------
    seed : int, numpy Generator or None
        Seeds the internal :class:`numpy.random.Generator` used by
        :meth:`sample`. Pass a Generator to share one with the rest of a
        pipeline.
    blend_width : float
        Width of the error-function ramp blending the small-mean and large-mean
        branches of the mean-to-rate mapping.
    nu_range : tuple of float or None
        Half-open range ``[lo, hi)`` over which the mean-to-rate mapping is
        trusted. ``None`` disables the check entirely.
    on_invalid_nu : {'raise', 'warn', 'ignore'}
        What to do with a ``nu`` outside ``nu_range``. Defaults to ``'raise'``:
        outside the validated window the mapping does not merely lose accuracy,
        it can return NaN and yield all-zero counts, which is far worse to
        inherit silently than an exception.

    Examples
    --------
    >>> from cmpdist import CMP
    >>> cmp = CMP(seed=0)
    >>> cmp.sample(5.0, 1.0)                     # doctest: +SKIP
    4
    >>> cmp.sample(np.full(1000, 5.0), 1.5)      # doctest: +SKIP
    array([3, 5, 4, ...])
    """

    def __init__(self, seed=None, blend_width=0.2, nu_range=(0.5, 2.0),
                 on_invalid_nu="raise"):
        if on_invalid_nu not in _NU_POLICIES:
            raise ValueError(
                f"on_invalid_nu must be one of {sorted(_NU_POLICIES)}, got {on_invalid_nu!r}"
            )
        self.rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        self.blend_width = float(blend_width)
        self.nu_range = None if nu_range is None else (float(nu_range[0]), float(nu_range[1]))
        self.on_invalid_nu = on_invalid_nu
        self._logfac = _k.build_log_factorial(_INITIAL_TABLE)

    def __repr__(self):
        return (f"{type(self).__name__}(blend_width={self.blend_width!r}, "
                f"nu_range={self.nu_range!r}, on_invalid_nu={self.on_invalid_nu!r})")

    # -- public API --------------------------------------------------------

    def sample(self, lam, nu, size=None):
        """Draw CMP counts with mean ``lam`` and dispersion ``nu``.

        Parameters
        ----------
        lam : float or array_like
            Mean of the distribution — the Poisson expectation to reproduce.
            An array draws one independent count per entry, of any shape.
        nu : float or array_like
            Dispersion. A scalar applies to every draw; an array must match the
            number of draws element for element, which is how you let nu vary
            with a covariate.
        size : int or None
            Number of draws when ``lam`` is scalar. Ignored (must match) when
            ``lam`` is an array.

        Returns
        -------
        int or ndarray of int64
            A plain int when ``lam`` is scalar and ``size`` is None, otherwise
            an array shaped like ``lam`` (or ``(size,)``).

        Notes
        -----
        Draws are made by inverting the CDF truncated at :meth:`max_count`, one
        uniform deviate per count, taken from :attr:`rng`.
        """
        mu, scalar_lam, shape = _as_1d_float(lam, "lam")

        if size is not None:
            size = int(size)
            if scalar_lam:
                mu = np.full(size, mu[0])
                shape = (size,)
            elif mu.size != size:
                raise ValueError(f"size={size} does not match len(lam)={mu.size}")
            scalar_lam = False

        n = mu.size
        nu_arg = self._prepare_nu(nu, n)

        out = np.empty(n, dtype=np.int64)
        if n:
            need = _k.max_count_over(mu)
            self._ensure_table(need)
            work = np.empty(need + 1)
            u = self.rng.random(n)
            _k.sample_into(out, mu, nu_arg, u, work, self._logfac, self.blend_width)

        if scalar_lam and size is None:
            return int(out[0])
        return out.reshape(shape)

    def pmf(self, k, lam, nu):
        """Probability of each count in ``k`` for mean ``lam``, dispersion ``nu``.

        ``lam`` and ``nu`` are scalars; ``k`` may be any array_like of counts.
        Masses outside ``0 .. max_count(lam)`` are reported as 0, consistent
        with the truncation :meth:`sample` draws from.
        """
        lam = _as_scalar_float(lam, "lam")
        nu = _as_scalar_float(nu, "nu")
        self._validate_nu(np.asarray(nu))

        counts = np.asarray(k)
        flat = np.ascontiguousarray(counts.ravel(), dtype=np.int64)

        max_x = _k.max_count(lam)
        self._ensure_table(max_x)

        rate = _k.rate_from_mean(lam, nu, self.blend_width)
        log_z = _k.log_norm(rate, nu, max_x, self._logfac)

        out = np.empty(flat.size)
        _k.pmf_into(out, flat, rate, nu, log_z, self._logfac)
        out[flat > max_x] = 0.0
        return out.reshape(counts.shape) if counts.ndim else out[0]

    def rate(self, lam, nu):
        """CMP rate λ whose distribution has mean ``lam``.

        This is the parameter appearing in ``rate**k / k!**nu``, not the mean.
        """
        lam = _as_scalar_float(lam, "lam")
        nu = _as_scalar_float(nu, "nu")
        self._validate_nu(np.asarray(nu))
        return _k.rate_from_mean(lam, nu, self.blend_width)

    def log_norm(self, rate, nu, max_x=None):
        """``log Z(rate, nu)``, taking the CMP **rate** (see :meth:`rate`).

        ``max_x`` defaults to a truncation wide enough for that rate.
        """
        rate = _as_scalar_float(rate, "rate")
        nu = _as_scalar_float(nu, "nu")
        if max_x is None:
            max_x = _k.max_count(rate ** (1.0 / nu) if rate > 0 else 0.0)
        max_x = int(max_x)
        self._ensure_table(max_x)
        return _k.log_norm(rate, nu, max_x, self._logfac)

    @staticmethod
    def max_count(lam):
        """Count at which the PMF/CDF is truncated for mean ``lam``."""
        return int(_k.max_count(_as_scalar_float(lam, "lam")))

    # -- internals ---------------------------------------------------------

    def _prepare_nu(self, nu, n):
        """Coerce ``nu`` to the scalar or length-``n`` array the kernel expects."""
        if np.ndim(nu) == 0:
            value = _as_scalar_float(nu, "nu")
            self._validate_nu(np.asarray(value))
            return value

        arr = np.ascontiguousarray(nu, dtype=np.float64).ravel()
        if arr.size != n:
            raise ValueError(
                f"nu must be a scalar or match the number of draws: "
                f"len(nu)={arr.size}, draws={n}"
            )
        self._validate_nu(arr)
        return arr

    def _validate_nu(self, arr):
        """Enforce ``nu_range`` according to :attr:`on_invalid_nu`.

        The range is half-open, ``lo <= nu < hi``.
        """
        if self.nu_range is None or self.on_invalid_nu == "ignore" or arr.size == 0:
            return
        lo, hi = self.nu_range
        bad = (arr < lo) | (arr >= hi)
        if not bad.any():
            return

        offenders = np.unique(arr[bad])
        shown = ", ".join(f"{v:g}" for v in offenders[:5])
        if offenders.size > 5:
            shown += f", ... ({offenders.size} distinct values)"
        message = (
            f"nu must satisfy {lo} <= nu < {hi}; got {shown}. Outside that range "
            f"the mean-to-rate mapping is not valid and can return NaN, giving "
            f"meaningless counts. Pass on_invalid_nu='warn' or 'ignore' to "
            f"override, or widen nu_range."
        )
        if self.on_invalid_nu == "raise":
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=3)

    def _ensure_table(self, k_max):
        """Grow the log-factorial table so index ``k_max`` is valid."""
        if k_max >= self._logfac.size:
            self._logfac = _k.build_log_factorial(max(2 * self._logfac.size, k_max + 1))


def _as_1d_float(x, name):
    """Return ``(contiguous 1-D float array, was_scalar, original shape)``."""
    arr = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    if np.any(arr < 0):
        raise ValueError(f"{name} must be non-negative")
    return np.ascontiguousarray(arr.ravel()), arr.ndim == 0, arr.shape


def _as_scalar_float(x, name):
    if np.ndim(x) != 0:
        raise TypeError(f"{name} must be a scalar here, got shape {np.shape(x)}")
    value = float(x)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value
