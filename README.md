# cmpdist

Conway-Maxwell-Poisson (CMP) counts, parameterised by the **mean** you actually
want.

The CMP law generalises the Poisson distribution with a dispersion parameter
$\nu$:

$$
P(k \mid \lambda, \nu) = \frac{\lambda^{k}}{(k!)^{\nu}\, Z(\lambda, \nu)}
\qquad
Z(\lambda, \nu) = \sum_{k \ge 0} \frac{\lambda^{k}}{(k!)^{\nu}}
$$

$\nu = 1$ recovers Poisson, $\nu > 1$ narrows the counts (sub-Poisson) and
$\nu < 1$ widens them (super-Poisson), all at fixed mean. It is the natural
choice for count data that is over- or under-dispersed relative to Poisson but
still controlled by a single extra parameter.

The catch is that $\lambda$ is *not* the mean unless $\nu = 1$. This package
takes the mean you want and inverts the mapping for you.

## Install

```bash
pip install git+https://github.com/antoine-rocher/cmpdist
```

Or from a local checkout:

```bash
pip install -e .
```

Requires numpy, numba and scipy.

## Use

```python
import numpy as np
from cmpdist import cmp          # ready-made instance

counts = cmp.sample(lam, nu)     # lam = the mean, one entry per draw
```

Control the seed with your own instance:

```python
from cmpdist import CMP

d = CMP(seed=42)
d.sample(5.0, 1.2)                        # -> a single int
d.sample(5.0, 1.2, size=1000)             # -> 1000 draws
d.sample(np.full(10_000, 5.0), 1.2)       # -> one draw per entry
d.sample(lam_2d, 1.2)                     # -> same shape as lam_2d
```

`nu` may be a scalar shared by every draw, or one value per draw — so the
dispersion can vary across the sample with any covariate you like:

```python
nu = 0.6 + 0.2 * np.log10(x / x_ref)
counts = d.sample(lam, nu)                # len(nu) == len(lam)
```

Other methods:

```python
d.pmf(np.arange(20), lam=5.0, nu=1.3)   # mass function for that mean
d.rate(lam=5.0, nu=1.3)                 # the CMP rate behind that mean
d.log_norm(rate=4.1, nu=1.3)            # log Z, taking the rate
d.max_count(lam=5.0)                    # where the CDF is truncated
```

See [`examples/getting_started.ipynb`](examples/getting_started.ipynb) for a
worked walkthrough with plots.

## Validity range

The mean-to-rate mapping is enforced on `0.5 <= nu <= 4` (both endpoints usable), and by default an
out-of-range `nu` raises:

```python
d.sample(lam, 0.3)
# ValueError: nu must satisfy 0.5 <= nu <= 4.0; got 0.3. ...
```

The two ends fail for different reasons, which is worth knowing before you
override anything:

- **Below 0.5** the mapping *breaks*. The large-mean branch
  `(lam + (nu-1)/(2*nu))**nu` has a negative base whenever
  `lam < (1-nu)/(2*nu)`, which is only possible for `nu < 1`; there it returns
  NaN and every draw collapses to 0. At `nu = 0.3, lam = 1.0` you get all
  zeros for a distribution whose mean is supposed to be 1.
- **Above 4** it merely *drifts*. For `nu > 1` the offset is positive, so the
  base is never negative and nothing ever goes NaN — the recovered mean just
  loses accuracy gradually (~5% at `nu = 4`, ~8% by `nu = 10`). If you need
  large `nu`, `on_invalid_nu="warn"` is a reasonable override there.

Relax it per instance if you know what you are doing:

```python
CMP(on_invalid_nu="warn")        # RuntimeWarning instead of ValueError
CMP(on_invalid_nu="ignore")      # no check at all
CMP(nu_range=(0.4, 3.0))         # different bounds, still enforced
```

The check is elementwise, so a single bad entry in a per-draw `nu` array is
caught.

## Accuracy

`nu = 1` is exact: the rate is the Poisson mean to machine precision and the
pmf matches `scipy.stats.poisson`.

Elsewhere the mean-to-rate mapping is approximate. The table below is the
**analytic** mean of the resulting distribution divided by the mean you asked
for — no Monte Carlo noise, so these are the real errors:

| lam | nu=0.5 | nu=0.6 | nu=0.8 | nu=1.0 | nu=1.5 | nu=2.0 | nu=3.0 | nu=4.0 |
|----:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| 0.05 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.5 | 0.974 | 0.993 | 0.999 | 1.000 | 0.999 | 0.999 | 0.994 | 0.985 |
| 0.8 | **0.909** | 0.972 | 1.002 | 1.000 | 0.981 | 0.970 | 0.959 | 0.951 |
| 1.0 | 0.949 | 0.996 | 1.011 | 1.000 | 0.972 | 0.960 | 0.962 | 0.973 |
| 2.0 | 1.027 | 1.022 | 1.009 | 1.000 | 0.991 | 0.990 | 0.991 | 0.991 |
| 5.0 | 1.010 | 1.006 | 1.002 | 1.000 | 0.999 | 0.999 | 0.999 | 0.998 |
| 20 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 100 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Two things to read off it:

- **The error lives in a narrow `lam` window.** It is exact at `nu = 1`, and
  for `lam` above ~5 or below ~0.2 it is exact at every `nu`. Everything
  concentrates in `0.6 < lam < 1.5`, where the small-mean series and the
  large-mean asymptotic are blended. Outside that window even `nu = 10` is
  accurate.
- **`nu = 0.5` is the one point inside the range that exceeds 5%**, reaching
  **9.1% at `lam ≈ 0.8`**. It is kept in the default range because it is a
  common choice, but it is not covered by the 5% figure. Everywhere else in
  `0.6 <= nu <= 4.0` the worst case is 5%.

Measured worst-case `|mean/lam - 1|` over `lam` in `1e-3 .. 1e3`:

| nu | 0.5 | 0.55 | 0.6 | 1.0 | 2.0 | 3.0 | 4.0 | 4.2 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| worst error | 9.1% | 5.2% | 2.9% | 0% | 4.0% | 4.4% | 4.9% | 5.0% | 7.9% |

If much of your data sits at `lam ≈ 1` with strongly non-Poisson `nu`,
calibrate against these tables rather than assuming the mean is reproduced.

Counts are drawn by inverting the CDF truncated at `max_count(lam)`, which
leaves a neglected tail below 1e-8 across the valid `nu` range.

## Notes

- Sampling uses numpy's `Generator`, so `CMP(seed=...)` is reproducible in the
  ordinary way. (numba's internal RNG is separate and cannot be seeded from
  Python.)
- Kernels are numba-compiled with `cache=True`, so the JIT cost is paid once
  per machine rather than once per process.
- The log-factorial table is built lazily and grows on demand, rather than
  allocating a fixed 8 MB at import.
- Throughput is roughly 3 million draws per second on one core.
- This package as been re-written by AI Claude based on previous code from the author

## Tests

```bash
pip install -e ".[test]"
pytest
```
