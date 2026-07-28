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

The mean-to-rate mapping is only valid for `0.5 <= nu < 2`. Outside that window
it can return NaN and hand back all-zero counts, so by default an out-of-range
`nu` raises:

```python
d.sample(lam, 0.3)
# ValueError: nu must satisfy 0.5 <= nu < 2.0; got 0.3. ...
```

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

| lam | nu=0.5 | nu=0.6 | nu=0.8 | nu=1.0 | nu=1.5 | nu=1.9 |
|----:|-------:|-------:|-------:|-------:|-------:|-------:|
| 0.05 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.5 | 0.974 | 0.993 | 0.999 | 1.000 | 0.999 | 0.999 |
| 0.8 | 0.909 | 0.972 | 1.002 | 1.000 | 0.981 | 0.972 |
| 1.0 | 0.949 | 0.996 | 1.011 | 1.000 | 0.972 | 0.962 |
| 2.0 | 1.027 | 1.022 | 1.009 | 1.000 | 0.991 | 0.990 |
| 5.0 | 1.010 | 1.006 | 1.002 | 1.000 | 0.999 | 0.999 |
| 20 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 100 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

The mapping is exact at `nu = 1` and for `lam` above ~5 or below ~0.2. The
error concentrates in `0.5 < lam < 2`, where the small-mean series and the
large-mean asymptotic are blended, and grows towards the edges of the valid
`nu` window — up to **9% at `lam = 0.8, nu = 0.5`**. If much of your data sits
in that corner with strongly non-Poisson `nu`, calibrate against this table
rather than assuming the mean is reproduced.

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

## Tests

```bash
pip install -e ".[test]"
pytest
```
