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

The validity range for the mean-to-rate mapping is `0.55 <= nu <= 4`,
with both endpoints included. Values `nu < 0.55` or `nu > 4` are outside
this range. With these bounds enforced in the implementation, an
out-of-range `nu` raises:

```python
d.sample(lam, 0.3)
# ValueError: nu must satisfy 0.55 <= nu <= 4.0; got 0.3. ...
```

The two ends fail for different reasons, which is worth knowing before you
override anything:

- **At or below 0.55**, mean recovery becomes less accurate near `lam = 1`.
  At sufficiently small `nu`, the mapping can also break. The large-mean branch
  `(lam + (nu-1)/(2*nu))**nu` has a negative base whenever
  `lam < (1-nu)/(2*nu)`, which is only possible for `nu < 1`; there it returns
  NaN and every draw collapses to 0. At `nu = 0.3, lam = 1.0` you get all
  zeros for a distribution whose mean is supposed to be 1.
- **At or above 4**, the mapping is outside the validity range and can *drift*. For `nu > 1` the offset is positive, so the
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

Elsewhere the mean-to-rate mapping is approximate. The table below reports
the obtained mean divided by the requested mean, `lam`, here denoted by
$\langle N_{\rm sat}(M)\rangle$. Unity indicates agreement with the requested
mean. Values are rounded to four decimal places.

| $\langle N_{\rm sat}(M)\rangle$ | nu=0.4 | nu=0.5 | nu=0.6 | nu=0.8 | nu=1.2 | nu=2 | nu=4 | nu=5 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.9950 | 0.9923 | 0.9989 | 0.9990 | 0.9932 | 1.0043 | 1.0048 | 1.0062 |
| 0.2 | 1.0030 | 0.9994 | 0.9980 | 1.0016 | 1.0030 | 0.9979 | 1.0009 | 1.0008 |
| 0.5 | 0.9878 | 0.9936 | 0.9976 | 0.9987 | 0.9978 | 1.0020 | 0.9882 | 0.9786 |
| 0.85 | 0.8152 | 0.9321 | 0.9722 | 0.9987 | 0.9959 | 0.9900 | 0.9518 | 0.9328 |
| 0.9 | 0.7639 | 0.9223 | 0.9761 | 1.0038 | 0.9918 | 0.9771 | 0.9537 | 0.9509 |
| 0.95 | 0.7582 | 0.9316 | 0.9856 | 1.0090 | 0.9881 | 0.9655 | 0.9640 | 0.9716 |
| 1.0 | 0.7868 | 0.9471 | 0.9966 | 1.0119 | 0.9865 | 0.9607 | 0.9726 | 0.9855 |
| 2.0 | 1.0209 | 1.0271 | 1.0212 | 1.0081 | 0.9934 | 0.9898 | 0.9910 | 0.9909 |
| 5.0 | 1.0175 | 1.0101 | 1.0053 | 1.0016 | 0.9999 | 0.9986 | 0.9986 | 0.9985 |
| 20 | 1.0000 | 0.9996 | 1.0000 | 1.0003 | 1.0001 | 1.0000 | 0.9998 | 1.0000 |
| 100 | 0.9998 | 1.0000 | 1.0000 | 1.0001 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

The largest departures in this table occur near `lam = 1`:

- At `nu = 0.4`, the largest tabulated underestimate is 24.18% at `lam = 0.95`.
- At `nu = 0.5`, it is 7.77% at `lam = 0.9`.
- For the tabulated values `nu = 0.6, 0.8, 1.2, 2, 4`, all listed ratios are
  within 5% of unity.
- At `nu = 5`, the largest tabulated underestimate is 6.72% at `lam = 0.85`.

The `nu = 0.4`, `0.5`, `4`, and `5` columns lie outside the strict validity
range `0.55 < nu < 4` and are included for comparison. These are results at the listed parameter
values, not worst-case bounds over a continuous parameter range. A ratio
rounded to 1.0000 does not establish exact agreement.

Maximum absolute relative deviation, `abs(mean / lam - 1)`, among the rows
shown above (calculated from the rounded ratios):

| nu | 0.4 | 0.5 | 0.6 | 0.8 | 1.2 | 2 | 4 | 5 |
|---|---|---|---|---|---|---|---|---|
| maximum tabulated deviation | 24.18% | 7.77% | 2.78% | 1.19% | 1.35% | 3.93% | 4.82% | 6.72% |

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
- This package as been re-written by AI Claude based on previous code from the author.
All scientific ideas and decisions are from the author, and the auhtor take full
responsibility for any errors or inaccuracies.

## Tests

```bash
pip install -e ".[test]"
pytest
```

## Citation

If you use this package, please cite
[Rocher (2026)](https://arxiv.org/abs/2605.28644):

```bibtex
@misc{rocher2026exploringnonpoissonsatelliteoccupation,
      title={Exploring non-Poisson satellite occupation in HOD models and its impact on 2- and 3-point galaxy clustering},
      author={Antoine Rocher},
      year={2026},
      eprint={2605.28644},
      archivePrefix={arXiv},
      primaryClass={astro-ph.CO},
      url={https://arxiv.org/abs/2605.28644},
}
```
