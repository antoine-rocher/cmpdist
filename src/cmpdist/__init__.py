"""Conway-Maxwell-Poisson counts: over- and under-dispersed alternatives to Poisson.

The CMP law generalises the Poisson distribution with a dispersion parameter
``nu``: nu = 1 is Poisson, nu > 1 narrows the counts (sub-Poisson) and nu < 1
widens them (super-Poisson), all at fixed mean.

Usage::

    from cmpdist import cmp

    counts = cmp.sample(lam, nu)        # lam = the mean you want

``cmp`` is a ready-made :class:`CMP` instance. Build your own to control the
seed::

    from cmpdist import CMP

    counts = CMP(seed=42).sample(lam, nu)
"""

from .core import CMP

__version__ = "0.1.0"

#: Module-level default instance, seeded from the OS entropy pool.
cmp = CMP()

__all__ = ["CMP", "cmp", "__version__"]
