#!/usr/bin/env python3
"""DYTemplate.LogPolyFit reaches the minimum of its likelihood.

The fits run on noise-free (Asimov) shapes, so the best fit is the true function. Each shape is a DNN
tail like the SR DY tails: exp(-0.7 t - 0.5 t^2 + a3 t^3) in the fit's own variable t, fitted above
the peak (the mask) at degree 3.
- a small cubic term with many events: started at 0, the leading coefficient -exp(p) sat on a
  plateau (p = log 1e-6, gradient ~1e-6) where migrad stopped. Here that left the fit invalid, and
  tail_template would have fallen back to the degree-2 fit. In the real tails it also left "valid"
  fits up to ~100 in NLL short of the minimum, which is why the result depended on the CPU (numpy's
  AVX2 vs AVX-512 arithmetic);
- no cubic term: the data want the leading coefficient at 0, the boundary of -exp(p). The fit has
  to be valid and still give the true function.

    python3 test/test_dy_template.py    (needs numpy and iminuit, e.g. in the CMSSW environment;
                                         under pytest without iminuit it is skipped)
"""

import os
import sys
import unittest

import numpy as np

try:
    import iminuit  # noqa: F401
except ImportError:
    # FLAF's CI image has no iminuit; the test runs where the preprocessing does (CMSSW)
    import pytest

    pytest.skip("needs iminuit", allow_module_level=True)

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Analysis")
)
from DYTemplate import LogPolyFit  # noqa: E402

X = (np.arange(100) + 0.5) / 100  # 100 fine bins of the DNN score
TAIL = X >= 0.44


def tail_shape(a3, total=6000.0):
    """The shape, with its t taken from a first fit (t depends on the data the fit sees)."""
    y = np.exp(-((X - 0.3) ** 2) / 0.02)
    first = LogPolyFit(X, y, y.copy(), 3, mask=TAIL, merge=False)
    t = (X - first.x0) / first.s
    y = np.exp(-0.7 * t - 0.5 * t**2 + a3 * t**3)
    return y * total / y[TAIL].sum()


class LogPolyFitTest(unittest.TestCase):
    def fit(self, y):
        fit = LogPolyFit(X, y, y.copy(), 3, mask=TAIL, merge=False)
        self.assertTrue(fit.ok, "the fit is not valid")
        keep = TAIL & (y > 1e-3 * y[TAIL].max())
        worst = float(np.max(np.abs(fit.predict()[keep] / y[keep] - 1)))
        self.assertLess(
            worst, 1e-3, f"the fit misses the true shape by up to {worst:.3g}"
        )
        return fit

    def test_small_cubic_term_is_found(self):
        self.fit(tail_shape(-0.03))

    def test_zero_cubic_term_is_a_valid_fit(self):
        self.fit(tail_shape(0.0))


if __name__ == "__main__":
    unittest.main()
