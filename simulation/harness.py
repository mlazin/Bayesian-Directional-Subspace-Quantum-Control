"""Evaluation logging wrapper for quantum-control optimization."""

import time

import numpy as np
from scipy.optimize import approx_fprime


# Set to a small value (for example, 1e-6) to enable finite-difference
# gradient checking. Leave as None for production runs.
checkgradeps = None


class harness:
    """Record simulator evaluations, gradients, fields, and timing information."""

    def __init__(self, f):
        self.f = f
        self.xs = []
        self.fs = []
        self.gs = []
        self.fids = []
        self.evalt = []  # cumulative simulator evaluation time
        self.t = []      # cumulative elapsed time
        self.t0 = None
        self.bestP = 0.0
        self.bestPs = []
        self.Es = []

    def start(self):
        self.t0 = time.time()

    def bounds(self):
        return self.f.bounds()

    def __call__(self, x, fid=0, grad=False):
        if self.t0 is None:
            self.t0 = time.time()

        t0 = time.time()
        if grad:
            fval, gval = self.f(x, fid=fid, grad=True)
        else:
            fval = self.f(x, fid=fid, grad=False)
            gval = None
        t1 = time.time()

        self.xs.append(x)
        self.Es.append(self.f.E)

        if np.abs(fval) > self.bestP:
            self.bestP = np.abs(fval)
        self.bestPs.append(self.bestP)

        self.fs.append(fval)
        self.gs.append(gval)
        self.fids.append(fid)
        self.evalt.append(
            (0 if len(self.evalt) == 0 else self.evalt[-1]) + (t1 - t0)
        )
        self.t.append(t1 - self.t0)

        if grad and checkgradeps is not None:
            tt0 = time.time()
            approxgval = approx_fprime(
                x,
                lambda xx: self.f(xx, fid=fid, grad=False),
                checkgradeps,
            )
            if np.abs(approxgval - gval).max() > 1e-3:
                raise RuntimeError(
                    "Gradient check failed.\n"
                    f"finite difference: {approxgval}\n"
                    f"returned: {gval}"
                )
            tt1 = time.time()
            t0 += tt1 - tt0  # retained from original implementation

        if grad:
            return fval, gval
        return fval
