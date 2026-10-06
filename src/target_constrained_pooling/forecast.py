"""Common endpoint correction and a policy learned only from calibration C."""
from dataclasses import dataclass, field
import numpy as np

from ._projection import branch
from ._precision import HP
from ._shrink import shrink_weight
from ._strength import fit_alpha


def common_branch(free, boundary, endpoint, *, return_bridge=False):
    """Apply the residual bridge followed by bounded, boundary-anchored PAVA.

    ``free`` contains future values at observation indices m+1,...,L; boundary
    is the final observed sample, not a fitted model value. The bridge is
    free[j] + (j+1)/H * (endpoint-free[-1]). Project its interior onto monotone
    tails between the fixed boundary and endpoint, append the exact endpoint.
    A zero span gives a constant tail; H=1 gives just the endpoint. Historical
    observations are not part of the output. No fitted model is evaluated.
    """
    free = np.asarray(free, dtype=float)
    if free.ndim != 1 or not len(free) or not np.isfinite(free).all():
        raise ValueError("free must be a finite, nonempty one-dimensional future vector")
    if not np.isfinite(boundary) or not np.isfinite(endpoint):
        raise ValueError("boundary and endpoint must be finite scalars")
    projected, bridge = branch(free, float(boundary), float(endpoint))
    return (projected, bridge) if return_bridge else projected


@dataclass(frozen=True)
class Policy:
    """Fixed branch strengths and U mixture weight learned on C only."""
    alpha_uniform: float
    alpha_matched: float
    weight_uniform: float
    diagnostics: dict = field(default_factory=dict, compare=False)

    def __post_init__(self):
        if any(not np.isfinite(x) or not 0 <= x <= 1 for x in (self.alpha_uniform, self.alpha_matched, self.weight_uniform)):
            raise ValueError("all policy parameters must be finite and in [0,1]")


def calibrate(free, uniform_hard, matched_hard, truth):
    """Fit one policy from calibration continuations C; never pass E truth.

    Four finite matrices must have identical (C trajectories, H future values)
    shape and shared response coordinates. Loss is mean PER-TRAJECTORY tail
    RMSE, including the endpoint, with equal trajectory weights. Strength fits
    preserve the main implementation: SciPy bounded search, xatol=1e-8, then
    compare 0, 1 and its returned candidate; exact float objective ties favor
    smaller strength. This is not the rerun's exact-alpha contrast.

    The weight uses the audited 200-bit one-sided derivative harm minimizer,
    closest-to-half ties, then directed shrinkage until mean RMSE is no worse
    than equal averaging. It uses a 2^-150 sign zero band and 1e-15 bisection
    resolution; ordinary high precision, not an interval certificate.
    """
    arrays = [np.asarray(x, dtype=float) for x in (free, uniform_hard, matched_hard, truth)]
    if any(x.ndim != 2 or not x.shape[0] or not x.shape[1] or not np.isfinite(x).all() for x in arrays):
        raise ValueError("calibration inputs must be finite, nonempty (trajectories,H) matrices")
    if any(x.shape != arrays[0].shape for x in arrays[1:]):
        raise ValueError("all calibration matrices must share a shape")
    free, uniform_hard, matched_hard, truth = arrays
    au, ui = fit_alpha(list(zip(free, uniform_hard, truth)))
    am, mi = fit_alpha(list(zip(free, matched_hard, truth)))
    weight, wi = shrink_weight(HP(truth - free, uniform_hard - free, matched_hard - free, au, am))
    return Policy(au, am, weight, {"uniform_strength": ui, "matched_strength": mi, "weight": wi})


def predict(free, boundary, endpoints, policy):
    """Return TCP from cached free future values, projected endpoints and policy.

    Returns a future vector only. Preserve the prefix externally. Arithmetic
    matches the saved main arms: form each calibrated branch, then mix them.
    Calibrated branches/TCP can be nonmonotone; they are never projected again.
    Frozen policies can be reused on arbitrarily many evaluation prefixes.
    """
    uniform = common_branch(free, boundary, endpoints.uniform)
    matched = common_branch(free, boundary, endpoints.matched)
    free = np.asarray(free, dtype=float)
    ucal = free + policy.alpha_uniform * (uniform - free)
    mcal = free + policy.alpha_matched * (matched - free)
    return policy.weight_uniform * ucal + (1 - policy.weight_uniform) * mcal
