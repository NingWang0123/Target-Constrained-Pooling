"""Uniform source levels and five-point matched changes, in one coordinate."""
from dataclasses import dataclass, field
import numpy as np

from ._matching import endpoint


def _sign(direction):
    if direction not in ("fwd", "bwd"):
        raise ValueError("direction must be 'fwd' or 'bwd'")
    return 1.0 if direction == "fwd" else -1.0


def project_endpoint(value, boundary, direction):
    """Project an endpoint onto boundary + direction * nonnegative change."""
    if not np.isfinite(value) or not np.isfinite(boundary):
        raise ValueError("endpoint and boundary must be finite scalars")
    sign = _sign(direction)
    return float(boundary + sign * max(sign * (value - boundary), 0.0))


@dataclass(frozen=True)
class ResponseCoordinate:
    """Positive affine map; use the same instance for every response and loss."""
    offset: float
    scale: float

    def __post_init__(self):
        if not np.isfinite(self.offset) or not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("coordinate needs finite offset and positive scale")

    def transform(self, values):
        """Convert raw responses or cached free forecasts to shared coordinates."""
        return (np.asarray(values) - self.offset) / self.scale

    def inverse(self, values):
        """Convert shared-coordinate predictions back to raw responses."""
        return self.offset + self.scale * np.asarray(values)


def _inputs(prefix, donors):
    prefix, donors = np.asarray(prefix, dtype=float), np.asarray(donors, dtype=float)
    if prefix.ndim != 1 or donors.ndim != 2 or not len(donors) or not 0 < len(prefix) < donors.shape[1]:
        raise ValueError("prefix is one-dimensional; donors are nonempty full curves with a later horizon")
    if not np.isfinite(prefix).all() or not np.isfinite(donors).all():
        raise ValueError("prefix and permitted donors must be finite")
    return prefix, donors


def response_coordinate(prefix, donors):
    """Paper packet map from permitted donor prefixes plus the target prefix.

    Inputs are already oriented. No target continuation is accepted. Use the
    returned map for donors, target prefix, cached free forecasts, calibration
    truth and predictions; never refit or separately normalize a forecaster.
    """
    prefix, donors = _inputs(prefix, donors)
    heads = np.vstack([donors[:, :len(prefix)], prefix])
    lo = float(heads.min())
    scale = float(heads.max() - lo)
    return ResponseCoordinate(lo, scale if scale > 1e-12 else 1.0)


@dataclass(frozen=True)
class Endpoints:
    """Raw and direction-projected U/M endpoint values with matching evidence."""
    uniform_raw: float
    matched_raw: float
    uniform: float
    matched: float
    diagnostics: dict = field(default_factory=dict, compare=False)


def estimate_endpoints(prefix, donors, direction="fwd"):
    """Estimate U and N1 M using only a target prefix and permitted full donors.

    All responses must already share a positive affine coordinate. Five-point
    matching considers source stages p=5,...,m; exact distance ties use later p.
    It transfers source[p+H] - source[p], where H=L-m, not necessarily the
    source's original terminal change. Bandwidth fits donor-only leave-one-out
    endpoint MAE by bounded local search with the frozen near-uniform probe.
    Two donors are required: singleton leave-one-out bandwidth is undefined.
    """
    prefix, donors = _inputs(prefix, donors)
    sign = _sign(direction)
    if len(prefix) < 5:
        raise ValueError("N1 matching requires at least five observed target points")
    if len(donors) < 2:
        raise ValueError("N1 donor-only leave-one-out bandwidth is undefined for fewer than two donors")
    target = np.full(donors.shape[1], np.nan)
    target[:len(prefix)] = prefix
    matched = endpoint("N1", np.vstack([donors, target]), len(prefix), sign)
    uniform_raw = float(donors[:, -1].mean())
    return Endpoints(
        uniform_raw, matched["V_raw"],
        project_endpoint(uniform_raw, float(prefix[-1]), direction), matched["V"], matched,
    )
