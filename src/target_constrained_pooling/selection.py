"""Deterministic, permission-explicit calibration and donor selection."""
import hashlib
import numpy as np

from ._descriptors import descriptors
from ._herding import greedy_mmd
from ._knee import knee, mmd2_path, valid_numerics


def identity_folds(target_ids):
    """Return the paper's identity-hashed order and five-fold assignment.

    IDs must be unique strings denoting physical trajectories. Forward and
    backward evaluations must share these IDs. No random generator is used.
    The historical hash salt and five folds are fixed, not tunable parameters.
    """
    ids = list(target_ids)
    if any(not isinstance(x, str) for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("target_ids must be unique strings")
    order = sorted(range(len(ids)), key=lambda i: hashlib.sha256(
        ("full-distance-v1-20260923|" + ids[i]).encode()).hexdigest())
    assignment = {i: position % 5 for position, i in enumerate(order)}
    return order, assignment


def select_representatives(features):
    """Herd a finite nonempty feature matrix, then take its exact knee prefix.

    Returns local row indices and diagnostics. Kernel bandwidth is the median
    pairwise Euclidean distance (or one at <=1e-12). Herding ties follow input
    order; the smallest exact knee maximizer wins. No positive knee score,
    singleton, or zero discrepancy retains one representative. Material
    negative MMD squared or a nonzero full-pool residual fails the numerical
    gate instead of changing the selected size.
    """
    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or not len(features) or not features.shape[1] or not np.isfinite(features).all():
        raise ValueError("features must be a finite nonempty two-dimensional matrix")
    order, clipped_path, bandwidth, kernel = greedy_mmd(features)
    raw_path = mmd2_path(kernel, order)
    if not np.array_equal(np.maximum(raw_path, 0.0), clipped_path):
        raise ArithmeticError("herding and cumulative MMD paths disagree")
    diagnostics = knee(raw_path)
    if not valid_numerics(diagnostics):
        raise ArithmeticError("MMD roundoff gate failed")
    indices = [int(i) for i in order[:diagnostics["m"]]]
    return indices, {"order": order, "bandwidth": bandwidth, "knee": diagnostics}


def _indices(values, n, name):
    values = list(values)
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) for x in values):
        raise ValueError(f"{name} must contain integer row indices")
    if len(set(values)) != len(values) or any(x < 0 or x >= n for x in values):
        raise ValueError(f"{name} must contain distinct in-range row indices")
    return values


def select_pools(oriented_curves, boundary_count, permitted_indices, valid_indices):
    """Select C from permitted valid prefixes, then D from permitted rows minus C.

    ``oriented_curves`` is (trajectories, observation positions), already
    reversed for backward forecasting. ``permitted_indices`` is the ordered
    out-of-fold pool O; ``valid_indices`` denotes targets eligible in BOTH
    directions. Evaluation identities must be absent from O. Only P_C prefix
    values and P_D full curves are inspected; other rows may contain NaNs.
    The caller owns identity/content permissions and the eligibility decision.

    Returns a dict of original row indices and diagnostics. An empty P_C or
    P_D raises ValueError; no fallback or minimum donor size is introduced.
    A singleton D is selectable, but N1 endpoint estimation requires >=2.
    """
    rows = np.asarray(oriented_curves)
    if rows.ndim != 2 or not isinstance(boundary_count, (int, np.integer)) or not 3 <= boundary_count < rows.shape[1]:
        raise ValueError("provide a matrix and integer boundary_count in [3, horizon length)")
    permitted = _indices(permitted_indices, len(rows), "permitted_indices")
    valid = set(_indices(valid_indices, len(rows), "valid_indices"))
    candidates = [i for i in permitted if i in valid]
    if not candidates:
        raise ValueError("empty permitted calibration candidate pool P_C")
    prefixes = rows[candidates, :boundary_count]
    if not np.isfinite(prefixes).all():
        raise ValueError("nonfinite permitted calibration prefix")
    lo = float(prefixes.min())
    scale = float(np.ptp(prefixes))
    scale = scale if scale > 1e-12 else 1.0
    features = descriptors(prefixes, lo, scale)
    mean, standard_deviation = features.mean(0), features.std(0)
    standard_deviation = np.where(standard_deviation > 1e-12, standard_deviation, 1.0)
    features = (features - mean) / standard_deviation
    selected, c_diagnostics = select_representatives(features)
    calibration = [candidates[i] for i in selected]
    sources = [i for i in permitted if i not in set(calibration)]
    if not sources:
        raise ValueError("empty permitted source pool P_D after excluding selected C")
    source_rows = rows[sources]
    if not np.isfinite(source_rows).all():
        raise ValueError("nonfinite permitted source curve")
    lo = float(source_rows[:, :boundary_count].min())
    scale = float(np.ptp(source_rows[:, :boundary_count]))
    scale = scale if scale > 1e-12 else 1.0
    selected, d_diagnostics = select_representatives((source_rows - lo) / scale)
    return {
        "O": permitted, "P_C": candidates, "C": calibration,
        "P_D": sources, "D": [sources[i] for i in selected],
        "C_diagnostics": c_diagnostics, "D_diagnostics": d_diagnostics,
    }
