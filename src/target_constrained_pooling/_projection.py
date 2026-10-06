"""Anchored residual bridge and bounded monotone Euclidean projection."""
import numpy as np

def pava(x):
    """Unweighted Euclidean projection onto nondecreasing sequences."""
    x = np.asarray(x, dtype=float)
    assert x.ndim == 1 and np.isfinite(x).all()
    means = []
    counts = []
    for v in x:
        means.append(float(v))
        counts.append(1)
        while len(means) > 1 and means[-2] > means[-1]:
            n = counts[-2] + counts[-1]
            v = (counts[-2] * means[-2] + counts[-1] * means[-1]) / n
            means[-2:] = [v]
            counts[-2:] = [n]
    return np.repeat(means, counts)

def branch(f, b, v):
    """Saved future grid tau+1,...,T; observed boundary is a discrete anchor.

    q is the L2 projection of f+r*(v-f[-1]) onto monotone tails between
    fixed b and v, with last value exactly v. No fitted f(tau) is inferred.
    """
    f = np.asarray(f, dtype=float)
    assert f.ndim == 1 and len(f) > 0 and np.isfinite(f).all()
    assert np.isfinite(b) and np.isfinite(v)
    h = len(f)
    r = np.arange(1, h + 1, dtype=float) / h
    a = f + r * (v - f[-1])
    a[-1] = v
    s = float(np.sign(v - b))
    span = abs(v - b)
    if s == 0:
        q = np.full(h, b, dtype=float)
    elif np.all(np.diff(np.r_[b, a]) * s >= 0):
        q = a.copy()
    else:
        q = np.r_[b + s * np.clip(pava(s * (a[:-1] - b)), 0, span), v]
    q[-1] = v
    assert np.isfinite(q).all()
    return (q, a)
