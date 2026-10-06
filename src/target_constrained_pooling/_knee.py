"""Exact float64 knee rule and roundoff gates."""
import numpy as np
EPS = float(np.finfo(np.float64).eps)
NEG_FLAG_FACTOR = 64

def flag_bound(n):
    return NEG_FLAG_FACTOR * max(int(n), 1) * EPS

def mmd2_path(K, order):
    """Unclipped cumulative MMD^2 of prefixes of `order` against the whole pool of kernel K (float64)."""
    K = np.asarray(K, dtype=np.float64)
    n = len(K)
    order = [int(x) for x in order]
    assert K.shape == (n, n) and sorted(order) == list(range(n))
    mu = K.mean(1)
    kbar = float(K.mean())
    sumK = np.zeros(n)
    selfK = 0.0
    musum = 0.0
    out = np.empty(n, dtype=np.float64)
    for (t, x) in enumerate(order):
        m1 = t + 1
        selfK += 2 * sumK[x] + K[x, x]
        sumK += K[:, x]
        musum += mu[x]
        out[t] = selfK / m1 ** 2 - 2 * musum / m1 + kbar
    return out

def valid_numerics(k):
    """Aggregation gate on numerical validity only (not a selection rule): no material negative MMD^2 and a full-pool
    residual within B_N. knee() never changes m because of these flags; the aggregator must reject a pool that fails."""
    return k.get('N', 0) == 0 or (not k['material_negative_mmd2'] and k['full_pool_residual_ok'])

def knee(d2):
    """Knee choice from the unclipped MMD^2 path d2[m-1], m = 1..N. Returns a JSON-serialisable dict."""
    raw = np.asarray(d2, dtype=np.float64)
    N = int(raw.size)
    if N == 0:
        return dict(status='INFEASIBLE_EMPTY_POOL', N=0, m=None, branch='empty_pool')
    assert np.isfinite(raw).all(), 'non-finite MMD^2'
    B = flag_bound(N)
    d = np.sqrt(np.maximum(raw, 0.0))
    d[-1] = 0.0
    out = dict(status='PASS', N=N, mmd2_raw=raw.tolist(), mmd=d.tolist(), flag_bound=B, negative_mmd2_count=int((raw < 0).sum()), min_mmd2_raw=float(raw.min()), material_negative_mmd2=bool((raw < -B).any()), full_pool_mmd2_computed=float(raw[-1]), full_pool_residual_ok=bool(abs(raw[-1]) <= B))
    if N == 1:
        out.update(m=1, branch='singleton', x=[0.0], y=[None], score=[None])
        return out
    x = np.arange(N, dtype=np.float64) / (N - 1)
    if d[0] == d[-1]:
        out.update(m=1, branch='degenerate_d1_eq_dN', x=x.tolist(), y=None, score=None)
        return out
    y = (d - d[-1]) / (d[0] - d[-1])
    score = 1.0 - x - y
    m = int(np.argmax(score)) + 1
    assert score[0] == 0.0 and score[-1] == 0.0, (score[0], score[-1])
    assert 1 <= m < N, (m, N)
    top = float(score[m - 1])
    ties = int(np.sum(score == score.max()))
    out.update(m=m, branch='interior_knee' if top > 0 else 'no_interior_knee', x=x.tolist(), y=y.tolist(), score=score.tolist(), max_score=top, exact_argmax_ties=ties, nonmonotone=bool(np.any(np.diff(d) > 0)), y_above_1=bool(np.any(y > 1.0)))
    return out
