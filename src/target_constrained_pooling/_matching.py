"""Frozen N1 five-point matched-increment endpoint; no fitters or variants."""
import math
import numpy as np
from scipy.optimize import minimize_scalar
WIN = 5
D1 = np.diff(np.eye(WIN), axis=0)
BRANCH = {"N1": dict(lam=0.0, curv=False, rhos=(1.0,), amp=False)}

def smoother(lam):
    assert lam == 0.0
    return np.eye(WIN)

def feature_rows(lam, curv):
    A = smoother(lam)
    assert not curv
    R = np.vstack([A[-1:], D1 @ A])
    return (A, R, np.linalg.norm(R, axis=1))

def candidates(tau, L, H, rhos):
    """Supported (p, rho): ceil(1 + 4 rho) <= p <= tau (backward window) and p + rho H <= L (forward horizon); p is a 1-based stage end."""
    out = []
    for r in rhos:
        for p in range(max(1, math.ceil(1 + (WIN - 1) * r - 1e-12)), tau + 1):
            if p + r * H <= L + 1e-12:
                out.append((p, r))
    return out

def donor_window(y, p, r):
    """5 chronological values at positions p - r(4-j), j=0..4, by linear interpolation of y TRUNCATED at p (only positions 1..p enter)."""
    t = np.arange(1, p + 1, dtype=float)
    x = p - r * (WIN - 1 - np.arange(WIN))
    assert x[0] >= 1 - 1e-12
    return np.interp(x, t, y[:p])

def motion(y, p, r, H):
    L = len(y)
    x = p + r * H
    assert x <= L + 1e-12
    return float(np.interp(x, np.arange(1, L + 1, dtype=float), y) - y[p - 1])

def prepare(branch, donors, tau, H):
    """N1 candidate features split around the stage level, with amplitude one."""
    cfg = BRANCH[branch]
    (A, R, sig) = feature_rows(cfg['lam'], cfg['curv'])
    L = donors.shape[1]
    cand = candidates(tau, L, H, cfg['rhos'])
    (n, C) = (len(donors), len(cand))
    phi0 = np.empty((n, C, len(sig)))
    phi1 = np.empty((n, C, len(sig)))
    mot = np.empty((n, C))
    for (c, (p, r)) in enumerate(cand):
        for k in range(n):
            z = donor_window(donors[k], p, r)
            base = donors[k, p - 1]
            phi0[k, c] = R @ np.full(WIN, base)
            phi1[k, c] = R @ (z - base)
            mot[k, c] = motion(donors[k], p, r, H)
    return dict(cfg=cfg, A=A, R=R, sig=sig, cand=cand, phi0=phi0, phi1=phi1, mot=mot)

def align(P, v_target):
    """Best candidate per donor for one (pseudo-)target window v_target (5 chronological values). Returns d, motion, stage, rho, amplitude, flags."""
    (A, R, sig) = (P['A'], P['R'], P['sig'])
    ft = R @ v_target
    (n, C) = P['mot'].shape
    a = np.ones((n, C))
    flat = np.zeros((n, C), bool)
    clipped = np.zeros((n, C), bool)
    fk = P['phi0'] + a[..., None] * P['phi1']
    d2 = np.mean(((ft - fk) / sig) ** 2, axis=-1)
    rho = np.array([c[1] for c in P['cand']])
    pst = np.array([c[0] for c in P['cand']])
    best = np.empty(n, int)
    for k in range(n):
        cand_idx = np.flatnonzero(d2[k] == d2[k].min())
        order = sorted(cand_idx, key=lambda c: (abs(math.log(rho[c])), -pst[c]))
        best[k] = order[0]
    ar = np.arange(n)
    return dict(d=np.sqrt(d2[ar, best]), motion=a[ar, best] * P['mot'][ar, best], stage=pst[best], rho=rho[best], amp=a[ar, best], amp_flat=flat[ar, best], amp_clipped=clipped[ar, best], ties=int(sum((int(np.sum(d2[k] == d2[k].min()) > 1) for k in range(n)))))

def endpoint(branch, Y, tau, sg, P=None):
    """Y: donors (rows 0..n-1, full curves) + target (last row, NaN future). Mirrors the frozen three_motion.endpoint bandwidth/weights/projection."""
    L = Y.shape[1]
    H = L - tau
    donors = Y[:-1]
    yt = Y[-1]
    n = len(donors)
    assert np.isnan(yt[tau:]).all() and np.isfinite(donors).all() and np.isfinite(yt[:tau]).all()
    P = prepare(branch, donors, tau, H) if P is None else P
    bt = float(yt[tau - 1])
    T = align(P, yt[tau - WIN:tau])
    Db = np.zeros((n, n))
    Mb = np.zeros((n, n))
    for j in range(n):
        a_j = align(P, donors[j, tau - WIN:tau])
        Db[j] = a_j['d']
        Mb[j] = a_j['motion']
    off = ~np.eye(n, dtype=bool)
    Db = np.where(off, Db, 0.0)
    Mb = np.where(off, Mb, 0.0)
    bq = donors[:, tau - 1].copy()
    truth = donors[:, -1].copy()
    pos = Db[off]
    med = float(np.median(pos[pos > 0])) if np.any(pos > 0) else 1.0
    loo_med = np.array([np.median(Mb[j, off[j]]) for j in range(n)])

    def cv(logm):
        h = max(med * np.exp(logm), 1e-12)
        Wm = np.exp(-(Db / h) ** 2)
        Wm[~off] = 0.0
        den = Wm.sum(1)
        pred = bq + np.where(den > 1e-12, (Wm * Mb).sum(1) / np.maximum(den, 1e-12), loo_med)
        return float(np.mean(np.abs(pred - truth)))
    r = minimize_scalar(cv, bounds=(np.log(0.001), np.log(50.0)), method='bounded', options={'xatol': 0.001})
    m = float(np.exp(r.x))
    h = max(med * m, 1e-12)
    uni = cv(np.log(1000000.0)) <= r.fun
    if uni:
        h = med * 1000000.0
    (dt, Mt) = (T['d'], T['motion'])
    wu = np.exp(-(dt / h) ** 2)
    s = float(wu.sum())
    fb = s <= 1e-12
    pooled = float(np.median(Mt)) if fb else float(np.sum(wu * Mt) / s)
    Vraw = bt + pooled
    V = float(bt + sg * max(sg * (Vraw - bt), 0.0))
    w = None if fb else wu / s
    dbar = float(np.mean(dt)) if fb else float(np.sum(w * dt))
    med_pos = float(np.median(pos[pos > 0])) if np.any(pos > 0) else 0.0
    u = (1.0 if dbar == 0 else 0.0) if med_pos == 0 else float(1.0 / (1.0 + (dbar / med_pos) ** 2))
    Vuni_raw = bt + float(np.mean(Mt))
    Vuni = float(bt + sg * max(sg * (Vuni_raw - bt), 0.0))
    return dict(b=bt, H=H, V_raw=Vraw, V=V, pooled_motion=pooled, projection_active=bool(sg * (Vraw - bt) < 0), median_fallback=bool(fb), m=float(h / med), h=float(h), median_distance=med, uniform_limit=bool(uni), ess=float(1 / np.sum(w ** 2)) if w is not None else None, kernel_mass=s, support_u=u, dbar=dbar, V_uniform_raw=Vuni_raw, V_uniform=Vuni, n_candidates=len(P['cand']), candidates_by_rho={str(rr): sum((1 for c in P['cand'] if c[1] == rr)) for rr in P['cfg']['rhos']}, ties=T['ties'], donors=[dict(k=k, stage=int(T['stage'][k]), rho=float(T['rho'][k]), amp=float(T['amp'][k]), amp_flat=bool(T['amp_flat'][k]), amp_clipped=bool(T['amp_clipped'][k]), dist=float(dt[k]), motion=float(Mt[k]), w=float(w[k]) if w is not None else None) for k in range(n)])
