"""Harm minimization followed by directed accuracy-preserving shrinkage."""
from . import _precision as V3
RES, sgn = V3.RES, V3.sgn

def harm_argmin(hp):
    """proj_M(0.5), M = argmin_[0,1] G. Returns (w_H, info)."""
    (_, p5, m5, s5) = hp.d('G', 0.5)
    (sp, sm) = (sgn(p5, s5), sgn(m5, s5))
    if sm <= 0 <= sp:
        return (0.5, dict(case='0.5 in M', flat=bool(sp == 0 or sm == 0), bracket=None))
    if sp < 0:
        (_, p1, _, s1) = hp.d('G', 1.0)
        if sgn(p1, s1) < 0:
            return (1.0, dict(case='endpoint 1', flat=False, bracket=None))
        (a, b) = (0.5, 1.0)
        while abs(b - a) > RES:
            m = 0.5 * (a + b)
            if m in (a, b):
                break
            (_, pm, _, sc) = hp.d('G', m)
            (a, b) = (m, b) if sgn(pm, sc) < 0 else (a, m)
        (_, pb, _, sb) = hp.d('G', b)
        return (b, dict(case='interior right' if b < 1.0 else 'endpoint 1 (kink)', flat=bool(sgn(pb, sb) == 0), bracket=[a, b]))
    (_, _, m0, s0) = hp.d('G', 0.0)
    if sgn(m0, s0) > 0:
        return (0.0, dict(case='endpoint 0', flat=False, bracket=None))
    (a, b) = (0.0, 0.5)
    while abs(b - a) > RES:
        m = 0.5 * (a + b)
        if m in (a, b):
            break
        (_, _, mm, sc) = hp.d('G', m)
        (a, b) = (a, m) if sgn(mm, sc) > 0 else (m, b)
    (_, _, ma, sa) = hp.d('G', a)
    return (a, dict(case='interior left' if a > 0.0 else 'endpoint 0 (kink)', flat=bool(sgn(ma, sa) == 0), bracket=[a, b]))

def shrink_weight(hp):
    """Returns (w_star, info). info: w_H, harm-argmin case, shrink flag, rho, feasible/infeasible ends of the directed root."""
    LA = hp.val('L', 0.5)
    feas = lambda w: sgn(hp.val('L', w) - LA, abs(LA)) <= 0
    (wH, hi) = harm_argmin(hp)
    if feas(wH):
        return (wH, dict(hi, wH=wH, shrink=False, rho=1.0, w_out=None, evals=hp.evals))
    (a, b) = (0.5, wH)
    while abs(b - a) > RES:
        m = 0.5 * (a + b)
        if m in (a, b):
            break
        (a, b) = (m, b) if feas(m) else (a, m)
    return (a, dict(hi, wH=wH, shrink=True, rho=(a - 0.5) / (wH - 0.5), w_out=b, evals=hp.evals))
