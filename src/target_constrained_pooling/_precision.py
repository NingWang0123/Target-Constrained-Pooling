"""Audited 200-bit one-sided derivative primitives; ordinary high precision."""
import math
import mpmath as _mpmath
import numpy as np
mp = _mpmath.mp.clone()
mp.prec = 200
ZB = mp.mpf(2) ** -150
RES = 1e-15
EPS = float(np.finfo(float).eps)

class HP:

    def __init__(s, R, DU, DN, aU, aN):
        (R, DU, DN) = (np.atleast_2d(R), np.atleast_2d(DU), np.atleast_2d(DN))
        (s.n, s.H) = R.shape
        (A, B) = (mp.mpf(float(aU)), mp.mpf(float(aN)))
        s.U = [[mp.mpf(float(R[i, j])) - A * mp.mpf(float(DU[i, j])) for j in range(s.H)] for i in range(s.n)]
        s.N = [[mp.mpf(float(R[i, j])) - B * mp.mpf(float(DN[i, j])) for j in range(s.H)] for i in range(s.n)]
        s.D = [[u - v for (u, v) in zip(Ui, Ni)] for (Ui, Ni) in zip(s.U, s.N)]
        s.sqH = mp.sqrt(s.H)
        s.phi = [mp.sqrt(mp.fsum((mp.mpf(float(x)) ** 2 for x in R[i]))) / s.sqH for i in range(s.n)]
        s.dn = [mp.sqrt(mp.fsum((d * d for d in Di))) / s.sqH for Di in s.D]
        z = R - float(aN) * DN
        v = float(aU) * DU - float(aN) * DN
        s.zn = np.sqrt((z * z).sum(1))
        s.vn = np.sqrt((v * v).sum(1))
        s.phif = np.sqrt((R * R).sum(1)) / math.sqrt(s.H)
        s.c = 8.0 * (s.H + 4) * EPS
        s.evals = 0

    def tau(s, which, w):
        t = s.c * float(np.mean((s.zn + abs(w) * s.vn) / math.sqrt(s.H)))
        return t + (s.c * float(np.mean(s.phif)) if which == 'G' else 0.0)

    def d(s, which, w):
        """exact (200-bit) f(w), f'_+(w), f'_-(w) and the cancellation scale of the derivative sums"""
        s.evals += 1
        X = mp.mpf(float(w))
        O = 1 - X
        (f, dp, dm, sc) = ([], [], [], [])
        for (Ui, Ni, Di, dn, ph) in zip(s.U, s.N, s.D, s.dn, s.phi):
            e = [X * u + O * v for (u, v) in zip(Ui, Ni)]
            ne = mp.sqrt(mp.fsum((x * x for x in e)))
            r = ne / s.sqH
            if ne == 0:
                (gp, gm) = (dn, -dn)
            else:
                gp = gm = mp.fsum((x * y for (x, y) in zip(e, Di))) / (ne * s.sqH)
            if which == 'L':
                f.append(r)
                dp.append(gp)
                dm.append(gm)
            else:
                t = r - ph
                if abs(t) <= ZB * max(1, ph):
                    f.append(mp.mpf(0))
                    dp.append(max(gp, mp.mpf(0)))
                    dm.append(min(gm, mp.mpf(0)))
                elif t > 0:
                    f.append(t)
                    dp.append(gp)
                    dm.append(gm)
                else:
                    f.append(mp.mpf(0))
                    dp.append(mp.mpf(0))
                    dm.append(mp.mpf(0))
            sc.append(abs(gp) + abs(gm))
        return (mp.fsum(f) / s.n, mp.fsum(dp) / s.n, mp.fsum(dm) / s.n, mp.fsum(sc) / s.n)

    def val(s, which, w):
        return s.d(which, w)[0]

def sgn(x, scale):
    return 0 if abs(x) <= ZB * max(1, scale) else 1 if x > 0 else -1
