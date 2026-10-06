"""Gaussian kernel herding without replacement; pool-order ties."""
import numpy as np

def greedy_mmd(Z):
    """Greedy MMD minimization (kernel herding without replacement); RBF kernel, median-heuristic bandwidth; full nested order."""
    n = len(Z)
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    bw = float(np.median(np.sqrt(d2[np.triu_indices(n, 1)]))) if n > 1 else 1.0
    bw = bw if bw > 1e-12 else 1.0
    K = np.exp(-d2 / (2 * bw * bw))
    mu = K.mean(1)
    kbar = float(K.mean())
    chosen = np.zeros(n, bool)
    sumK = np.zeros(n)
    selfK = 0.0
    musum = 0.0
    order = []
    mmd = []
    for t in range(n):
        m1 = t + 1
        score = (selfK + 2 * sumK + np.diag(K)) / m1 ** 2 - 2 * (musum + mu) / m1
        score[chosen] = np.inf
        x = int(np.argmin(score))
        chosen[x] = True
        order.append(x)
        selfK += 2 * sumK[x] + K[x, x]
        sumK += K[:, x]
        musum += mu[x]
        mmd.append(max(selfK / m1 ** 2 - 2 * musum / m1 + kbar, 0.0))
    return (order, mmd, bw, K)
