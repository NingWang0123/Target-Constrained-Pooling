"""Five frozen prefix descriptors."""
import numpy as np

def descriptors(heads, lo, scale):
    import numpy as np
    h = (heads - lo) / scale
    x = np.linspace(0, 1, h.shape[1])
    coef = np.polynomial.polynomial.polyfit(x, h.T, 2)
    slope = np.sum((x - x.mean()) * (h - h.mean(1, keepdims=True)), axis=1) / np.sum((x - x.mean()) ** 2)
    return np.column_stack([h[:, -1], slope, 2 * coef[2], h.mean(1), np.ptp(h, axis=1)])
