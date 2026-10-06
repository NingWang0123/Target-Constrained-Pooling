"""Source-faithful bounded SciPy strength fit with endpoint comparison."""
import numpy as np
from scipy.optimize import minimize_scalar

def fit_alpha(pairs):

    def objective(a):
        return float(np.mean([np.sqrt(np.mean((y - f - a * (h - f)) ** 2)) for (f, h, y) in pairs]))
    opt = minimize_scalar(objective, bounds=(0, 1), method='bounded', options={'xatol': 1e-08})
    assert opt.success
    a = min([0.0, 1.0, float(opt.x)], key=lambda a: (objective(a), a))
    return (a, {'mean_C_rmse': objective(a), 'free_C_rmse': objective(0), 'hard_C_rmse': objective(1)})
