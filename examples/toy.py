"""Small synthetic usage example; no external data or model training."""
import numpy as np
from target_constrained_pooling import (
    calibrate, common_branch, estimate_endpoints, predict, response_coordinate,
)


def main():
    rng = np.random.default_rng(7)
    positions = np.arange(1, 13, dtype=float)
    donors = np.array([0.05 * k + (0.8 + 0.04 * k) * (1 - np.exp(-positions / (4 + k)))
                       for k in range(8)])
    observed_count = 7
    calibration_free, uniform_hard, matched_hard, calibration_truth = [], [], [], []
    for k in range(4):
        # Independent synthetic pseudo-targets; their truth belongs to C only.
        trajectory = (0.85 + 0.05 * k) * (1 - np.exp(-positions / (4.5 + k)))
        prefix = trajectory[:observed_count]
        coordinate = response_coordinate(prefix, donors)
        prefix, sources = coordinate.transform(prefix), coordinate.transform(donors)
        free = coordinate.transform(trajectory[observed_count:] + rng.normal(0, 0.04, 5))
        endpoints = estimate_endpoints(prefix, sources)
        calibration_free.append(free)
        uniform_hard.append(common_branch(free, prefix[-1], endpoints.uniform))
        matched_hard.append(common_branch(free, prefix[-1], endpoints.matched))
        calibration_truth.append(coordinate.transform(trajectory[observed_count:]))
    policy = calibrate(calibration_free, uniform_hard, matched_hard, calibration_truth)

    # Evaluation supplies only the observed prefix and an already-made free tail.
    prefix = np.array([0.10, 0.22, 0.31, 0.39, 0.45, 0.50, 0.54])
    free_raw = np.array([0.58, 0.62, 0.67, 0.71, 0.75])
    coordinate = response_coordinate(prefix, donors)
    observed = coordinate.transform(prefix)
    endpoints = estimate_endpoints(observed, coordinate.transform(donors))
    future = predict(coordinate.transform(free_raw), observed[-1], endpoints, policy)
    print("strengths and U weight:", policy.alpha_uniform, policy.alpha_matched, policy.weight_uniform)
    print("TCP future:", coordinate.inverse(future))


if __name__ == "__main__":
    main()
