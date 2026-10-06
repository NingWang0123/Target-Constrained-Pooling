# Target-Constrained Pooling

TCP transfers permitted source endpoint information into an existing free
forecast. This package implements the current forecast-level method: knee
selection, uniform and matched-increment endpoints, directional projection,
the anchored residual bridge with bounded PAVA, calibration strengths, and
harm-first weighting with accuracy-preserving shrinkage.

It accepts cached free future forecasts from any forecaster. It contains no
base-model fitters, datasets, experimental outputs, cluster runners, or paper
build files. The example and test inputs are synthetic.

## Install and run

Requires Python 3.10+ and NumPy, SciPy, and mpmath.

```sh
python -m pip install .
python examples/toy.py
python -m unittest discover -s tests -v
```

The release was checked with NumPy 2.2.6, SciPy 1.15.3 and mpmath 1.3.0.
The same algorithms are used across supported versions; identical last bits
across library versions or hardware are not promised.

## Workflow and API

1. Orient observations toward the forecasting horizon. Reverse the physical
   order first for `direction="bwd"`; a prefix always comes first. Target
   and source rows must share this orientation. `fwd` fixes response sign
   sigma=+1 (increase), and `bwd` fixes sigma=-1 (decrease). These are supplied
   response assumptions, not signs inferred from data; physical reversal
   alone does not establish the expected response sign.
2. Establish physical identity permissions before selection. `identity_folds`
   reproduces the fixed identity-hashed five-fold order. `select_pools` takes
   an explicit ordered out-of-fold pool O and valid target indices (eligible
   in both directions), selects C using five z-scored prefix descriptors,
   then selects D from O minus the selected C using full source curves.
   Identity exclusions are the caller's responsibility; duplicate curve
   contents are preserved. Only permitted P_C prefixes and P_D full curves
   are read. Evaluation continuations must not enter the permitted pool.
3. Put prefix, D sources, cached free tail and C truth into the same positive
   affine response coordinate. `response_coordinate` reproduces the packet
   map using donor prefixes plus the target prefix. Retain existing maps for
   cached forecasts; do not renormalize or refit their originating models.
4. Estimate endpoints, build hard branches on C, and call `calibrate` with
   C truth only. Reuse the resulting `Policy` on evaluation targets.
5. `predict` returns only future values. Preserve the observed prefix
   externally and inverse-transform the future for raw-response output.

```python
from target_constrained_pooling import (
    common_branch, estimate_endpoints, calibrate, predict,
)

# All arrays below already share the saved response coordinate.
# C_free, C_U, C_M, C_truth: shape (calibration trajectories, H).
policy = calibrate(C_free, C_U, C_M, C_truth)
endpoints = estimate_endpoints(observed_prefix, permitted_donors, "fwd")
future = predict(cached_free_tail, observed_prefix[-1], endpoints, policy)
```

See [examples/toy.py](examples/toy.py) for a runnable complete example,
including hard-branch construction and coordinate conversion.
`common_branch(free, boundary, endpoint, return_bridge=True)` also exposes
the residual bridge for inspection. `select_representatives(features)`
exposes the herding/knee primitive with diagnostic MMD paths.

## Numerical method and boundaries

Selection uses a Gaussian kernel with median pairwise distance bandwidth.
Herding ties follow pool order; the knee chooses the smallest exact float64
maximizer. Singleton, zero-discrepancy and no-positive-score pools retain one
member. Empty pools fail. Material negative MMD squared or a full-pool
residual outside `64*N*eps` fails the numerical gate without altering selection.

U is the mean selected donor terminal level. M matches the last five target
points against source stages p=5,...,m using last level and four differences,
scaled by row norms 1 and sqrt(2); distance ties choose later p. It transfers
`source[p+H] - source[p]`, where H=L-m and p is one-based. This translated
horizon equals the original terminal only at p=m. Source-only leave-one-out
endpoint MAE chooses the bandwidth with bounded SciPy search in
log(kappa)=[log(1e-3),log(50)], tolerance 1e-3. The finite kappa=1e6 probe
wins when no worse; kernel mass <=1e-12 uses the median increment. This
nonconvex bounded search has no global-optimum certificate. Matching needs
five observed points and at least two donors. A selected singleton remains
valid selection, but its leave-one-out bandwidth is undefined: estimation
raises a precise error rather than adding a new fallback.

Projected endpoints respect the given response direction. Each hard branch
adds `j/H * (V-free[-1])` to the free tail, then applies unweighted PAVA to
its interior in signed boundary coordinates and clips to [0,abs(V-b)]. The
observed boundary and exact terminal are fixed. A zero span is constant;
H=1 is the endpoint. This is a projection of bridge VALUES, not a joint
slope-minimization problem. Calibrated branches and TCP need not be monotone;
they are not projected again.

Calibration loss is mean per-trajectory tail RMSE, including the endpoint,
with equal trajectory weights and a common H within each policy fit. Source
main strengths use SciPy bounded search with xatol=1e-8, compare its returned
candidate with 0 and 1, and resolve exact float objective ties toward smaller
strength. This does not claim exact global smallest-minimizer calibration.
The harm solver minimizes mean positive RMSE increase relative to the free
forecast, resolves ties toward weight 1/2, then moves toward that minimizer
as far as mean RMSE no worse than equal averaging permits. Its audited
200-bit mpmath primitives use sign zero band 2^-150 and bisection resolution
1e-15, with a private precision context. These are ordinary high-precision
numerics, not interval certificates. Diagnostics are retained in `Policy`.

All method steps are deterministic. No seeds are drawn or accepted; seeds
used to produce external free forecasts belong to their forecasters. The
toy generator uses an explicit seed solely to make its synthetic input
reproducible. Tests cover synthetic reference parity, permissions, matching
ties, degenerate branches, loss units, and calibration/shrinkage behavior.
