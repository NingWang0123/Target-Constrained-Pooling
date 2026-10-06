"""Scientific invariants and synthetic-only golden source-parity checks."""
import json
import unittest
from pathlib import Path
import numpy as np

from target_constrained_pooling import (
    Endpoints, Policy, calibrate, common_branch, estimate_endpoints,
    identity_folds, predict, project_endpoint, response_coordinate,
    select_pools, select_representatives,
)
from target_constrained_pooling._knee import knee
from target_constrained_pooling._precision import HP
from target_constrained_pooling._shrink import shrink_weight

REFERENCE = json.loads((Path(__file__).parent / "fixtures/source_parity.json").read_text())


class SourceParity(unittest.TestCase):
    def test_branch_golden_values_and_constraints(self):
        for record in REFERENCE["branches"]:
            with self.subTest(H=len(record["free"]), boundary=record["boundary"], endpoint=record["endpoint"]):
                q, bridge = common_branch(record["free"], record["boundary"], record["endpoint"], return_bridge=True)
                np.testing.assert_array_equal(q, record["q"])
                np.testing.assert_array_equal(bridge, record["bridge"])
                self.assertEqual(q[-1], record["endpoint"])
                b, v = record["boundary"], record["endpoint"]
                sign = np.sign(v-b)
                self.assertTrue(np.all(sign * np.diff(np.r_[b, q]) >= -1e-15))
                self.assertTrue(np.all(q >= min(b, v)-1e-15))
                self.assertTrue(np.all(q <= max(b, v)+1e-15))

    def test_selection_source_parity_and_permissions(self):
        record = REFERENCE["selection"]
        rows = np.vstack([record["rows"], np.full((2, 12), np.nan)])
        result = select_pools(rows, 7, list(range(11)), list(range(9)))
        source = record["reference"]
        for key in ["O", "P_C", "C", "P_D", "D"]:
            self.assertEqual(result[key], source[key])
        for pool in ["C", "D"]:
            diagnostics = result[pool + "_diagnostics"]
            allowed = result["P_" + pool]
            self.assertEqual([allowed[i] for i in diagnostics["order"]], source[pool + "_order"])
            self.assertEqual(diagnostics["bandwidth"], source[pool + "_bandwidth"])
            self.assertEqual(diagnostics["knee"], source[pool + "_knee"])
        self.assertFalse(set(result["C"]) & set(result["D"]))
        self.assertFalse((set(result["C"]) | set(result["D"])) & {11, 12})
        rows[11:] = 1e20
        rows[result["C"], 7:] = 1e15
        self.assertEqual(select_pools(rows, 7, list(range(11)), list(range(9))), result)

    def test_matched_endpoint_source_parity(self):
        record = REFERENCE["matching"]
        result = estimate_endpoints(record["prefix"], record["donors"])
        source = record["reference"]
        self.assertEqual(result.uniform_raw, np.asarray(record["donors"])[:, -1].mean())
        self.assertAlmostEqual(result.matched_raw, source["V_raw"], places=12)
        self.assertAlmostEqual(result.matched, source["V"], places=12)
        for actual, expected in zip(result.diagnostics["donors"], source["donors"]):
            self.assertEqual(actual["stage"], expected["stage"])
            for key in ["dist", "motion", "w"]:
                self.assertAlmostEqual(actual[key], expected[key], places=12)
        self.assertEqual(result.diagnostics["uniform_limit"], source["uniform_limit"])

    def test_calibration_source_parity_and_accuracy_constraint(self):
        self.assertTrue(any(r["weight_diagnostics"]["shrink"] for r in REFERENCE["calibration"]))
        for record in REFERENCE["calibration"]:
            f, u, m, y = [np.asarray(record[key]) for key in ["free", "uniform", "matched", "truth"]]
            result = calibrate(f, u, m, y)
            self.assertAlmostEqual(result.alpha_uniform, record["alpha_uniform"], places=12)
            self.assertAlmostEqual(result.alpha_matched, record["alpha_matched"], places=12)
            self.assertAlmostEqual(result.weight_uniform, record["weight_uniform"], places=14)
            self.assertEqual(result.diagnostics["weight"]["case"], record["weight_diagnostics"]["case"])
            self.assertEqual(result.diagnostics["weight"]["shrink"], record["weight_diagnostics"]["shrink"])
            for branch, strength in [(u, result.alpha_uniform), (m, result.alpha_matched)]:
                risks = [np.sqrt(np.mean((y-f-a*(branch-f))**2, axis=1)).mean() for a in [0, 1, strength]]
                self.assertLessEqual(risks[-1], min(risks[:2])+1e-14)
            hp = HP(y-f, u-f, m-f, result.alpha_uniform, result.alpha_matched)
            self.assertLessEqual(float(hp.val("L", result.weight_uniform)-hp.val("L", 0.5)), 1e-14)
            self.assertLessEqual(float(hp.val("G", result.weight_uniform)-hp.val("G", 0.5)), 1e-14)


class MethodInvariants(unittest.TestCase):
    def test_feasible_bridge_is_unchanged(self):
        q, bridge = common_branch([0.2, 0.5, 0.8], 0, 1, return_bridge=True)
        np.testing.assert_array_equal(q, bridge)

    def test_pava_projection_known_solution(self):
        # Endpoint already matches; adjacent 0.8/0.2 values pool to 0.5.
        np.testing.assert_array_equal(common_branch([0.8, 0.2, 1], 0, 1), [0.5, 0.5, 1])
        np.testing.assert_array_equal(common_branch([-3, 2, 1], 0, 1), [0, 1, 1])

    def test_strength_and_weight_ties(self):
        f = np.array([[1., 2.], [3., 4.]])
        result = calibrate(f, f, f, f)
        self.assertEqual((result.alpha_uniform, result.alpha_matched, result.weight_uniform), (0, 0, 0.5))

    def test_loss_is_equal_weight_per_trajectory_rmse(self):
        # Mean row RMSE=(sqrt(2)+0)/2; flattened RMSE=1.
        hp = HP(np.array([[2., 0.], [0., 0.]]), np.zeros((2, 2)), np.zeros((2, 2)), 0, 0)
        self.assertAlmostEqual(float(hp.val("L", 0.5)), np.sqrt(2)/2, places=14)
        self.assertNotAlmostEqual(float(hp.val("L", 0.5)), 1.0)

    def test_matching_later_stage_tie_and_translated_horizon(self):
        donors = np.array([[1.]*7+[2., 3., 4.], [2.]*7+[3., 4., 5.]])
        result = estimate_endpoints(np.ones(7), donors)
        self.assertEqual([x["stage"] for x in result.diagnostics["donors"]], [7, 7])
        donors = np.array([[0., 1., 2., 3., 4., 8., 16., 32., 64., 128.],
                           [0., 1., 2., 3., 4., 9., 18., 36., 72., 144.]])
        prefix = np.array([-2., -1., 0., 1., 2., 3., 4.])
        result = estimate_endpoints(prefix, donors)
        for k, detail in enumerate(result.diagnostics["donors"]):
            self.assertEqual(detail["stage"], 5)
            self.assertEqual(detail["motion"], donors[k, 7]-donors[k, 4])
            self.assertNotEqual(detail["motion"], donors[k, -1]-donors[k, 4])

    def test_knee_degenerate_and_exact_tie_rules(self):
        self.assertEqual(select_representatives(np.ones((4, 2)))[0], [0])
        self.assertEqual(select_representatives(np.ones((1, 2)))[0], [0])
        self.assertEqual(knee([1., 1., 0.])["m"], 1)
        self.assertEqual(knee([])["status"], "INFEASIBLE_EMPTY_POOL")
        # Scores at m=2 and 3 both equal 1/3; exact tie chooses m=2.
        self.assertEqual(knee([1., 1/9, 0., 0.])["m"], 2)

    def test_response_coordinate_and_direction(self):
        prefix = np.array([2., 3., 4., 5., 6.])
        donors = np.array([[1., 2., 3., 4., 5., 100.], [3., 4., 5., 6., 7., 200.]])
        coordinate = response_coordinate(prefix, donors)
        self.assertEqual((coordinate.offset, coordinate.scale), (1, 6))
        np.testing.assert_array_equal(coordinate.inverse(coordinate.transform(prefix)), prefix)
        self.assertEqual(project_endpoint(9, 3, "bwd"), 3)
        self.assertEqual(project_endpoint(1, 3, "fwd"), 3)

    def test_prediction_uses_frozen_calibrated_arithmetic(self):
        f = np.array([0.5, 0.2, 0.7])
        endpoints = Endpoints(1, 0.8, 1, 0.8)
        policy = Policy(0.3, 0.6, 0.2)
        u, m = common_branch(f, 0, 1), common_branch(f, 0, 0.8)
        expected = 0.2*(f+0.3*(u-f)) + (1-0.2)*(f+0.6*(m-f))
        np.testing.assert_array_equal(predict(f, 0, endpoints, policy), expected)

    def test_no_seed_draws_or_global_precision_mutation(self):
        import mpmath
        before = mpmath.mp.prec
        f = np.zeros((2, 2))
        calibrate(f, f, f, f)
        self.assertEqual(mpmath.mp.prec, before)
        order, assignments = identity_folds(["physical-b", "physical-a", "physical-c"])
        self.assertEqual((order, assignments), identity_folds(["physical-b", "physical-a", "physical-c"]))

    def test_invalid_inputs_fail_without_fallback(self):
        for call in [lambda: common_branch([], 0, 1), lambda: common_branch([np.nan], 0, 1),
                     lambda: select_representatives(np.empty((0, 2))),
                     lambda: estimate_endpoints(np.ones(5), np.ones((1, 7))),
                     lambda: calibrate([[0, 0]], [[0]], [[0, 0]], [[0, 0]]),
                     lambda: identity_folds(["same", "same"])]:
            with self.assertRaises(ValueError):
                call()


if __name__ == "__main__":
    unittest.main()
