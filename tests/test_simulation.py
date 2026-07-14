from __future__ import annotations

import unittest

import numpy as np
import scipy.sparse as sp

from compass.model import LdChromosomeBlock
from compass.simulation import generate_sparse_context_truth, ld_propagate, simulate_noncentral_chisq
from compass.sldsc import fit_sldsc_wls


class SimulationTest(unittest.TestCase):
    def test_sparse_context_truth_is_seeded_and_normalized(self):
        scores = np.array([[1.0, 0.0], [2.0, 3.0], [4.0, 5.0], [0.0, 7.0]])
        target = np.array([0.14, 0.06])
        first = generate_sparse_context_truth(scores, target, causal_fraction=1.0, seed=7)
        second = generate_sparse_context_truth(scores, target, causal_fraction=1.0, seed=7)
        np.testing.assert_allclose(first.direct_h2, second.direct_h2)
        self.assertAlmostEqual(float(first.direct_h2.sum()), 0.20, places=7)
        np.testing.assert_array_equal(first.selected_counts, [3, 3])

    def test_ld_propagation_and_noncentral_sampling_are_seeded(self):
        matrix = sp.csr_matrix(np.array([[1.0, 0.25], [0.25, 1.0]], dtype=np.float32))
        block = LdChromosomeBlock(chrom=1, rows=np.array([0, 1]), R2=matrix)
        propagated = ld_propagate([block], np.array([0.10, 0.02], dtype=np.float32))
        np.testing.assert_allclose(propagated, [0.105, 0.045])
        chisq_a, nonc_a = simulate_noncentral_chisq(propagated, n_eff=100_000, seed=9)
        chisq_b, nonc_b = simulate_noncentral_chisq(propagated, n_eff=100_000, seed=9)
        np.testing.assert_allclose(chisq_a, chisq_b)
        np.testing.assert_allclose(nonc_a, nonc_b)
        np.testing.assert_allclose(nonc_a, [10_500, 4_500])

    def test_sldsc_recovers_a_toy_infinitesimal_model(self):
        matrix = sp.identity(4, format="csr", dtype=np.float32)
        block = LdChromosomeBlock(chrom=1, rows=np.arange(4), R2=matrix)
        scores = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        n_eff = 1_000.0
        coefficients = np.array([0.10, 0.20, 0.03])
        chisq = 1.0 + n_eff * (scores @ coefficients[:2] + coefficients[2])
        result = fit_sldsc_wls([block], scores, ["first", "second"], chisq, n_eff)
        np.testing.assert_allclose(result.coefficients, coefficients, rtol=1e-7, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
