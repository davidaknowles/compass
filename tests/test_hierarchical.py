from __future__ import annotations

import unittest

import numpy as np
import scipy.sparse as sp

from compass.model import (
    CompassDataset,
    LdChromosomeBlock,
    aggregate_context_annotations,
    fit_hierarchical_nuclear,
    fit_hierarchical_nuclear_path,
)


class HierarchicalModelTest(unittest.TestCase):
    def test_aggregate_context_annotations_supports_binary_and_sum(self):
        annotation = sp.csr_matrix(
            np.array(
                [
                    [0.2, 0.0, 0.3, 0.4],
                    [0.0, 0.5, 0.0, 0.7],
                ],
                dtype=np.float32,
            )
        )
        binary = aggregate_context_annotations(annotation, 2, 2, mode="binary")
        summed = aggregate_context_annotations(annotation, 2, 2, mode="sum")
        np.testing.assert_array_equal(binary.toarray(), [[1.0, 1.0], [0.0, 1.0]])
        np.testing.assert_allclose(summed.toarray(), [[0.5, 0.4], [0.0, 1.2]])

    def test_context_effects_are_not_shrunk_with_gene_deviations(self):
        n_variants = 12
        n_genes = 3
        n_mechanisms = 2
        dense_annotation = np.zeros((n_variants, n_genes * n_mechanisms), dtype=np.float32)
        for row in range(8):
            mechanism = row % n_mechanisms
            gene = row % n_genes
            dense_annotation[row, gene * n_mechanisms + mechanism] = 1.0
        annotation = sp.csr_matrix(dense_annotation)
        contexts = aggregate_context_annotations(annotation, n_genes, n_mechanisms, mode="binary")
        expected_context = np.array([0.02, 0.08], dtype=np.float32)
        expected_tau = 0.005
        n_samples = 1000.0
        chisq = 1.0 + n_samples * (contexts.toarray() @ expected_context + expected_tau)
        block = LdChromosomeBlock(
            chrom=1,
            rows=np.arange(n_variants),
            R2=sp.identity(n_variants, format="csr", dtype=np.float32),
        )
        dataset = CompassDataset(
            A=annotation,
            chisq=chisq.astype(np.float32),
            chrom=np.ones(n_variants, dtype=np.int64),
            n_samples=n_samples,
            ld_blocks=[block],
            sample_weight=np.ones(n_variants, dtype=np.float32),
            context_annotations=contexts,
        )
        deviations, context_effects, context_se, tau, _, _ = fit_hierarchical_nuclear(
            dataset,
            n_genes,
            n_mechanisms,
            lambda_value=1000.0,
            lr=1e-3,
            max_iter=40,
            tol=1e-8,
            objective_improve_tol=1e-12,
            device="cpu",
        )
        np.testing.assert_allclose(context_effects, expected_context, rtol=1e-4, atol=1e-6)
        self.assertAlmostEqual(tau, expected_tau, places=6)
        np.testing.assert_allclose(context_se, 0.0, atol=1e-8)
        np.testing.assert_allclose(deviations, 0.0, atol=1e-8)

        fixed = np.array([0.01, 0.09], dtype=np.float32)
        _, fixed_effects, fixed_se, _, _, metadata = fit_hierarchical_nuclear(
            dataset,
            n_genes,
            n_mechanisms,
            lambda_value=1000.0,
            fixed_context_effects=fixed,
            fixed_context_effect_se=np.array([0.001, 0.002], dtype=np.float32),
            lr=1e-3,
            max_iter=12,
            device="cpu",
        )
        np.testing.assert_array_equal(fixed_effects, fixed)
        np.testing.assert_allclose(fixed_se, [0.001, 0.002])
        self.assertTrue(metadata["context_effects_fixed"])

        profile = expected_context / 2.0
        _, scaled_effects, _, _, _, scaled_metadata = fit_hierarchical_nuclear(
            dataset,
            n_genes,
            n_mechanisms,
            lambda_value=1000.0,
            fixed_context_effects=profile,
            fixed_context_effect_se=np.zeros(2, dtype=np.float32),
            scale_fixed_context_effects=True,
            lr=1e-3,
            max_iter=12,
            device="cpu",
        )
        np.testing.assert_allclose(scaled_effects, expected_context, rtol=1e-4, atol=1e-6)
        self.assertTrue(scaled_metadata["context_effects_scaled"])

        dataset.cv_groups = np.arange(n_variants) % 2
        dataset.cv_score_groups = dataset.cv_groups.copy()
        path = fit_hierarchical_nuclear_path(
            dataset,
            n_genes,
            n_mechanisms,
            lambdas=[1000.0],
            fixed_context_effects=profile,
            fixed_context_effect_se=np.zeros(2, dtype=np.float32),
            scale_fixed_context_effects=True,
            max_lambda_extensions=0,
            lr=1e-3,
            max_iter=12,
            device="cpu",
        )
        np.testing.assert_allclose(path.context_effects, expected_context, rtol=1e-4, atol=1e-6)
        self.assertEqual(path.best_lambda, 1000.0)


if __name__ == "__main__":
    unittest.main()
