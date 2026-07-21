from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import scipy.sparse as sp
import torch

from compass.model import (
    CompassDataset,
    LdChromosomeBlock,
    aggregate_context_annotations,
    context_heritability_components,
    fit_hierarchical_nuclear,
    fit_hierarchical_nuclear_path,
    gene_deviation_heritability_from_masses,
    nuclear_prox,
    nuclear_prox_lower_bound,
)


class HierarchicalModelTest(unittest.TestCase):
    def test_shifted_nuclear_prox_enforces_total_nonnegativity(self):
        value = torch.tensor(
            [[0.8, -0.7], [-0.5, 0.4], [0.2, -0.3]], dtype=torch.float32
        )
        lower = torch.tensor([[-0.2, -0.1]], dtype=torch.float32)
        threshold = 0.35
        result = nuclear_prox_lower_bound(value, threshold, lower, max_iter=200, tol=1e-8)
        naive = torch.maximum(nuclear_prox(value, threshold), lower)

        self.assertTrue(torch.all(result >= lower - 1e-6))
        self.assertTrue(torch.any(result < 0))

        def objective(candidate):
            return 0.5 * torch.sum((candidate - value) ** 2) + threshold * torch.linalg.matrix_norm(
                candidate, ord="nuc"
            )

        self.assertLessEqual(float(objective(result)), float(objective(naive)) + 1e-5)

    def test_gene_deviation_heritability_from_masses(self):
        mass = np.array([[2.0, 3.0], [4.0, 5.0]])
        coefficients = np.array([[0.1, 0.2], [0.3, 0.4]])
        np.testing.assert_allclose(
            gene_deviation_heritability_from_masses(mass, coefficients),
            [[0.2, 0.6], [1.2, 2.0]],
        )
        with self.assertRaisesRegex(ValueError, "matching gene-by-context"):
            gene_deviation_heritability_from_masses(mass[:, :1], coefficients)

    def test_context_heritability_components_separates_global_and_deviation_terms(self):
        annotation = sp.csr_matrix(
            np.array(
                [
                    [1.0, 0.0, 2.0, 1.0],
                    [0.0, 3.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            )
        )
        contexts = sp.csr_matrix(np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float32))
        B = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        summary = context_heritability_components(
            annotation, B, contexts, np.array([0.5, 0.25], dtype=np.float32)
        )
        np.testing.assert_allclose(summary["global_h2"], [0.5, 0.5])
        np.testing.assert_allclose(summary["deviation_h2"], [1.0, 1.0])
        np.testing.assert_allclose(summary["total_h2"], [1.5, 1.5])
        np.testing.assert_allclose(summary["fraction"], [0.5, 0.5])

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

    def test_signed_deviations_recover_downweights_with_nonnegative_totals(self):
        n_genes = 2
        n_contexts = 2
        annotation = sp.vstack(
            [
                sp.eye(4, format="csr", dtype=np.float32),
                sp.csr_matrix((4, 4), dtype=np.float32),
            ],
            format="csr",
        )
        contexts = aggregate_context_annotations(annotation, n_genes, n_contexts, mode="sum")
        context_effects = np.array([0.08, 0.06], dtype=np.float32)
        expected_deviations = np.array([[-0.03, 0.02], [0.01, -0.02]], dtype=np.float32)
        n_samples = 100.0
        chisq = 1.0 + n_samples * (
            annotation @ (context_effects[None, :] + expected_deviations).ravel()
        )
        block = LdChromosomeBlock(
            chrom=1,
            rows=np.arange(8),
            R2=sp.eye(8, format="csr", dtype=np.float32),
        )
        dataset = CompassDataset(
            A=annotation,
            chisq=np.asarray(chisq, dtype=np.float32),
            chrom=np.ones(8, dtype=np.int64),
            n_samples=n_samples,
            ld_blocks=[block],
            sample_weight=np.ones(8, dtype=np.float32),
            context_annotations=contexts,
        )
        deviations, fitted_contexts, _, tau, _, metadata = fit_hierarchical_nuclear(
            dataset,
            n_genes,
            n_contexts,
            lambda_value=0.0,
            fixed_context_effects=context_effects,
            init_tau=0.0,
            lr=1e-3,
            max_iter=100,
            tol=1e-8,
            objective_relative_tol=1e-10,
            objective_window=20,
            device="cpu",
            deviation_constraint="total_nonnegative",
        )
        np.testing.assert_allclose(deviations, expected_deviations, atol=1e-6)
        np.testing.assert_allclose(fitted_contexts, context_effects)
        self.assertAlmostEqual(tau, 0.0, places=7)
        self.assertTrue(np.all(fitted_contexts[None, :] + deviations >= -1e-7))
        self.assertEqual(metadata["deviation_constraint"], "total_nonnegative")

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

        _, _, _, _, adaptive_losses, adaptive_metadata = fit_hierarchical_nuclear(
            dataset,
            n_genes,
            n_mechanisms,
            lambda_value=0.01,
            lr=0.1,
            max_iter=20,
            step_backtrack_patience=1,
            device="cpu",
        )
        self.assertGreaterEqual(adaptive_metadata["step_backtracks"], 1)
        self.assertTrue(np.all(np.diff(adaptive_losses) <= 1e-5))

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
        np.testing.assert_allclose(
            scaled_metadata["initial_context_effects"], profile, rtol=0, atol=0
        )

        warm_context = np.array([0.5, 1.0], dtype=np.float32)
        _, _, _, _, _, warm_metadata = fit_hierarchical_nuclear(
            dataset,
            n_genes,
            n_mechanisms,
            lambda_value=1_000.0,
            init_context_effects=warm_context,
            fixed_context_effects=profile,
            fixed_context_effect_se=np.zeros(2, dtype=np.float32),
            scale_fixed_context_effects=True,
            max_iter=12,
            lr=1e-2,
            tol=1e-8,
        )
        np.testing.assert_allclose(
            warm_metadata["initial_context_effects"], warm_context, rtol=0, atol=0
        )

        _, _, _, _, objective_losses, objective_metadata = fit_hierarchical_nuclear(
            dataset,
            n_genes,
            n_mechanisms,
            lambda_value=1_000.0,
            fixed_context_effects=profile,
            scale_fixed_context_effects=True,
            max_iter=40,
            tol=0.0,
            objective_relative_tol=1.0,
            objective_window=2,
            lr=1e-3,
        )
        self.assertLess(len(objective_losses), 40)
        self.assertEqual(objective_metadata["convergence_reason"], "relative_objective")

        frozen_path = fit_hierarchical_nuclear_path(
            dataset,
            n_genes,
            n_mechanisms,
            lambdas=[1_000.0, 0.1],
            cv=False,
            fixed_context_effects=profile,
            fixed_context_effect_se=np.zeros(2, dtype=np.float32),
            scale_fixed_context_effects=True,
            freeze_scaled_context_effects=True,
            lr=1e-3,
            max_iter=12,
            device="cpu",
        )
        np.testing.assert_allclose(
            frozen_path.context_effects, expected_context, rtol=1e-4, atol=1e-6
        )
        self.assertTrue(frozen_path.metadata["freeze_scaled_context_effects"])
        self.assertTrue(frozen_path.metadata[0.1]["context_effects_fixed"])

        dataset.cv_groups = np.arange(n_variants) % 2
        dataset.cv_score_groups = dataset.cv_groups.copy()
        with TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "hierarchical_cv.npz"
            path = fit_hierarchical_nuclear_path(
                dataset,
                n_genes,
                n_mechanisms,
                lambdas=[1000.0],
                fixed_context_effects=profile,
                fixed_context_effect_se=np.zeros(2, dtype=np.float32),
                scale_fixed_context_effects=True,
                freeze_scaled_context_effects=True,
                cv_checkpoint_path=checkpoint,
                max_lambda_extensions=0,
                lr=1e-3,
                max_iter=12,
                device="cpu",
            )
            np.testing.assert_allclose(path.context_effects, expected_context, rtol=1e-4, atol=1e-6)
            self.assertEqual(path.best_lambda, 1000.0)
            with np.load(checkpoint) as saved:
                np.testing.assert_array_equal(saved["next_lambda_index"], [1, 1])
                self.assertTrue(np.isfinite(saved["scores"]).all())
                self.assertTrue(saved["freeze_scaled_context_effects"].item())

            resumed = fit_hierarchical_nuclear_path(
                dataset,
                n_genes,
                n_mechanisms,
                lambdas=[1000.0],
                fixed_context_effects=profile,
                fixed_context_effect_se=np.zeros(2, dtype=np.float32),
                scale_fixed_context_effects=True,
                freeze_scaled_context_effects=True,
                cv_checkpoint_path=checkpoint,
                max_lambda_extensions=0,
                lr=1e-3,
                max_iter=12,
                device="cpu",
            )
            self.assertEqual(resumed.cv_scores, path.cv_scores)

            subset_checkpoint = Path(temporary_directory) / "hierarchical_cv_fold1.npz"
            subset = fit_hierarchical_nuclear_path(
                dataset,
                n_genes,
                n_mechanisms,
                lambdas=[1000.0],
                fixed_context_effects=profile,
                fixed_context_effect_se=np.zeros(2, dtype=np.float32),
                scale_fixed_context_effects=True,
                freeze_scaled_context_effects=True,
                cv_checkpoint_path=subset_checkpoint,
                cv_fold_subset=[1],
                max_lambda_extensions=0,
                lr=1e-3,
                max_iter=12,
                device="cpu",
            )
            self.assertEqual(subset.metadata["cv_folds"], [1])
            self.assertEqual(subset.metadata["cv_fold_subset"], [1])
            with np.load(subset_checkpoint) as saved:
                np.testing.assert_array_equal(saved["folds"], [1])
                np.testing.assert_array_equal(saved["next_lambda_index"], [1])
                self.assertTrue(np.isfinite(saved["scores"]).all())

            with self.assertRaisesRegex(ValueError, "unavailable folds"):
                fit_hierarchical_nuclear_path(
                    dataset,
                    n_genes,
                    n_mechanisms,
                    lambdas=[1000.0],
                    cv_fold_subset=[2],
                    max_lambda_extensions=0,
                    max_iter=1,
                    device="cpu",
                )
            with self.assertRaisesRegex(ValueError, "checkpoint is incompatible"):
                fit_hierarchical_nuclear_path(
                    dataset,
                    n_genes,
                    n_mechanisms,
                    lambdas=[1000.0],
                    fixed_context_effects=profile[::-1].copy(),
                    scale_fixed_context_effects=True,
                    freeze_scaled_context_effects=True,
                    cv_checkpoint_path=checkpoint,
                    max_lambda_extensions=0,
                    max_iter=12,
                    device="cpu",
                )


if __name__ == "__main__":
    unittest.main()
