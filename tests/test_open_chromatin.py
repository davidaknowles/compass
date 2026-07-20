from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from compass.data import load_open_chromatin_tss_annotations, load_peak_context_annotations


class OpenChromatinTssAnnotationTest(unittest.TestCase):
    def test_flat_peak_context_annotations_include_union_nuisance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            astrocyte = root / "astrocyte.bed"
            microglia = root / "microglia.bed"
            astrocyte.write_text("chr1\t90\t110\n")
            microglia.write_text("chr1\t190\t210\n")
            matrix = load_peak_context_annotations(
                {"astrocyte": astrocyte, "microglia": microglia},
                np.array([1, 1, 1]),
                np.array([100, 200, 300]),
                ["intercept", "astrocyte", "microglia"],
            )
        np.testing.assert_array_equal(
            matrix.toarray(),
            [[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        )

    def test_links_peak_variants_to_expressed_nearby_tss(self):
        gwas = pd.DataFrame(
            {
                "chrom": [1, 1, 1],
                "pos": [100, 150, 300_000],
                "snp": ["rs1", "rs2", "rs3"],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            peaks = tmp_path / "atac.bed"
            peaks.write_text("chr1\t99\t151\n")
            tss = tmp_path / "tss.tsv.gz"
            pd.DataFrame(
                {
                    "chrom": ["chr1", "chr1", "chr1"],
                    "tss": [200_151, 100, 100],
                    "gene": ["FAR", "EXPRESSED", "OTHER_CONTEXT"],
                    "CellType": ["microglia", "microglia", "astrocyte"],
                }
            ).to_csv(tss, sep="\t", index=False, compression="gzip")
            annotation = load_open_chromatin_tss_annotations(
                {"microglia": peaks},
                tss,
                gwas,
                tss_window=100_000,
            )
        self.assertEqual(annotation.variants.shape[0], 3)
        self.assertEqual(annotation.mechanisms, ["intercept", "microglia"])
        self.assertEqual(annotation.genes["gene"].tolist(), ["EXPRESSED", "FAR"])
        linked = annotation.triples[annotation.triples["mechanism_idx"].eq(1)]
        self.assertEqual(linked["variant_idx"].tolist(), [0, 1])
        self.assertEqual(linked["gene_idx"].tolist(), [0, 0])
