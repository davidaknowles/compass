from __future__ import annotations

import gzip
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from compass.glass_abc import prepare_glass_abc_v2


class GlassAbcPreparationTest(unittest.TestCase):
    def test_preserves_compact_column_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            prediction = source / "ABC_results_microglia_v2" / "microglia" / "Predictions"
            prediction.mkdir(parents=True)
            raw = prediction / "EnhancerPredictionsAllPutative.ForVariantOverlap.shrunk150bp.tsv.gz"
            frame = pd.DataFrame(
                {
                    "chr": ["chr1"],
                    "start": [99],
                    "end": [100],
                    "name": ["promoter|chr1:99-100"],
                    "class": ["promoter"],
                    "TargetGene": ["GENE1"],
                    "isSelfPromoter": [True],
                    "ABC.Score": [1.0],
                    "CellType": ["source_label"],
                }
            )
            frame.to_csv(raw, sep="\t", index=False, compression="gzip")
            chain = root / "identity.chain"
            chain.write_text("test\n")
            liftover = root / "fake_liftover.sh"
            liftover.write_text("#!/bin/sh\ncp \"$3\" \"$5\"\n: > \"$6\"\n")
            liftover.chmod(0o755)
            output = root / "glass.tsv.gz"
            prepare_glass_abc_v2(source, output, chain, str(liftover), ("microglia",))
            with gzip.open(output, "rt") as handle:
                result = pd.read_csv(handle, sep="\t")
        self.assertEqual(
            result.columns.tolist(),
            ["chr", "start", "end", "TargetGene", "ABC.Score", "CellType", "class", "isSelfPromoter"],
        )
        self.assertEqual(result.loc[0, "TargetGene"], "GENE1")
        self.assertEqual(result.loc[0, "CellType"], "microglia")
        self.assertEqual(result.loc[0, "class"], "promoter")
        self.assertTrue(result.loc[0, "isSelfPromoter"])
