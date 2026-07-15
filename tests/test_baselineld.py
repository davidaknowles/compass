from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from compass.baselineld import ABC_COLUMN_NAMES, write_abc_annotations


class BaselineLdAnnotationTest(unittest.TestCase):
    def test_writes_abc_scores_in_bim_order(self):
        contexts = list(ABC_COLUMN_NAMES)
        bim = pd.DataFrame(
            {
                "CHR": [1, 1],
                "SNP": ["rs2", "rs1"],
                "CM": [0.0, 0.0],
                "BP": [200, 100],
                "A1": ["A", "A"],
                "A2": ["G", "G"],
            }
        )
        abc = pd.DataFrame(
            {
                "chr": ["chr1"] * len(contexts),
                "start": [100] * len(contexts),
                "end": [100] * len(contexts),
                "TargetGene": ["GENE1"] * len(contexts),
                "ABC.Score": [0.02 + index * 0.01 for index in range(len(contexts))],
                "CellType": contexts,
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            abc_path = tmp_path / "abc.tsv.gz"
            abc.to_csv(abc_path, sep="\t", index=False, compression="gzip")
            write_abc_annotations({1: bim}, abc_path, contexts, tmp_path / "out")
            result = pd.read_csv(tmp_path / "out" / "abc_baselineld.1.annot.gz", sep="\t")
        columns = [ABC_COLUMN_NAMES[context] for context in contexts]
        self.assertEqual(result["SNP"].tolist(), ["rs2", "rs1"])
        self.assertEqual(result.loc[0, columns].sum(), 0.0)
        self.assertAlmostEqual(result.loc[1, columns[0]], 0.02)
        self.assertAlmostEqual(result.loc[1, columns[-1]], 0.06)
