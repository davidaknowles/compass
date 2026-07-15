from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from compass.baselineld import ABC_COLUMN_NAMES, write_abc_annotations, write_continuous_abc_annotations, write_peak_annotations


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

    def test_writes_binary_peak_membership_with_bed_coordinates(self):
        bim = pd.DataFrame(
            {
                "CHR": [1, 1, 1, 1],
                "SNP": ["rs1", "rs2", "rs3", "rs4"],
                "CM": [0.0] * 4,
                "BP": [100, 101, 150, 201],
                "A1": ["A"] * 4,
                "A2": ["G"] * 4,
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            peaks_path = tmp_path / "peaks.bed"
            peaks_path.write_text("chr1\t99\t101\nchr1\t149\t201\n")
            write_peak_annotations({1: bim}, {"ATAC_PU1": peaks_path}, tmp_path / "out")
            result = pd.read_csv(tmp_path / "out" / "peaks_baselineld.1.annot.gz", sep="\t")
        self.assertEqual(result["ATAC_PU1"].tolist(), [1.0, 1.0, 1.0, 1.0])

    def test_writes_custom_context_columns(self):
        bim = pd.DataFrame(
            {
                "CHR": [1],
                "SNP": ["rs1"],
                "CM": [0.0],
                "BP": [100],
                "A1": ["A"],
                "A2": ["G"],
            }
        )
        abc = pd.DataFrame(
            {
                "chr": ["chr1"],
                "start": [100],
                "end": [100],
                "TargetGene": ["GENE1"],
                "ABC.Score": [0.02],
                "CellType": ["microglia"],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            abc_path = tmp_path / "abc.tsv.gz"
            abc.to_csv(abc_path, sep="\t", index=False, compression="gzip")
            write_continuous_abc_annotations(
                {1: bim},
                abc_path,
                ["microglia"],
                tmp_path / "out",
                column_names=["ABC_glass_microglia"],
                prefix="glass_abc_baselineld",
            )
            result = pd.read_csv(tmp_path / "out" / "glass_abc_baselineld.1.annot.gz", sep="\t")
        self.assertAlmostEqual(result.loc[0, "ABC_glass_microglia"], 0.02)
