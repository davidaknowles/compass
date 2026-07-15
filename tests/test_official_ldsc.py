from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from compass.model import LdChromosomeBlock
from compass.official_ldsc import write_ldsc_reference


class OfficialLdscWriterTest(unittest.TestCase):
    def test_writes_reference_weight_and_annotation_files(self):
        block = LdChromosomeBlock(
            chrom=1,
            rows=np.array([0, 1]),
            R2=sp.csr_matrix(np.array([[1.0, 0.25], [0.25, 1.0]], dtype=np.float32)),
        )
        variants = pd.DataFrame({"CHR": [1, 1], "BP": [10, 20], "SNP": ["rs1", "rs2"]})
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_ldsc_reference(tmp, [block], variants, np.array([[2.0], [3.0]]), ["abc"])
            reference = pd.read_csv(Path(paths["reference_prefix"] + "1.l2.ldscore.gz"), sep="\t")
            annotation = pd.read_csv(Path(paths["reference_prefix"] + "1.annot.gz"), sep="\t")
            weights = pd.read_csv(Path(paths["weight_prefix"] + "1.l2.ldscore.gz"), sep="\t")
            self.assertEqual(reference.columns.tolist(), ["CHR", "SNP", "BP", "base", "abc"])
            np.testing.assert_allclose(reference["abc"], [2.75, 3.5])
            np.testing.assert_allclose(annotation["abc"], [2.0, 3.0])
            np.testing.assert_allclose(weights["LD_weights"], [1.25, 1.25])
