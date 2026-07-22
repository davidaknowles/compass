from pathlib import Path

import numpy as np
import pandas as pd

from compass.predicted_eqtl import (
    PREDICTED_EQTL_CELLS,
    load_predicted_eqtl_annotations,
    write_predicted_eqtl_annotations,
)


def _write_links(root: Path) -> None:
    partition = root / "chrom=1"
    partition.mkdir(parents=True)
    pd.DataFrame(
        {
            "chrom": np.array([1, 1, 1, 1], dtype=np.int16),
            "pos": [100, 100, 200, 200],
            "gene": ["ENSG1", "ENSG2", "ENSG1", "ENSG3"],
            "mechanism": ["astrocyte", "astrocyte", "microglia", "microglia"],
            "score": np.array([0.95, 0.91, 0.99, 0.89], dtype=np.float32),
        }
    ).to_parquet(partition / "links.parquet", index=False)


def test_loads_high_confidence_links_and_keeps_all_gwas_rows(tmp_path: Path) -> None:
    _write_links(tmp_path)
    gwas = pd.DataFrame(
        {"chrom": [1, 1, 1], "pos": [100, 200, 300], "snp": ["a", "b", "c"]}
    )
    annotation = load_predicted_eqtl_annotations(tmp_path, gwas, min_score=0.9)

    assert annotation.variants["pos"].tolist() == [100, 200, 300]
    assert annotation.mechanisms == ["intercept", *PREDICTED_EQTL_CELLS.values()]
    assert set(annotation.genes["gene"]) == {"ENSG1", "ENSG2"}
    context = annotation.triples[annotation.triples["mechanism_idx"].ne(0)]
    intercept = annotation.triples[annotation.triples["mechanism_idx"].eq(0)]
    assert context.shape[0] == 3
    assert intercept.shape[0] == 3
    assert not annotation.triples["variant_idx"].eq(2).any()


def test_writes_gene_sums_in_bim_order(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    _write_links(prepared)
    bim = pd.DataFrame(
        {
            "CHR": [1, 1, 1],
            "SNP": ["rs300", "rs100", "rs200"],
            "CM": [0.0, 0.0, 0.0],
            "BP": [300, 100, 200],
            "A1": ["A", "A", "A"],
            "A2": ["G", "G", "G"],
        }
    )
    output = tmp_path / "output"
    write_predicted_eqtl_annotations({1: bim}, prepared, output, min_score=0.9)
    result = pd.read_csv(output / "predicted_eqtl_baselineld.1.annot.gz", sep="\t")

    assert result["SNP"].tolist() == ["rs300", "rs100", "rs200"]
    np.testing.assert_allclose(result["PredEQTL_astrocyte"], [0.0, 1.86, 0.0])
    np.testing.assert_allclose(result["PredEQTL_microglia"], [0.0, 0.0, 0.99])
