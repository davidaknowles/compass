"""Write official LDSC file formats from cached COMPASS annotations and LD."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


def write_ldsc_reference(
    output_dir: str | Path,
    ld_blocks,
    variants: pd.DataFrame,
    annotation_scores: np.ndarray,
    annotation_names: list[str],
) -> dict[str, str]:
    """Write chromosome-split reference, weights, and annotation files for LDSC.

    ``annotation_scores`` contains the gene-summed continuous ABC annotations.
    An all-SNP ``base`` annotation is prepended to represent non-ABC-mediated
    heritability. The supplied LD blocks are already the analysis reference
    (including its configured R2 threshold).
    """

    output_dir = Path(output_dir)
    reference_dir = output_dir / "reference"
    weights_dir = output_dir / "weights"
    reference_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    scores = np.asarray(annotation_scores, dtype=np.float32)
    required = {"CHR", "BP", "SNP"}
    missing = required.difference(variants.columns)
    if missing:
        raise ValueError(f"variants missing LDSC columns: {sorted(missing)}")
    if variants.shape[0] != scores.shape[0]:
        raise ValueError("variants and annotation_scores must have different row counts")
    if scores.shape[1] != len(annotation_names):
        raise ValueError("annotation_names must match annotation_scores columns")
    if variants["SNP"].duplicated().any():
        raise ValueError("LDSC SNP IDs must be unique")

    all_names = ["base"] + list(annotation_names)
    for block in ld_blocks:
        rows = np.asarray(block.rows, dtype=np.int64)
        matrix = block.R2.tocsr()
        if matrix.dtype == np.float16:
            matrix = sp.csr_matrix(
                (matrix.data.astype(np.float32), matrix.indices, matrix.indptr), shape=matrix.shape
            )
        annotation = np.column_stack((np.ones(rows.size, dtype=np.float32), scores[rows]))
        ld_scores = matrix @ annotation
        info = variants.iloc[rows][["CHR", "SNP", "BP"]].copy()
        if not np.all(info["CHR"].to_numpy(np.int64) == int(block.chrom)):
            raise ValueError(f"variant chromosome mismatch in block {block.chrom}")
        reference = pd.concat((info.reset_index(drop=True), pd.DataFrame(ld_scores, columns=all_names)), axis=1)
        annotation_frame = pd.concat((info.reset_index(drop=True), pd.DataFrame(annotation, columns=all_names)), axis=1)
        weights = pd.concat(
            (info.reset_index(drop=True), pd.DataFrame({"LD_weights": ld_scores[:, 0]})), axis=1
        )
        prefix = reference_dir / f"abc_ukbb.{int(block.chrom)}"
        reference.to_csv(f"{prefix}.l2.ldscore.gz", sep="\t", index=False, compression="gzip")
        annotation_frame.to_csv(f"{prefix}.annot.gz", sep="\t", index=False, compression="gzip")
        m_values = annotation.sum(axis=0)
        Path(f"{prefix}.l2.M").write_text(" ".join(f"{value:.10g}" for value in m_values) + "\n")
        Path(f"{prefix}.l2.M_5_50").write_text(" ".join(f"{value:.10g}" for value in m_values) + "\n")
        weight_prefix = weights_dir / f"weights_ukbb.{int(block.chrom)}"
        weights.to_csv(f"{weight_prefix}.l2.ldscore.gz", sep="\t", index=False, compression="gzip")
    return {
        "reference_prefix": str(reference_dir / "abc_ukbb."),
        "weight_prefix": str(weights_dir / "weights_ukbb."),
        "annotation_names": ",".join(all_names),
    }
