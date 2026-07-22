import gzip
import stat
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from download_neuropsychiatric_gwas import extract_archive_member  # noqa: E402
from prepare_neuropsychiatric_gwas import normalize  # noqa: E402


class NeuropsychiatricGwasTest(unittest.TestCase):
    def test_pd_archive_extraction_and_liftover(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            member = "GP2_ALL_EUR_CLINICAL_ONLY_HG38_12162024.txt.gz"
            source = root / member
            with gzip.open(source, "wt") as stream:
                stream.write(
                    "chromosome base_pair_position SNP_ID effect_allele other_allele "
                    "effect_allele_frequency n_datasets p_value beta standard_error "
                    "p_value(random) beta(random) I\n"
                    "1 100 rs1 A G 0.2 10 0.1 0.25 0.05 0.1 0.25 0\n"
                )
            archive = root / "source.zip"
            with zipfile.ZipFile(archive, "w") as target:
                target.write(source, arcname=f"nested/{member}")
            source.unlink()
            extract_archive_member(archive, member, source)

            chain = root / "chain"
            chain.write_text("fixture\n")
            liftover = root / "liftOver"
            liftover.write_text(
                "#!/bin/sh\n"
                "awk 'BEGIN{OFS=\"\\t\"} {$2+=10;$3+=10;print}' \"$1\" > \"$3\"\n"
                ": > \"$4\"\n"
            )
            liftover.chmod(liftover.stat().st_mode | stat.S_IXUSR)

            output = root / "pd.hg19.tsv.gz"
            manifest = normalize("pd", source, output, chain, str(liftover))
            with gzip.open(output, "rt") as stream:
                rows = [line.rstrip("\n").split("\t") for line in stream]

            self.assertEqual(rows[1][:3], ["1", "110", "rs1"])
            expected_n = 4 * 34933 * 31009 / (34933 + 31009)
            self.assertAlmostEqual(float(rows[1][7]), expected_n, places=4)
            self.assertEqual(manifest["output_records"], 1)
            self.assertEqual(manifest["unmapped_records"], 0)


if __name__ == "__main__":
    unittest.main()
