from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from compass.data import cv_cache_path, load_cv_cache, write_cv_cache


def test_cv_cache_round_trip_and_shape_validation():
    with TemporaryDirectory() as temporary_directory:
        arrays = Path(temporary_directory) / "dataset.arrays.npz"
        path = cv_cache_path(arrays, n_folds=3, r2_threshold=0.01)
        groups = np.array([0, 1, 2, 0], dtype=np.int64)
        score_groups = np.array([0, -1, 2, 0], dtype=np.int64)
        metadata = {"cv_components": 7, "cv_fold_rows": [2, 1, 1]}

        write_cv_cache(path, groups, score_groups, metadata)
        loaded = load_cv_cache(path, n_variants=4)
        assert loaded is not None
        loaded_groups, loaded_score_groups, loaded_metadata = loaded
        np.testing.assert_array_equal(loaded_groups, groups)
        np.testing.assert_array_equal(loaded_score_groups, score_groups)
        assert loaded_metadata == metadata
        assert load_cv_cache(path, n_variants=5) is None
