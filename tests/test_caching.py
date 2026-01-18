import json
import tempfile
from pathlib import Path
import numpy as np
import pytest
from nmr_analysis.core.caching import CacheManager
from nmr_analysis.core.types import AnalysisResult, ExperimentType


def test_cache_key_generation():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"content")
        path = Path(f.name)

    try:
        params1 = {"a": 1, "b": "test"}
        params2 = {"a": 1, "b": "test"}
        params3 = {"a": 2}

        # Test consistency
        key1 = CacheManager.compute_cache_key(path, params1)
        key2 = CacheManager.compute_cache_key(path, params2)
        assert key1 == key2

        # Test differentiation
        key3 = CacheManager.compute_cache_key(path, params3)
        assert key1 != key3

        # Test differentiation by file
        with tempfile.NamedTemporaryFile(delete=False) as f2:
            f2.write(b"different content")
            path2 = Path(f2.name)
        try:
            key4 = CacheManager.compute_cache_key(path2, params1)
            # Keys might collide if only hashing path string, but here we hash size+mtime
            # Writing new file -> new mtime/size usually.
            assert key1 != key4
        finally:
            path2.unlink()

    finally:
        path.unlink()


def test_cache_save_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        target_file = tmp_path / "data.txt"
        target_file.write_text("dummy data")

        params = {"experiment": "test"}

        # Create Dummy Result
        result = AnalysisResult(
            experiment_type=ExperimentType.T2,
            dataset_name="Test Result",
            params={"T2": 0.5, "M0": 100.0},
            fit_curve=np.array([1.0, 0.5, 0.2]),
            residuals=np.array([0.01, -0.01, 0.0]),
            r_squared=0.99,
            param_errors={"T2": 0.01},
            metadata={"info": "cached"},
        )

        # Set Cache
        CacheManager.set(target_file, params, result)

        # Check cache file exists
        key = CacheManager.compute_cache_key(target_file, params)
        cache_file = target_file.parent / CacheManager.CACHE_DIR_NAME / key
        assert cache_file.exists()

        # Get Cache
        loaded_result = CacheManager.get(target_file, params)
        assert loaded_result is not None
        assert loaded_result.dataset_name == "Test Result"
        assert loaded_result.params["T2"] == 0.5
        np.testing.assert_array_equal(loaded_result.fit_curve, result.fit_curve)
        assert loaded_result.metadata["info"] == "cached"

        # Test Miss
        params_diff = {"experiment": "other"}
        assert CacheManager.get(target_file, params_diff) is None
