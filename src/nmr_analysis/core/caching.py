import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
from nmr_analysis.core.types import AnalysisResult, ExperimentType


class CacheManager:
    """
    Manages caching of AnalysisResult objects to disk.

    Cache files are stored in a hidden subdirectory `.nmr_cache` relative to the data file,
    or in a global cache directory if input is a directory.

    Format: .npz compressed archive containing:
        - arrays: fit_curve, residuals, etc.
        - metadata: JSON string containing params, errors, metadata, experiment_type
    """

    CACHE_DIR_NAME = ".nmr_cache"

    @staticmethod
    def _compute_file_hash(file_path: Path) -> str:
        """Compute a fast hash based on file metadata (size + mtime)."""
        stat = file_path.stat()
        # Mix size and mtime_ns for a quick unique identifier
        # We don't read full content for speed, assuming explicit user change updates mtime
        identifier = f"{stat.st_size}_{stat.st_mtime_ns}_{file_path.name}"
        return hashlib.sha256(identifier.encode("utf-8")).hexdigest()

    @staticmethod
    def compute_cache_key(target_path: Path, params: Dict[str, Any]) -> str:
        """
        Generate a unique cache filename based on input file and analysis parameters.

        Args:
            target_path: Path to the input file or directory.
            params: Dictionary of analysis parameters (e.g. channel, experiment, smoothing, etc.)
        """
        # 1. Input Hash
        if target_path.is_file():
            input_hash = CacheManager._compute_file_hash(target_path)
        else:
            # For directories, maybe hash the directory name + modification time?
            # Or simplified: just path name + recent mtime of content?
            # For simplicity for now, directory caching might be tricky if content changes.
            # But the 'montage' command calls _run_analysis on a directory which then processes files.
            # Usually _run_analysis is called on files or a directory.
            # If called on directory, it currently runs on files inside.
            # If we cache at _run_analysis level for a directory, it's complex.
            # We should probably cache at the lowest level (per file) OR if _run_analysis handles single targets.
            # Let's support directory by hashing its path string for now + mtime.
            stat = target_path.stat()
            input_hash = hashlib.sha256(
                f"{target_path.absolute()}_{stat.st_mtime_ns}".encode("utf-8")
            ).hexdigest()

        # 2. Params Hash
        # Sort keys to ensure consistent order
        params_str = json.dumps(params, sort_keys=True, default=str)
        params_hash = hashlib.sha256(params_str.encode("utf-8")).hexdigest()

        # Combine
        return f"{input_hash}_{params_hash}.npz"

    @staticmethod
    def get_cache_path(target_path: Path, cache_key: str) -> Path:
        """Get the absolute path to the cache file."""
        # Store cache in a sibling directory or subdirectory
        parent_dir = target_path.parent if target_path.is_file() else target_path
        cache_dir = parent_dir / CacheManager.CACHE_DIR_NAME
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / cache_key

    @staticmethod
    def get(
        target_path: Path, analysis_params: Dict[str, Any]
    ) -> Optional[AnalysisResult]:
        """
        Retrieve cached result if available.
        """
        key = CacheManager.compute_cache_key(target_path, analysis_params)
        cache_file = CacheManager.get_cache_path(target_path, key)

        if not cache_file.exists():
            return None

        try:
            with np.load(cache_file, allow_pickle=True) as data:
                # Load JSON metadata
                meta_json = str(data["metadata_json"])
                meta_dict = json.loads(meta_json)

                # Reconstruct AnalysisResult
                experiment_type = ExperimentType(meta_dict["experiment_type"])
                dataset_name = meta_dict["dataset_name"]
                params = meta_dict["params"]
                param_errors = meta_dict["param_errors"]
                metadata = meta_dict["metadata"]
                r_squared = float(meta_dict["r_squared"])

                fit_curve = data["fit_curve"]
                residuals = data["residuals"]

                return AnalysisResult(
                    experiment_type=experiment_type,
                    dataset_name=dataset_name,
                    params=params,
                    fit_curve=fit_curve,
                    residuals=residuals,
                    r_squared=r_squared,
                    param_errors=param_errors,
                    metadata=metadata,
                )
        except Exception:
            # If cache is corrupt or version mismatch, ignore it
            return None

    @staticmethod
    def set(target_path: Path, analysis_params: Dict[str, Any], result: AnalysisResult):
        """
        Save result to cache.
        """
        try:
            key = CacheManager.compute_cache_key(target_path, analysis_params)
            cache_file = CacheManager.get_cache_path(target_path, key)

            # Serialize metadata to JSON
            meta_dict = {
                "experiment_type": result.experiment_type.value,
                "dataset_name": result.dataset_name,
                "params": result.params,
                "param_errors": result.param_errors,
                "metadata": result.metadata,
                "r_squared": result.r_squared,
            }
            meta_json = json.dumps(meta_dict, default=str)

            # Save compressed
            np.savez_compressed(
                cache_file,
                fit_curve=result.fit_curve,
                residuals=result.residuals,
                metadata_json=meta_json,
            )
        except Exception:
            # Don't crash if caching fails (e.g. read-only fs)
            pass
