"""Lightweight file-based model registry.

Trained models are saved as versioned artifacts (never overwritten in place)
alongside a JSON metadata index (registry.json) recording metrics, training
date, and data version for each kept version. Only the last N versions are
retained on disk; older ones are pruned. This is deliberately simple (no
external model registry service) since MLflow's tracking store already holds
the full experiment history — this registry is specifically what the serving
layer reads to find "the current production model artifact."
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ModelVersionMetadata:
    version: str  # e.g. "20240915T120000Z"
    model_path: str
    lookup_artifacts_path: str
    mlflow_run_id: str
    data_version: str
    trained_at: str
    metrics: dict
    params: dict
    is_active: bool = False


class ModelRegistry:
    def __init__(self, registry_dir: Path, metadata_file: Path, keep_last_n: int):
        self.registry_dir = Path(registry_dir)
        self.metadata_file = Path(metadata_file)
        self.keep_last_n = keep_last_n
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def _read_index(self) -> list[dict]:
        if not self.metadata_file.exists():
            return []
        with open(self.metadata_file) as f:
            return json.load(f)

    def _write_index(self, index: list[dict]) -> None:
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_file, "w") as f:
            json.dump(index, f, indent=2, default=str)

    def register(self, metadata: ModelVersionMetadata) -> None:
        """Add a new version to the index, mark it active, deactivate others,
        and prune old versions beyond keep_last_n."""
        index = self._read_index()
        for entry in index:
            entry["is_active"] = False

        metadata.is_active = True
        index.append(asdict(metadata))
        index.sort(key=lambda e: e["trained_at"])

        while len(index) > self.keep_last_n:
            stale = index.pop(0)
            self._delete_artifact(stale)

        self._write_index(index)

    def _delete_artifact(self, entry: dict) -> None:
        for key in ("model_path", "lookup_artifacts_path"):
            p = Path(entry[key])
            if p.exists():
                p.unlink()

    def get_active(self) -> dict | None:
        index = self._read_index()
        active = [e for e in index if e.get("is_active")]
        return active[-1] if active else None

    def list_versions(self) -> list[dict]:
        return sorted(self._read_index(), key=lambda e: e["trained_at"], reverse=True)
