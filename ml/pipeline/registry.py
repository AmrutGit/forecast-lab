"""Lightweight file-based model registry with champion/challenger promotion.

Trained models are saved as versioned artifacts (never overwritten in place)
alongside a JSON metadata index (registry.json) recording metrics, training
date, and data version for each kept version. Only the last N versions are
retained on disk; older ones are pruned. This is deliberately simple (no
external model registry service) since MLflow's tracking store already holds
the full experiment history — this registry is specifically what the serving
layer reads to find "the current production model artifact."

Promotion policy: a freshly trained model ("challenger") is registered for
every version, but it only becomes the *active* ("champion") model if it
beats the current champion on the comparison metric (MAE by default, lower
is better) — evaluated on the same held-out test set construction, so the
comparison is apples-to-apples. This prevents a retrain from silently
regressing production quality just because it's the most recent run; a
challenger that loses stays registered (visible in `list_versions`) but
never serves traffic.
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


@dataclass
class PromotionResult:
    promoted: bool
    reason: str
    challenger_version: str
    challenger_metric: float
    champion_version: str | None
    champion_metric: float | None


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
        """Add a new version to the index as an inactive challenger, and
        prune old versions beyond keep_last_n (oldest first, active version
        is never pruned regardless of age)."""
        index = self._read_index()
        index.append(asdict(metadata))
        index.sort(key=lambda e: e["trained_at"])

        while len(index) > self.keep_last_n:
            candidates = [e for e in index if not e.get("is_active")]
            if not candidates:
                break
            stale = min(candidates, key=lambda e: e["trained_at"])
            index.remove(stale)
            self._delete_artifact(stale)

        self._write_index(index)

    def promote_if_better(
        self, challenger_version: str, comparison_metric: str = "mae", lower_is_better: bool = True
    ) -> PromotionResult:
        """Compare the named challenger version against the current active
        champion on `comparison_metric` (both evaluated on the same held-out
        test set by the caller) and promote it only if it wins. If there is
        no current champion, the challenger is promoted unconditionally
        (bootstrapping the very first version)."""
        index = self._read_index()
        challenger = next((e for e in index if e["version"] == challenger_version), None)
        if challenger is None:
            raise ValueError(f"Unknown challenger version: {challenger_version!r}")

        challenger_metric = challenger["metrics"][comparison_metric]
        champion = next((e for e in index if e.get("is_active")), None)

        if champion is None:
            promoted, reason = True, "no existing champion — bootstrapping first active version"
        else:
            champion_metric = champion["metrics"][comparison_metric]
            better = challenger_metric < champion_metric if lower_is_better else challenger_metric > champion_metric
            promoted = better
            reason = (
                f"challenger {comparison_metric}={challenger_metric:.4f} "
                f"{'beats' if better else 'does not beat'} "
                f"champion {comparison_metric}={champion_metric:.4f}"
            )

        if promoted:
            for entry in index:
                entry["is_active"] = entry["version"] == challenger_version
            self._write_index(index)

        return PromotionResult(
            promoted=promoted,
            reason=reason,
            challenger_version=challenger_version,
            challenger_metric=challenger_metric,
            champion_version=champion["version"] if champion else None,
            champion_metric=champion["metrics"][comparison_metric] if champion else None,
        )

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
