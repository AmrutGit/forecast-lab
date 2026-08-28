from __future__ import annotations

import pytest

from ml.pipeline.registry import ModelRegistry, ModelVersionMetadata


def _metadata(version: str, mae: float) -> ModelVersionMetadata:
    return ModelVersionMetadata(
        version=version,
        model_path=f"/tmp/model_{version}.joblib",
        lookup_artifacts_path=f"/tmp/lookup_{version}.joblib",
        mlflow_run_id=f"run_{version}",
        data_version="abc123",
        trained_at=f"2024-01-{version}T00:00:00Z",
        metrics={"mae": mae, "rmse": mae * 1.3},
        params={"n_estimators": 100},
    )


@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(
        registry_dir=tmp_path / "models",
        metadata_file=tmp_path / "models" / "registry.json",
        keep_last_n=5,
    )


def test_first_version_is_promoted_unconditionally(registry):
    registry.register(_metadata("01", mae=10.0))
    result = registry.promote_if_better("01")

    assert result.promoted is True
    assert result.champion_version is None
    assert registry.get_active()["version"] == "01"


def test_challenger_with_worse_mae_is_not_promoted(registry):
    registry.register(_metadata("01", mae=10.0))
    registry.promote_if_better("01")

    registry.register(_metadata("02", mae=15.0))  # worse MAE
    result = registry.promote_if_better("02")

    assert result.promoted is False
    assert registry.get_active()["version"] == "01"


def test_challenger_with_better_mae_is_promoted(registry):
    registry.register(_metadata("01", mae=10.0))
    registry.promote_if_better("01")

    registry.register(_metadata("02", mae=7.5))  # better MAE
    result = registry.promote_if_better("02")

    assert result.promoted is True
    assert registry.get_active()["version"] == "02"


def test_non_promoted_challenger_remains_listed_but_inactive(registry):
    registry.register(_metadata("01", mae=10.0))
    registry.promote_if_better("01")

    registry.register(_metadata("02", mae=15.0))
    registry.promote_if_better("02")

    versions = {v["version"]: v["is_active"] for v in registry.list_versions()}
    assert versions == {"01": True, "02": False}


def test_promote_unknown_version_raises(registry):
    registry.register(_metadata("01", mae=10.0))
    with pytest.raises(ValueError, match="Unknown challenger version"):
        registry.promote_if_better("does-not-exist")


def test_active_version_is_never_pruned_even_if_oldest(tmp_path):
    registry = ModelRegistry(
        registry_dir=tmp_path / "models", metadata_file=tmp_path / "models" / "registry.json", keep_last_n=2
    )
    registry.register(_metadata("01", mae=10.0))
    registry.promote_if_better("01")  # 01 becomes champion, is oldest

    registry.register(_metadata("02", mae=15.0))
    registry.promote_if_better("02")  # loses, 01 remains champion

    registry.register(_metadata("03", mae=20.0))
    registry.promote_if_better("03")  # loses, 01 remains champion — pruning kicks in (keep_last_n=2)

    versions = [v["version"] for v in registry.list_versions()]
    assert "01" in versions  # active version survives pruning despite being oldest
    assert registry.get_active()["version"] == "01"
