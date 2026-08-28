from __future__ import annotations

from ml.config.taxonomy import CATEGORIES, CATEGORY_ATTRIBUTES, REGIONS


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_get_regions(client):
    resp = client.get("/api/regions")
    assert resp.status_code == 200
    assert resp.json()["regions"] == REGIONS


def test_get_categories(client):
    resp = client.get("/api/categories")
    assert resp.status_code == 200
    assert resp.json()["categories"] == CATEGORIES


def test_get_attribute_types_known_category(client):
    resp = client.get("/api/categories/Outerwear/attribute-types")
    assert resp.status_code == 200
    assert set(resp.json()["attribute_types"]) == set(CATEGORY_ATTRIBUTES["Outerwear"].keys())


def test_get_attribute_types_unknown_category(client):
    resp = client.get("/api/categories/NotACategory/attribute-types")
    assert resp.status_code == 404
    assert "NotACategory" in resp.json()["detail"]


def test_get_predictions_valid_region_and_category(client):
    resp = client.get("/api/predictions", params={"region": "Region A", "category": "Outerwear"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["region"] == "Region A"
    assert body["category"] == "Outerwear"
    assert body["model_version"]
    assert len(body["groups"]) == len(CATEGORY_ATTRIBUTES["Outerwear"])

    for group in body["groups"]:
        ranks = [p["rank"] for p in group["predictions"]]
        assert ranks == sorted(ranks)
        preds = [p["predicted_units"] for p in group["predictions"]]
        assert preds == sorted(preds, reverse=True)

        for prediction in group["predictions"]:
            # p10 <= p50 <= p90 should hold for the vast majority of
            # predictions from independently-trained quantile models; we
            # don't assert it as a hard invariant here (that's covered more
            # rigorously in ml/tests), just that the fields are present and
            # sane in shape.
            assert prediction["predicted_units_low"] <= prediction["predicted_units_high"]
            assert isinstance(prediction["top_factors"], list)
            assert len(prediction["top_factors"]) > 0
            for factor in prediction["top_factors"]:
                assert "feature" in factor
                assert "impact" in factor


def test_get_predictions_unknown_region(client):
    resp = client.get("/api/predictions", params={"region": "Nowhere", "category": "Outerwear"})
    assert resp.status_code == 404


def test_get_predictions_unknown_category(client):
    resp = client.get("/api/predictions", params={"region": "Region A", "category": "NotACategory"})
    assert resp.status_code == 404


def test_get_predictions_missing_query_param(client):
    resp = client.get("/api/predictions", params={"region": "Region A"})
    assert resp.status_code == 422


def test_get_model_metadata(client):
    resp = client.get("/api/model/metadata")
    assert resp.status_code == 200
    body = resp.json()
    assert "version" in body
    assert set(body["metrics"].keys()) >= {"mae", "rmse", "ndcg_at_5", "interval_coverage"}


def test_get_model_drift(client):
    resp = client.get("/api/model/drift")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in {"stable", "moderate_shift", "significant_shift"}
    assert isinstance(body["retrain_recommended"], bool)
    assert len(body["feature_psi"]) > 0
    for entry in body["feature_psi"]:
        assert entry["psi"] >= 0


def test_retrain_status_unknown_job(client):
    resp = client.get("/api/retrain/status/not-a-real-job-id")
    assert resp.status_code == 404


def test_openapi_schema_is_served(client):
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    paths = resp.json()["paths"]
    assert "/api/predictions" in paths
    assert "/api/model/metadata" in paths
    assert "/api/model/drift" in paths
