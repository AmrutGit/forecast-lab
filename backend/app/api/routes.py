from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.app.schemas.predictions import (
    AttributePrediction,
    AttributeTypeGroup,
    AttributeTypesResponse,
    CategoriesResponse,
    DriftReportResponse,
    FeatureDrift,
    ModelMetadataResponse,
    PredictionResponse,
    RegionsResponse,
    RetrainResult,
    RetrainStartResponse,
    RetrainStatusResponse,
)
from backend.app.services.model_serving import ModelNotAvailableError, model_server
from backend.app.services.retraining import retraining_service
from ml.config.taxonomy import CATEGORIES, CATEGORY_ATTRIBUTES, REGIONS

router = APIRouter(prefix="/api")


@router.get("/regions", response_model=RegionsResponse)
def get_regions() -> RegionsResponse:
    return RegionsResponse(regions=REGIONS)


@router.get("/categories", response_model=CategoriesResponse)
def get_categories() -> CategoriesResponse:
    return CategoriesResponse(categories=CATEGORIES)


@router.get("/categories/{category}/attribute-types", response_model=AttributeTypesResponse)
def get_attribute_types(category: str) -> AttributeTypesResponse:
    if category not in CATEGORY_ATTRIBUTES:
        raise HTTPException(status_code=404, detail=f"Unknown category: {category!r}")
    return AttributeTypesResponse(attribute_types=list(CATEGORY_ATTRIBUTES[category].keys()))


@router.get("/predictions", response_model=PredictionResponse)
def get_predictions(
    region: str = Query(..., description="Region code, e.g. 'Region A'"),
    category: str = Query(..., description="Product category, e.g. 'Outerwear'"),
) -> PredictionResponse:
    if region not in REGIONS:
        raise HTTPException(status_code=404, detail=f"Unknown region: {region!r}")
    if category not in CATEGORY_ATTRIBUTES:
        raise HTTPException(status_code=404, detail=f"Unknown category: {category!r}")

    try:
        served_model = model_server.get_active()
    except ModelNotAvailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    groups = []
    for attribute_type in CATEGORY_ATTRIBUTES[category]:
        predictions = served_model.predict_group(region, category, attribute_type)
        groups.append(
            AttributeTypeGroup(
                attribute_type=attribute_type,
                predictions=[AttributePrediction(**p) for p in predictions],
            )
        )

    as_of_week = served_model.lookup_artifacts.as_of_week
    return PredictionResponse(
        region=region,
        category=category,
        model_version=served_model.version,
        as_of_week=str(as_of_week.date()),
        groups=groups,
    )


@router.get("/model/metadata", response_model=ModelMetadataResponse)
def get_model_metadata() -> ModelMetadataResponse:
    try:
        served_model = model_server.get_active()
    except ModelNotAvailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    meta = served_model.metadata
    return ModelMetadataResponse(
        version=meta["version"],
        trained_at=meta["trained_at"],
        data_version=meta["data_version"],
        metrics=meta["metrics"],
        params=meta["params"],
    )


@router.post("/retrain", response_model=RetrainStartResponse, status_code=202)
def trigger_retrain() -> RetrainStartResponse:
    job_id = retraining_service.start_job()
    return RetrainStartResponse(status="started", job_id=job_id)


@router.get("/retrain/status/{job_id}", response_model=RetrainStatusResponse)
def get_retrain_status(job_id: str) -> RetrainStatusResponse:
    job = retraining_service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown retrain job_id: {job_id!r}")

    result_model = RetrainResult(**job.result) if job.result else None
    return RetrainStatusResponse(job_id=job.job_id, status=job.status, detail=job.detail, result=result_model)


@router.get("/model/drift", response_model=DriftReportResponse)
def get_model_drift() -> DriftReportResponse:
    """Population Stability Index of key features vs. the training-time
    reference distribution — a signal for whether a retrain is warranted,
    independent of whether ground-truth labels for the recent period exist
    yet."""
    try:
        served_model = model_server.get_active()
    except ModelNotAvailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    report = served_model.drift_report()
    return DriftReportResponse(
        model_version=served_model.version,
        status=report.status,
        max_psi=report.max_psi,
        max_psi_feature=report.max_psi_feature,
        retrain_recommended=report.retrain_recommended,
        feature_psi=[FeatureDrift(feature=k, psi=v) for k, v in report.feature_psi.items()],
    )
