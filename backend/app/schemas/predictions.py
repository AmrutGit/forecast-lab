from __future__ import annotations

from pydantic import BaseModel, Field


class FeatureContribution(BaseModel):
    feature: str
    value: float | str
    impact: float = Field(..., description="Signed SHAP value, in units of predicted units_sold")


class AttributePrediction(BaseModel):
    attribute_value: str
    predicted_units: float
    predicted_units_low: float = Field(..., description="p10 — 10th percentile of the prediction interval")
    predicted_units_high: float = Field(..., description="p90 — 90th percentile of the prediction interval")
    historical_avg_units: float
    rank: int
    top_factors: list[FeatureContribution] = Field(
        default_factory=list, description="Top SHAP feature contributions driving this prediction"
    )


class AttributeTypeGroup(BaseModel):
    attribute_type: str
    predictions: list[AttributePrediction]


class PredictionResponse(BaseModel):
    region: str
    category: str
    model_version: str
    as_of_week: str
    groups: list[AttributeTypeGroup]


class RegionsResponse(BaseModel):
    regions: list[str]


class CategoriesResponse(BaseModel):
    categories: list[str]


class AttributeTypesResponse(BaseModel):
    attribute_types: list[str]


class ModelMetadataResponse(BaseModel):
    version: str
    trained_at: str
    data_version: str
    metrics: dict[str, float]
    params: dict


class RetrainStartResponse(BaseModel):
    status: str
    job_id: str


class RetrainResult(ModelMetadataResponse):
    promoted: bool = Field(..., description="Whether this newly trained version beat the prior champion and went live")
    promotion_reason: str


class RetrainStatusResponse(BaseModel):
    job_id: str
    status: str
    detail: str | None = None
    result: RetrainResult | None = None


class FeatureDrift(BaseModel):
    feature: str
    psi: float


class DriftReportResponse(BaseModel):
    model_version: str
    status: str = Field(..., description="'stable' | 'moderate_shift' | 'significant_shift'")
    max_psi: float
    max_psi_feature: str
    retrain_recommended: bool
    feature_psi: list[FeatureDrift]


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Human-readable error message")
