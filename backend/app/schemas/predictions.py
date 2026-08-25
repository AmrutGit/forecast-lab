from __future__ import annotations

from pydantic import BaseModel, Field


class AttributePrediction(BaseModel):
    attribute_value: str
    predicted_units: float
    historical_avg_units: float
    rank: int


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


class RetrainStatusResponse(BaseModel):
    job_id: str
    status: str
    detail: str | None = None
    result: ModelMetadataResponse | None = None


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Human-readable error message")
