/**
 * One function per backend endpoint. All requests go through `apiRequest`,
 * which normalizes errors into `ApiError` and applies the configured base URL.
 */

import { apiRequest } from "./client";
import type {
  AttributeTypesResponse,
  CategoriesResponse,
  DriftReportResponse,
  ModelMetadataResponse,
  PredictionsResponse,
  RegionsResponse,
  RetrainStartResponse,
  RetrainStatusResponse,
} from "../types/api";

export function getRegions(signal?: AbortSignal): Promise<RegionsResponse> {
  return apiRequest<RegionsResponse>("/api/regions", { signal });
}

export function getCategories(signal?: AbortSignal): Promise<CategoriesResponse> {
  return apiRequest<CategoriesResponse>("/api/categories", { signal });
}

export function getAttributeTypes(
  category: string,
  signal?: AbortSignal,
): Promise<AttributeTypesResponse> {
  return apiRequest<AttributeTypesResponse>(
    `/api/categories/${encodeURIComponent(category)}/attribute-types`,
    { signal },
  );
}

export function getPredictions(
  region: string,
  category: string,
  signal?: AbortSignal,
): Promise<PredictionsResponse> {
  return apiRequest<PredictionsResponse>("/api/predictions", {
    query: { region, category },
    signal,
  });
}

export function getModelMetadata(
  signal?: AbortSignal,
): Promise<ModelMetadataResponse> {
  return apiRequest<ModelMetadataResponse>("/api/model/metadata", { signal });
}

export function getModelDrift(signal?: AbortSignal): Promise<DriftReportResponse> {
  return apiRequest<DriftReportResponse>("/api/model/drift", { signal });
}

export function startRetrain(
  signal?: AbortSignal,
): Promise<RetrainStartResponse> {
  return apiRequest<RetrainStartResponse>("/api/retrain", {
    method: "POST",
    signal,
  });
}

export function getRetrainStatus(
  jobId: string,
  signal?: AbortSignal,
): Promise<RetrainStatusResponse> {
  return apiRequest<RetrainStatusResponse>(
    `/api/retrain/status/${encodeURIComponent(jobId)}`,
    { signal },
  );
}
