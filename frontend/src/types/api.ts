/**
 * Types mirroring the Forecast Lab backend API contract (FastAPI service).
 * Keep these in sync with the backend's response schemas.
 */

import type {
  AttributeType,
  AttributeValue,
  Category,
  Region,
} from "./domain";

export interface RegionsResponse {
  regions: Region[];
}

export interface CategoriesResponse {
  categories: Category[];
}

export interface AttributeTypesResponse {
  attribute_types: AttributeType[];
}

export interface AttributePrediction {
  attribute_value: AttributeValue;
  predicted_units: number;
  historical_avg_units: number;
  rank: number;
}

export interface AttributePredictionGroup {
  attribute_type: AttributeType;
  predictions: AttributePrediction[];
}

export interface PredictionsResponse {
  region: Region;
  category: Category;
  model_version: string;
  as_of_week: string;
  groups: AttributePredictionGroup[];
}

export interface ModelMetrics {
  mae: number;
  rmse: number;
  ndcg_at_5: number;
}

export interface ModelParams {
  [key: string]: number | string | boolean;
}

export interface ModelMetadataResponse {
  version: string;
  trained_at: string;
  data_version: string;
  metrics: ModelMetrics;
  params: ModelParams;
}

export type RetrainJobStatus = "running" | "completed" | "failed";

export interface RetrainStartResponse {
  status: "started";
  job_id: string;
}

export interface RetrainStatusResponse {
  status: RetrainJobStatus;
  job_id?: string;
  detail?: string;
  [key: string]: unknown;
}

/** FastAPI's default error envelope, e.g. `{"detail": "Category not found"}`. */
export interface ApiErrorBody {
  detail: string;
}
