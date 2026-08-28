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

export interface FeatureContribution {
  feature: string;
  value: number;
  impact: number;
}

export interface AttributePrediction {
  attribute_value: AttributeValue;
  predicted_units: number;
  predicted_units_low: number;
  predicted_units_high: number;
  historical_avg_units: number;
  rank: number;
  top_factors: FeatureContribution[];
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
  interval_coverage: number;
  interval_coverage_target: number;
  mean_interval_width: number;
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

export type DriftStatus = "stable" | "moderate_shift" | "significant_shift";

export interface FeatureDrift {
  feature: string;
  psi: number;
}

export interface DriftReportResponse {
  model_version: string;
  status: DriftStatus;
  max_psi: number;
  max_psi_feature: string;
  retrain_recommended: boolean;
  feature_psi: FeatureDrift[];
}

export type RetrainJobStatus = "running" | "completed" | "failed";

export interface RetrainStartResponse {
  status: "started";
  job_id: string;
}

export interface RetrainJobResult {
  promoted: boolean;
  promotion_reason: string;
  [key: string]: unknown;
}

export interface RetrainStatusResponse {
  status: RetrainJobStatus;
  job_id?: string;
  detail?: string;
  result?: RetrainJobResult;
  [key: string]: unknown;
}

/** FastAPI's default error envelope, e.g. `{"detail": "Category not found"}`. */
export interface ApiErrorBody {
  detail: string;
}
