import type { ModelMetadataResponse } from "../types/api";
import { Card } from "./Card";

interface ModelPanelProps {
  metadata: ModelMetadataResponse;
}

function formatDateTime(isoDate: string): string {
  const parsed = new Date(isoDate);
  if (Number.isNaN(parsed.getTime())) return isoDate;
  return parsed.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatMetric(value: number, digits = 2): string {
  return value.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function ModelPanel({ metadata }: ModelPanelProps) {
  return (
    <Card title="Live model" subtitle={`Version ${metadata.version}`}>
      <dl className="model-meta">
        <div className="model-meta__row">
          <dt>Trained at</dt>
          <dd>{formatDateTime(metadata.trained_at)}</dd>
        </div>
        <div className="model-meta__row">
          <dt>Data version</dt>
          <dd>{metadata.data_version}</dd>
        </div>
      </dl>

      <div className="metric-row">
        <div className="metric-tile">
          <span className="metric-tile__label">MAE</span>
          <span className="metric-tile__value">
            {formatMetric(metadata.metrics.mae)}
          </span>
        </div>
        <div className="metric-tile">
          <span className="metric-tile__label">RMSE</span>
          <span className="metric-tile__value">
            {formatMetric(metadata.metrics.rmse)}
          </span>
        </div>
        <div className="metric-tile">
          <span className="metric-tile__label">NDCG@5</span>
          <span className="metric-tile__value">
            {formatMetric(metadata.metrics.ndcg_at_5, 3)}
          </span>
        </div>
      </div>
    </Card>
  );
}
