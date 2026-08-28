import type { DriftReportResponse, DriftStatus, ModelMetadataResponse } from "../types/api";
import type { AsyncState } from "../api/useApi";
import { Card } from "./Card";
import { ErrorBlock, LoadingBlock } from "./StatusStates";

interface ModelPanelProps {
  metadata: ModelMetadataResponse;
  driftState: AsyncState<DriftReportResponse>;
}

const DRIFT_LABELS: Record<DriftStatus, string> = {
  stable: "Stable",
  moderate_shift: "Moderate shift",
  significant_shift: "Significant shift",
};

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

function formatPercent(value: number, digits = 1): string {
  return value.toLocaleString(undefined, {
    style: "percent",
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function DriftIndicator({ driftState }: { driftState: AsyncState<DriftReportResponse> }) {
  if (driftState.status === "loading") {
    return <LoadingBlock label="Checking drift…" />;
  }
  if (driftState.status === "error") {
    return <ErrorBlock error={driftState.error} />;
  }

  const drift = driftState.data;
  return (
    <div className="drift">
      <span className={`drift__badge drift__badge--${drift.status}`}>
        <span className="drift__dot" aria-hidden="true" />
        {DRIFT_LABELS[drift.status]}
      </span>
      <p className="drift__detail">
        Largest shift in <strong>{drift.max_psi_feature}</strong> (PSI{" "}
        {formatMetric(drift.max_psi, 3)})
      </p>
    </div>
  );
}

export function ModelPanel({ metadata, driftState }: ModelPanelProps) {
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
        <div className="metric-tile">
          <span className="metric-tile__label">Interval coverage</span>
          <span className="metric-tile__value">
            {formatPercent(metadata.metrics.interval_coverage)}
          </span>
          <span className="metric-tile__hint">
            target {formatPercent(metadata.metrics.interval_coverage_target, 0)}
          </span>
        </div>
      </div>

      <DriftIndicator driftState={driftState} />
    </Card>
  );
}
