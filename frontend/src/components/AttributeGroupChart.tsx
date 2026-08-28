import {
  Bar,
  BarChart,
  CartesianGrid,
  ErrorBar,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { TooltipProps } from "recharts";
import type { AttributePredictionGroup, FeatureContribution } from "../types/api";
import { EmptyBlock } from "./StatusStates";

interface AttributeGroupChartProps {
  group: AttributePredictionGroup;
}

const PREDICTED_COLOR = "var(--series-predicted)";
const HISTORICAL_COLOR = "var(--series-historical)";
const INTERVAL_COLOR = "var(--text-muted)";

// Human-readable labels for the raw internal feature names surfaced in
// `top_factors`. Anything not listed here falls back to the raw name.
const FEATURE_LABELS: Record<string, string> = {
  lag_1: "Last week's demand",
  rolling_mean_4: "Recent 4-week average",
  rolling_mean_12: "Recent 12-week average",
  group_avg_price: "Average price",
  region_category_share: "Share of regional demand",
  trend_slope_12: "12-week trend",
  month: "Seasonality",
  quarter: "Seasonality",
  week_of_year: "Seasonality",
  is_holiday_quarter: "Seasonality",
};

function featureLabel(feature: string): string {
  return FEATURE_LABELS[feature] ?? feature;
}

function formatUnits(value: number): string {
  return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
}

function formatImpact(value: number): string {
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toLocaleString(undefined, {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  })} units`;
}

interface ChartRow {
  attribute_value: string;
  "Predicted units": number;
  "Historical avg units": number;
  range: [number, number];
  predicted_units_low: number;
  predicted_units_high: number;
  top_factors: FeatureContribution[];
}

function ChartTooltip({ active, payload, label }: TooltipProps<number, string>) {
  if (!active || !payload || payload.length === 0) return null;

  const row = payload[0]?.payload as ChartRow | undefined;

  return (
    <div className="chart-tooltip">
      <p className="chart-tooltip__label">{label}</p>
      {payload
        .filter((entry) => entry.dataKey !== "range")
        .map((entry) => (
          <div className="chart-tooltip__row" key={entry.dataKey as string}>
            <span
              className="chart-tooltip__key"
              style={{ backgroundColor: entry.color }}
              aria-hidden="true"
            />
            <span className="chart-tooltip__name">{entry.name}</span>
            <span className="chart-tooltip__value">
              {formatUnits(entry.value as number)} units
            </span>
          </div>
        ))}

      {row && (
        <p className="chart-tooltip__interval">
          Range {formatUnits(row.predicted_units_low)}
          {" – "}
          {formatUnits(row.predicted_units_high)} units
        </p>
      )}

      {row && row.top_factors.length > 0 && (
        <div className="chart-tooltip__factors">
          <p className="chart-tooltip__factors-title">Top factors</p>
          {row.top_factors.map((factor) => (
            <div className="chart-tooltip__factor-row" key={factor.feature}>
              <span className="chart-tooltip__factor-name">
                {featureLabel(factor.feature)}
              </span>
              <span
                className={`chart-tooltip__factor-impact${
                  factor.impact < 0 ? " chart-tooltip__factor-impact--negative" : ""
                }`}
              >
                {formatImpact(factor.impact)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function AttributeGroupChart({ group }: AttributeGroupChartProps) {
  if (group.predictions.length === 0) {
    return (
      <EmptyBlock>
        No predictions available for &ldquo;{group.attribute_type}&rdquo; in this
        region/category combination.
      </EmptyBlock>
    );
  }

  const data: ChartRow[] = [...group.predictions]
    .sort((a, b) => a.rank - b.rank)
    .map((prediction) => ({
      attribute_value: prediction.attribute_value,
      "Predicted units": prediction.predicted_units,
      "Historical avg units": prediction.historical_avg_units,
      // ErrorBar wants the +/- offset from the bar value, not absolute bounds.
      range: [
        prediction.predicted_units - prediction.predicted_units_low,
        prediction.predicted_units_high - prediction.predicted_units,
      ],
      predicted_units_low: prediction.predicted_units_low,
      predicted_units_high: prediction.predicted_units_high,
      top_factors: prediction.top_factors,
    }));

  const chartHeight = Math.max(180, data.length * 56);

  return (
    <div>
      <ResponsiveContainer width="100%" height={chartHeight}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 4, right: 24, bottom: 4, left: 0 }}
          barGap={2}
          barCategoryGap="28%"
        >
          <CartesianGrid
            horizontal={false}
            stroke="var(--chart-grid)"
            strokeDasharray="0"
          />
          <XAxis
            type="number"
            tick={{ fill: "var(--text-muted)", fontSize: 12 }}
            tickLine={false}
            axisLine={{ stroke: "var(--chart-axis)" }}
            tickFormatter={formatUnits}
          />
          <YAxis
            type="category"
            dataKey="attribute_value"
            width={132}
            tick={{ fill: "var(--text-secondary)", fontSize: 13 }}
            tickLine={false}
            axisLine={{ stroke: "var(--chart-axis)" }}
          />
          <Tooltip
            content={<ChartTooltip />}
            cursor={{ fill: "var(--chart-hover-fill)" }}
          />
          <Legend
            verticalAlign="top"
            align="left"
            height={32}
            iconType="rect"
            iconSize={10}
            wrapperStyle={{ fontSize: 13, color: "var(--text-secondary)" }}
          />
          <Bar
            dataKey="Predicted units"
            fill={PREDICTED_COLOR}
            radius={[0, 4, 4, 0]}
            maxBarSize={20}
          >
            <ErrorBar
              dataKey="range"
              width={4}
              direction="x"
              stroke={INTERVAL_COLOR}
              strokeWidth={1.5}
            />
          </Bar>
          <Bar
            dataKey="Historical avg units"
            fill={HISTORICAL_COLOR}
            radius={[0, 4, 4, 0]}
            maxBarSize={20}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
