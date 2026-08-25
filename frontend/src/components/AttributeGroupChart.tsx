import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { TooltipProps } from "recharts";
import type { AttributePredictionGroup } from "../types/api";
import { EmptyBlock } from "./StatusStates";

interface AttributeGroupChartProps {
  group: AttributePredictionGroup;
}

const PREDICTED_COLOR = "var(--series-predicted)";
const HISTORICAL_COLOR = "var(--series-historical)";

function formatUnits(value: number): string {
  return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
}

function ChartTooltip({ active, payload, label }: TooltipProps<number, string>) {
  if (!active || !payload || payload.length === 0) return null;

  return (
    <div className="chart-tooltip">
      <p className="chart-tooltip__label">{label}</p>
      {payload.map((entry) => (
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

  const data = [...group.predictions]
    .sort((a, b) => a.rank - b.rank)
    .map((prediction) => ({
      attribute_value: prediction.attribute_value,
      "Predicted units": prediction.predicted_units,
      "Historical avg units": prediction.historical_avg_units,
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
          />
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
