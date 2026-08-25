import type { PredictionsResponse } from "../types/api";
import { AttributeGroupChart } from "./AttributeGroupChart";
import { Card } from "./Card";
import { EmptyBlock } from "./StatusStates";

interface PredictionsPanelProps {
  data: PredictionsResponse;
}

function formatAttributeType(attributeType: string): string {
  return attributeType
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function formatAsOfWeek(isoDate: string): string {
  const parsed = new Date(isoDate);
  if (Number.isNaN(parsed.getTime())) return isoDate;
  return parsed.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

export function PredictionsPanel({ data }: PredictionsPanelProps) {
  if (data.groups.length === 0) {
    return (
      <Card
        title={`${data.category} in ${data.region}`}
        subtitle={`Forecast as of ${formatAsOfWeek(data.as_of_week)} · model ${data.model_version}`}
      >
        <EmptyBlock>
          No attribute predictions are available yet for this region and
          category. Try a different combination.
        </EmptyBlock>
      </Card>
    );
  }

  return (
    <div className="predictions-grid">
      {data.groups.map((group) => (
        <Card
          key={group.attribute_type}
          title={formatAttributeType(group.attribute_type)}
          subtitle="Predicted vs. historical average units, next quarter"
        >
          <AttributeGroupChart group={group} />
        </Card>
      ))}
    </div>
  );
}
