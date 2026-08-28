import { useEffect, useState } from "react";
import {
  getAttributeTypes,
  getCategories,
  getModelDrift,
  getModelMetadata,
  getPredictions,
  getRegions,
} from "../api/endpoints";
import { useAsync } from "../api/useApi";
import { CategorySelect } from "../components/CategorySelect";
import { ModelPanel } from "../components/ModelPanel";
import { PredictionsPanel } from "../components/PredictionsPanel";
import { RegionSelect } from "../components/RegionSelect";
import { ErrorBlock, LoadingBlock, Skeleton } from "../components/StatusStates";

export function Dashboard() {
  const regionsState = useAsync((signal) => getRegions(signal), []);
  const categoriesState = useAsync((signal) => getCategories(signal), []);

  const [region, setRegion] = useState<string>("");
  const [category, setCategory] = useState<string>("");
  const [predictionsRetryKey, setPredictionsRetryKey] = useState(0);

  // Seed the selectors once their option lists arrive.
  useEffect(() => {
    if (regionsState.status === "success" && !region) {
      setRegion(regionsState.data.regions[0] ?? "");
    }
  }, [regionsState, region]);

  useEffect(() => {
    if (categoriesState.status === "success" && !category) {
      setCategory(categoriesState.data.categories[0] ?? "");
    }
  }, [categoriesState, category]);

  // Attribute types are fetched per category (mainly to keep the taxonomy
  // API-driven end to end); refetched whenever the category changes.
  const attributeTypesState = useAsync(
    (signal) => {
      if (!category) return Promise.resolve({ attribute_types: [] });
      return getAttributeTypes(category, signal);
    },
    [category],
  );

  const predictionsState = useAsync(
    (signal) => {
      if (!region || !category) {
        return Promise.reject(new Error("Region or category not selected yet."));
      }
      return getPredictions(region, category, signal);
    },
    [region, category, predictionsRetryKey],
  );

  const modelMetadataState = useAsync((signal) => getModelMetadata(signal), []);
  const driftState = useAsync((signal) => getModelDrift(signal), []);

  const selectorsReady =
    regionsState.status === "success" && categoriesState.status === "success";

  return (
    <div className="dashboard">
      <section className="filter-bar" aria-label="Region and category filters">
        {regionsState.status === "loading" || categoriesState.status === "loading" ? (
          <LoadingBlock label="Loading regions and categories…" />
        ) : regionsState.status === "error" ? (
          <ErrorBlock error={regionsState.error} />
        ) : categoriesState.status === "error" ? (
          <ErrorBlock error={categoriesState.error} />
        ) : (
          <>
            <RegionSelect
              regions={regionsState.data.regions}
              value={region}
              onChange={setRegion}
            />
            <CategorySelect
              categories={categoriesState.data.categories}
              value={category}
              onChange={setCategory}
            />
            {attributeTypesState.status === "success" && (
              <p className="filter-bar__hint">
                Tracking {attributeTypesState.data.attribute_types.length} attribute
                {attributeTypesState.data.attribute_types.length === 1 ? "" : "s"} for{" "}
                {category}:{" "}
                {attributeTypesState.data.attribute_types
                  .map((type) => type.replace(/_/g, " "))
                  .join(", ")}
              </p>
            )}
          </>
        )}
      </section>

      <div className="dashboard__layout">
        <div className="dashboard__main">
          {!selectorsReady || !region || !category ? (
            <Skeleton height={320} />
          ) : predictionsState.status === "loading" ? (
            <div className="predictions-grid">
              <Skeleton />
              <Skeleton />
            </div>
          ) : predictionsState.status === "error" ? (
            <ErrorBlock
              error={predictionsState.error}
              onRetry={() => setPredictionsRetryKey((key) => key + 1)}
            />
          ) : (
            <PredictionsPanel data={predictionsState.data} />
          )}
        </div>

        <aside className="dashboard__sidebar">
          {modelMetadataState.status === "loading" ? (
            <Skeleton height={220} />
          ) : modelMetadataState.status === "error" ? (
            <ErrorBlock error={modelMetadataState.error} />
          ) : (
            <ModelPanel metadata={modelMetadataState.data} driftState={driftState} />
          )}
        </aside>
      </div>
    </div>
  );
}
