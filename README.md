# Forecast Lab

A regional demand-forecasting tool for retail inventory planning. Given a
**region** and a **product category**, it predicts which product
**attributes** (material, fit, closure type, sleeve type, ...) will sell
best next quarter — so regional buyers can stock the right mix instead of
guessing from last year's memory.

> "In Region A, for Outerwear, `fleece` material and `heavy` weight-class
> are trending up — with wool and down close behind."

This is a from-scratch, synthetic-data portfolio project: a clean-room
schema, a deterministic data generator, a full ML lifecycle (EDA → features
→ training → evaluation → registry), a FastAPI serving layer, and a React
dashboard.

## Problem framing

See [docs/problem_framing.md](docs/problem_framing.md) for the full
objective, target variable, and metric justification. In short: the model
predicts weekly `units_sold` per `(region, category, attribute_type,
attribute_value)` and is evaluated on **MAE**, **RMSE**, and **NDCG@5**
(since the product surfaces a top-5 ranked list of attributes, ranking
quality matters as much as point-forecast accuracy).

## Architecture

```mermaid
flowchart LR
    subgraph Offline["Offline / Training"]
        GEN["Data generator\n(ml/data_generator)"] --> RAW[("data/raw/\ntransactions.csv")]
        RAW --> EDA["EDA\n(ml/eda)"]
        RAW --> PIPE["Pipeline\n(ml/pipeline)"]
        PIPE -->|"load, aggregate,\nfeature engineer,\ntime-split"| FEAT[("data/processed/\ntrain/val/test.parquet")]
        FEAT --> TRAIN["Train + HPO + CV\n(LightGBM + Optuna)"]
        TRAIN --> MLF[("MLflow tracking\nml/mlruns/")]
        TRAIN --> EVAL["Evaluate\nMAE / RMSE / NDCG@5"]
        EVAL --> REG["Model registry\n(models/registry.json)"]
        TRAIN --> LOOKUP[("Lookup artifacts\n(serving-time features)")]
        LOOKUP --> REG
    end

    subgraph Online["Online / Serving"]
        REG --> API["FastAPI backend\n(backend/app)"]
        LOOKUP --> API
        API --> UI["React dashboard\n(frontend/)"]
    end

    RETRAIN["POST /api/retrain"] -.->|triggers background job| TRAIN
```

**Design principle:** training and serving are fully decoupled. The API
never re-runs feature engineering or training inline — it loads a versioned
model + a versioned "lookup artifact" (the most recent historical feature
snapshot per group) from the registry. This is what prevents train/serve
skew: any feature derived from historical aggregates (lags, rolling means,
regional demand share) is computed once, persisted, and reused identically
at inference time.

## Repository layout

```
ml/
  config/          # config.yaml (paths, cutoffs, hyperparameters) + domain taxonomy
  data_generator/  # deterministic synthetic data generator
  eda/             # EDA script -> docs/eda/ (stats + plots)
  pipeline/        # data loading, splitting, features, training, evaluation, registry
  tests/           # pytest unit tests + training smoke test
backend/
  app/
    api/           # FastAPI routes
    core/          # settings
    schemas/       # pydantic request/response models
    services/      # model serving + retraining orchestration
  tests/           # FastAPI TestClient tests
frontend/
  src/
    api/           # typed fetch client
    components/    # dashboard UI components
    pages/         # Dashboard page
    types/         # API/domain TypeScript types
docs/
  problem_framing.md
  eda/             # generated EDA artifacts (stats json + plots)
  screenshots/     # dashboard screenshots
```

## Data

Synthetic order-line data — no real company, brand, city, or person
appears anywhere. Schema:

| column                     | meaning                                            |
|----------------------------|-----------------------------------------------------|
| `order_id`                 | unique order line id                                |
| `order_date`               | date of the order line                              |
| `region_code`              | `"Region A"` .. `"Region F"`                        |
| `product_category`         | `Outerwear`, `Footwear`, `Tops`, `Bottoms`, `Accessories` |
| `product_attribute_type`   | e.g. `material`, `closure_type`, `fit`, `sleeve_type` |
| `product_attribute_value`  | e.g. `wool`, `zip`, `slim`                          |
| `quantity`                 | units sold on this order line                       |
| `unit_price`               | price per unit                                      |

The generator ([ml/data_generator/generate.py](ml/data_generator/generate.py))
injects real structure so the ML stage has signal to find:

- **Seasonality** — e.g. Outerwear peaks in Dec/Jan, Tops peak in Jul/Aug,
  scaled by each region's climate profile.
- **Regional preference skew** — e.g. cold regions favor `wool`/`heavy`
  attributes; every attribute value also gets a smaller, stable per-region
  affinity so no two regions look identical.
- **A slow multi-year trend** per category.
- Poisson noise on top, so the signal is real but not noiseless.

It's fully deterministic (fixed seed) — `python -m ml.data_generator.generate`
regenerates byte-identical output.

## Quickstart (local, no Docker)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Generate synthetic data (deterministic, ~80k rows, 2 years)
python -m ml.data_generator.generate

# 2. Run EDA (writes docs/eda/*.json and *.png)
python -m ml.eda.run_eda

# 3. Run the full training pipeline (feature engineering, HPO+CV, eval, registry)
python -m ml.pipeline.run_training

# 4. Run tests
pytest ml/tests backend/tests -v

# 5. Start the API
uvicorn backend.app.main:app --reload
# -> docs at http://localhost:8000/docs
```

```bash
cd frontend
npm install
cp .env.example .env   # VITE_API_BASE_URL=http://localhost:8000
npm run dev
# -> http://localhost:5173
```

## One-command demo (Docker)

```bash
docker compose up --build
```

This builds and starts both services. The backend container generates the
synthetic dataset and trains a model on first boot if none exists yet
(`backend/entrypoint.sh`), then serves on `:8000`; the frontend serves on
`:5173`.

## Screenshots

Region A / Outerwear:

![Dashboard — Region A, Outerwear](docs/screenshots/dashboard_outerwear.png)

Switching to Region E / Tops re-fetches attribute types and predictions:

![Dashboard — Region E, Tops](docs/screenshots/dashboard_tops.png)

## ML lifecycle details

- **Splitting**: strictly time-based (`ml/pipeline/splitting.py`) — train
  through 2024-06-30, validation through 2024-09-30, test after. No
  shuffling, no leakage across the boundary.
- **Features**: calendar features, lags (1/4/52 weeks), rolling
  mean/std, trailing price, trend slope, and each attribute value's trailing
  share of its parent group's demand — all computed with `shift(1)` before
  any window operation, so no feature ever sees its own label
  (`ml/pipeline/features.py`, verified by `ml/tests/test_features.py`).
- **Training**: LightGBM + Optuna hyperparameter search, scored via
  expanding-window `TimeSeriesSplit` cross-validation on the training set
  only (`ml/pipeline/training.py`).
- **Tracking**: every run's params/metrics/model are logged to MLflow
  (`ml/mlruns/`, local file-based tracking store).
- **Registry**: trained models are saved as versioned artifacts
  (`models/model_<version>.joblib` + `models/lookup_<version>.joblib`),
  indexed in `models/registry.json` with metrics/params/training date. Only
  the last N versions are kept (`ml/pipeline/registry.py`).
- **Evaluation**: MAE, RMSE, and NDCG@5 computed on the held-out test set
  (`ml/pipeline/evaluation.py`) — see
  [docs/problem_framing.md](docs/problem_framing.md) for why these three.

## Testing & CI

- `ml/tests/` — unit tests for data loading, feature engineering (including
  explicit no-leakage checks), splitting, evaluation metrics, and a training
  smoke test that fits a real (tiny) model end-to-end and round-trips it
  through `joblib`.
- `backend/tests/` — FastAPI `TestClient` tests covering all endpoints,
  validation errors, and 404s.
- `.github/workflows/ci.yml` runs `ruff` lint, the full pytest suite (with a
  fast CI-only training config, `ml/config/config.ci.yaml`, so a real model
  exists for the backend tests), and a separate frontend job (typecheck,
  lint, build).

## Configuration

All paths, date cutoffs, and pipeline parameters live in
[ml/config/config.yaml](ml/config/config.yaml) — nothing is hardcoded across
the individual pipeline modules. The domain taxonomy (regions, categories,
attribute types/values) lives in
[ml/config/taxonomy.py](ml/config/taxonomy.py) and is the single source of
truth shared by the data generator, feature pipeline, and API.
