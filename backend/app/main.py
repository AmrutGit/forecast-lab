from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import router
from backend.app.core.config import settings

app = FastAPI(
    title="Forecast Lab API",
    description=(
        "Serves regional demand forecasts by product attribute for a "
        "synthetic retail chain. Predictions are produced by a versioned "
        "LightGBM model loaded from the model registry; training and "
        "serving are fully decoupled — this API never re-runs the training "
        "pipeline inline, only triggers it as a background job via /retrain."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
