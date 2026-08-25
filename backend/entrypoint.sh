#!/bin/sh
set -e

if [ ! -f /srv/data/raw/transactions.csv ]; then
  echo "No raw data found — generating synthetic dataset..."
  python -m ml.data_generator.generate
fi

if [ ! -f /srv/models/registry.json ]; then
  echo "No trained model found — running training pipeline..."
  python -m ml.pipeline.run_training
fi

exec uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
