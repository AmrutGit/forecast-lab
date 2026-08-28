# Problem Framing

## Business context

A mid-size retail chain operates stores grouped into regions. Regional
buyers place next-quarter stocking orders per product category (e.g.
Outerwear, Footwear). Buyers currently choose product attributes — material,
fit, closure type, etc. — largely on instinct and last year's memory. This
leads to overstocking attributes that are falling out of favor in a region
and understocking attributes that are trending up.

## Objective

Given a **region** and a **product category**, rank the product attribute
values (across all attribute types relevant to that category — material,
closure type, fit, ...) by predicted demand over the next period, so a buyer
can see "in Region A, for Outerwear, `heavy` weight-class and `wool` material
are trending up" at a glance.

## Target variable

`units_sold`: the number of units sold for a given
`(region_code, product_category, product_attribute_type, product_attribute_value, period)`
combination, aggregated at a **weekly** grain. Weekly aggregation smooths
day-to-day order noise while remaining responsive enough to catch a
trending attribute within a quarter (~13 weekly observations).

The model is trained as a regression over this aggregated demand series,
then its predictions are used to **rank** attribute values within each
`(region, category, attribute_type)` group for the "top predicted
attributes" product surface.

Rather than a single point estimate, three quantile regressors (p10/p50/p90)
are trained per version. p50 (the median) is the point prediction used for
ranking and for the MAE/RMSE/NDCG@5 metrics below; p10/p90 give a prediction
*interval* around it. This matters for the actual buying decision: "40
units" and "40 units, likely between 25 and 60" call for different order
sizes, and a buyer stocking against a bare point estimate has no way to
tell a confidently-forecast attribute from a noisy one.

## Evaluation metrics

Three metrics are computed on the held-out test set, each answering a
different product question:

1. **MAE (Mean Absolute Error)** — on `units_sold`. Answers "on average, how
   many units off is a single forecast?" in units a buyer can sanity-check
   directly. Chosen over MSE as the headline metric because it's directly
   interpretable in the same units as the business decision (how many units
   to stock) and is not dominated by a handful of high-volume outlier weeks.

2. **RMSE (Root Mean Squared Error)** — reported alongside MAE. RMSE
   penalizes large misses more heavily than MAE; tracking both surfaces
   whether errors are consistently small-but-frequent (MAE ≈ RMSE) or
   occasionally large (RMSE >> MAE), which matters because a single very
   wrong forecast can cause a costly overstock/understock decision.

3. **NDCG@5 (Normalized Discounted Cumulative Gain at 5)** — computed per
   `(region, category, attribute_type)` group by ranking attribute values on
   predicted demand and comparing against the ranking implied by actual
   demand. This is the metric that matters most for the actual product
   surface: buyers act on the **top-5 ranked attributes**, not the raw
   forecast number, so a model can have modest MAE/RMSE but still be
   useless if it gets the *ranking* of top attributes wrong (or vice
   versa — ranking can be robust to absolute-value noise that hurts MAE).

Regression accuracy (MAE/RMSE) and ranking quality (NDCG@5) are tracked
separately because they can diverge: a model that's uniformly biased high
across all attributes in a group can still rank them perfectly, and a model
with low average error can still shuffle a close top-5.

A fourth metric, **prediction interval coverage**, checks calibration
rather than accuracy: the fraction of held-out actuals that actually fall
within the p10–p90 interval, which should sit close to 80%. This is tracked
separately from MAE/RMSE/NDCG@5 because an interval can be well-calibrated
around a mediocre point forecast, or badly miscalibrated (too narrow, too
wide) around an accurate one — they answer different questions ("how
accurate is the number" vs. "how honest is the uncertainty around it") and
a stocking decision needs both.

## Model governance

Two additional practices, beyond a single train-and-ship run, are part of
how this model is expected to be operated over time:

- **Champion/challenger promotion** — a newly trained model only replaces
  the currently active one if it beats it on held-out test MAE. This
  prevents a routine retrain (e.g. on a schedule, or after a data refresh)
  from silently degrading production quality just because it's the most
  recent run.
- **Drift monitoring** — the Population Stability Index between each
  monitored feature's training-time distribution and its current values is
  checked independently of model accuracy, since a meaningful shift in the
  regions/categories/attributes being sold is itself a signal that a
  retrain is due, even before enough time has passed to measure fresh
  accuracy against new ground truth.

## Non-goals

- SKU-level or single-product forecasting — the product operates one level
  up, at the attribute level, by design.
- Real-time/streaming inference — retraining and serving both operate on a
  batch/quarterly cadence appropriate for stocking decisions.
- Price optimization or elasticity modeling — `unit_price` is used only as
  a feature, not a decision variable.
