# Mastercard AI Defense Lab

Mastercard Innovation Challenge submission: an AI Defense Lab for payment
security, built as three pillars.

- **`identify/`** - `attack-taxonomy.md`, fourteen GenAI-era payment fraud
  attack vectors across five categories, each grounded in a documented
  real-world mechanism. Shared input for the other two pillars.
- **`generate/`** - synthetic labeled dataset with one simulated instance per
  taxonomy vector, plus a legitimate baseline, for training and evaluating
  fraud detection. See `generate/README.md`.
- **`defend/`** - XGBoost fraud classifier trained on Generate's output, with
  per-category/per-vector evaluation. Currently covers 11 of 14 vectors (E1
  excluded - blocked by a bug in `generate/`, see `defend/README.md` "Known
  limitations"). See `defend/README.md` and `defend/results/report.md`.
