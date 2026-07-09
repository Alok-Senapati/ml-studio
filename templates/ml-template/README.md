# ML Project Template

This template provides a lightweight starting point for small-to-medium machine
learning projects. It favors a familiar directory layout, `uv`-based dependency
management, and explicit entry points for training, evaluation, and batch
prediction.

## Structure

```text
.
├── configs/             # Experiment and runtime configuration files
├── data/
│   ├── raw/             # Immutable source data
│   ├── interim/         # Temporary cleaned or joined data
│   ├── processed/       # Model-ready datasets and features
│   └── external/        # Third-party or manually sourced inputs
├── experiments/         # Scratch artifacts or experiment notes
├── models/              # Saved model checkpoints and exports
├── notebooks/           # Exploratory analysis notebooks
├── reports/
│   ├── figures/         # Charts and visual outputs
│   └── metrics/         # Serialized metrics and evaluation summaries
├── scripts/             # CLI entry points for the main workflows
├── src/                 # Reusable project code
└── tests/               # Automated tests
```

## Quick Start

```bash
make install
make train
make evaluate
make predict
make test
```

## Workflow

1. Put source data in `data/raw`.
2. Explore ideas in `notebooks/`.
3. Move reusable logic into `src/`.
4. Keep orchestration code in `scripts/`.
5. Save trained artifacts in `models/`.
6. Store charts and metrics under `reports/`.

## Script Conventions

The starter scripts are placeholders with small CLI surfaces so new projects can
replace the internals without changing the command shape immediately.

- `scripts/train.py` accepts a config path and model output directory.
- `scripts/evaluate.py` accepts a model path and metrics output directory.
- `scripts/predict.py` accepts a model path, input data path, and output path.

## Customization Tips

- Add project-specific configuration files under `configs/`.
- Move feature engineering, training, and inference utilities into `src/`.
- Replace the placeholder prints in `scripts/` with your real pipeline logic.
- Add focused tests for data transforms, metrics, and inference behavior.
