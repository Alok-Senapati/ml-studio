# AI Workstation Science Layer

The science layer extends `ai-python` with a shared scientific computing stack
for notebooks, classical machine learning, and feature engineering workloads.

## Base Image

- `ai-python:1.0.0`

## Installed Packages

- NumPy
- Pandas
- SciPy
- Scikit-Learn
- Matplotlib
- PyArrow
- Polars
- DuckDB

## Commands

```bash
make build-science
make verify-science
make shell-science
```
