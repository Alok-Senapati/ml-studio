# AI Workstation MLflow Layer

The MLflow layer provides a lightweight tracking server image that can be run
locally with Docker Compose.

## Base Image

- `ai-python:1.0.0`

## Installed Packages

- `mlflow`
- `psycopg2-binary`

## Commands

```bash
make build-mlflow
make verify-mlflow
make up-mlflow
make logs-mlflow
make down-mlflow
```

The MLflow UI is served at `http://localhost:5000`.
