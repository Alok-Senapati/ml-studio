# AI Workstation - Python Layer

## Purpose

This layer extends `ai-base` by adding Python development tooling.

## Base Image

- ai-base:1.0.0

## Responsibilities

- Python package management
- uv
- pip compatibility
- Python tooling

## Does NOT include

- NumPy
- Pandas
- SciPy
- PyTorch
- TensorFlow
- Jupyter

## Build

```bash
make build-python
```

## Verify

```bash
make verify-python
```

## Run

```bash
make run-python
```

## Acceptance Criteria

- [x] Image builds successfully
- [x] Python available
- [x] pip available
- [x] uv available
- [x] Verification passes