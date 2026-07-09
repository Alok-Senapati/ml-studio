# AI Workstation

AI Workstation is a layered Docker-based machine learning environment for local
GPU development. It provides reusable images for Python, scientific computing,
PyTorch, TensorFlow, JupyterLab, and MLflow, plus a small ML project template
for starting new experiments quickly.

## Repository Layout

```text
.
├── compose/                 # Docker Compose services for Jupyter and MLflow
├── configs/                 # Runtime configuration and persisted MLflow state
├── datasets/                # Local datasets mounted into containers
├── docker/                  # Layered Docker images and verification scripts
├── docs/                    # Architecture and workflow documentation
├── notebooks/               # Shared notebooks mounted into Jupyter
├── projects/                # Local project workspace mounted into containers
└── templates/ml-template/   # Reusable ML project scaffold
```

## Image Stack

The images are intentionally layered so shared dependencies are built once and
reused across higher-level environments.

1. `ai-base` installs Ubuntu, CUDA runtime support, Python, and core CLI tools.
2. `ai-python` adds `uv` for Python environment and package management.
3. `ai-science` adds the shared scientific Python stack.
4. `ai-pytorch` adds the CUDA-enabled PyTorch toolchain.
5. `ai-tensorflow` provides a standalone TensorFlow GPU image because its CUDA
   requirements differ from the PyTorch stack.
6. `ai-jupyter-*` adds JupyterLab on top of the framework images.
7. `ai-mlflow` provides an MLflow tracking server image.

More detail is available in [docs/architecture.md](docs/architecture.md).

## Prerequisites

- Docker with Compose support
- NVIDIA Container Toolkit for GPU-enabled images
- GNU Make
- A Linux or WSL-based filesystem path for the mounted workspace directories

## Quick Start

1. Review `compose/.env` and adjust the paths or ports for your machine.
2. Build the images you need:

```bash
make build-base
make build-python
make build-science
make build-pytorch
make build-jupyter-pytorch
```

3. Verify the images:

```bash
make verify-all
make verify-jupyter-pytorch
```

4. Start a Jupyter service:

```bash
make up-pytorch
```

5. Stop the service when you are done:

```bash
make down-pytorch
```

## Common Commands

```bash
make help
make build-all
make verify-all
make shell-pytorch
make shell-tensorflow
make up-pytorch
make up-tensorflow
make up-mlflow
```

## Compose Services

### PyTorch JupyterLab

```bash
make up-pytorch
make logs-pytorch
make down-pytorch
```

### TensorFlow JupyterLab

```bash
make up-tensorflow
make logs-tensorflow
make down-tensorflow
```

### MLflow

```bash
make up-mlflow
make logs-mlflow
make down-mlflow
```

The MLflow UI is exposed at `http://localhost:5000`.

## Template Project

The repository includes a reusable ML project scaffold in
`templates/ml-template`. It provides a conventional directory structure, a
minimal `pyproject.toml`, and starter entry points for training, evaluation,
and prediction workflows.

See [templates/ml-template/README.md](templates/ml-template/README.md) for setup details.
