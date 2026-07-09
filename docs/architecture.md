# Architecture Overview

AI Workstation is organized as a small stack of Docker images plus a few
Compose-managed services for interactive development.

## Image Dependency Graph

```text
ai-base
  └── ai-python
        ├── ai-science
        │     └── ai-pytorch
        │           └── ai-jupyter-pytorch
        └── ai-mlflow

ai-tensorflow
  └── ai-jupyter-tensorflow
```

## Why TensorFlow Is Separate

The PyTorch stack is built on top of the shared `ai-science` image. TensorFlow
is kept independent because its CUDA and base image requirements currently
differ from the rest of the layered stack.

## Mounted Directories

The Jupyter Compose services mount shared workspace folders so experiments and
data persist outside the containers.

- `projects/` for code and project-specific assets
- `datasets/` for local datasets
- `notebooks/` for shared notebooks
- `configs/` for runtime configuration

## Verification Strategy

Each image includes a `verify.sh` script stored alongside its Dockerfile. These
checks are intentionally lightweight and focus on proving that the expected
tools, Python packages, and GPU integrations are available.

## Operational Flow

1. Build the lowest layer you need.
2. Verify that image before building higher layers.
3. Launch JupyterLab or MLflow with Docker Compose.
4. Develop inside mounted workspace directories so artifacts persist locally.
