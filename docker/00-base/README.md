# AI Workstation Base Image

The base image provides the shared operating system, CUDA runtime support, and
command-line tooling used by the higher layers.

## Includes

- Ubuntu 24.04
- NVIDIA CUDA runtime image
- Python, pip, and virtual environment support
- Common developer utilities such as `git`, `curl`, `wget`, and `tini`

## Excludes

- Scientific Python libraries
- Deep learning frameworks
- JupyterLab
- MLflow

## Commands

```bash
make build-base
make verify-base
make shell-base
```
