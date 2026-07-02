# AI Workstation Base Image

This image is the foundation of the AI Workstation.

## Responsibilities

- Ubuntu 24.04
- CUDA Runtime
- cuDNN Runtime
- Python
- Linux utilities
- Common environment variables

## Does NOT include

- NumPy
- Pandas
- SciPy
- PyTorch
- TensorFlow
- Jupyter
- MLflow

## Build

docker build \
    -t ai-base:1.0.0 \
    -f Dockerfile .

## Run

docker run --rm -it ai-base:1.0.0