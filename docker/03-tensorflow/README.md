# AI Workstation TensorFlow Layer

The TensorFlow layer is a standalone GPU image for TensorFlow workloads whose
CUDA requirements differ from the PyTorch-based stack.

## Base Image

- `nvidia/cuda:12.5.1-cudnn-runtime-ubuntu22.04`

## Installed Components

- Python 3.12 managed by `uv`
- TensorFlow
- The same scientific stack used by the science layer

## Verification Focus

The verification script confirms that TensorFlow can detect a GPU and execute a
matrix multiplication on `/GPU:0`.

## Commands

```bash
make build-tensorflow
make verify-tensorflow
make shell-tensorflow
```
