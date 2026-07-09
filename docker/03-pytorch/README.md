# AI Workstation PyTorch Layer

The PyTorch layer extends `ai-science` with GPU-enabled PyTorch packages for
training and experimentation.

## Base Image

- `ai-science:1.0.0`

## Installed Packages

- `torch`
- `torchvision`
- `torchaudio`

## Verification Focus

The verification script checks CUDA availability, detects a GPU, and executes a
small matrix multiplication on the device.

## Commands

```bash
make build-pytorch
make verify-pytorch
make shell-pytorch
```
