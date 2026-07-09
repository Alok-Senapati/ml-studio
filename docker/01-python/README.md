# AI Workstation Python Layer

The Python layer extends `ai-base` by installing `uv` and preparing the image
for Python dependency management.

## Base Image

- `ai-base:1.0.0`

## Responsibilities

- Install `uv`
- Preserve compatibility with standard `pip` workflows
- Provide a clean base for higher Python-focused layers

## Commands

```bash
make build-python
make verify-python
make shell-python
```
