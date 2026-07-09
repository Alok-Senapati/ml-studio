# AI Workstation Jupyter Layer

The Jupyter layer adds JupyterLab and notebook tooling on top of either the
PyTorch or TensorFlow runtime image.

## Images Produced

- `ai-jupyter-pytorch:1.0.0`
- `ai-jupyter-tensorflow:1.0.0`

## Installed Tools

- JupyterLab
- Notebook
- IPykernel
- JupyterLab Git
- Ipywidgets
- Plotly

## Commands

```bash
make build-jupyter-pytorch
make build-jupyter-tensorflow
make verify-jupyter-pytorch
make verify-jupyter-tensorflow
```
