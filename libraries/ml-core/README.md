# ml-core

`ml-core` is a reusable machine learning library developed as part of the **Machine Learning Expert Track**.

The goal is not to compete with scikit-learn, but to build a production-inspired educational library that teaches how machine learning algorithms work internally.

---

## Objectives

- Learn machine learning from first principles
- Implement algorithms using NumPy
- Build reusable, modular components
- Follow clean architecture and SOLID principles
- Maintain production-quality engineering practices

---

## Current Status

🚧 Under active development

The first implementation target is **Logistic Regression**, which will establish the reusable architecture for future algorithms.

---

## Planned Modules

- Base abstractions
- Activation functions
- Loss functions
- Optimizers
- Metrics
- Linear models
- Data preprocessing
- Mathematical utilities
- Validation
- Exception hierarchy

---

## Development

Install dependencies:

```bash
uv sync
```

Run tests:

```bash
make test
```

Run all quality checks:

```bash
make check
```

---

## License

MIT