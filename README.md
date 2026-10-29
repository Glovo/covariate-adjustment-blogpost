<div align="center">

# 🛠️ covariate-adjustment-blogpost

**Simulations behind the Glovo Engineering blog post on covariate adjustment in A/B tests**

This repo holds the simulation code and notebooks that back the covariate adjustment blog post: it compares
variance-reduction methods (CUPED, CUPAC, multivariable regression, doubly robust) on synthetic A/B test data.
It is **not** a library or a service: nothing is packaged, published or deployed.

[![Python](https://img.shields.io/badge/Python-%5E3.8-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Poetry](https://img.shields.io/badge/Poetry-managed-60A5FA?logo=poetry&logoColor=white)](pyproject.toml)
[![JupyterLab](https://img.shields.io/badge/JupyterLab-%5E3.4.5-F37626?logo=jupyter&logoColor=white)](pyproject.toml)

</div>

---

## Table of contents

- [Overview](#overview)
- [Build & local development](#build--local-development)
- [Repository layout](#repository-layout)
- [Contributing & conventions](#contributing--conventions)
- [Glossary](#glossary)
- [Ownership & support](#ownership--support)

---

## Overview

The blog post describes methods that use **covariates** to reduce the variance of the estimated effect in an
A/B test. This repo shows, through simulations, how those methods perform. Helper code in [`src/`](src)
generates synthetic experiments and estimates the average treatment effect (ATE); the notebooks in
[`notebooks/`](notebooks) run the simulations and draw the plots and DAGs used in the post.

| Notebook | What it shows |
|---|---|
| [`notebooks/covariate_adjustment.ipynb`](notebooks/covariate_adjustment.ipynb) | Compares the adjustment methods over many simulated experiments: linear effects, a random (useless) covariate, and non-linear effects |
| [`notebooks/good_bad_covariates.ipynb`](notebooks/good_bad_covariates.ipynb) | Compares good and bad covariates with those methods and generates the blog post's DAGs |

| Module | Role |
|---|---|
| [`src/generator.py`](src/generator.py) | `generate_samples()`: synthetic outcome `Y`, treatment `T`, pre-experiment metric `Y_before`, covariates `X_i` and a random covariate `R_1` |
| [`src/adjustment_methods.py`](src/adjustment_methods.py) | ATE estimators: `get_ate_cuped`, `get_ate_ols`, `get_ate_cupac`, `get_ate_doubly_robust` |
| [`src/utils.py`](src/utils.py) | `split_and_preprocess()`: splits data 50/50 into a pre-experiment set (treatment effect removed) and an experiment set |
| [`src/imbalance.py`](src/imbalance.py) | Standalone `# %%` script: estimator error with and without a covariate, by group imbalance |

| | |
|---|---|
| Language / build | Python `^3.8`, Poetry ([`pyproject.toml`](pyproject.toml), [`poetry.lock`](poetry.lock)) |
| Main libraries | pandas, numpy, statsmodels, scikit-learn, seaborn |
| Runtime | Local JupyterLab |

---

## Build & local development

> [!NOTE]
> Prerequisites: Python 3.8+ and [Poetry](https://python-poetry.org/).

```bash
poetry install                                                           # create the env from poetry.lock
poetry run python -m ipykernel install --user --name=covariate-adjustment   # register the env as a Jupyter kernel
poetry run jupyter lab                                                   # then pick the "covariate-adjustment" kernel
```

### Run the simulations

- Open [`notebooks/covariate_adjustment.ipynb`](notebooks/covariate_adjustment.ipynb) for the method comparison.
- Open [`notebooks/good_bad_covariates.ipynb`](notebooks/good_bad_covariates.ipynb) for good vs bad covariates and the DAGs.
- The first notebook imports from `src/` with `sys.path.append('../src')`, so run it with `notebooks/` as the
  working directory (JupyterLab's default when you open it from there).
- [`src/imbalance.py`](src/imbalance.py) imports `tqdm`, which is **not** in [`pyproject.toml`](pyproject.toml) or
  [`poetry.lock`](poetry.lock); install it in the env before running that script.

---

## Repository layout

```text
.
├── notebooks/                      # ⭐ the simulations behind the blog post
│   ├── covariate_adjustment.ipynb  # method comparison
│   └── good_bad_covariates.ipynb   # good vs bad covariates, DAGs
├── src/                            # helpers imported by the notebooks
│   ├── adjustment_methods.py       # ATE estimators
│   ├── generator.py                # synthetic data generator
│   ├── imbalance.py                # standalone imbalance simulation
│   └── utils.py                    # pre-experiment / experiment split
├── pyproject.toml                  # Poetry project and dependencies
└── poetry.lock                     # locked versions
```

---

## Contributing & conventions

- Add dependencies with `poetry add` so [`pyproject.toml`](pyproject.toml) and [`poetry.lock`](poetry.lock) stay in sync.
- The repo is **public**: anything committed is visible outside Glovo.
- No CODEOWNERS file exists, so reviews are not enforced by GitHub.

---

## Glossary

| Term | Meaning |
|---|---|
| ATE | Average treatment effect: the difference in mean outcome between treatment and control |
| Covariate | A variable measured alongside the outcome, used to explain part of its variance |
| CUPED | Controlled-experiment Using Pre-Experiment Data: adjusts the outcome with its pre-experiment value |
| CUPAC | CUPED with a model prediction (here a gradient-boosting regressor) as the control variate |
| OLS | Ordinary least squares regression of `Y` on `T` plus covariates |
| Doubly robust | Estimator combining outcome models per group with propensity weighting |
| DAG | Directed acyclic graph of causal relations between variables |

---

## Ownership & support

- **Contacts** (from the previous README), for doubts or suggestions:
  - victor.bouzas@glovoapp.com
  - david.masip@glovoapp.com
