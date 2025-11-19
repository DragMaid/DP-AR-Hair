# Project Directory Structure

This document provides a definitive overview of the project's directory layout and the purpose of each major component. It establishes the organizational standard for maintainability and reproducibility in our ML/AR environment.

---

## Root Directory

These files govern the overall project environment and documentation.

| File/Directory | Purpose |
| :--- | :--- |
| `LICENSE` | Defines the licensing terms for the project code. |
| `Makefile` | Scriptable tasks for common actions: environment setup, data processing, model training, and documentation build. |
| `README.md` | The main entry point for the project; provides high-level overview and setup instructions. |
| `mkdocs.yaml` | **Documentation Configuration:** Primary configuration file for building the project documentation site. |
| `requirements.txt` | **Environment Reproducibility:** Lists all Python package dependencies required to run the project. |
| `setup.py` | Package definition file allowing the project to be installed as an editable Python package (`pip install -e .`). |
| `docs` | **Documentation Source:** Markdown files and assets used by MkDocs to generate the website. |

---

## Data Management

This directory follows a clear separation-of-concerns standard to ensure **data provenance** and **immutability**.

| Subdirectory | Purpose |
| :--- | :--- |
| `data/raw` | **Immutable Data:** Original, untouched data dumps or sensor readings. Files here are never modified. |
| `data/external` | **Third-Party Data:** Datasets from external, third-party sources (e.g., public benchmarks, external APIs). |
| `data/interim` | **Feature Engineering:** Intermediate transformed data, often cleaned or feature-engineered but not yet finalized for training. |
| `data/processed` | **Canonical Datasets:** Final, version-controlled datasets ready for direct input into model training scripts. |

---

## Model & AR Artifacts

These directories store the products of the machine learning and augmented reality workflows.

| Directory | Purpose |
| :--- | :--- |
| `models` | **Model Weights/Artifacts:** Stores serialized models, trained weights (e.g., `.pkl`, `.h5`), prediction results, and experiment summaries. |
| `ar_assets` | **AR Assets:** Contains specific assets for the Augmented Reality application, such as 3D models, textures, shaders, and configuration files for AR tracking. |
| `notebooks` | **Exploration & Prototyping:** Jupyter notebooks for initial data exploration, quick experiments, and demonstrating model output. Naming convention: `<order>-<initials>-<description>`. |
| `references` | **Explanatory Materials:** Manuals, academic papers, detailed data dictionaries, and supplementary explanatory materials relevant to the project or algorithms. |
| `reports` | **Reporting Outputs:** Generated artifacts for reporting and presentation. |
| `reports/figures` | Automatically generated graphics, plots, and figures for analysis and reports. |
| `reports/analysis` | Final analysis documents in formats like HTML, PDF, or LaTeX. |

---

## Source Code

The `src` directory holds all the reusable, production-ready Python code organized into functional sub-modules.

| Subdirectory | Purpose |
| :--- | :--- |
| `src/data` | Scripts for data handling, downloading, versioning, and the main feature engineering logic (`make_dataset.py`). |
| `src/models` | Core machine learning code: **model definition**, training scripts (`train_model.py`), and inference logic (`predict_model.py`). |
| `src/ar` | Code specifically for the Augmented Reality implementation: camera handling, tracking logic, and rendering pipelines. |
| `src/pipelines` | High-level orchestration of the ML and AR workflow, defining the sequential steps for both training and deployment. |
| `src/utils` | General-purpose, project-wide utility functions (e.g., logging, configuration parsing, common math helpers). |
| `src/main.py` | The project's command-line entry point for processing arguments and launching major pipelines (e.g., `python src/main.py --train`). |

---

## Summary

This professional structure, inspired by ML best practices, ensures a clear separation between **Data**, **Code**, **Models**, and **Documentation**. This organizational clarity is essential for project scaling, team collaboration, and ensuring the reproducibility of all machine learning experiments. Below is a more condensed version of all the above documentation:

```
├── LICENSE
├── Makefile
├── README.md
├── mkdocs.yaml                   # Configurations for documentation
├── data
│   ├── external                  # Data from third‑party sources
│   ├── interim                   # Intermediate transformed data
│   ├── processed                 # Final canonical datasets for modeling
│   └── raw                       # Original, immutable data dumps
├── docs                          # Sphinx project (optional), can coexist with MkDocs
├── models                        # Serialized models, predictions, or summaries
├── notebooks                     # Jupyter notebooks; naming: `<order>-<initials>-<short_description>`
├── references                    # Manuals, data dictionaries, and explanatory materials
├── reports
│   ├── figure.png                # Generated graphics for reporting
│   └── analysis.tex              # Analysis in HTML, PDF, LaTeX, etc.
├── requirements.txt              # Reproducible environment file
├── setup.py                      # Allows `pip install -e` installation
├── src
│   ├── __init__.py
│   ├── data
│   │   └── make_dataset.py       # Data download/generation logic
│   ├── utils
│   │   └── util.py               # Include all the utility functions
│   ├── models
│   │   ├── predict_model.py      # Inference
│   │   └── train_model.py        # Training scripts
│   └── visualization
│       └── visualize.py          # Visualization utilities
│   └── pipelines
│       └── training.py           # Pipeline for training
│       └── inference.py          # Pipeline for inference
│   └── main.py                   # Process arguments to set environment and choose pipeline
```
