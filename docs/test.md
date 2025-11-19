## Testing Plan and Conventions for ML/AR Project

This document outlines the **testing strategy** for the Machine Learning and Augmented Reality project. It defines the types of tests, their placement, execution standards, and conventions to ensure code stability and model performance.

-----

## Testing Philosophy

Every change, regardless of scope, must be accompanied by relevant testing to ensure **functional correctness** (code works as intended) and **performance fidelity** (model meets evaluation metrics). Testing is mandatory for merging any code into the `dev` branch.

### 1\. Test Placement

Tests should be organized in a dedicated directory at the root level, typically named `tests/`. The structure within this directory should mirror the structure of the `src/` directory.

| Directory | Content | Purpose |
| :--- | :--- | :--- |
| `tests/src/data/` | Tests for data validation and pre-processing scripts. | Ensures data pipelines are deterministic and clean. |
| `tests/src/models/` | Tests for model layer definitions, training logic, and inference correctness. | Verifies the ML pipeline is correctly implemented. |
| `tests/src/ar/` | Tests for AR tracking logic, rendering, and device interaction. | Ensures the augmented reality interface functions on target hardware. |
| `tests/src/utils/` | Tests for general utility functions (e.g., helpers, logging). | Verifies core project utilities are reliable. |

### 2\. Naming Conventions

All test files and functions must follow clear, conventional naming rules for discoverability and execution.

  * **File Naming:** Test files must be prefixed with `test_`.
      * **Example:** `tests/src/models/test_training_logic.py`
  * **Function Naming:** Test functions must be prefixed with `test_`.
      * **Example:** `def test_model_output_shape():`

-----

## Types of Testing

The project requires three main categories of tests, corresponding to the software development and ML lifecycles:

### 1\. Unit Tests (Code Integrity)

These tests verify the smallest parts of the code base (individual functions, classes, or modules) in isolation.

  * **Focus:** Input validation, utility function correctness, data transformation logic, and AR component initialization.
  * **Execution:** Run automatically by the Continuous Integration (CI) pipeline on every Pull Request (PR).
  * **Tooling:** Typically executed using **PyTest**.

### 2\. Integration Tests (Pipeline Flow)

These tests verify that different components of the system work together correctly, particularly focusing on the ML and AR pipelines.

  * **Examples:**
      * Testing the data pipeline from `raw` to `processed` directory.
      * Verifying that a trained model correctly loads into the AR inference module.
      * Testing the sequence: Data Ingestion → Feature Engineering → Model Prediction.
  * **Execution:** Executed on the `dev` branch before merging to `main`.

### 3\. ML-Specific Tests (Performance & Stability)

These specialized tests focus on the model's behavior and performance, using statistical rigor.

| Test Type | Purpose | Standard/Threshold |
| :--- | :--- | :--- |
| **Data Validation** | Check for data drift, missing values, or schema violations in the input data. | Schema must match the expected feature set. |
| **Model Sanity Check** | Ensure the model can overfit a tiny subset of data (debug). | Must achieve near-perfect metrics (e.g., $99\%+$ accuracy) on the tiny set. |
| **Performance Regression** | Compare current model metrics (e.g., $F_1$ score, latency) against the last successful release/baseline. | Metrics must not degrade by more than **$X\%$** (e.g., $0.5\%$) relative to the current `main` branch model. |
| **Inference Latency** | Measure the time taken for a single prediction in the AR environment. | Must meet the strict real-time requirement (e.g., $\le 10$ milliseconds). |

-----

## Testing Workflow & PR Requirements

To merge a feature branch into `dev`, the following testing criteria **must be met** and verified by the CI system:

1.  **Code Coverage:** All new or modified code must be covered by **Unit Tests** and pass the minimum code coverage threshold (e.g., $80\%$).

2.  **No Regression:** All existing **Unit and Integration Tests** must pass without failure.

3.  **Model Stability:** If the PR changes model code, the **Performance Regression Test** must pass, showing no significant degradation in metrics.

4.  **Local Verification:** Developers must run the full test suite locally before pushing and opening a PR:

```bash
# Run all tests using the project's preferred tool (e.g., pytest)
pytest --cov=src --cov-report term-missing tests/
```
