# Testing Plan and Conventions for ML/AR Project

This document defines the testing strategy for the Machine Learning + Augmented Reality project. It covers test structure, conventions, execution workflow, and standards for ensuring software correctness and model performance across the entire pipeline.

---

## Testing Philosophy

Every code change must include appropriate testing. The goal is to guarantee:

* **Functional correctness** — code behaves consistently and as intended
* **Pipeline reliability** — components integrate without breaking
* **Model performance fidelity** — ML models continue to meet or exceed required metrics

No changes are merged into the `dev` branch without passing all required tests.

---

## Test Directory Structure

All tests are placed in a root-level directory named `tests/`.
Its internal structure mirrors the `src/` directory for clarity and discoverability.

| Directory           | Contains                                             | Purpose                                         |
| ------------------- | ---------------------------------------------------- | ----------------------------------------------- |
| `tests/src/data/`   | Data loaders, parsers, preprocessing tests           | Ensures deterministic and clean data pipelines  |
| `tests/src/models/` | Model components, training routines, inference logic | Validates ML pipeline behavior                  |
| `tests/src/ar/`     | AR tracking, rendering, device IO logic              | Ensures AR features function on target hardware |
| `tests/src/utils/`  | Helper functions, logging, system utilities          | Guarantees stability of shared utilities        |

This mirroring ensures that every file in `src/` has a predictable testing location.

---

## Naming Conventions

To ensure automatic discovery by test runners such as PyTest:

* **Test files** must be prefixed with `test_`

  * Example: `tests/src/models/test_training_logic.py`
* **Test functions** must also be prefixed with `test_`

  * Example:

    ```python
    def test_model_output_shape():
        ...
    ```

Following these conventions ensures CI systems can automatically locate and execute all tests.

---

## Types of Testing

The project relies on three complementary categories of tests.

---

### Unit Tests — Code Integrity

These tests verify small, isolated pieces of logic.

**Focus areas include:**

* Data transformation correctness
* Input validation
* Utility function behavior
* AR component initialization logic

**Execution:**
Automatically run in CI for every Pull Request.

**Tools:**
Primarily **PyTest**.

---

### Integration Tests — Pipeline Flow

Integration tests verify correct coordination between components.

**Typical scenarios include:**

* Full data pipeline: raw → cleaned → processed
* Loading a trained model into AR inference
* Full sequence validation:
  Data Ingestion → Feature Engineering → Model Prediction

**Execution:**
Performed on `dev` before merging into `main`.

---

### ML-Specific Tests — Performance & Stability

These tests target ML behavior, evaluation metrics, and runtime constraints.

| Test Type                  | Purpose                                                  | Standard / Threshold                        |
| -------------------------- | -------------------------------------------------------- | ------------------------------------------- |
| **Data Validation**        | Detect data drift, schema mismatches, missing values     | Feature schema must match baseline          |
| **Sanity Overfit Test**    | Check model correctness by overfitting on a tiny dataset | Must reach near-perfect accuracy/score      |
| **Performance Regression** | Ensure new models do not degrade metrics                 | No > **X%** drop (e.g., 0.5%) from baseline |
| **Inference Latency**      | Validate real-time AR inference speed                    | Must meet required runtime (e.g., ≤ 10 ms)  |

These ensure both code and model performance remain stable and predictable.

---

## Testing Workflow and PR Requirements

A feature branch can only be merged into `dev` if the CI system verifies all of the following:

---

### 1. Code Coverage

All new and modified lines must be covered by **Unit Tests**, meeting the coverage threshold (e.g., 80%).

---

### 2. No Regression

All existing Unit Tests and Integration Tests must pass without introducing new failures.

---

### 3. Model Stability

If a PR affects model code:

* Performance Regression Tests must pass
* Updated metrics must remain within the allowed tolerance

Regression failures block the merge.

---

### 4. Local Verification

Before pushing any branch, developers must run the entire test suite locally:

```bash
pytest --cov=src --cov-report=term-missing tests/
```

This ensures failures are caught before CI evaluation.

---
