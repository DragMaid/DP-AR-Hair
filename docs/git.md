# Git Usage Standards & Branching Strategy for ML/AR Project

This document defines the **Git workflow**, **branch naming conventions**, and **merge policies** for collaborating on this Machine Learning and Augmented Reality project.

-----

## Global Conventions

  * Use **feature branches** for all work.
  * All changes must go through a **Pull Request (PR)** into `dev`, except for documentation-only branches.
  * The `dev` and `main` branches are **protected**:
      * Cannot be pushed to directly (except for allowed `docs/*` merges to `main`).
      * Must be updated via PR.
  * Every PR **must be validated via CI/CD pipelines** (e.g., model training or AR build) before merge.
  * All code merged into `main`:
      * **Requires a PR.**
      * **Must be reviewed by at least one peer.**
      * Exception: `docs/*` branches can be locally merged and pushed directly to `main` with a merge commit.

-----

## Branch Naming Convention

Branches must follow the pattern:

```
<role>/<content>
```

### Role Prefixes

| Prefix | Purpose |
| :--- | :--- |
| `feat/` | New model features, major algorithm changes, new AR functionality. |
| `model/` | Work specific to model training, evaluation, hyperparameter tuning, or dataset versioning. |
| `infra/` | Changes to deployment infrastructure, training environment setup, CI/CD, cloud resources. |
| `data/` | Data collection, pre-processing scripts, data labeling improvements, or managing data pipelines. |
| `ar/` | Augmented Reality specific code, tracking logic, rendering, or integration with hardware. |
| `docs/` | Documentation updates. |
| `chore/` | Tooling, setup, virtual environment management, or minor dependency updates. |
| `fix/` | Bugfixes (include in relevant scope, e.g., `fix/model-loss-spike`). |

### Examples

| Branch Name | Meaning |
| :--- | :--- |
| `model/new-cnn-architecture` | Implementing a new Convolutional Neural Network architecture. |
| `ar/hand-tracking-optimization` | Optimizing hand tracking performance in the AR module. |
| `data/rebalance-dataset-v2` | Scripts for rebalancing the second version of the training dataset. |
| `infra/aws-sagemaker-config` | Configuration changes for the AWS SageMaker training environment. |
| `feat/add-realtime-inference` | Implementing a new feature for real-time model inference. |

-----

## Pull Request Workflow

1.  **Create a branch** from `dev`:

    ```bash
    git checkout dev
    git pull origin dev
    git checkout -b model/new-training-loop
    ```

2.  **Commit changes** using conventional commit format:

    ```
    feat(model): implement new CNN architecture
    fix(ar): correct coordinate system bug
    chore(infra): update dependency versions
    ```

3.  **Validate the code** before opening PR (using local scripts or commands):

    ```bash
    # Example validation command
    python run_tests.py && python run_lint.py
    ```

4.  **Push your branch** and open a PR into `dev`:

    ```bash
    git push -u origin model/new-training-loop
    ```

5.  Review will ensure:

      * Code passes CI/CD tests and static analysis.
      * Model runs locally and, if applicable, the AR build is stable.
      * Team review and approval are granted.

6.  After approval, the PR can be merged into `dev`.

-----

## Release Process

> Only leads or release engineers should perform this process.

1.  Ensure `dev` is stable, passes all end-to-end tests, and contains the final model weights for release.
2.  Create a PR from `dev` → `main`.
3.  PR must be:
      * Reviewed by at least one team member.
      * Merged using **Squash & Merge**.
4.  Tag release (e.g., `v1.0.0`) and push.

> **Exception:** If the change is **only documentation (`docs/*`)**, you may:
>
>   * Merge locally into `main`.
>   * Push the merge commit directly (no PR required).

-----

## Additional Tips

  * Keep PRs **small and focused** (1 topic or experiment at a time).
  * Always **pull latest `dev`** before creating a new branch.
  * Run the project's **local verification steps** to ensure your environment is working.
  * Add meaningful descriptions and attach relevant **experiment tracking links** (e.g., Weights & Biases, MLflow) to all PRs.

-----

## Summary

| Rule | Required |
| :--- | :--- |
| Branch follows `<role>/<content>` | ✅ Yes |
| PR required to merge to `dev` | ✅ Yes |
| PR required to merge to `main` (non-docs) | ✅ Yes |
| Peer review required before merging to `main` | ✅ Yes |
| `docs/*` can be merged locally to `main` | ✅ Yes (with merge commit) |
| `dev`/`main` are protected | ✅ Yes |
| CI/CD validation before PR merge | ✅ Yes |
| Conventional commits | ✅ Yes |
| Tests and lint must pass | ✅ Yes |
