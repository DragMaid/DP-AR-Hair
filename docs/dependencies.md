# Poetry — Python Dependency Management & Packaging

Poetry is a modern tool for Python projects that handles **dependency management**, **virtual environments**, and **packaging** in a clean, declarative way using `pyproject.toml`.

* **Official website:** [https://python-poetry.org](https://python-poetry.org)
* **Source repository:** [https://github.com/python-poetry/poetry](https://github.com/python-poetry/poetry)

---

## Why Use Poetry

Poetry provides a unified, consistent workflow for Python development:

* Manages dependencies through a single `pyproject.toml` file
* Automatically creates and manages virtual environments
* Generates a `poetry.lock` file for reproducible installations
* Integrates packaging, building, and publishing tools
* Supports dependency groups for separating runtime, test, docs, and dev tools
* Provides deterministic dependency resolution

These features make Poetry suitable for both simple apps and large-scale production systems.

---

## Installation

Poetry can be installed using the official installer:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Or through `pipx`:

```bash
pipx install poetry
```

Check the installation:

```bash
poetry --version
```

---

## Getting Started

### Creating a New Project

```bash
poetry new my_project
```

This generates a project structure including:

* `pyproject.toml`
* `my_project/` package folder
* `tests/` directory

### Initializing an Existing Project

If you already have code:

```bash
poetry init
```

You will be guided through creating `pyproject.toml`.

---

## Managing Dependencies

### Adding a Dependency

```bash
poetry add requests
```

Poetry will:

* resolve compatible versions
* update `pyproject.toml`
* install the dependency into the virtual environment
* update the lock file

### Adding Development Dependencies

For packages only used during development:

```bash
poetry add --dev pytest
```

These are excluded from production installs unless explicitly included.

### Removing a Dependency

```bash
poetry remove requests
```

This removes it from both `pyproject.toml` and the lock file.

---

## Dependency Groups

Poetry supports grouping dependencies into categories.

Example in `pyproject.toml`:

```toml
[tool.poetry.group.test]
optional = true

[tool.poetry.group.test.dependencies]
pytest = "^7.0"
```

Install with specific groups:

```bash
poetry install --with test
```

Install everything:

```bash
poetry install --all-groups
```

Skip optional groups:

```bash
poetry install --without docs
```

---

## Installing and Syncing

### Install All Dependencies

```bash
poetry install
```

If `poetry.lock` exists, Poetry installs the exact locked versions.
If it doesn't, Poetry resolves dependencies and creates it.

### Syncing Environments

```bash
poetry sync
```

This ensures your environment matches the lock file **exactly**, removing any extra packages.

---

## Locking Dependencies

Generate or update the lock file:

```bash
poetry lock
```

Force a full regeneration:

```bash
poetry lock --regenerate
```

This is useful when resolving conflicts or updating version constraints.

---

## Updating Dependencies

Update everything allowed by constraints:

```bash
poetry update
```

Update specific packages:

```bash
poetry update numpy pandas
```

If a new version is outside your version spec, update the spec in `pyproject.toml`.

---

## Virtual Environments

Poetry manages virtual environments automatically.

Run a script:

```bash
poetry run python main.py
```

Open a shell:

```bash
poetry shell
```

Poetry will ensure the environment uses the correct interpreter (`python = ">=3.10.5,<3.11"`).

---

## Building Packages

Build your project into a wheel and source distribution:

```bash
poetry build
```

Outputs:

* `dist/my_project-version-py3-none-any.whl`
* `dist/my_project-version.tar.gz`

---

## Publishing Packages

Publish to PyPI:

```bash
poetry publish --build
```

Poetry supports configuration of custom registries in `pyproject.toml` or `poetry config`.

---

## Useful Commands

| Command              | Description                               |
| -------------------- | ----------------------------------------- |
| `poetry show`        | View all installed dependencies           |
| `poetry show --tree` | Display dependency tree                   |
| `poetry self update` | Update Poetry itself                      |
| `poetry config`      | Manage Poetry’s settings                  |
| `poetry export`      | Export dependencies to `requirements.txt` |

---

## Best Practices

* Commit both `pyproject.toml` and `poetry.lock` to version control
* Use dependency groups to separate runtime vs dev/test tools
* Keep version constraints flexible (e.g., `^2.0` instead of pinning exact versions)
* Run `poetry update` periodically to pull security updates
* Use `poetry sync` to ensure consistent environments during CI/CD
* After packaging, test installation in a clean environment to verify dependencies

---

## Troubleshooting

**Dependencies don’t resolve**

* Loosen version constraints or regenerate the lock file
* Use `poetry debug resolve` to view resolution issues

**Poetry not using correct Python version**

* Configure explicitly:

  ```bash
  poetry env use 3.10.5
  ```

**Environment drift**

* Run `poetry sync` to restore consistency

---
