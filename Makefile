.PHONY: install poetry ninja submodules test

install: poetry ninja submodules

# 1. Install Poetry if not present
poetry:
	@command -v poetry >/dev/null 2>&1 || (echo "Installing Poetry..." && curl -sSL https://install.python-poetry.org | python3 -)
	@echo "Installing project dependencies..."
	poetry install

# 2. Install Ninja
ninja:
	@command -v ninja >/dev/null 2>&1 || ( \
		echo "Installing Ninja..."; \
		wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip -O /tmp/ninja.zip; \
		sudo unzip -o /tmp/ninja.zip -d /usr/local/bin/; \
		sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force; \
	)
	@echo "Ninja is ready"

# 3. Update git submodules
submodules:
	git submodule update --init --recursive

# 4. Run tests
test:
	export PYTHONPATH=src:libs
	poetry run pytest

