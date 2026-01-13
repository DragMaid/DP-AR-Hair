.PHONY: install poetry ninja submodules test download

install: poetry ninja submodules download

# Install Poetry if not present
# Will have to manually download python 3.10.5 and set poetry to use it
poetry:
	command -v poetry >/dev/null 2>&1 || \
		echo "Installing Poetry..."; \
		curl -sSL https://install.python-poetry.org \
		| python3 - \
		| grep -o 'export PATH="[^"]*"' \
		| sh
	)

install:


# Install Ninja
ninja:
	command -v ninja >/dev/null 2>&1 || ( \
		echo "Installing Ninja..."; \
		wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip -O /tmp/ninja.zip; \
		sudo unzip -o /tmp/ninja.zip -d /usr/local/bin/; \
		sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force; \
	)
	@echo "Ninja is ready"

# Run tests
test:
	PYTHONPATH=src:libs poetry run pytest

# Download all weights for dataset
download:
	git clone https://huggingface.co/AIRI-Institute/HairFastGAN
	mv HairFastGAN/pretrained_models/ .
	rm -rf HairFastGAN
