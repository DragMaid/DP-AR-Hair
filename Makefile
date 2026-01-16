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

# Install Ninja
ninja:
	command -v ninja >/dev/null 2>&1 || ( \
		echo "Installing Ninja..."; \
		wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip -O /tmp/ninja.zip; \
		sudo unzip -o /tmp/ninja.zip -d /usr/local/bin/; \
		sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force; \
	)
	@echo "Ninja is ready"

init-roots = PYTHONPATH=src:libs 
# Run tests
test:
	$(init-roots) poetry run pytest

# Download all weights for dataset
download:
	git clone https://huggingface.co/AIRI-Institute/HairFastGAN
	mv HairFastGAN/pretrained_models/ .
	rm -rf HairFastGAN

backend-docker = src/manager/docker-compose.yml
backend-dir = src.manager
db-dir = ./src/manager/db

dbup:
	docker compose -f $(backend-docker) up db

dbclear:
	docker compose -f $(backend-docker) down -v db

backend:
	$(init-roots) poetry run uvicorn $(backend-dir).server:app --host 0.0.0.0 --port 8000 --reload

nginx:
	docker compose -f $(backend-docker) down -v nginx
	docker compose -f $(backend-docker) up nginx

seed:
	$(init-roots) poetry run python -m $(backend-dir).seed

seeddebug:
	$(init-roots) poetry run python -m $(backend-dir).seed -d

cron:
	docker compose -f $(backend-docker) up cronjob

migrate:
	dbmate -d $(db-dir)/migrations -s $(db-dir)schema.sql migrate

tui:
	$(init-roots) poetry run python -m $(backend-dir).tui

worker:
	$(init-roots) poetry run python -m $(backend-dir).worker

make dbinsert:
	$(init-roots) poetry run python -m $(backend-dir).inserter
