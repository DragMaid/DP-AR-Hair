# =========================
# Bootstrap / tooling
# =========================

.PHONY: bootstrap bootstrap-tools bootstrap-models \
        tool-poetry tool-ninja tool-dbmate

bootstrap: bootstrap-tools bootstrap-models

bootstrap-tools: tool-poetry tool-ninja tool-dbmate

bootstrap-models: models-download

# ---- Tools ----

tool-poetry:
	command -v poetry >/dev/null 2>&1 || ( \
		echo "Installing Poetry..."; \
		curl -sSL https://install.python-poetry.org | python3 - \
	)

tool-ninja:
	command -v ninja >/dev/null 2>&1 || ( \
		echo "Installing Ninja..."; \
		wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip -O /tmp/ninja.zip; \
		sudo unzip -o /tmp/ninja.zip -d /usr/local/bin/; \
		sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force; \
	)
	@echo "Ninja is ready"

tool-dbmate:
	command -v dbmate >/dev/null 2>&1 || ( \
		echo "Installing dbmate..."; \
		sudo curl -fsSL -o /usr/local/bin/dbmate \
			https://github.com/amacneil/dbmate/releases/latest/download/dbmate-linux-amd64; \
		sudo chmod +x /usr/local/bin/dbmate; \
	)

# =========================
# Global configuration
# =========================

COMPOSE_FILE := src/manager/docker-compose.yml
COMPOSE      := docker compose -f $(COMPOSE_FILE)

BACKEND_PKG  := src.manager
DB_DIR       := src/manager/db

POETRY_RUN   := poetry run python -m
INIT_ROOT    := PYTHONPATH=src:libs

# =========================
# Safety
# =========================

.PHONY: help \
        stack-up stack-down stack-destroy \
        svc-backend svc-nginx svc-cron \
        db-up db-down db-reset db-migrate \
        seed seed-debug \
        worker tui db-insert

# =========================
# Help
# =========================

help:
	@echo ""
	@echo "Available targets:"
	@echo ""
	@echo "Stack:"
	@echo "  stack-up           Start all services"
	@echo "  stack-down         Stop all services"
	@echo "  stack-destroy      Stop services AND remove volumes (DANGEROUS)"
	@echo ""
	@echo "Services:"
	@echo "  svc-backend        Start backend service"
	@echo "  svc-nginx          Recreate nginx service"
	@echo "  svc-cron           Start cron job service"
	@echo ""
	@echo "Database:"
	@echo "  db-up              Start database"
	@echo "  db-down            Stop database"
	@echo "  db-reset           Drop DB volumes and restart (DANGEROUS)"
	@echo "  db-migrate         Run dbmate migrations"
	@echo ""
	@echo "Dev tools:"
	@echo "  seed               Seed database"
	@echo "  seed-debug         Seed database (debug)"
	@echo "  worker             Run worker"
	@echo "  tui                Run TUI"
	@echo "  db-insert          Run inserter"
	@echo ""

# =========================
# Stack lifecycle
# =========================

stack-up:
	$(COMPOSE) up -d db
	$(COMPOSE) up -d migration
	$(COMPOSE) up -d backend nginx cronjob

stack-down:
	$(COMPOSE) down

stack-destroy:
	$(COMPOSE) down -v

# =========================
# Services
# =========================

svc-backend:
	$(COMPOSE) up backend

svc-nginx:
	$(COMPOSE) down nginx
	$(COMPOSE) up nginx

svc-cron:
	$(COMPOSE) up cronjob

# =========================
# Database
# =========================

db-up:
	$(COMPOSE) up db

db-down:
	$(COMPOSE) down db

db-reset:
	@echo "⚠️  This will DELETE database volumes."
	@read -p "Continue? [y/N] " ans; \
	if [ "$$ans" = "y" ]; then \
		$(COMPOSE) down -v db; \
	else \
		echo "Aborted."; \
	fi

db-migrate:
	dbmate -d $(DB_DIR)/migrations -s $(DB_DIR)/schema.sql --env-file "./src/manager/.env" migrate

# =========================
# Dev / tooling
# =========================

seed:
	$(INIT_ROOT) $(POETRY_RUN) $(BACKEND_PKG).seed

seed-debug:
	$(INIT_ROOT) $(POETRY_RUN) $(BACKEND_PKG).seed -d

worker:
	$(INIT_ROOT) $(POETRY_RUN) $(BACKEND_PKG).worker

tui-install:
	@if command -v poetry >/dev/null 2>&1; then \
		poetry config virtualenvs.in-project true; \
		poetry install --with dqs; \
	elif command -v pip >/dev/null 2>&1; then \
		pip install -r src/manager/client/requirements.txt; \
	else \
		echo "Error: neither poetry nor pip found in PATH" >&2; \
		exit 1; \
	fi

tui:
	@if command -v poetry >/dev/null 2>&1; then \
		$(INIT_ROOT) $(POETRY_RUN) $(BACKEND_PKG).tui; \
	elif command -v python >/dev/null 2>&1; then \
		$(INIT_ROOT) python -m $(BACKEND_PKG).tui; \
	else \
		echo "Error: neither poetry nor python found in PATH" >&2; \
		exit 1; \
	fi


db-insert:
	$(INIT_ROOT) $(POETRY_RUN) $(BACKEND_PKG).utils --action insert

db-cache:
	$(INIT_ROOT) $(POETRY_RUN) $(BACKEND_PKG).utils --action retrieve --cache auto


# =========================
# Dataset for hosting
# =========================

ASSET_DIR := ./src/manager/assets
REF_ZIP := $(ASSET_DIR)/celebahq-resized-256x256.zip
REF_TMP := $(ASSET_DIR)/reference_images_tmp
REF_OUT := $(ASSET_DIR)/reference_images

dataset-download: tool-download download-prepare ref-download drive-download cache-download

tool-download:
	pip install gdown
	sudo apt install -y unrar

download-prepare:
	mkdir -p $(ASSET_DIR)
	mkdir -p $(ASSET_DIR)/driving_images
	mkdir -p $(REF_OUT)

ref-download:
	@if [ ! -f "$(REF_ZIP)" ]; then \
		echo "[INFO] Downloading CelebA-HQ dataset..."; \
		curl -L -o "$(REF_ZIP)" \
			https://www.kaggle.com/api/v1/datasets/download/badasstechie/celebahq-resized-256x256; \
	else \
		echo "[INFO] Zip already exists, skipping download"; \
	fi
	@rm -rf "$(REF_TMP)"
	@mkdir -p "$(REF_TMP)"
	@unzip -q "$(REF_ZIP)" -d "$(REF_TMP)"
	@mkdir -p "$(REF_OUT)"
	@rsync -a "$(REF_TMP)/celeba_hq_256/" "$(REF_OUT)/"
	@rm -rf "$(REF_TMP)"

drive-download:
	@if [ ! -d "$(ASSET_DIR)/driving_images" ]; then \
		echo "[INFO] Downloading driving images..."; \
		gdown --fuzzy https://drive.google.com/file/d/1ZV3pdgHbTpToFesBbvns_mk-yZrimYVP/view -O "$(ASSET_DIR)/driving_images.rar"; \
	else \
		echo "[INFO] Driving images already exist, skipping"; \
	fi
	@unrar e "$(ASSET_DIR)/driving_images.rar" "$(ASSET_DIR)/driving_images/"; \

cache-download:
	# TODO: set this fucking link again
	# TODO: retreive this file from the final db instead
	gdown --fuzzy https://drive.google.com/file/d/1FegzETY0U0iMuRs74B7_gTCzDUlka9a0/view -O "$(ASSET_DIR)/"
