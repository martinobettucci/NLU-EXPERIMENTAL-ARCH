.DEFAULT_GOAL := help
SHELL := /bin/bash

UV ?= uv
RUN ?= $(UV) run
SEED ?= 42
CONFIG ?= config/benchmark/cpu.yaml
ARCHITECTURES ?= all

.PHONY: help setup lint format typecheck test coverage models-download \
        generate-data validate-data doctors train build-index \
        benchmark-smoke benchmark-text benchmark-audio benchmark-e2e benchmark-full \
        report update-readme reproduce clean

help: ## Liste les cibles disponibles
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup: ## Installe l'environnement complet (CPU uniquement)
	$(UV) sync --all-extras

lint: ## Verifie le style
	$(RUN) ruff check src tests
	$(RUN) ruff format --check src tests

format: ## Applique le formatage
	$(RUN) ruff format src tests
	$(RUN) ruff check --fix src tests

typecheck: ## Verifie les types en mode strict
	$(RUN) mypy

test: ## Tests ne necessitant aucun poids de modele
	$(RUN) pytest -m "not models and not slow"

test-models: ## Tests sur modeles reels (poids requis)
	$(RUN) pytest -m models

coverage: ## Tests avec rapport de couverture
	$(RUN) pytest -m "not models and not slow" --cov --cov-report=term-missing

models-download: ## Telecharge explicitement les poids reels (aucun telechargement implicite ailleurs)
	$(RUN) ivr-bench models download

doctors: ## Genere le catalogue synthetique de praticiens
	$(RUN) ivr-bench doctors generate --seed $(SEED)

generate-data: doctors ## Genere les corpus index/train/validation/test
	$(RUN) ivr-bench data generate --seed $(SEED)
	$(RUN) ivr-bench data validate

validate-data: ## Valide schemas, manifestes et rapport de contamination
	$(RUN) ivr-bench data validate

build-index: ## Construit les index semantiques (embeddings et prototypes)
	$(RUN) ivr-bench index build --config $(CONFIG)

train: ## Entraine les composants qui le necessitent (DIET, FunctionGemma specialise)
	$(RUN) ivr-bench diet train --seed $(SEED)
	$(RUN) ivr-bench functiongemma train --seed $(SEED)

benchmark-smoke: ## Campagne courte, modeles reels, corpus reduit
	$(RUN) ivr-bench benchmark text --config config/benchmark/smoke.yaml --seed $(SEED)

benchmark-text: ## Campagne texte complete
	$(RUN) ivr-bench benchmark text --architectures $(ARCHITECTURES) --config $(CONFIG) --seed $(SEED)

benchmark-audio: ## Campagne audio (francais uniquement)
	$(RUN) ivr-bench benchmark audio --architectures $(ARCHITECTURES) --config $(CONFIG) --seed $(SEED)

benchmark-e2e: ## Campagne bout en bout audio vers action metier
	$(RUN) ivr-bench benchmark e2e --architectures $(ARCHITECTURES) --config $(CONFIG) --seed $(SEED)

benchmark-full: ## Toutes les campagnes
	$(RUN) ivr-bench benchmark all --config $(CONFIG) --seed $(SEED)

report: ## Produit tableaux, graphiques et rapports
	$(RUN) ivr-bench report build

update-readme: ## Met a jour uniquement la zone generee du README
	$(RUN) ivr-bench readme update

reproduce: ## Rejoue la chaine complete depuis un depot propre
	$(RUN) ivr-bench reproduce --config $(CONFIG) --seed $(SEED)

clean: ## Supprime les artefacts non versionnes
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
