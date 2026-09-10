# SENTINEL-WM  -  research / backend / frontend
.DEFAULT_GOAL := help
.PHONY: help install research-run bundle backend-dev backend-test up down docker-build

help:  ## show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## editable-install research + backend into the current venv
	pip install -e ./research
	pip install -e "./backend[dev]"

research-run:  ## full training + benchmark study -> runs/
	cd research && python -m sentinel_wm.research all

bundle:  ## assemble the portable model bundle -> models/
	cd research && python -m sentinel_wm.cli bundle ../models

backend-dev:  ## run the API with autoreload (needs models/)
	cd backend && SENTINEL_WM_MODEL_DIR=../models uvicorn app.main:app --reload

backend-test:  ## backend unit tests (throwaway random bundle, no training needed)
	cd backend && python -m pytest -q

docker-build:  ## build the backend image
	docker build -f backend/Dockerfile -t sentinel-wm-backend .

up:  ## docker compose up (mounts ./models)
	docker compose up --build

down:
	docker compose down
