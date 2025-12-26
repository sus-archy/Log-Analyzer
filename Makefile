# LogMind AI - Makefile
# Commands for development, testing, and running the application

.PHONY: help install setup-models run dev test lint clean smoke venv

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# Default target
help:
	@echo "LogMind AI - Available commands:"
	@echo ""
	@echo "  make install       - Install all dependencies (backend + frontend)"
	@echo "  make setup-models  - Pull required Ollama models"
	@echo "  make run           - Run both backend and frontend"
	@echo "  make backend       - Run backend only"
	@echo "  make frontend      - Run frontend only"
	@echo "  make dev           - Run in development mode with auto-reload"
	@echo "  make test          - Run all tests"
	@echo "  make smoke         - Run smoke tests"
	@echo "  make lint          - Run linting"
	@echo "  make clean         - Clean generated files"
	@echo "  make ingest        - Ingest logs from default folder"
	@echo ""

# Create virtual environment
venv:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating virtual environment..."; \
		python -m venv $(VENV); \
	fi

# Install dependencies
install: venv
	@echo "Installing backend dependencies..."
	$(PIP) install -r apps/api/requirements.txt
	@echo ""
	@echo "Installing frontend dependencies..."
	cd apps/ui && npm install
	@echo ""
	@echo "Installation complete!"
	@echo "Virtual environment created at $(VENV)"

# Setup Ollama models
setup-models:
	@chmod +x scripts/setup_models.sh
	@./scripts/setup_models.sh

# Run backend
backend:
	$(VENV)/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --app-dir apps/api

# Run frontend
frontend:
	cd apps/ui && npm run dev

# Run both (using background process)
run:
	@echo "Starting LogMind AI..."
	@echo "Backend: http://localhost:8000"
	@echo "Frontend: http://localhost:3000"
	@echo ""
	@make backend &
	@sleep 2
	@make frontend

# Development mode with auto-reload
dev:
	@echo "Starting development mode..."
	$(VENV)/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --app-dir apps/api &
	@sleep 2
	cd apps/ui && npm run dev

# Run tests
test:
	@echo "Running backend tests..."
	cd apps/api && $(PYTHON) -m pytest tests/ -v
	@echo ""
	@echo "Running frontend tests..."
	cd apps/ui && npm test

# Smoke tests
smoke:
	@chmod +x scripts/smoke.sh
	@./scripts/smoke.sh

# Lint
lint:
	@echo "Linting backend..."
	cd apps/api && $(PYTHON) -m ruff check app/
	@echo ""
	@echo "Linting frontend..."
	cd apps/ui && npm run lint

# Clean generated files
clean:
	@echo "Cleaning..."
	rm -rf data/logmind.sqlite
	rm -rf data/faiss_index.bin
	rm -rf apps/api/__pycache__
	rm -rf apps/api/app/__pycache__
	rm -rf apps/api/app/**/__pycache__
	rm -rf apps/ui/.next
	rm -rf apps/ui/node_modules/.cache
	@echo "Clean complete!"

# Ingest logs
ingest:
	$(PYTHON) scripts/ingest_folder.py

# Create data directory
init:
	mkdir -p data
	mkdir -p Logs