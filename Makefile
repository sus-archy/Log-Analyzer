# LogMind AI - Makefile
# Commands for development, testing, and running the application

.PHONY: help install setup-models run dev test lint clean smoke venv train

# Directories
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
SCRIPTS := scripts
TOOLS := scripts/tools

# Default target
help:
	@echo "╔══════════════════════════════════════════════════════════╗"
	@echo "║               LogMind AI - Commands                      ║"
	@echo "╠══════════════════════════════════════════════════════════╣"
	@echo "║  Setup                                                   ║"
	@echo "║    make install       Install all dependencies           ║"
	@echo "║    make setup-models  Pull required Ollama models        ║"
	@echo "║    make init          Create data directories            ║"
	@echo "║                                                          ║"
	@echo "║  Run                                                     ║"
	@echo "║    make run           Run backend and frontend           ║"
	@echo "║    make backend       Run backend only                   ║"
	@echo "║    make frontend      Run frontend only                  ║"
	@echo "║    make dev           Development mode with reload       ║"
	@echo "║                                                          ║"
	@echo "║  Data & ML                                               ║"
	@echo "║    make ingest        Ingest logs from folder            ║"
	@echo "║    make train         Train ML models                    ║"
	@echo "║                                                          ║"
	@echo "║  Testing & Quality                                       ║"
	@echo "║    make test          Run all tests                      ║"
	@echo "║    make smoke         Run smoke tests                    ║"
	@echo "║    make lint          Run linting                        ║"
	@echo "║    make typecheck     Run type checking                  ║"
	@echo "║                                                          ║"
	@echo "║  Maintenance                                             ║"
	@echo "║    make clean         Clean generated files              ║"
	@echo "║    make clean-all     Clean everything including venv    ║"
	@echo "╚══════════════════════════════════════════════════════════╝"
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
	@chmod +x $(TOOLS)/setup_models.sh
	@./$(TOOLS)/setup_models.sh

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
	@chmod +x $(TOOLS)/smoke.sh
	@./$(TOOLS)/smoke.sh

# Type checking
typecheck:
	@echo "Running type checks..."
	cd apps/api && $(PYTHON) -m pyright app/ --project ../../../config/pyrightconfig.json

# Lint
lint:
	@echo "Linting backend..."
	cd apps/api && $(PYTHON) -m ruff check app/
	@echo ""
	@echo "Linting frontend..."
	cd apps/ui && npm run lint

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf data/logmind.sqlite data/logmind.sqlite-*
	rm -rf data/faiss.index data/models
	rm -rf apps/api/__pycache__
	rm -rf apps/api/app/__pycache__
	find apps/api -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf apps/ui/.next
	rm -rf apps/ui/node_modules/.cache
	rm -rf .pytest_cache
	@echo "Clean complete!"

# Clean everything including venv
clean-all: clean
	@echo "Removing virtual environment..."
	rm -rf $(VENV)
	rm -rf apps/ui/node_modules
	@echo "Full clean complete!"

# Train ML models
train:
	@echo "Training ML models..."
	$(PYTHON) $(SCRIPTS)/train_models.py

# Ingest logs
ingest:
	$(PYTHON) $(SCRIPTS)/ingest_folder.py

# Create data directory
init:
	mkdir -p data
	mkdir -p data/models
	mkdir -p Logs
	@echo "Data directories created!"