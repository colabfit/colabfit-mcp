.PHONY: help setup build start stop restart logs clean test

# Load environment variables from .env if it exists
-include .env
export

# Set defaults
COLABFIT_DATA_ROOT ?= ./colabfit_data
USER_ID ?= $(shell id -u)
GROUP_ID ?= $(shell id -g)

help:
	@echo "ColabFit MCP - Available Commands"
	@echo "=================================="
	@echo "  make setup      - Create data directories and .env file"
	@echo "  make build      - Build Docker images with current user ID"
	@echo "  make start      - Start all services"
	@echo "  make stop       - Stop all services"
	@echo "  make restart    - Restart all services"
	@echo "  make logs       - Follow container logs (Ctrl+C to exit)"
	@echo "  make clean      - Stop services and remove containers"
	@echo "  make test       - Check system status"
	@echo ""
	@echo "First time setup:"
	@echo "  1. Copy example.env to .env and customize if needed"
	@echo "  2. Run: make setup && make build && make start"
	@echo ""
	@echo "Current configuration:"
	@echo "  Data directory: ${COLABFIT_DATA_ROOT}"
	@echo "  User ID:        ${USER_ID}"
	@echo "  Group ID:       ${GROUP_ID}"

setup:
	@echo "Setting up ColabFit MCP..."
	@if [ ! -f .env ]; then \
		cp example.env .env; \
		echo "✓ Created .env file from example.env"; \
		echo "  → Edit .env to customize settings"; \
	else \
		echo "✓ .env file already exists"; \
	fi
	@echo "Creating data directories with correct ownership..."
	@mkdir -p "${COLABFIT_DATA_ROOT}/datasets" "${COLABFIT_DATA_ROOT}/models"
	@echo "✓ Created data directories in: ${COLABFIT_DATA_ROOT}"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Review/edit .env file if needed"
	@echo "  2. Run: make build"
	@echo "  3. Run: make start"

fix-permissions:
	@echo "Fixing permissions on existing data directories..."
	@docker run --rm -v "${PWD}/${COLABFIT_DATA_ROOT}:/data" alpine sh -c "chown -R ${USER_ID}:${GROUP_ID} /data && chmod -R 755 /data"
	@echo "✓ Permissions fixed"

build:
	@echo "Building Docker images with USER_ID=${USER_ID} GROUP_ID=${GROUP_ID}..."
	USER_ID=${USER_ID} GROUP_ID=${GROUP_ID} docker compose build
	@echo "✓ Build complete"

start: setup
	@echo "Starting ColabFit MCP services..."
	USER_ID=${USER_ID} GROUP_ID=${GROUP_ID} docker compose up -d
	@echo "✓ Services started"
	@echo ""
	@echo "View logs with: make logs"
	@echo "Check status with: make test"

stop:
	@echo "Stopping services..."
	docker compose down
	@echo "✓ Services stopped"

restart: stop start

logs:
	@echo "Following container logs (Ctrl+C to exit)..."
	@echo "=========================================="
	docker compose logs -f server

clean:
	@echo "Cleaning up..."
	docker compose down -v
	@echo "✓ Containers and volumes removed"
	@echo "Note: Data in ${COLABFIT_DATA_ROOT} is preserved"

test:
	@echo "Checking ColabFit MCP status..."
	@echo "================================"
	@docker compose ps
	@echo ""
	@echo "Testing MCP connection..."
	@docker compose exec -T server python3 -c "import colabfit_mcp; print('✓ MCP module loaded successfully')" 2>/dev/null || echo "✗ MCP not running or not ready"
