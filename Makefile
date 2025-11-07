.PHONY: help test test-cov up-api up-mcp up-both down status clean install verify examples

# Default target
help:
	@echo "Vision RAG - Available Make Targets"
	@echo "===================================="
	@echo ""
	@echo "Development:"
	@echo "  make install     - Install dependencies with uv"
	@echo "  make verify      - Run verification script"
	@echo "  make test        - Run tests"
	@echo "  make test-cov    - Run tests with coverage report"
	@echo ""
	@echo "Services (detached mode):"
	@echo "  make up-api      - Start FastAPI service in background"
	@echo "  make up-mcp      - Start MCP server in background"
	@echo "  make up-both     - Start both services in background"
	@echo "  make down        - Stop all running services"
	@echo "  make status      - Check status of running services"
	@echo ""
	@echo "Examples:"
	@echo "  make examples    - Run example demonstrations"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean       - Remove cache and temporary files"

# Installation
install:
	uv sync

# Run verification script
verify:
	python scripts/verify.py

# Testing
test:
	uv run pytest tests/

test-cov:
	uv run pytest tests/ --cov=vision_rag --cov-report=term-missing

# Service management (detached mode)
up-api:
	@echo "Starting FastAPI service in background..."
	@nohup python scripts/run_service.py --mode api > logs/api.log 2>&1 & echo $$! > .api.pid
	@sleep 2
	@if ps -p $$(cat .api.pid) > /dev/null 2>&1; then \
		echo "✓ FastAPI service started (PID: $$(cat .api.pid))"; \
		echo "  Logs: tail -f logs/api.log"; \
		echo "  Docs: http://localhost:8001/docs"; \
	else \
		echo "✗ Failed to start FastAPI service"; \
		cat logs/api.log; \
		rm -f .api.pid; \
		exit 1; \
	fi

up-mcp:
	@echo "Starting MCP server in background..."
	@nohup python scripts/run_service.py --mode mcp > logs/mcp.log 2>&1 & echo $$! > .mcp.pid
	@sleep 2
	@if ps -p $$(cat .mcp.pid) > /dev/null 2>&1; then \
		echo "✓ MCP server started (PID: $$(cat .mcp.pid))"; \
		echo "  Logs: tail -f logs/mcp.log"; \
	else \
		echo "✗ Failed to start MCP server"; \
		cat logs/mcp.log; \
		rm -f .mcp.pid; \
		exit 1; \
	fi

up-both:
	@echo "Starting both services in background..."
	@nohup python scripts/run_service.py --mode both > logs/both.log 2>&1 & echo $$! > .both.pid
	@sleep 2
	@if ps -p $$(cat .both.pid) > /dev/null 2>&1; then \
		echo "✓ Both services started (PID: $$(cat .both.pid))"; \
		echo "  Logs: tail -f logs/both.log"; \
		echo "  API Docs: http://localhost:8001/docs"; \
	else \
		echo "✗ Failed to start services"; \
		cat logs/both.log; \
		rm -f .both.pid; \
		exit 1; \
	fi

down:
	@echo "Stopping services..."
	@if [ -f .api.pid ]; then \
		if ps -p $$(cat .api.pid) > /dev/null 2>&1; then \
			kill $$(cat .api.pid) && echo "✓ Stopped FastAPI service"; \
		fi; \
		rm -f .api.pid; \
	fi
	@if [ -f .mcp.pid ]; then \
		if ps -p $$(cat .mcp.pid) > /dev/null 2>&1; then \
			kill $$(cat .mcp.pid) && echo "✓ Stopped MCP server"; \
		fi; \
		rm -f .mcp.pid; \
	fi
	@if [ -f .both.pid ]; then \
		if ps -p $$(cat .both.pid) > /dev/null 2>&1; then \
			kill $$(cat .both.pid) && echo "✓ Stopped both services"; \
		fi; \
		rm -f .both.pid; \
	fi
	@echo "All services stopped"

status:
	@echo "Service Status:"
	@echo "==============="
	@if [ -f .api.pid ]; then \
		if ps -p $$(cat .api.pid) > /dev/null 2>&1; then \
			echo "FastAPI:  ✓ Running (PID: $$(cat .api.pid))"; \
		else \
			echo "FastAPI:  ✗ Not running (stale PID file)"; \
			rm -f .api.pid; \
		fi; \
	else \
		echo "FastAPI:  ✗ Not running"; \
	fi
	@if [ -f .mcp.pid ]; then \
		if ps -p $$(cat .mcp.pid) > /dev/null 2>&1; then \
			echo "MCP:      ✓ Running (PID: $$(cat .mcp.pid))"; \
		else \
			echo "MCP:      ✗ Not running (stale PID file)"; \
			rm -f .mcp.pid; \
		fi; \
	else \
		echo "MCP:      ✗ Not running"; \
	fi
	@if [ -f .both.pid ]; then \
		if ps -p $$(cat .both.pid) > /dev/null 2>&1; then \
			echo "Both:     ✓ Running (PID: $$(cat .both.pid))"; \
		else \
			echo "Both:     ✗ Not running (stale PID file)"; \
			rm -f .both.pid; \
		fi; \
	else \
		echo "Both:     ✗ Not running"; \
	fi

# Examples
examples:
	@echo "Running example demonstrations..."
	@echo ""
	@echo "1. Main demo:"
	python examples/main.py
	@echo ""
	@echo "2. Client demo (requires service running):"
	@echo "   Start service first: make up-api"
	@echo "   Then run: python examples/client_demo.py"

# Cleanup
clean:
	@echo "Cleaning up..."
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf *.egg-info
	rm -rf build
	rm -rf dist
	rm -rf .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✓ Cleaned up cache and temporary files"
