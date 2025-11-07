.PHONY: help test test-cov up-api up-mcp up-both down status clean clean-rag clean-images install verify examples example-main example-client demo demo-simple demo-full demo-multi

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
	@echo "  make examples       - List available example scripts"
	@echo "  make example-main   - Run main demo script"
	@echo "  make example-client - Run client demo (requires service)"
	@echo ""
	@echo "Demonstrations:"
	@echo "  make demo           - List available demonstrations"
	@echo "  make demo-simple    - Run simple visualization demo (fast, 50 images)"
	@echo "  make demo-full      - Run full visualization demo (1000 images)"
	@echo "  make demo-multi     - Run multi-dataset demonstration"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         - Remove cache and temporary files"
	@echo "  make clean-rag     - Remove RAG database directories"
	@echo "  make clean-images  - Remove image store directories"

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
	@mkdir -p logs
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
	@mkdir -p logs
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
	@mkdir -p logs
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
	@echo "Available Example Scripts:"
	@echo "=========================="
	@echo ""
	@echo "1. Main Demo (standalone):"
	@echo "   make example-main"
	@echo "   - Downloads MedMNIST dataset"
	@echo "   - Encodes images with CLIP"
	@echo "   - Stores in ChromaDB"
	@echo "   - Performs similarity search"
	@echo ""
	@echo "2. Client Demo (requires service):"
	@echo "   make up-api          # Start service first"
	@echo "   make example-client  # Then run client demo"
	@echo "   - Tests FastAPI endpoints"
	@echo "   - Tests MCP agent server"
	@echo "   - Demonstrates service integration"

example-main:
	@echo "Running main demo..."
	python examples/main.py

example-client:
	@echo "Running client demo..."
	@if curl -s http://localhost:8001/health > /dev/null 2>&1; then \
		echo "✓ Service is already running"; \
		python examples/client_demo.py; \
	else \
		echo "⚠️  Service not running, starting it..."; \
		$(MAKE) up-api; \
		echo ""; \
		echo "Waiting for service to be ready..."; \
		sleep 3; \
		python examples/client_demo.py; \
	fi

# Demonstrations
demo:
	@echo "Available Demonstrations"
	@echo "======================="
	@echo ""
	@echo "Simple Visualization Demo (fast, 50 images):"
	@echo "   make demo-simple"
	@echo "   - Quick demonstration of core functionality"
	@echo "   - Creates 5 visualization files in simple_visualizations/"
	@echo "   - Perfect for understanding basic concepts"
	@echo ""
	@echo "Full Visualization Demo (1000 images):"
	@echo "   make demo-full"
	@echo "   - Comprehensive demonstration with full analysis"
	@echo "   - Creates 9 visualization files in visualizations/"
	@echo "   - Includes embedding space analysis (t-SNE)"
	@echo "   - Shows multiple search scenarios"
	@echo ""
	@echo "Multi-Dataset Demo:"
	@echo "   make demo-multi"
	@echo "   - Demonstrates using multiple MedMNIST datasets"
	@echo "   - Compares PathMNIST and ChestMNIST"
	@echo "   - Shows dataset configuration usage"

demo-simple:
	@echo "Running simple visualization demo..."
	@echo "This will create visualizations in demonstrations/simple_visualizations/"
	cd demonstrations && uv run python simple_visualization_example.py

demo-full:
	@echo "Running full visualization demo..."
	@echo "This will create visualizations in demonstrations/visualizations/"
	@echo "⚠️  This demo uses 1000 images and may take a few minutes..."
	cd demonstrations && uv run python demo_with_visualization.py

demo-multi:
	@echo "Running multi-dataset demonstration..."
	cd demonstrations && uv run python multi_dataset_demo.py

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

clean-rag:
	@echo "Removing RAG database directories..."
	@if ls chroma_db_* > /dev/null 2>&1; then \
		rm -rf chroma_db_*; \
		echo "✓ Removed all chroma_db_* directories"; \
	else \
		echo "✓ No chroma_db_* directories found"; \
	fi
	@if ls demonstrations/chroma_db_* > /dev/null 2>&1; then \
		rm -rf demonstrations/chroma_db_*; \
		echo "✓ Removed demonstrations/chroma_db_* directories"; \
	else \
		echo "✓ No demonstrations/chroma_db_* directories found"; \
	fi

clean-images:
	@echo "Removing image store directories..."
	@if ls image_store_* > /dev/null 2>&1; then \
		rm -rf image_store_*; \
		echo "✓ Removed all image_store_* directories"; \
	else \
		echo "✓ No image_store_* directories found"; \
	fi
	@if ls demonstrations/image_store_* > /dev/null 2>&1; then \
		rm -rf demonstrations/image_store_*; \
		echo "✓ Removed demonstrations/image_store_* directories"; \
	else \
		echo "✓ No demonstrations/image_store_* directories found"; \
	fi
