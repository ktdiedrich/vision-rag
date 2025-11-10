#!/usr/bin/env python3
"""
Startup script for Vision RAG service.

Usage:
    # Start FastAPI service
    python scripts/run_service.py --mode api --port 8001
    
    # Start MCP agent server
    python scripts/run_service.py --mode mcp
    
    # Start both
    python scripts/run_service.py --mode both
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_api_service(host: str = "0.0.0.0", port: int = 8001):
    """Run the FastAPI service."""
    import uvicorn
    from vision_rag.service import app
    
    print(f"🚀 Starting FastAPI service on {host}:{port}")
    print(f"📖 API docs available at http://{host}:{port}/docs")
    
    uvicorn.run(app, host=host, port=port)


async def run_mcp_server():
    """Run the MCP agent server."""
    from vision_rag.mcp_server import main
    
    await main()


def run_both(api_host: str = "0.0.0.0", api_port: int = 8001):
    """Run both API and MCP server."""
    import multiprocessing
    import sys
    
    print("=" * 60, file=sys.stderr)
    print("🔬 Vision RAG Service - Both Modes", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    
    # Start API in separate process
    api_process = multiprocessing.Process(
        target=run_api_service,
        args=(api_host, api_port),
    )
    api_process.start()
    
    # Run MCP server in main process
    try:
        asyncio.run(run_mcp_server())
    except KeyboardInterrupt:
        print("\n⚠️  Shutting down...", file=sys.stderr)
        api_process.terminate()
        api_process.join()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Vision RAG Service Runner")
    parser.add_argument(
        "--mode",
        choices=["api", "mcp", "both"],
        default="api",
        help="Service mode: api (FastAPI), mcp (MCP agent), or both",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="API host address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="API port (default: 8001)",
    )
    
    args = parser.parse_args()
    
    # Only print to stderr for MCP mode to avoid interfering with JSON-RPC
    output = sys.stderr if args.mode == "mcp" else sys.stdout
    
    print("=" * 60, file=output)
    print("🔬 Vision RAG Service", file=output)
    print("=" * 60, file=output)
    
    if args.mode == "api":
        run_api_service(host=args.host, port=args.port)
    elif args.mode == "mcp":
        asyncio.run(run_mcp_server())
    elif args.mode == "both":
        run_both(api_host=args.host, api_port=args.port)


if __name__ == "__main__":
    main()
