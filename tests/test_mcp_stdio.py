#!/usr/bin/env python3
"""Test MCP server through stdio interface to simulate Claude Desktop interaction."""

import asyncio
import json
from pathlib import Path

async def test_mcp_stdio():
    """Test MCP server by sending JSON-RPC messages to its stdio interface."""
    
    # Start the MCP server as a subprocess
    # Use parent of tests directory as cwd (project root)
    project_root = Path(__file__).parent.parent
    proc = await asyncio.create_subprocess_exec(
        "uv", "run", "python", "-m", "vision_rag.mcp_server",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=project_root,
    )
    
    async def read_response():
        """Read a JSON-RPC response from the server."""
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            try:
                return json.loads(line.decode())
            except json.JSONDecodeError:
                continue
        return None
    
    async def send_request(method: str, params: dict = None, request_id: int = 1):
        """Send a JSON-RPC request to the server."""
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params:
            request["params"] = params
        
        message = json.dumps(request) + "\n"
        proc.stdin.write(message.encode())
        await proc.stdin.drain()
        
        return await read_response()
    
    try:
        print("🔧 Starting MCP server via stdio...")
        
        # Wait a bit for server to start
        await asyncio.sleep(2)
        
        # Initialize the connection
        print("\n1️⃣ Sending initialize request...")
        init_response = await send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            },
            request_id=1
        )
        print(f"✅ Initialize response: {json.dumps(init_response, indent=2)}")
        
        # List tools
        print("\n2️⃣ Listing available tools...")
        tools_response = await send_request("tools/list", {}, request_id=2)
        print(f"✅ Tools response:")
        if tools_response and "result" in tools_response:
            tools = tools_response["result"].get("tools", [])
            for tool in tools:
                print(f"   - {tool['name']}")
        
        # Test list_available_datasets
        print("\n3️⃣ Calling list_available_datasets...")
        list_datasets_response = await send_request(
            "tools/call",
            {
                "name": "list_available_datasets",
                "arguments": {}
            },
            request_id=3
        )
        print(f"✅ List datasets response: {json.dumps(list_datasets_response, indent=2)[:500]}...")
        
        # Test preload_dataset with small sample
        print("\n4️⃣ Calling preload_dataset (BloodMNIST, 25 images)...")
        preload_response = await send_request(
            "tools/call",
            {
                "name": "preload_dataset",
                "arguments": {
                    "dataset_name": "BloodMNIST",
                    "split": "train",
                    "max_images": 25,
                    "size": 224
                }
            },
            request_id=4
        )
        
        # Wait for preload to complete (might take a while)
        print("⏳ Waiting for preload to complete...")
        await asyncio.sleep(30)  # Give it time to download and process
        
        print(f"✅ Preload response: {json.dumps(preload_response, indent=2)}")
        
        # Get statistics
        print("\n5️⃣ Getting statistics...")
        stats_response = await send_request(
            "tools/call",
            {"name": "get_statistics", "arguments": {}},
            request_id=5
        )
        print(f"✅ Stats response: {json.dumps(stats_response, indent=2)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        
        # Print stderr output first (to diagnose why server died)
        try:
            stderr_output = await asyncio.wait_for(proc.stderr.read(), timeout=1.0)
            if stderr_output:
                print("\n📝 Server stderr output:")
                print(stderr_output.decode())
        except asyncio.TimeoutError:
            print("⏱️ Timeout reading stderr")
        
        # Terminate only if still running
        if proc.returncode is None:
            proc.terminate()
            await proc.wait()
        else:
            print(f"⚠️ Process already exited with code {proc.returncode}")

if __name__ == "__main__":
    asyncio.run(test_mcp_stdio())
