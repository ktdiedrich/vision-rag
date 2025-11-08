# Vision RAG MCP Server

The Vision RAG system provides a Model Context Protocol (MCP) server for AI agent communication.

## What is MCP?

The Model Context Protocol (MCP) is an open protocol that enables seamless integration between AI applications and external data sources. MCP servers expose tools that AI agents can discover and call.

## Available Tools

The Vision RAG MCP server provides 5 tools:

### 1. `search_similar_images`
Search for visually similar medical images using CLIP embeddings.

**Parameters:**
- `image_base64` (string, required): Base64 encoded image to search for
- `n_results` (integer, optional): Number of similar images to return (default: 5)

**Returns:** Search results with distances, metadata, and human-readable labels

### 2. `search_by_label`
Search for medical images by organ label.

**Parameters:**
- `label` (integer, required): Organ label (0-10)
  - 0: bladder
  - 1: femur-left
  - 2: femur-right
  - 3: heart
  - 4: kidney-left
  - 5: kidney-right
  - 6: liver
  - 7: lung-left
  - 8: lung-right
  - 9: pancreas
  - 10: spleen
- `n_results` (integer, optional): Limit on number of results
- `return_images` (boolean, optional): If true, return base64 encoded images (default: false)

**Returns:** Images with matching label, optionally including base64 encoded image data

### 3. `add_image`
Add a medical image to the RAG store.

**Parameters:**
- `image_base64` (string, required): Base64 encoded image to add
- `metadata` (object, optional): Optional metadata (e.g., label, patient_id)

**Returns:** Status, assigned ID, and total embeddings count

### 4. `get_statistics`
Get statistics about the vision RAG store.

**Parameters:** None

**Returns:**
- `total_embeddings`: Number of images indexed
- `total_images`: Number of images stored on disk
- `collection_name`: ChromaDB collection name
- `persist_directory`: Database directory
- `image_store_directory`: Image files directory
- `encoder_model`: CLIP model being used
- `embedding_dimension`: Vector dimension

### 5. `list_available_labels`
List all available organ labels with human-readable names.

**Parameters:** None

**Returns:**
- `labels`: Dictionary mapping label IDs to names
- `count`: Total number of labels

## Running the MCP Server

### Option 1: Using Make (Recommended)

```bash
# Start MCP server in background
make up-mcp

# Check status
make status

# Stop server
make down
```

### Option 2: Direct Python Execution

```bash
# Run MCP server
python scripts/run_service.py --mode mcp

# Or using uv
uv run python scripts/run_service.py --mode mcp
```

### Option 3: Run Both API and MCP

```bash
# Start both FastAPI and MCP server
make up-both

# Check status
make status

# Stop all services
make down
```

## Using the MCP Server

### From Claude Desktop

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "vision-rag": {
      "command": "python",
      "args": [
        "/path/to/vision-rag/scripts/run_service.py",
        "--mode",
        "mcp"
      ],
      "env": {
        "VISION_RAG_DATASET": "OrganSMNIST",
        "VISION_RAG_IMAGE_SIZE": "224"
      }
    }
  }
}
```

### From Python Client

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    # Connect to MCP server
    server_params = StdioServerParameters(
        command="python",
        args=["scripts/run_service.py", "--mode", "mcp"],
        env=None
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize connection
            await session.initialize()
            
            # List available tools
            tools = await session.list_tools()
            print(f"Available tools: {[tool.name for tool in tools.tools]}")
            
            # Get statistics
            result = await session.call_tool("get_statistics", {})
            print(f"Statistics: {result}")
            
            # Search by label
            result = await session.call_tool("search_by_label", {
                "label": 3,  # heart
                "n_results": 5,
                "return_images": True
            })
            print(f"Found {result['count']} heart images")

if __name__ == "__main__":
    asyncio.run(main())
```

### From Other AI Agents

Any MCP-compatible AI agent can discover and use the Vision RAG tools. The server uses stdio transport for communication.

## Configuration

The MCP server uses these directories:
- **Database**: `./chroma_db_mcp` (ChromaDB vector store)
- **Images**: `./image_store_mcp` (Saved image files)
- **Logs**: `./logs/mcp.log` (When run in background)

Environment variables (optional):
- `VISION_RAG_DATASET`: Dataset to use (default: "OrganSMNIST")
- `VISION_RAG_IMAGE_SIZE`: Image resize dimension (default: 224)
- `VISION_RAG_MEDMNIST_SIZE`: MedMNIST download size (default: 224)
- `VISION_RAG_CLIP_MODEL`: CLIP model name (default: "clip-ViT-B-32")

## Architecture

```
AI Agent (Claude, custom agent, etc.)
    ↓
MCP Protocol (stdio transport)
    ↓
Vision RAG MCP Server
    ↓
├── CLIP Encoder (sentence-transformers)
├── ChromaDB (vector store)
├── Image Store (file system)
└── MedMNIST Data Loader
```

## Testing

Run MCP server tests:
```bash
uv run pytest tests/test_mcp_server.py -v
```

47 comprehensive tests cover:
- Tool registration and discovery
- Image search (similar and by label)
- Image addition with metadata
- Statistics retrieval
- Label listing
- Error handling
- Concurrent operations
- Image resizing
- Base64 encoding/decoding

## Comparison: MCP vs FastAPI

| Feature | MCP Server | FastAPI Server |
|---------|-----------|----------------|
| **Protocol** | MCP (stdio) | REST (HTTP) |
| **Use Case** | AI agent integration | Web applications, direct HTTP calls |
| **Port** | None (stdio) | 8001 |
| **Discovery** | Auto via MCP | Manual via OpenAPI docs |
| **Image Return** | Base64 in response | Base64 or file upload |
| **Stateful** | Yes | Yes |
| **Best For** | Agent-to-agent communication | Human/app API access |

## Troubleshooting

### Server won't start
```bash
# Check Python version (requires 3.12+)
python --version

# Install dependencies
uv sync

# Check for errors
python scripts/run_service.py --mode mcp
```

### Can't connect from Claude
- Verify path in `claude_desktop_config.json` is absolute
- Check Python is in PATH
- Restart Claude Desktop after config changes

### Tools not appearing
- Ensure server is running: `make status`
- Check logs: `tail -f logs/mcp.log`
- Verify MCP SDK is installed: `uv sync`

## Further Reading

- [MCP Documentation](https://modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Vision RAG API Documentation](./API.md)
