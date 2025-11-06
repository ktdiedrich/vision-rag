# Vision RAG Service Quick Start

## 🚀 Starting the Service

### Option 1: FastAPI REST Service
```bash
python run_service.py --mode api --port 8001
```
- API Docs: http://localhost:8001/docs
- Health: http://localhost:8001/health

### Option 2: MCP Agent Server
```bash
python run_service.py --mode mcp
```

### Option 3: Both Services
```bash
python run_service.py --mode both
```

## 📡 API Endpoints

### Health Check
```bash
curl http://localhost:8001/health
```

### Search Similar Images
```bash
curl -X POST http://localhost:8001/search \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "iVBORw0KGgoAAAANSUhEUg...",
    "n_results": 5
  }'
```

### Search by Label
```bash
curl -X POST http://localhost:8001/search/label \
  -H "Content-Type: application/json" \
  -d '{
    "label": 6,
    "n_results": 10
  }'
```

### Add Image
```bash
curl -X POST http://localhost:8001/add \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "iVBORw0KGgoAAAANSUhEUg...",
    "metadata": {"label": 6, "patient_id": "P001"}
  }'
```

### Get Statistics
```bash
curl http://localhost:8001/stats
```

## 🤖 MCP Agent Communication

### Python Client
```python
from vision_rag.mcp_server import VisionRAGMCPServer

# Initialize server
server = VisionRAGMCPServer()

# Get available tools
tools = server.get_tool_definitions()

# Search similar images
result = await server.handle_tool_call(
    "search_similar_images",
    {
        "image_base64": base64_image,
        "n_results": 5
    }
)

# Search by label
result = await server.handle_tool_call(
    "search_by_label",
    {"label": 6, "n_results": 10}
)

# Add image
result = await server.handle_tool_call(
    "add_image",
    {
        "image_base64": base64_image,
        "metadata": {"label": 3}
    }
)

# Get statistics
result = await server.handle_tool_call("get_statistics", {})

# List available labels
result = await server.handle_tool_call("list_available_labels", {})
```

## 🏥 Organ Labels

| Label | Organ |
|-------|-------|
| 0 | bladder |
| 1 | femur-left |
| 2 | femur-right |
| 3 | heart |
| 4 | kidney-left |
| 5 | kidney-right |
| 6 | liver |
| 7 | lung-left |
| 8 | lung-right |
| 9 | pancreas |
| 10 | spleen |

## 🧪 Testing

Run the demo client:
```bash
PYTHONPATH=/home/ktdiedrich/code/vision-rag uv run python examples/client_demo.py
```

Run unit tests:
```bash
uv run pytest tests/
```

Run with coverage:
```bash
uv run pytest tests/ --cov=vision_rag --cov-report=term-missing
```

## 📊 Example Workflow

1. Start the service:
   ```bash
   python run_service.py --mode api
   ```

2. Add medical images:
   ```python
   import httpx
   import base64
   
   async with httpx.AsyncClient() as client:
       response = await client.post(
           "http://localhost:8001/add",
           json={"image_base64": img_b64, "metadata": {"label": 6}}
       )
   ```

3. Search for similar images:
   ```python
   response = await client.post(
       "http://localhost:8001/search",
       json={"image_base64": query_img_b64, "n_results": 5}
   )
   ```

4. Filter by organ type:
   ```python
   response = await client.post(
       "http://localhost:8001/search/label",
       json={"label": 6, "n_results": 10}
   )
   ```
