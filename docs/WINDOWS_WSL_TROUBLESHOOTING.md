# Windows/WSL MCP Server Troubleshooting Guide

## Summary

This guide helps diagnose and fix issues when running the Vision RAG MCP server on WSL (Windows Subsystem for Linux) with Claude Desktop on Windows.

## Quick Diagnostic

Run the diagnostic tool to check your setup:

```bash
cd /path/to/vision-rag
uv run python scripts/test_mcp_windows.py
```

This will check:
- ✅ Environment and paths
- ✅ Dataset file integrity
- ✅ MCP server functionality
- ✅ Windows/WSL compatibility

## Common Issues and Solutions

### 1. "File is not a zip file" Error

**Problem**: Corrupted dataset files (`.npz` files)

**Symptoms**:
```
Error preloading dataset: File is not a zip file
```

**Solution**:
1. Find corrupted files:
   ```bash
   cd vision-rag
   uv run python -c "
   import numpy as np
   from pathlib import Path
   
   for f in Path('data').glob('*.npz'):
       try:
           np.load(str(f)).close()
           print(f'✅ {f.name}')
       except:
           print(f'❌ {f.name} - CORRUPTED')
   "
   ```

2. Remove corrupted files:
   ```bash
   rm data/<corrupted_file>.npz
   ```

3. The datasets will be re-downloaded automatically on next use.

**Root Cause**: Interrupted downloads, disk errors, or file system corruption.

### 2. Claude Desktop Can't Connect to MCP Server

**Problem**: Claude Desktop on Windows can't communicate with MCP server on WSL

**Check Claude Desktop Configuration**:

The config file should be at:
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- Path: `C:\Users\<username>\AppData\Roaming\Claude\claude_desktop_config.json`

**Correct configuration**:
```json
{
  "mcpServers": {
    "vision-rag": {
      "command": "wsl",
      "args": [
        "-d", "Ubuntu",
        "--cd", "/home/<username>/code/medical-research-workspace/vision-rag",
        "--", "bash", "-c",
        "source ~/.bashrc && uv run python -m vision_rag.mcp_server"
      ]
    }
  }
}
```

**Key points**:
- Use `wsl` command to launch into WSL
- Specify distribution with `-d Ubuntu` (or your WSL distro name)
- Use `--cd` to set working directory to WSL path
- Source `.bashrc` to get proper environment (uv, Python, etc.)

### 3. Permission Errors

**Problem**: WSL can't access files or directories

**Solution**:
1. Ensure you're on WSL filesystem (not `/mnt/c/`):
   ```bash
   pwd  # Should show /home/... not /mnt/c/...
   ```

2. Check file permissions:
   ```bash
   ls -la data/
   ls -la chroma_db_mcp/
   ls -la image_store_mcp/
   ```

3. Fix permissions if needed:
   ```bash
   chmod -R u+rw data/ chroma_db_mcp/ image_store_mcp/
   ```

### 4. Multi-Label Dataset Issues

**Problem**: Some datasets (ChestMNIST, etc.) have multi-label classifications

**Fixed**: Labels are now automatically handled:
- Single-label datasets: Stored as `int`
- Multi-label datasets: Stored as JSON string (e.g., `"[0, 0, 1, 0, ...]"`)

**No action needed** - the server handles this automatically.

### 5. Slow Performance

**Problem**: Operations are slower on WSL than native Linux

**Solutions**:
1. **Use WSL2** (not WSL1):
   ```powershell
   # In PowerShell (Windows)
   wsl --list --verbose
   # Should show VERSION 2
   ```

2. **Store data on WSL filesystem** (not Windows filesystem):
   - ✅ Good: `/home/username/code/vision-rag/`
   - ❌ Slow: `/mnt/c/Users/username/code/vision-rag/`

3. **Increase WSL memory** (`.wslconfig` file):
   ```ini
   # C:\Users\<username>\.wslconfig
   [wsl2]
   memory=8GB
   processors=4
   ```

### 6. Claude Desktop Logs

**Where to find logs**:

1. **MCP Server logs** (stderr):
   - Check Claude Desktop's MCP server output
   - Look for stderr messages with emoji indicators (🔄, ✅, ❌)

2. **Test manually**:
   ```bash
   cd vision-rag
   uv run python -m vision_rag.mcp_server
   # Server will wait for JSON-RPC input on stdin
   # stderr will show startup messages
   ```

## Testing Without Claude Desktop

You can test the MCP server independently:

### Test 1: Direct MCP Server Test
```bash
cd vision-rag
uv run python -c "
import asyncio
from vision_rag.mcp_server import VisionRAGMCPServer

async def test():
    server = VisionRAGMCPServer()
    
    # List datasets
    datasets = await server.list_available_datasets()
    print(f'Datasets: {datasets[\"count\"]}')
    
    # Preload small dataset
    result = await server.preload_dataset(
        dataset_name='PneumoniaMNIST',
        split='train',
        max_images=10,
        size=28
    )
    print(f'Preload result: {result}')

asyncio.run(test())
"
```

### Test 2: MCP Protocol Test
```bash
cd vision-rag
uv run python test_mcp_stdio.py
```

This simulates how Claude Desktop communicates with the server via JSON-RPC.

### Test 3: Diagnostic Test
```bash
cd vision-rag
uv run python scripts/test_mcp_windows.py
```

Comprehensive check of environment, files, and functionality.

## Verification Checklist

Before asking Claude Desktop to use the MCP server:

- [ ] Diagnostic test passes (`test_mcp_windows.py`)
- [ ] No corrupted dataset files
- [ ] Working directory is on WSL filesystem (not `/mnt/c/`)
- [ ] Claude Desktop config uses correct WSL paths
- [ ] ChromaDB and image store directories exist
- [ ] Can manually run server: `uv run python -m vision_rag.mcp_server`

## Expected Behavior

When working correctly:

1. **Server startup** (stderr):
   ```
   🤖 Vision RAG MCP Server Starting...
   📊 Total embeddings: 0
   🔧 Available tools: ['search_similar_images', 'search_by_label', ...]
   ✅ MCP Server ready for agent communication
   ```

2. **Preload operation** (stderr):
   ```
   🔄 Preloading PneumoniaMNIST (train split)...
   📦 Loaded 4708 images from PneumoniaMNIST
   ✂️  Limited to 100 images
   💾 Saved 100 images to disk
   🧠 Encoding images with CLIP...
   ✅ Encoded 100 images
   📊 Adding 100 embeddings to RAG store...
   ✅ Successfully loaded 100 images from PneumoniaMNIST (train)
   📊 Total embeddings in store: 100
   ```

3. **Claude Desktop**: Should see Vision RAG tools available in MCP menu

## Still Having Issues?

If problems persist:

1. **Check WSL version**: `wsl --list --verbose` (should be WSL2)
2. **Restart WSL**: `wsl --shutdown` (in PowerShell), then restart
3. **Restart Claude Desktop**: Fully quit and reopen
4. **Check firewall**: Ensure WSL can access network for downloads
5. **Update dependencies**: `cd vision-rag && uv sync`

## Dataset Information

Successfully tested datasets:
- ✅ PneumoniaMNIST (single-label, binary classification)
- ✅ ChestMNIST (multi-label, 14 diseases)
- ✅ OrganSMNIST (single-label, 11 organs)
- ✅ OrganAMNIST (single-label, axial view)
- ✅ OrganCMNIST (single-label, coronal view)
- ✅ PathMNIST (single-label, 9 tissue types)

All 12 MedMNIST datasets should work with the current implementation.

## Performance Expectations

Typical performance on WSL2:

| Operation | Time (WSL2) | Notes |
|-----------|-------------|-------|
| Download dataset (224px) | 2-5 min | 500MB-2GB files |
| Load 100 images | 1-2 sec | Depends on size |
| Encode 100 images (CLIP) | 5-10 sec | CPU-based |
| Store embeddings | <1 sec | ChromaDB is fast |
| Search query | <1 sec | CLIP + similarity |

## Additional Resources

- **MCP Documentation**: https://modelcontextprotocol.io/
- **WSL Documentation**: https://docs.microsoft.com/en-us/windows/wsl/
- **MedMNIST**: https://medmnist.com/
- **ChromaDB**: https://docs.trychroma.com/

## Contact

If you find additional issues or solutions, please update this document!
