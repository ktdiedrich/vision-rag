#!/usr/bin/env python3
"""Debug script to check all paths when MCP server starts."""

import sys
from pathlib import Path

print("=" * 70, file=sys.stderr)
print("PATH DEBUGGING INFORMATION", file=sys.stderr)
print("=" * 70, file=sys.stderr)

print(f"\n📁 Working Directory: {Path.cwd()}", file=sys.stderr)
print(f"📁 Script Location: {Path(__file__).parent}", file=sys.stderr)
print(f"🐍 Python Executable: {sys.executable}", file=sys.stderr)
print(f"📦 Python Path:", file=sys.stderr)
for p in sys.path[:5]:
    print(f"   - {p}", file=sys.stderr)

# Check data directory
data_dir = Path.cwd() / "data"
print(f"\n📂 Data Directory (relative): ./data", file=sys.stderr)
print(f"📂 Data Directory (absolute): {data_dir.resolve()}", file=sys.stderr)
print(f"   Exists: {data_dir.exists()}", file=sys.stderr)
if data_dir.exists():
    npz_files = list(data_dir.glob("*.npz"))
    print(f"   NPZ files: {len(npz_files)}", file=sys.stderr)

# Check image_store_mcp directory
image_store = Path.cwd() / "image_store_mcp"
print(f"\n🖼️  Image Store (relative): ./image_store_mcp", file=sys.stderr)
print(f"🖼️  Image Store (absolute): {image_store.resolve()}", file=sys.stderr)
print(f"   Exists: {image_store.exists()}", file=sys.stderr)
if image_store.exists():
    png_files = list(image_store.glob("*.png"))
    jpg_files = list(image_store.glob("*.jpg"))
    print(f"   PNG files: {len(png_files)}", file=sys.stderr)
    print(f"   JPG files: {len(jpg_files)}", file=sys.stderr)
    print(f"   Total: {len(png_files) + len(jpg_files)}", file=sys.stderr)
    
    # Show first few files
    all_images = png_files + jpg_files
    if all_images:
        print(f"   First 3 files:", file=sys.stderr)
        for img in all_images[:3]:
            size_kb = img.stat().st_size / 1024
            print(f"      - {img.name} ({size_kb:.1f} KB)", file=sys.stderr)

# Check chroma_db_mcp directory
chroma_db = Path.cwd() / "chroma_db_mcp"
print(f"\n💾 ChromaDB (relative): ./chroma_db_mcp", file=sys.stderr)
print(f"💾 ChromaDB (absolute): {chroma_db.resolve()}", file=sys.stderr)
print(f"   Exists: {chroma_db.exists()}", file=sys.stderr)
if chroma_db.exists():
    db_files = list(chroma_db.glob("*"))
    print(f"   Files: {len(db_files)}", file=sys.stderr)
    for f in db_files:
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"      - {f.name} ({size_kb:.1f} KB)", file=sys.stderr)

print("\n" + "=" * 70, file=sys.stderr)

# Now start the actual server
print("Starting MCP server...", file=sys.stderr)
import asyncio
from vision_rag.mcp_server import main

asyncio.run(main())
