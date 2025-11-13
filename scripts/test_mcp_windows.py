#!/usr/bin/env python3
"""
Test script to diagnose MCP server issues when running on WSL with Claude Desktop on Windows.

This script helps identify:
1. File path issues (Windows vs WSL paths)
2. Dataset file corruption
3. MCP protocol communication
4. Directory permissions
"""

import asyncio
import sys
from pathlib import Path
import platform

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from vision_rag.mcp_server import VisionRAGMCPServer
from vision_rag.config import AVAILABLE_DATASETS


async def test_environment():
    """Test the environment and file system."""
    print("=" * 70)
    print("🔍 Environment Check")
    print("=" * 70)
    
    print(f"\n💻 Platform: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {platform.python_version()}")
    print(f"📁 Working directory: {Path.cwd()}")
    print(f"📁 Script directory: {Path(__file__).parent}")
    
    # Check data directory
    data_dir = Path(__file__).parent.parent / "data"
    print(f"\n📦 Data directory: {data_dir}")
    print(f"   Exists: {data_dir.exists()}")
    if data_dir.exists():
        print(f"   Readable: {data_dir.is_dir()}")
        npz_files = list(data_dir.glob("*.npz"))
        print(f"   NPZ files: {len(npz_files)}")
        for npz in npz_files[:5]:
            size_mb = npz.stat().st_size / (1024*1024)
            print(f"      - {npz.name} ({size_mb:.1f} MB)")
    
    # Check ChromaDB directory
    chroma_dir = Path("./chroma_db_mcp")
    print(f"\n💾 ChromaDB directory: {chroma_dir.resolve()}")
    print(f"   Exists: {chroma_dir.exists()}")
    
    # Check image store directory
    image_dir = Path("./image_store_mcp")
    print(f"\n🖼️  Image store directory: {image_dir.resolve()}")
    print(f"   Exists: {image_dir.exists()}")


async def test_dataset_files():
    """Test dataset files for corruption."""
    print("\n" + "=" * 70)
    print("🔬 Dataset File Validation")
    print("=" * 70)
    
    import numpy as np
    
    data_dir = Path(__file__).parent.parent / "data"
    if not data_dir.exists():
        print("❌ Data directory does not exist!")
        return
    
    npz_files = sorted(data_dir.glob("*.npz"))
    print(f"\nChecking {len(npz_files)} NPZ files...\n")
    
    valid_count = 0
    corrupt_count = 0
    
    for npz_file in npz_files:
        try:
            data = np.load(str(npz_file))
            keys = list(data.keys())
            data.close()
            valid_count += 1
            print(f"✅ {npz_file.name:35s} - Valid ({len(keys)} keys)")
        except Exception as e:
            corrupt_count += 1
            size_mb = npz_file.stat().st_size / (1024*1024)
            print(f"❌ {npz_file.name:35s} - CORRUPTED ({size_mb:.1f} MB)")
            print(f"   Error: {str(e)[:60]}")
    
    print(f"\n📊 Summary: {valid_count} valid, {corrupt_count} corrupted")
    
    if corrupt_count > 0:
        print("\n⚠️  WARNING: Corrupted files detected!")
        print("   These files should be deleted and re-downloaded.")
        print("   Run: rm data/<corrupted_file>.npz")


async def test_mcp_server():
    """Test MCP server initialization and basic operations."""
    print("\n" + "=" * 70)
    print("🤖 MCP Server Test")
    print("=" * 70)
    
    try:
        print("\n1️⃣ Initializing MCP Server...")
        server = VisionRAGMCPServer()
        print("   ✅ Server initialized successfully")
        
        print("\n2️⃣ Getting statistics...")
        stats = await server.get_statistics()
        print(f"   Total embeddings: {stats['total_embeddings']}")
        print(f"   Total images: {stats['total_images']}")
        print(f"   ChromaDB: {stats['persist_directory']}")
        print(f"   Image store: {stats['image_store_directory']}")
        
        print("\n3️⃣ Listing available datasets...")
        datasets = await server.list_available_datasets()
        print(f"   Found {datasets['count']} datasets")
        for name in list(datasets['datasets'].keys())[:5]:
            print(f"      - {name}")
        
        print("\n4️⃣ Testing preload with smallest dataset (10 images)...")
        # Use ChestMNIST as it's typically small
        result = await server.preload_dataset(
            dataset_name="ChestMNIST",
            split="train",
            max_images=10,
            size=28  # Use smallest size for faster testing
        )
        
        if result.get("success"):
            print(f"   ✅ Preload successful!")
            print(f"   Images loaded: {result.get('images_loaded')}")
            print(f"   Total embeddings: {result.get('total_embeddings')}")
        else:
            print(f"   ❌ Preload failed: {result.get('error')}")
            return False
        
        print("\n5️⃣ Verifying final state...")
        final_stats = await server.get_statistics()
        print(f"   Total embeddings: {final_stats['total_embeddings']}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ MCP Server test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_windows_wsl_paths():
    """Test Windows/WSL path compatibility."""
    print("\n" + "=" * 70)
    print("🪟 Windows/WSL Path Check")
    print("=" * 70)
    
    # Check if running in WSL
    try:
        with open("/proc/version", "r") as f:
            version = f.read()
            if "microsoft" in version.lower() or "wsl" in version.lower():
                print("\n✅ Running in WSL")
                print(f"   Version: {version.strip()}")
            else:
                print("\n❌ Not running in WSL")
    except FileNotFoundError:
        print("\n❌ Not running in WSL (no /proc/version)")
    
    # Check for Windows-style paths
    cwd = str(Path.cwd())
    if cwd.startswith("/mnt/"):
        print(f"\n⚠️  WARNING: Working directory is on Windows filesystem!")
        print(f"   Path: {cwd}")
        print(f"   This may cause permission or performance issues.")
        print(f"   Recommendation: Use WSL filesystem (/home/...) instead.")
    else:
        print(f"\n✅ Working directory is on WSL filesystem")
        print(f"   Path: {cwd}")


async def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "MCP SERVER DIAGNOSTIC TOOL" + " " * 27 + "║")
    print("║" + " " * 17 + "Windows/WSL Compatibility" + " " * 26 + "║")
    print("╚" + "=" * 68 + "╝")
    
    await test_environment()
    await test_windows_wsl_paths()
    await test_dataset_files()
    
    success = await test_mcp_server()
    
    print("\n" + "=" * 70)
    print("📋 Summary")
    print("=" * 70)
    
    if success:
        print("\n✅ All tests passed!")
        print("\nThe MCP server is working correctly on your system.")
        print("If Claude Desktop on Windows still has issues, check:")
        print("  1. Claude Desktop MCP configuration paths")
        print("  2. WSL network connectivity")
        print("  3. File permissions in WSL")
    else:
        print("\n❌ Some tests failed!")
        print("\nPlease check the error messages above and:")
        print("  1. Remove any corrupted dataset files")
        print("  2. Ensure data directory is accessible")
        print("  3. Check file permissions")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
