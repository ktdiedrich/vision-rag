"""
Demonstration of dataset preloading via the FastAPI service.

This script shows how to use the REST API endpoints to preload MedMNIST datasets.

Usage:
    1. Start the API service: python -m vision_rag.service
    2. Run this script: python examples/api_preload_demo.py
"""

import requests


API_BASE_URL = "http://localhost:8001"


def list_available_datasets():
    """List all available MedMNIST datasets."""
    print("\n" + "="*60)
    print("Listing Available Datasets")
    print("="*60)
    
    response = requests.get(f"{API_BASE_URL}/datasets")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nFound {data['count']} available datasets:\n")
        
        for name, info in data['datasets'].items():
            print(f"  • {name}")
            print(f"    Description: {info['description']}")
            print(f"    Classes: {info['n_classes']}")
            print(f"    Image size: {info['image_size']}x{info['image_size']}")
            print(f"    Channels: {info['channels']}")
            print()
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)


def preload_dataset(dataset_name: str, split: str = "train", max_images: int = None):
    """Preload a dataset into the RAG store."""
    print(f"\n📦 Preloading {dataset_name} ({split})...")
    
    payload = {
        "dataset_name": dataset_name,
        "split": split,
    }
    
    if max_images is not None:
        payload["max_images"] = max_images
    
    response = requests.post(
        f"{API_BASE_URL}/preload",
        json=payload,
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ Success!")
        print(f"   Dataset: {data['dataset_name']}")
        print(f"   Split: {data['split']}")
        print(f"   Images loaded: {data['images_loaded']}")
        print(f"   Total embeddings: {data['total_embeddings']}")
        print(f"   Message: {data['message']}")
        return data
    else:
        print(f"\n❌ Error: {response.status_code}")
        print(response.text)
        return None


def get_stats():
    """Get current RAG store statistics."""
    response = requests.get(f"{API_BASE_URL}/stats")
    
    if response.status_code == 200:
        data = response.json()
        print("\n📊 Current Statistics:")
        print(f"   Total embeddings: {data['total_embeddings']}")
        print(f"   Total images: {data['total_images']}")
        print(f"   Collection: {data['collection_name']}")
        return data
    else:
        print(f"❌ Error getting stats: {response.status_code}")
        return None


def search_by_label(label: int, n_results: int = 5):
    """Search for images by label."""
    print(f"\n🔍 Searching for images with label {label}...")
    
    response = requests.post(
        f"{API_BASE_URL}/search/label",
        json={"label": label, "n_results": n_results},
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n   Found {data['count']} images")
        print(f"   Label: {data['query_info'].get('human_readable_label', 'Unknown')}")
        if data['results']:
            print(f"   First result ID: {data['results'][0]['id']}")
        return data
    else:
        print(f"❌ Error: {response.status_code}")
        return None


def clear_store():
    """Clear all data from the RAG store."""
    print("\n🗑️  Clearing RAG store...")
    
    response = requests.delete(f"{API_BASE_URL}/clear")
    
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ {data['message']}")
        return data
    else:
        print(f"❌ Error: {response.status_code}")
        return None


def main():
    """Run demonstration."""
    print("\n" + "="*60)
    print("Vision RAG API - Dataset Preloading Demo")
    print("="*60)
    print("\nMake sure the API service is running:")
    print("  python -m vision_rag.service")
    print("="*60)
    
    # Check service health
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code != 200:
            print("\n❌ API service not responding. Please start it first.")
            return
        print("\n✅ API service is running")
    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect to API service. Please start it first:")
        print("   python -m vision_rag.service")
        return
    
    # Demo 1: List available datasets
    list_available_datasets()
    
    # Demo 2: Get initial stats
    get_stats()
    
    # Demo 3: Preload a small dataset
    print("\n" + "="*60)
    print("DEMO: Preloading PneumoniaMNIST")
    print("="*60)
    preload_dataset("PneumoniaMNIST", split="train", max_images=50)
    
    # Demo 4: Get updated stats
    get_stats()
    
    # Demo 5: Search by label
    search_by_label(label=0, n_results=3)
    
    # Demo 6: Preload another dataset
    print("\n" + "="*60)
    print("DEMO: Preloading BreastMNIST")
    print("="*60)
    preload_dataset("BreastMNIST", split="train", max_images=30)
    
    # Final stats
    get_stats()
    
    print("\n" + "="*60)
    print("Demo completed!")
    print("="*60)
    print("\nTo clear the store, uncomment the line below:")
    print("# clear_store()")


if __name__ == "__main__":
    main()
