"""Tests for image file storage."""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from PIL import Image

from vision_rag.image_store import ImageFileStore


@pytest.fixture
def temp_store_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def image_store(temp_store_dir):
    """Create an ImageFileStore instance for testing."""
    return ImageFileStore(storage_dir=temp_store_dir)


@pytest.fixture
def sample_image():
    """Create a sample grayscale image."""
    img_array = np.random.randint(0, 255, size=(28, 28), dtype=np.uint8)
    return Image.fromarray(img_array, mode='L')


@pytest.fixture
def sample_rgb_image():
    """Create a sample RGB image."""
    img_array = np.random.randint(0, 255, size=(28, 28, 3), dtype=np.uint8)
    return Image.fromarray(img_array, mode='RGB')


@pytest.fixture
def sample_numpy_array():
    """Create a sample numpy array image."""
    return np.random.randint(0, 255, size=(28, 28), dtype=np.uint8)


class TestImageFileStoreInitialization:
    """Tests for ImageFileStore initialization."""
    
    def test_init_creates_directory(self, temp_store_dir):
        """Test that initialization creates the storage directory."""
        store_path = Path(temp_store_dir) / "test_store"
        ImageFileStore(storage_dir=str(store_path))
        
        assert store_path.exists()
        assert store_path.is_dir()
    
    def test_init_with_existing_directory(self, temp_store_dir):
        """Test initialization with an existing directory."""
        # Create directory first
        store_path = Path(temp_store_dir) / "existing_store"
        store_path.mkdir()
        
        # Should not raise error
        ImageFileStore(storage_dir=str(store_path))
        assert store_path.exists()


class TestSaveImage:
    """Tests for saving images."""
    
    def test_save_pil_image(self, image_store, sample_image):
        """Test saving a PIL Image."""
        image_path = image_store.save_image(sample_image)
        
        assert image_path is not None
        assert Path(image_path).exists()
        assert image_path.endswith('.png')
    
    def test_save_numpy_array_grayscale(self, image_store, sample_numpy_array):
        """Test saving a grayscale numpy array."""
        image_path = image_store.save_image(sample_numpy_array)
        
        assert image_path is not None
        assert Path(image_path).exists()
        
        # Verify image can be loaded
        loaded_img = Image.open(image_path)
        assert loaded_img.mode == 'L'
    
    def test_save_numpy_array_rgb(self, image_store):
        """Test saving an RGB numpy array."""
        rgb_array = np.random.randint(0, 255, size=(28, 28, 3), dtype=np.uint8)
        image_path = image_store.save_image(rgb_array)
        
        assert image_path is not None
        assert Path(image_path).exists()
        
        # Verify image can be loaded
        loaded_img = Image.open(image_path)
        assert loaded_img.mode == 'RGB'
    
    def test_save_numpy_array_rgba(self, image_store):
        """Test saving an RGBA numpy array."""
        rgba_array = np.random.randint(0, 255, size=(28, 28, 4), dtype=np.uint8)
        image_path = image_store.save_image(rgba_array)
        
        assert image_path is not None
        assert Path(image_path).exists()
        
        # Verify image can be loaded
        loaded_img = Image.open(image_path)
        assert loaded_img.mode == 'RGBA'
    
    def test_save_with_custom_id(self, image_store, sample_image):
        """Test saving with a custom image ID."""
        custom_id = "custom_123"
        image_path = image_store.save_image(sample_image, image_id=custom_id)
        
        assert custom_id in image_path
        assert Path(image_path).exists()
    
    def test_save_with_custom_prefix(self, image_store, sample_image):
        """Test saving with a custom prefix."""
        custom_prefix = "test"
        image_path = image_store.save_image(sample_image, prefix=custom_prefix)
        
        assert custom_prefix in Path(image_path).name
        assert Path(image_path).exists()
    
    def test_save_multiple_images(self, image_store):
        """Test saving multiple images."""
        paths = []
        for i in range(5):
            img_array = np.random.randint(0, 255, size=(28, 28), dtype=np.uint8)
            img = Image.fromarray(img_array, mode='L')
            path = image_store.save_image(img)
            paths.append(path)
        
        # All paths should be unique
        assert len(paths) == len(set(paths))
        
        # All files should exist
        for path in paths:
            assert Path(path).exists()
    
    def test_save_invalid_shape_raises_error(self, image_store):
        """Test that invalid array shape raises error."""
        invalid_array = np.random.rand(28, 28, 5)  # Invalid number of channels
        
        with pytest.raises(ValueError, match="Unsupported image shape"):
            image_store.save_image(invalid_array)


class TestLoadImage:
    """Tests for loading images."""
    
    def test_load_saved_image(self, image_store, sample_image):
        """Test loading a previously saved image."""
        # Save image
        image_path = image_store.save_image(sample_image)
        
        # Load image
        loaded_image = image_store.load_image(image_path)
        
        assert isinstance(loaded_image, Image.Image)
        assert loaded_image.size == sample_image.size
        assert loaded_image.mode == sample_image.mode
    
    def test_load_with_absolute_path(self, image_store, sample_image):
        """Test loading with absolute path."""
        image_path = image_store.save_image(sample_image)
        absolute_path = Path(image_path).absolute()
        
        loaded_image = image_store.load_image(str(absolute_path))
        assert isinstance(loaded_image, Image.Image)
    
    def test_load_nonexistent_image_raises_error(self, image_store):
        """Test that loading nonexistent image raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            image_store.load_image("nonexistent_image.png")
    
    def test_load_preserves_image_content(self, image_store, sample_numpy_array):
        """Test that loaded image has same content as original."""
        # Save numpy array
        image_path = image_store.save_image(sample_numpy_array)
        
        # Load and convert back to array
        loaded_image = image_store.load_image(image_path)
        loaded_array = np.array(loaded_image)
        
        # Should be identical
        np.testing.assert_array_equal(loaded_array, sample_numpy_array)


class TestDeleteImage:
    """Tests for deleting images."""
    
    def test_delete_existing_image(self, image_store, sample_image):
        """Test deleting an existing image."""
        image_path = image_store.save_image(sample_image)
        
        # Delete image
        result = image_store.delete_image(image_path)
        
        assert result is True
        assert not Path(image_path).exists()
    
    def test_delete_nonexistent_image(self, image_store):
        """Test deleting a nonexistent image."""
        result = image_store.delete_image("nonexistent.png")
        assert result is False
    
    def test_delete_with_absolute_path(self, image_store, sample_image):
        """Test deleting with absolute path."""
        image_path = image_store.save_image(sample_image)
        absolute_path = str(Path(image_path).absolute())
        
        result = image_store.delete_image(absolute_path)
        assert result is True
        assert not Path(image_path).exists()


class TestImageExists:
    """Tests for checking image existence."""
    
    def test_exists_for_saved_image(self, image_store, sample_image):
        """Test that exists returns True for saved image."""
        image_path = image_store.save_image(sample_image)
        assert image_store.image_exists(image_path) is True
    
    def test_exists_for_nonexistent_image(self, image_store):
        """Test that exists returns False for nonexistent image."""
        assert image_store.image_exists("nonexistent.png") is False
    
    def test_exists_after_deletion(self, image_store, sample_image):
        """Test that exists returns False after deletion."""
        image_path = image_store.save_image(sample_image)
        image_store.delete_image(image_path)
        
        assert image_store.image_exists(image_path) is False


class TestClear:
    """Tests for clearing the image store."""
    
    def test_clear_empty_store(self, image_store):
        """Test clearing an empty store."""
        count = image_store.clear()
        assert count == 0
    
    def test_clear_with_images(self, image_store):
        """Test clearing a store with images."""
        # Add multiple images
        for i in range(5):
            img_array = np.random.randint(0, 255, size=(28, 28), dtype=np.uint8)
            img = Image.fromarray(img_array, mode='L')
            image_store.save_image(img)
        
        # Clear store
        count = image_store.clear()
        
        assert count == 5
        assert image_store.count() == 0
    
    def test_clear_removes_all_files(self, image_store, sample_image):
        """Test that clear removes all image files."""
        # Save images
        paths = []
        for i in range(3):
            path = image_store.save_image(sample_image)
            paths.append(path)
        
        # Clear
        image_store.clear()
        
        # All paths should not exist
        for path in paths:
            assert not Path(path).exists()


class TestCount:
    """Tests for counting images."""
    
    def test_count_empty_store(self, image_store):
        """Test counting in empty store."""
        assert image_store.count() == 0
    
    def test_count_with_images(self, image_store):
        """Test counting with images."""
        # Add images
        for i in range(7):
            img_array = np.random.randint(0, 255, size=(28, 28), dtype=np.uint8)
            img = Image.fromarray(img_array, mode='L')
            image_store.save_image(img)
        
        assert image_store.count() == 7
    
    def test_count_after_deletion(self, image_store, sample_image):
        """Test count updates after deletion."""
        # Save 3 different images
        paths = []
        for i in range(3):
            # Create a unique image for each iteration
            img_array = np.random.randint(0, 255, size=(28, 28), dtype=np.uint8)
            img = Image.fromarray(img_array, mode='L')
            path = image_store.save_image(img)
            paths.append(path)
        
        assert image_store.count() == 3
        
        # Delete one
        image_store.delete_image(paths[0])
        
        assert image_store.count() == 2


class TestGenerateImageId:
    """Tests for image ID generation."""
    
    def test_same_image_generates_same_id(self, image_store, sample_numpy_array):
        """Test that the same image generates the same ID."""
        id1 = image_store._generate_image_id(sample_numpy_array)
        id2 = image_store._generate_image_id(sample_numpy_array)
        
        assert id1 == id2
    
    def test_different_images_generate_different_ids(self, image_store):
        """Test that different images generate different IDs."""
        img1 = np.random.randint(0, 255, size=(28, 28), dtype=np.uint8)
        img2 = np.random.randint(0, 255, size=(28, 28), dtype=np.uint8)
        
        id1 = image_store._generate_image_id(img1)
        id2 = image_store._generate_image_id(img2)
        
        # Highly unlikely to be the same for random images
        assert id1 != id2
    
    def test_pil_image_generates_id(self, image_store, sample_image):
        """Test that PIL images can generate IDs."""
        image_id = image_store._generate_image_id(sample_image)
        
        assert isinstance(image_id, str)
        assert len(image_id) == 16  # First 16 chars of hash


class TestImageResizing:
    """Tests for image resizing functionality."""
    
    def test_resize_on_save_grayscale(self, temp_store_dir):
        """Test that images are resized when image_size is set (grayscale)."""
        store = ImageFileStore(storage_dir=temp_store_dir, image_size=64)
        
        # Create a larger image
        large_image = Image.new('L', (200, 200), color=128)
        
        # Save image
        path = store.save_image(large_image)
        
        # Load and check size
        loaded_image = Image.open(path)
        assert loaded_image.size == (64, 64)
    
    def test_resize_on_save_rgb(self, temp_store_dir):
        """Test that RGB images are resized when image_size is set."""
        store = ImageFileStore(storage_dir=temp_store_dir, image_size=128)
        
        # Create a larger RGB image
        large_image = Image.new('RGB', (300, 300), color=(255, 0, 0))
        
        # Save image
        path = store.save_image(large_image)
        
        # Load and check size
        loaded_image = Image.open(path)
        assert loaded_image.size == (128, 128)
        assert loaded_image.mode == 'RGB'
    
    def test_resize_numpy_array(self, temp_store_dir):
        """Test that numpy arrays are resized correctly."""
        store = ImageFileStore(storage_dir=temp_store_dir, image_size=32)
        
        # Create a numpy array image
        large_array = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
        
        # Save image
        path = store.save_image(large_array)
        
        # Load and check size
        loaded_image = Image.open(path)
        assert loaded_image.size == (32, 32)
    
    def test_no_resize_when_none(self, temp_store_dir):
        """Test that images are not resized when image_size is None."""
        store = ImageFileStore(storage_dir=temp_store_dir, image_size=None)
        
        # Create an image with specific size
        original_image = Image.new('RGB', (150, 150), color=(0, 255, 0))
        
        # Save image
        path = store.save_image(original_image)
        
        # Load and check size - should be unchanged
        loaded_image = Image.open(path)
        assert loaded_image.size == (150, 150)
    
    def test_resize_upscaling(self, temp_store_dir):
        """Test that small images are upscaled when image_size is larger."""
        store = ImageFileStore(storage_dir=temp_store_dir, image_size=100)
        
        # Create a small image
        small_image = Image.new('L', (28, 28), color=200)
        
        # Save image
        path = store.save_image(small_image)
        
        # Load and check size - should be upscaled
        loaded_image = Image.open(path)
        assert loaded_image.size == (100, 100)
    
    def test_resize_preserves_mode(self, temp_store_dir):
        """Test that resizing preserves image mode."""
        store = ImageFileStore(storage_dir=temp_store_dir, image_size=50)
        
        # Test grayscale
        gray_image = Image.new('L', (100, 100), color=128)
        path1 = store.save_image(gray_image, prefix="gray")
        loaded1 = Image.open(path1)
        assert loaded1.mode == 'L'
        
        # Test RGB
        rgb_image = Image.new('RGB', (100, 100), color=(255, 128, 0))
        path2 = store.save_image(rgb_image, prefix="rgb")
        loaded2 = Image.open(path2)
        assert loaded2.mode == 'RGB'

