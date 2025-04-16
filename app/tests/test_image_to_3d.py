#!/usr/bin/env python3
"""
Test script for image-to-3D generation

Usage:
  python test_image_to_3d.py <image_path>

Example:
  python test_image_to_3d.py ../data/images/example.png
"""

import os
import sys
import logging
import json
from pathlib import Path
from dotenv import load_dotenv

# Add the parent directory to path so we can import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.stub import Stub
from core.image_to_3d import ImageTo3DGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_image_to_3d")

# Load environment variables
load_dotenv()


def test_image_to_3d(image_path):
    """
    Test image-to-3D generation with specified image

    Args:
        image_path: Path to the input image
    """
    # Validate image path
    image_path = Path(image_path)
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        sys.exit(1)

    # Get app ID from environment
    app_id = os.environ.get("IMAGE_TO_3D_APP_ID")
    if not app_id:
        logger.error("IMAGE_TO_3D_APP_ID environment variable not set")
        print("\nERROR: The IMAGE_TO_3D_APP_ID environment variable is not set.")
        print("Please set this variable with the appropriate Openfabric app ID.")
        print("You can do this by running:")
        print("export IMAGE_TO_3D_APP_ID=<your-app-id>")
        sys.exit(1)

    logger.info(f"Testing image-to-3D generation with app ID: {app_id}")
    logger.info(f"Input image: {image_path}")

    # Initialize Stub with app ID
    stub = Stub([app_id])

    # Create and initialize the generator
    generator = ImageTo3DGenerator(stub, app_id)

    # Optional parameters for 3D generation
    params = {
        "model_type": "textured",  # Options: textured, mesh, point_cloud
        "quality": "standard",  # Options: draft, standard, high
        "format": "glb",  # Output format
    }

    try:
        # Generate 3D model from image
        logger.info("Starting image-to-3D generation...")
        model_path, metadata_path = generator.generate(str(image_path), params)

        logger.info(f"Successfully generated 3D model: {model_path}")
        logger.info(f"Metadata saved to: {metadata_path}")

        # Display metadata
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        print("\n=== Generation Results ===")
        print(f"3D model: {metadata['file_path']}")
        print(f"Format: {metadata['format']}")
        print(f"Source image: {metadata['source_image']}")
        print(f"Has video preview: {metadata['has_video_preview']}")
        if metadata["has_video_preview"] and metadata["video_path"]:
            print(f"Video preview: {metadata['video_path']}")
        print(f"Parameters used: {metadata['parameters']}")

        return model_path, metadata_path

    except Exception as e:
        logger.error(f"Error during image-to-3D generation: {e}")
        print(f"\nERROR: Failed to generate 3D model: {e}")
        sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nAvailable images in data/images:")
        images_dir = Path(__file__).parent.parent / "data" / "images"
        for image_file in images_dir.glob("*.png"):
            print(f"  {image_file.name}")
        print("\nUse relative path from tools directory or absolute path.")
        sys.exit(1)

    image_path = sys.argv[1]
    test_image_to_3d(image_path)


if __name__ == "__main__":
    main()
