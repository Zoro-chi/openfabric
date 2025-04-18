import os
import logging
import base64
from typing import Dict, Optional, Any, Tuple
import json
from pathlib import Path
import time
import uuid
from dotenv import load_dotenv

from . import openfabric_logger
from .stub import Stub

# Load environment variables once at module level
load_dotenv()

# Initialize module-specific logger as a child of the openfabric logger
logger = openfabric_logger.getChild("text_to_image")


class TextToImageGenerator:
    """
    Handles the text-to-image generation using Openfabric's API.
    """

    def __init__(self, stub: Stub, app_id: str = None):
        """
        Initialize the text-to-image generator.

        Args:
            stub: Stub instance for communicating with Openfabric
            app_id: The app ID for the text-to-image service (default: from env var)
        """
        self.stub = stub
        self.app_id = app_id or os.environ.get("TEXT_TO_IMAGE_APP_ID")
        if not self.app_id:
            logger.error("No TEXT_TO_IMAGE_APP_ID provided or found in environment")
            raise ValueError("Missing TEXT_TO_IMAGE_APP_ID")

        # Set up output directory
        image_output_dir = os.environ.get("IMAGE_OUTPUT_DIR")
        self.output_dir = (
            Path(image_output_dir)
            if image_output_dir
            else Path(__file__).parent.parent / "data" / "images"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not image_output_dir:
            logger.warning(
                f"IMAGE_OUTPUT_DIR not set, using default: {self.output_dir}"
            )

        # Cache schemas without raising exceptions to allow fallback mode
        self._load_schemas()

    def _load_schemas(self):
        """Load API schemas without blocking initialization on failure"""
        try:
            self.input_schema = self.stub.schema(self.app_id, "input")
            self.output_schema = self.stub.schema(self.app_id, "output")
            self.manifest = self.stub.manifest(self.app_id)
            logger.info(
                f"Schema and manifest loaded for text-to-image app: {self.app_id}"
            )
        except Exception as e:
            logger.warning(f"Failed to load API schemas: {e}")
            self.input_schema = self.output_schema = self.manifest = None

    def generate(
        self,
        prompt: str,
        params: Optional[Dict[str, Any]] = None,
        original_prompt: str = None,
    ) -> Tuple[str, str]:
        """
        Generate an image from text prompt.

        Args:
            prompt: The text prompt (expanded by LLM)
            params: Additional parameters for image generation
            original_prompt: The original user prompt (used for naming files)

        Returns:
            Tuple of (image_path, metadata_path)
        """
        # Use original prompt for naming if provided, otherwise use expanded prompt
        file_naming_prompt = original_prompt if original_prompt else prompt

        # Prepare the request
        request_data = self._prepare_request(prompt, params)
        logger.info(f"Sending text-to-image request with prompt: {prompt[:100]}...")

        # Send the request to Openfabric
        try:
            start_time = time.time()
            result = self.stub.call(self.app_id, request_data)
            generation_time = time.time() - start_time
            logger.info(f"Text-to-image generation completed in {generation_time:.2f}s")

            # Process and save the result
            return self._process_result(result, prompt, file_naming_prompt)
        except Exception as e:
            logger.error(f"Failed to generate image: {e}")
            raise

    def _prepare_request(
        self, prompt: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Prepare the request payload based on the app's input schema.

        Args:
            prompt: Text prompt for image generation
            params: Additional parameters to override defaults

        Returns:
            Dict containing the properly formatted request payload
        """
        # Default parameters
        default_params = {
            "width": 512,
            "height": 512,
            "guidance_scale": 7.5,
            "num_inference_steps": 30,
            "seed": -1,  # Random seed
            "negative_prompt": "blurry, low quality, distorted, deformed",
        }

        # Override defaults with provided params
        request_params = {**default_params, **(params or {})}

        # Create request
        return {"prompt": prompt, **request_params}

    def _process_result(
        self, result: Dict[str, Any], prompt: str, file_naming_prompt: str
    ) -> Tuple[str, str]:
        """
        Process the result from the text-to-image app.

        Args:
            result: The API response
            prompt: The prompt used for generation
            file_naming_prompt: The prompt used for naming files

        Returns:
            Tuple of (image_path, metadata_path) - image_path may be None if needs download
        """
        try:
            # Generate a unique ID and timestamp
            image_id = str(uuid.uuid4())
            timestamp = int(time.time())

            # Create a descriptive filename from the prompt
            if file_naming_prompt:
                # Use first 15 chars of prompt, replacing spaces with underscores
                base_name = (
                    file_naming_prompt[:15].strip().replace(" ", "_").replace("/", "_")
                )
                # Remove any other non-alphanumeric characters
                base_name = "".join(c for c in base_name if c.isalnum() or c == "_")
            else:
                base_name = f"image_{timestamp}"

            # Create metadata filename
            metadata_filename = f"{base_name}_{timestamp}.json"
            metadata_path = self.output_dir / metadata_filename

            # Handle result format with blob ID (common case)
            if "result" in result:
                blob_id = result.get("result")
                logger.info(f"Image generation result ID: {blob_id}")

                # Create metadata for the image that includes the blob ID
                # Without creating an actual image file since it needs to be downloaded
                metadata = {
                    "id": image_id,
                    "timestamp": timestamp,
                    "prompt": prompt,
                    "parameters": result.get("parameters", {}),
                    "result_id": blob_id,
                    "type": "image",
                    "needs_download": True,
                    "base_name": base_name,
                }

                with open(metadata_path, "w") as meta_file:
                    json.dump(metadata, meta_file, indent=2)

                logger.info(f"Image metadata saved with result ID: {blob_id}")
                logger.info("Use blob_viewer.py to download the actual image")

                # Return only metadata path since image needs separate download
                return None, str(metadata_path)

            # Handle direct image data format (rare case)
            elif "image" in result:
                image_filename = f"{base_name}_{timestamp}.png"
                image_path = self.output_dir / image_filename

                # Process image data
                image_data = result.get("image")
                if isinstance(image_data, str) and image_data.startswith("data:image"):
                    # Extract base64 data after the comma
                    image_base64 = image_data.split(",", 1)[1]
                else:
                    image_base64 = image_data

                # Save the image
                image_bytes = base64.b64decode(image_base64)
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)

                # Save metadata
                metadata = {
                    "id": image_id,
                    "timestamp": timestamp,
                    "prompt": prompt,
                    "parameters": result.get("parameters", {}),
                    "file_path": str(image_path),
                    "type": "image",
                    "direct_image": True,
                }

                with open(metadata_path, "w") as meta_file:
                    json.dump(metadata, meta_file, indent=2)

                logger.info(f"Image saved to {image_path}")
                return str(image_path), str(metadata_path)

            else:
                raise KeyError(
                    f"Unexpected response format. Keys: {list(result.keys())}"
                )

        except Exception as e:
            logger.error(f"Failed to process image result: {e}")
            raise
