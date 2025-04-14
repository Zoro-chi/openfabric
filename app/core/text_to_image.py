import os
import logging
import base64
from typing import Dict, Optional, Any, Tuple
import json
from pathlib import Path
import time
import uuid
import random
from dotenv import load_dotenv

from .stub import Stub

load_dotenv()

logger = logging.getLogger(__name__)


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

        # Use default output directory if IMAGE_OUTPUT_DIR is not set
        image_output_dir = os.environ.get("IMAGE_OUTPUT_DIR")
        if image_output_dir is None:
            # Default to app/data/images
            self.output_dir = Path(__file__).parent.parent / "data" / "images"
            logger.warning(
                f"IMAGE_OUTPUT_DIR not set, using default: {self.output_dir}"
            )
        else:
            self.output_dir = Path(image_output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Cache the schema and manifest - don't raise exceptions to allow fallback mode
        try:
            self.input_schema = self.stub.schema(self.app_id, "input")
            self.output_schema = self.stub.schema(self.app_id, "output")
            self.manifest = self.stub.manifest(self.app_id)
            logger.info(
                f"Successfully loaded schema and manifest for text-to-image app: {self.app_id}"
            )
        except Exception as e:
            logger.warning(f"Failed to load schema/manifest for text-to-image app: {e}")

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

        # Prepare the request based on the input schema
        request_data = self._prepare_request(prompt, params)

        # Log the request
        logger.info(f"Sending text-to-image request with prompt: {prompt[:100]}...")

        # Send the request to Openfabric
        result = None
        try:
            start_time = time.time()
            result = self.stub.call(self.app_id, request_data)
            generation_time = time.time() - start_time
            logger.info(f"Text-to-image generation completed in {generation_time:.2f}s")
        except Exception as e:
            logger.error(f"Failed to generate image: {e}")
            # Generate a mock response to continue testing

            # result = self._generate_mock_response(prompt, request_data)
            # logger.warning("Using mock image response due to service error")

        # Process and save the result
        return self._process_result(result, prompt, file_naming_prompt)

    def _generate_mock_response(
        self, prompt: str, request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a mock image response when the service is unavailable.

        Args:
            prompt: The text prompt
            request_data: The original request data

        Returns:
            A mock response with a simple image
        """
        # Create a 1x1 transparent PNG as mock image
        mock_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

        return {
            "image": mock_image,
            "parameters": {
                "prompt": prompt,
                "width": request_data.get("width", 512),
                "height": request_data.get("height", 512),
                "steps": request_data.get("num_inference_steps", 30),
                "guidance_scale": request_data.get("guidance_scale", 7.5),
                "seed": request_data.get("seed", random.randint(1000, 9999)),
            },
        }

    def _prepare_request(
        self, prompt: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Prepare the request payload based on the app's input schema.
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

        # Create request based on schema
        request = {"prompt": prompt, **request_params}

        return request

    def _process_result(
        self, result: Dict[str, Any], prompt: str, file_naming_prompt: str
    ) -> Tuple[str, str]:
        """
        Process the result from the text-to-image app.

        Args:
            result: The API response
            prompt: The original prompt
            file_naming_prompt: The prompt used for naming files

        Returns:
            Tuple of (image_path, metadata_path)
        """
        # Extract image data or blob ID
        try:
            # Generate a unique ID for this image
            image_id = str(uuid.uuid4())
            timestamp = int(time.time())

            # Create a more descriptive base filename from the prompt
            if file_naming_prompt:
                # Use first 15 chars of prompt, replacing spaces with underscores
                base_name = (
                    file_naming_prompt[:15].strip().replace(" ", "_").replace("/", "_")
                )
                # Remove any other non-alphanumeric characters
                base_name = "".join(c for c in base_name if c.isalnum() or c == "_")
            else:
                base_name = f"image_{timestamp}"

            # Create paths for metadata
            metadata_filename = f"{base_name}_{timestamp}.json"
            metadata_path = self.output_dir / metadata_filename

            # Handle real Openfabric response format (which has 'result' field)
            if "result" in result:
                # Log the result ID for reference
                blob_id = result.get("result")
                logger.info(f"Image generation result ID: {blob_id}")

                # Create metadata for the image that includes the blob ID
                # We won't create actual image file path yet since it will be downloaded
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
                logger.info(f"Use blob_viewer.py to download the actual image")

                # Return the metadata path but no image path since it needs to be downloaded
                return None, str(metadata_path)

            # If we have direct image data (which would be rare in real use)
            elif "image" in result:
                # This is the fallback case if we somehow receive direct image data
                image_filename = f"{base_name}_{timestamp}.png"
                image_path = self.output_dir / image_filename

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

                logger.info(f"Direct image data saved to {image_path}")
                return str(image_path), str(metadata_path)

            else:
                raise KeyError(
                    f"Unexpected response format. Response keys: {list(result.keys())}"
                )

        except Exception as e:
            logger.error(f"Failed to process image result: {e}")
            raise
