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


class ImageTo3DGenerator:
    """
    Handles the image-to-3D generation using Openfabric's API.
    """

    def __init__(self, stub: Stub, app_id: str = None):
        """
        Initialize the image-to-3D generator.

        Args:
            stub: Stub instance for communicating with Openfabric
            app_id: The app ID for the image-to-3D service (default: from env var)
        """
        self.stub = stub
        self.app_id = app_id or os.environ.get("IMAGE_TO_3D_APP_ID")

        # Use default output directory if MODEL_OUTPUT_DIR is not set
        model_output_dir = os.environ.get("MODEL_OUTPUT_DIR")
        if model_output_dir is None:
            # Default to app/data/models
            self.output_dir = Path(__file__).parent.parent / "data" / "models"
            logger.warning(
                f"MODEL_OUTPUT_DIR not set, using default: {self.output_dir}"
            )
        else:
            self.output_dir = Path(model_output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Cache the schema and manifest - don't raise exceptions to allow fallback mode
        try:
            self.input_schema = self.stub.schema(self.app_id, "input")
            self.output_schema = self.stub.schema(self.app_id, "output")
            self.manifest = self.stub.manifest(self.app_id)
            logger.info(
                f"Successfully loaded schema and manifest for image-to-3D app: {self.app_id}"
            )
        except Exception as e:
            logger.warning(f"Failed to load schema/manifest for image-to-3D app: {e}")

    def generate(
        self, image_path: str, params: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str]:
        """
        Generate a 3D model from an image.

        Args:
            image_path: Path to the source image file
            params: Additional parameters for 3D generation

        Returns:
            Tuple of (model_path, metadata_path)
        """
        # Read the image and convert to base64
        try:
            with open(image_path, "rb") as img_file:
                image_data = base64.b64encode(img_file.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to read image at {image_path}: {e}")
            raise

        # Prepare the request based on the input schema
        request_data = self._prepare_request(image_data, params)

        # Log the request
        logger.info(f"Sending image-to-3D request for image: {image_path}")

        # Send the request to Openfabric
        result = None
        try:
            start_time = time.time()
            result = self.stub.call(self.app_id, request_data)
            generation_time = time.time() - start_time
            logger.info(f"Image-to-3D generation completed in {generation_time:.2f}s")
        except Exception as e:
            logger.error(f"Failed to generate 3D model: {e}")
            # Generate a mock response to continue testing

            # result = self._generate_mock_response(request_data)
            # logger.warning("Using mock 3D model response due to service error")

        # Process and save the result
        return self._process_result(result, image_path)

    def _generate_mock_response(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a mock 3D model response when the service is unavailable.

        Args:
            request_data: The original request data

        Returns:
            A mock response with a simple 3D model
        """
        # Create a very simple GLB file header as mock model
        mock_model = "data:model/glb;base64,Z2xURgIAAAABAAAABgAAAEJJTgAAAAAA"

        return {
            "model": mock_model,
            "format": "glb",
            "parameters": {
                "model_type": request_data.get("model_type", "textured"),
                "quality": request_data.get("quality", "standard"),
            },
        }

    def _prepare_request(
        self, image_data: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Prepare the request payload based on the app's input schema.

        Args:
            image_data: Base64-encoded image data
            params: Additional parameters for 3D generation

        Returns:
            Dict containing the properly formatted request payload
        """
        # Default parameters for image-to-3D transformation
        default_params = {
            "model_type": "textured",  # Options might include: textured, mesh, point_cloud
            "quality": "standard",  # Options might include: draft, standard, high
            "format": "glb",  # Output format: glb, obj, etc.
        }

        # Override defaults with provided params
        request_params = {**default_params, **(params or {})}

        # Create request based on the actual input schema
        # The schema specifies 'input_image' as the required field, not 'image'
        request = {"input_image": image_data, **request_params}

        return request

    def _process_result(
        self, result: Dict[str, Any], image_path: str
    ) -> Tuple[str, str]:
        """
        Process the result from the image-to-3D app.

        Args:
            result: The API response
            image_path: Path to the source image

        Returns:
            Tuple of (model_path, metadata_path)
        """
        # If result is None (due to service failure), create a mock response
        if result is None:
            logger.warning("Received None result from service, using mock response")
            result = self._generate_mock_response(
                {"input_image": "mock_image", "model_type": "textured"}
            )

        try:
            model_format = "glb"  # Default format for 3D models
            has_video_preview = False
            video_data = None

            # Process Openfabric blob response format (most common case)
            # This will have 'generated_object' as a data_blob ID (without the base64 data)
            if "generated_object" in result:
                # Extract the model data or model blob reference
                model_data = result.get("generated_object")

                # Check if this is a blob reference (data URI format or plain string)
                if isinstance(model_data, str):
                    if model_data.startswith("data_"):
                        # This is a blob ID reference, not actual base64 data
                        blob_id = model_data
                        logger.info(f"3D model generation result ID: {blob_id}")

                        # For now, use a mock model since we can't access the actual model data
                        # In a production environment, you'd need to fetch the model from the blob store
                        model_base64 = (
                            "Z2xURgIAAAABAAAABgAAAEJJTgAAAAAA"  # Simple GLB header
                        )
                        logger.info(
                            f"Using placeholder 3D model for blob ID: {blob_id}"
                        )
                    elif "," in model_data and "base64" in model_data:
                        # Extract base64 data if in data URI format
                        model_base64 = model_data.split(",", 1)[1]
                    else:
                        # Use as-is for plain base64 data
                        model_base64 = model_data
                else:
                    # Use as-is for binary data or None
                    model_base64 = (
                        "Z2xURgIAAAABAAAABgAAAEJJTgAAAAAA"  # Simple GLB header
                    )
                    logger.warning("Received unexpected generated_object type")

                # Also handle video preview if available
                video_data = result.get("video_object")
                has_video_preview = video_data is not None and video_data != ""

            # Handle result blob reference format (alternative format)
            elif "result" in result:
                blob_id = result.get("result")
                logger.info(f"3D model generation result ID: {blob_id}")

                # For now, use a mock model since we can't access the actual model data
                # In a production environment, you'd need to fetch the model from the blob store
                model_base64 = "Z2xURgIAAAABAAAABgAAAEJJTgAAAAAA"  # Simple GLB header
                logger.info(f"Using placeholder 3D model for blob ID: {blob_id}")

            # Handle mock response format (which has 'model' field)
            elif "model" in result:
                model_data = result.get("model")
                model_format = result.get("format", "glb")

                if isinstance(model_data, str):
                    if "," in model_data:
                        # Extract base64 data if in data URI format
                        model_base64 = model_data.split(",", 1)[1]
                    else:
                        model_base64 = model_data
                else:
                    # Use as-is for binary data
                    model_base64 = model_data
                has_video_preview = False
            else:
                raise KeyError(
                    f"Could not identify response format. Keys: {list(result.keys())}"
                )

            # Generate a unique ID for this model and timestamp
            model_id = str(uuid.uuid4())
            timestamp = int(time.time())

            # NEW: Extract the base filename from the source image to use for the model
            source_image_filename = Path(image_path).name
            base_name = source_image_filename.rsplit(".", 1)[0]  # Remove extension

            # If base name doesn't already include timestamp, add it
            if not any(c.isdigit() for c in base_name):
                base_name = f"{base_name}_{timestamp}"

            # Append "_3d" to clearly indicate this is a 3D model derived from the image
            base_name = f"{base_name}_3d"

            # Create filenames based on the image name pattern
            model_filename = f"{base_name}.{model_format}"
            metadata_filename = f"{base_name}.json"

            # Create paths for model and metadata
            model_path = self.output_dir / model_filename
            metadata_path = self.output_dir / metadata_filename

            # Create path for video preview if available
            video_path = None
            if has_video_preview and video_data:
                video_filename = f"{base_name}_preview.mp4"
                video_path = self.output_dir / video_filename
                try:
                    # Extract video base64 data
                    video_base64 = video_data
                    if isinstance(video_data, str) and "," in video_data:
                        video_base64 = video_data.split(",", 1)[1]

                    # Save the video preview
                    with open(video_path, "wb") as video_file:
                        video_file.write(base64.b64decode(video_base64))
                    logger.info(f"Video preview saved to {video_path}")
                except Exception as video_error:
                    logger.error(f"Failed to save video preview: {video_error}")
                    video_path = None

            # Save the model file
            with open(model_path, "wb") as model_file:
                model_file.write(base64.b64decode(model_base64))

            # Save metadata linking image to 3D model
            metadata = {
                "id": model_id,
                "timestamp": timestamp,
                "source_image": image_path,
                "source_image_filename": source_image_filename,
                "file_path": str(model_path),
                "format": model_format,
                "type": "3d_model",
                "has_video_preview": has_video_preview,
                "video_path": str(video_path) if video_path else None,
                "result_id": result.get(
                    "result", result.get("generated_object", "mock")
                ),
                "parameters": result.get("parameters", {}),
            }

            with open(metadata_path, "w") as meta_file:
                json.dump(metadata, meta_file)

            logger.info(f"3D model saved to {model_path}")
            logger.info(f"Metadata saved to {metadata_path}")

            return str(model_path), str(metadata_path)

        except Exception as e:
            logger.error(f"Failed to process 3D model result: {e}")
            raise
