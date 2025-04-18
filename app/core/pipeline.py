import logging
import os
import sys
import json
import requests
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import importlib
import importlib.util

# Use relative imports for modules in the same package
from .text_to_image import TextToImageGenerator
from .image_to_3d import ImageTo3DGenerator
from .stub import Stub

from llm.client import LLMClient

logger = logging.getLogger(__name__)


class PipelineResult:
    """Data class to store the results of a creation pipeline run"""

    def __init__(
        self,
        success: bool = False,
        original_prompt: str = None,
        expanded_prompt: str = None,
        image_path: str = None,
        image_metadata_path: str = None,
        model_path: str = None,
        model_metadata_path: str = None,
    ):
        self.success = success
        self.original_prompt = original_prompt
        self.expanded_prompt = expanded_prompt
        self.image_path = image_path
        self.image_metadata_path = image_metadata_path
        self.model_path = model_path
        self.model_metadata_path = model_metadata_path

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation"""
        return {
            "success": self.success,
            "original_prompt": self.original_prompt,
            "expanded_prompt": self.expanded_prompt,
            "image_path": self.image_path,
            "image_metadata_path": self.image_metadata_path,
            "model_path": self.model_path,
            "model_metadata_path": self.model_metadata_path,
        }


class CreativePipeline:
    """
    Orchestrates the end-to-end creative pipeline from prompt to 3D model.

    Flow:
    1. Take user prompt
    2. Enhance with local LLM
    3. Generate image from enhanced prompt
    4. Create 3D model from image
    5. Return comprehensive results
    """

    def __init__(self, stub: Stub):
        """
        Initialize the creative pipeline components.

        Args:
            stub: Stub instance for communicating with Openfabric apps
        """
        from . import openfabric_logger

        # Use the openfabric_logger for pipeline initialization
        of_logger = openfabric_logger.getChild("pipeline")
        of_logger.info("Initializing Creative Pipeline with Openfabric services")

        self.stub = stub

        # Log available app connections
        if hasattr(stub, "_connections"):
            app_ids = list(stub._connections.keys())
            of_logger.info(f"Openfabric connections available: {app_ids}")

            for app_id in app_ids:
                try:
                    # Get manifest information for this app
                    manifest = stub.manifest(app_id)
                    app_name = manifest.get("name", "Unknown")
                    app_version = manifest.get("version", "Unknown")
                    app_description = manifest.get(
                        "description", "No description available"
                    )

                    of_logger.info(f"Connected to Openfabric app: {app_id}")
                    of_logger.info(f"  App name: {app_name}")
                    of_logger.info(f"  App version: {app_version}")
                    of_logger.info(f"  Description: {app_description}")

                    # Try to log some schema information
                    try:
                        input_schema = stub.schema(app_id, "input")
                        output_schema = stub.schema(app_id, "output")
                        of_logger.info(f"  Schema loaded successfully for app {app_id}")
                    except Exception as schema_e:
                        of_logger.warning(
                            f"  Could not load schema for app {app_id}: {schema_e}"
                        )
                except Exception as e:
                    of_logger.error(f"Error getting manifest for app {app_id}: {e}")
        else:
            of_logger.warning("No Openfabric connections available in stub")

        # Initialize LLM client
        llm_service_url = os.environ.get("LLM_SERVICE_URL")
        self.llm_client = LLMClient(base_url=llm_service_url)
        of_logger.info(f"LLM client initialized with service URL: {llm_service_url}")

        # Initialize generators
        of_logger.info("Initializing Text-to-Image generator")
        self.text_to_image = TextToImageGenerator(stub)
        if hasattr(self.text_to_image, "app_id"):
            of_logger.info(
                f"Text-to-Image generator initialized with app ID: {self.text_to_image.app_id}"
            )

        of_logger.info("Initializing Image-to-3D generator")
        self.image_to_3d = ImageTo3DGenerator(stub)
        if hasattr(self.image_to_3d, "app_id"):
            of_logger.info(
                f"Image-to-3D generator initialized with app ID: {self.image_to_3d.app_id}"
            )

        # Ensure app/data directories exist
        data_dir = Path(__file__).parent.parent / "data"
        data_dir.mkdir(exist_ok=True)
        (data_dir / "images").mkdir(exist_ok=True)
        (data_dir / "models").mkdir(exist_ok=True)
        (data_dir / "downloads").mkdir(exist_ok=True)

        of_logger.info("Creative pipeline initialized successfully")
        logger.info("Creative pipeline initialized successfully")

    def create(self, prompt: str, params: Dict[str, Any] = None) -> PipelineResult:
        """
        Run the creative pipeline from text prompt to 3D model.

        Args:
            prompt: The user's text prompt
            params: Optional parameters for the pipeline stages

        Returns:
            PipelineResult object with paths to generated assets
        """
        try:
            # 1. Enhance the prompt with the LLM
            logger.info(f"Enhancing prompt: '{prompt}'")
            try:
                expanded_prompt = self.llm_client.expand_prompt(prompt)
                logger.info(f"Enhanced prompt: '{expanded_prompt}'")
            except Exception as e:
                logger.warning(f"Failed to enhance prompt: {e}")
                # Fall back to original prompt if enhancement fails
                expanded_prompt = prompt
                logger.info(f"Using original prompt: '{expanded_prompt}'")

            # 2. Generate image from the enhanced prompt
            image_params = params.get("image", {}) if params else {}
            image_path, image_metadata_path = self.text_to_image.generate(
                expanded_prompt, image_params, original_prompt=prompt
            )

            # If image_path is None but we have metadata, we need to download from blob store
            if image_path is None and image_metadata_path:
                try:
                    # Import the blob viewer module using a direct path approach
                    tools_dir = str(Path(__file__).parent.parent / "tools")
                    sys.path.append(tools_dir)

                    try:
                        from tools.blob_viewer import construct_resource_url
                    except ImportError:
                        # Fall back to the importlib approach if direct import fails
                        blob_viewer_path = os.path.join(tools_dir, "blob_viewer.py")
                        spec = importlib.util.spec_from_file_location(
                            "blob_viewer", blob_viewer_path
                        )
                        blob_viewer = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(blob_viewer)
                        construct_resource_url = blob_viewer.construct_resource_url

                    # Read metadata to get blob ID and other info
                    with open(image_metadata_path, "r") as f:
                        metadata = json.load(f)

                    if "result_id" in metadata and metadata["result_id"] != "mock":
                        blob_id = metadata["result_id"]
                        logger.info(f"Downloading image from blob store: {blob_id}")

                        # Parse the blob ID
                        parts = blob_id.split("/")
                        data_blob_id = parts[0]
                        execution_id = parts[2] if len(parts) > 2 else None

                        # Target directory
                        images_dir = Path(__file__).parent.parent / "data" / "images"

                        # Get the existing metadata filename and use it for the image
                        metadata_filename = Path(image_metadata_path).name
                        base_filename = metadata_filename.rsplit(".", 1)[0]
                        image_filename = f"{base_filename}.png"

                        # Prepare full paths
                        target_image_path = images_dir / image_filename

                        # Update the metadata file with additional information
                        metadata.update(
                            {
                                "original_prompt": prompt,
                                "expanded_prompt": expanded_prompt,
                                "needs_download": False,  # Mark as downloaded
                            }
                        )

                        # Write the updated metadata back to the file
                        with open(image_metadata_path, "w") as f:
                            json.dump(metadata, f, indent=2)

                        # Download the image
                        url = construct_resource_url(data_blob_id, execution_id)
                        response = requests.get(url)

                        if response.status_code == 200:
                            with open(target_image_path, "wb") as f:
                                f.write(response.content)
                            image_path = str(target_image_path)
                            logger.info(f"Generated image at {image_path}")
                        else:
                            logger.error(
                                f"Failed to download image: {response.status_code}"
                            )
                            return PipelineResult(
                                success=False,
                                original_prompt=prompt,
                                expanded_prompt=expanded_prompt,
                            )
                except Exception as e:
                    logger.error(f"Failed to download image from blob store: {e}")
                    return PipelineResult(
                        success=False,
                        original_prompt=prompt,
                        expanded_prompt=expanded_prompt,
                    )

            # Return early if we couldn't generate an image
            if not image_path:
                logger.error("Failed to generate image")
                return PipelineResult(
                    success=False,
                    original_prompt=prompt,
                    expanded_prompt=expanded_prompt,
                )

            logger.info(f"Generated image at {image_path}")

            # 3. Generate 3D model from the image
            model_params = params.get("model", {}) if params else {}
            try:
                logger.info(f"Starting 3D model generation from image: {image_path}")
                # The generate method will now handle all the asynchronous processing internally
                model_path, model_metadata_path = self.image_to_3d.generate(
                    image_path, model_params
                )

                # Verify the model was generated successfully
                if not Path(model_path).exists():
                    raise FileNotFoundError(
                        f"Generated model file not found at {model_path}"
                    )

                logger.info(f"Successfully generated 3D model at {model_path}")
                logger.info(f"Model metadata saved at {model_metadata_path}")

                # Load metadata to include additional details in the response
                try:
                    with open(model_metadata_path, "r") as f:
                        model_metadata = json.load(f)
                    logger.info(
                        f"3D model format: {model_metadata.get('format', 'unknown')}"
                    )

                    # Check for video preview
                    if model_metadata.get("has_video_preview") and model_metadata.get(
                        "video_path"
                    ):
                        logger.info(
                            f"3D model includes video preview at {model_metadata.get('video_path')}"
                        )
                except Exception as metadata_err:
                    logger.warning(f"Could not read model metadata: {metadata_err}")

                # Successful full pipeline
                return PipelineResult(
                    success=True,
                    original_prompt=prompt,
                    expanded_prompt=expanded_prompt,
                    image_path=image_path,
                    image_metadata_path=image_metadata_path,
                    model_path=model_path,
                    model_metadata_path=model_metadata_path,
                )
            except Exception as e:
                logger.error(f"Failed to generate 3D model: {e}")
                # Partial pipeline success (image only)
                return PipelineResult(
                    success=False,
                    original_prompt=prompt,
                    expanded_prompt=expanded_prompt,
                    image_path=image_path,
                    image_metadata_path=image_metadata_path,
                )

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return PipelineResult(success=False, original_prompt=prompt)

    def health_check(self) -> Dict[str, Any]:
        """
        Check health status of all components.

        Returns:
            Dictionary with health status of each component
        """
        health = {
            "pipeline": "initializing",
            "llm": "unknown",
            "text_to_image": "unknown",
            "image_to_3d": "unknown",
        }

        # Check LLM service
        try:
            llm_health = self.llm_client.health_check()
            health["llm"] = (
                "healthy" if llm_health.get("status") == "healthy" else "unhealthy"
            )
        except Exception:
            health["llm"] = "unavailable"

        # Check text-to-image service
        try:
            # Check if the service has a connection
            if (
                hasattr(self.text_to_image, "stub")
                and hasattr(self.text_to_image.stub, "_connections")
                and self.text_to_image.app_id in self.text_to_image.stub._connections
            ):
                health["text_to_image"] = "healthy"
            else:
                health["text_to_image"] = "degraded"
        except Exception:
            health["text_to_image"] = "unavailable"

        # Check image-to-3D service
        try:
            # Check if the service has a connection
            if (
                hasattr(self.image_to_3d, "stub")
                and hasattr(self.image_to_3d.stub, "_connections")
                and self.image_to_3d.app_id in self.image_to_3d.stub._connections
            ):
                health["image_to_3d"] = "healthy"
            else:
                health["image_to_3d"] = "degraded"
        except Exception:
            health["image_to_3d"] = "unavailable"

        # Overall health
        if all(v == "healthy" for v in health.values()):
            health["pipeline"] = "healthy"
        elif "unavailable" in health.values():
            health["pipeline"] = "degraded"
        else:
            health["pipeline"] = "partially available"

        return health
