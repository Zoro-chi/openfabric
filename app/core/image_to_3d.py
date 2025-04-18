import os
import logging
import base64
from typing import Dict, Optional, Any, Tuple
import json
from pathlib import Path
import time
import uuid
import requests
from dotenv import load_dotenv

from . import openfabric_logger
from .stub import Stub

# Load environment variables once at module level
load_dotenv()

# Initialize module-specific logger as a child of the openfabric logger
logger = openfabric_logger.getChild("image_to_3d")


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
        if not self.app_id:
            logger.error("No IMAGE_TO_3D_APP_ID provided or found in environment")
            raise ValueError("Missing IMAGE_TO_3D_APP_ID")

        # Maximum time to wait for job completion (in seconds)
        self.max_wait_time = 300  # 5 minutes
        self.polling_interval = 5  # Check every 5 seconds

        # Set up output directory
        model_output_dir = os.environ.get("MODEL_OUTPUT_DIR")
        self.output_dir = (
            Path(model_output_dir)
            if model_output_dir
            else Path(__file__).parent.parent / "data" / "models"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not model_output_dir:
            logger.warning(
                f"MODEL_OUTPUT_DIR not set, using default: {self.output_dir}"
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
                f"Schema and manifest loaded for image-to-3D app: {self.app_id}"
            )
        except Exception as e:
            logger.warning(f"Failed to load API schemas: {e}")
            self.input_schema = self.output_schema = self.manifest = None

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
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Read the image and convert to base64
        try:
            with open(image_path, "rb") as img_file:
                image_data = base64.b64encode(img_file.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to read image at {image_path}: {e}")
            raise

        # Prepare the request and send to Openfabric
        request_data = self._prepare_request(image_data, params)
        logger.info(f"Sending image-to-3D request for image: {image_path}")

        try:
            start_time = time.time()

            # Make the API call
            response = self.stub.call(self.app_id, request_data)

            # Extract the request ID from logs
            rid = self._extract_rid_from_logs()
            if not rid:
                logger.error("Could not extract request ID from logs")
                raise ValueError("Request ID extraction failed")

            logger.info(f"Submitted image-to-3D job with request ID: {rid}")

            # Poll for job completion
            qid, result = self._poll_for_completion(rid)

            generation_time = time.time() - start_time
            logger.info(f"Image-to-3D generation completed in {generation_time:.2f}s")

            return self._process_result(result, image_path)

        except Exception as e:
            logger.error(f"Failed to generate 3D model: {e}")
            raise

    def _extract_rid_from_logs(self) -> str:
        """Extract the request ID from recent queue entries"""
        # In production, we should rely on the queue API rather than parsing logs
        try:
            queue_url = f"https://{self.app_id}/queue/list"
            response = requests.get(queue_url, timeout=10)
            if response.status_code == 200:
                job_list = response.json()
                if job_list and isinstance(job_list, list) and len(job_list) > 0:
                    # Sort jobs by creation time, newest first
                    sorted_jobs = sorted(
                        job_list, key=lambda x: x.get("created_at", ""), reverse=True
                    )
                    # Get the most recent job's request ID
                    return sorted_jobs[0].get("rid")
            
            logger.warning("Could not find request ID in queue - job tracking may be unreliable")
        except Exception as e:
            logger.warning(f"Failed to get request ID from queue: {e}")
        
        return None

    def _poll_for_completion(self, rid: str) -> Tuple[str, Dict[str, Any]]:
        """Poll until the job is complete or the timeout is reached"""
        start_time = time.time()
        qid = None

        logger.info(f"Waiting for job completion (rid: {rid})...")

        while (time.time() - start_time) < self.max_wait_time:
            try:
                queue_url = f"https://{self.app_id}/queue/list"
                response = requests.get(queue_url, timeout=10)

                if response.status_code != 200:
                    logger.warning(f"Queue list request failed: {response.status_code}")
                    time.sleep(self.polling_interval)
                    continue

                job_list = response.json()
                our_job = next((job for job in job_list if job.get("rid") == rid), None)

                if not our_job:
                    logger.warning(f"Job with rid {rid} not found in queue")
                    time.sleep(self.polling_interval)
                    continue

                # Get queue ID if we don't have it yet
                if not qid:
                    qid = our_job.get("qid")
                    logger.info(f"Found job with qid: {qid}")

                # Check if job is finished
                if our_job.get("finished"):
                    if our_job.get("status") == "COMPLETED":
                        logger.info("Job completed successfully")

                        # Get the result
                        result_url = f"https://{self.app_id}/queue/get?qid={qid}"
                        result_response = requests.get(result_url, timeout=10)

                        if result_response.status_code == 200:
                            return qid, result_response.json()
                        else:
                            logger.error(
                                f"Failed to get result data: {result_response.status_code}"
                            )
                    else:
                        # Job failed
                        status = our_job.get("status")
                        error_msgs = [
                            m.get("content")
                            for m in our_job.get("messages", [])
                            if m.get("type") == "ERROR"
                        ]

                        error_msg = f"Job failed with status: {status}"
                        if error_msgs:
                            error_msg += f", errors: {'; '.join(error_msgs)}"

                        logger.error(error_msg)
                        raise ValueError(error_msg)

                # Job is still running, log status
                status = our_job.get("status")
                progress = (
                    our_job.get("bars", {}).get("default", {}).get("percent", "0")
                )
                logger.info(f"Job status: {status}, progress: {progress}%")

            except requests.RequestException as e:
                logger.error(f"Network error polling for job completion: {e}")
            except Exception as e:
                logger.error(f"Error polling for job completion: {e}")

            time.sleep(self.polling_interval)

        # If we get here, we timed out
        raise TimeoutError(
            f"Timed out waiting for job completion after {self.max_wait_time} seconds"
        )

    def _prepare_request(
        self, image_data: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Prepare the request payload for the API"""
        # Default parameters for image-to-3D transformation
        default_params = {
            "model_type": "textured",  # Options: textured, mesh, point_cloud
            "quality": "standard",  # Options: draft, standard, high
            "format": "glb",  # Always use GLB format for better compatibility
        }

        # Override defaults with provided params but ensure format is always glb
        request_params = {**default_params, **(params or {})}

        # Force GLB format regardless of what was provided in params
        request_params["format"] = "glb"

        # Create request
        return {"input_image": image_data, **request_params}

    def _process_result(
        self, result: Dict[str, Any], image_path: str
    ) -> Tuple[str, str]:
        """Process the API response and save the model file and metadata"""
        if not result:
            raise ValueError("No result received from image-to-3D generation service")

        try:
            # Always use GLB format for better compatibility with Gradio UI
            model_format = "glb"
            model_base64 = None

            # Handle different response formats
            if "generated_object" in result:
                model_data = result.get("generated_object")

                if isinstance(model_data, str):
                    # Handle blob reference or base64 data
                    if "/" in model_data or model_data.startswith("data_"):
                        # Download blob
                        resource_url = (
                            f"https://{self.app_id}/resource?reid={model_data}"
                        )
                        logger.info(f"Fetching 3D model from blob URL: {resource_url}")

                        response = requests.get(resource_url, timeout=30)
                        if response.status_code == 200:
                            model_binary = response.content
                            model_base64 = base64.b64encode(model_binary).decode(
                                "utf-8"
                            )

                            # Note the content type but still use GLB for saving
                            content_type = response.headers.get("Content-Type", "")
                            logger.info(
                                f"Received content type: {content_type}, saving as GLB format"
                            )
                        else:
                            raise ValueError(
                                f"Failed to fetch blob data: {response.status_code}"
                            )
                    elif "," in model_data and "base64" in model_data:
                        # Extract base64 from data URI
                        model_base64 = model_data.split(",", 1)[1]
                    else:
                        # Plain base64 data
                        model_base64 = model_data
            elif "result" in result:
                # Alternative format
                blob_id = result.get("result")
                resource_url = f"https://{self.app_id}/resource?reid={blob_id}"
                logger.info(f"Fetching 3D model from blob URL: {resource_url}")

                response = requests.get(resource_url, timeout=30)
                if response.status_code == 200:
                    model_binary = response.content
                    model_base64 = base64.b64encode(model_binary).decode("utf-8")

                    # Note the content type but still use GLB for saving
                    content_type = response.headers.get("Content-Type", "")
                    logger.info(
                        f"Received content type: {content_type}, saving as GLB format"
                    )
                else:
                    raise ValueError(
                        f"Failed to fetch blob data: {response.status_code}"
                    )
            elif "model" in result:
                # Direct model data format
                model_data = result.get("model")
                api_format = result.get("format", "glb")
                logger.info(
                    f"API returned model format: {api_format}, saving as GLB format"
                )

                if isinstance(model_data, str):
                    if "," in model_data:
                        model_base64 = model_data.split(",", 1)[1]
                    else:
                        model_base64 = model_data
            else:
                raise KeyError(f"Unknown response format. Keys: {list(result.keys())}")

            if not model_base64:
                raise ValueError("No model data found in the result")

            # Extract base filename from source image
            source_image_filename = Path(image_path).name
            base_name = source_image_filename.rsplit(".", 1)[0]  # Remove extension
            timestamp = int(time.time())

            # Use timestamp in name if not already present
            if not any(c.isdigit() for c in base_name):
                base_name = f"{base_name}_{timestamp}"

            # Append "_3d" to indicate this is a 3D model
            base_name = f"{base_name}_3d"

            # Create filenames - Always use GLB extension
            model_filename = f"{base_name}.glb"
            metadata_filename = f"{base_name}.json"

            # Create paths for model and metadata
            model_path = self.output_dir / model_filename
            metadata_path = self.output_dir / metadata_filename

            # Save the model file
            with open(model_path, "wb") as model_file:
                model_file.write(base64.b64decode(model_base64))

            # Save metadata - set format explicitly to GLB
            metadata = {
                "id": str(uuid.uuid4()),
                "timestamp": timestamp,
                "source_image": image_path,
                "source_image_filename": source_image_filename,
                "file_path": str(model_path),
                "format": "glb",  # Always use glb in metadata
                "type": "3d_model",
                "result_id": result.get("result", result.get("generated_object", "")),
                "parameters": result.get("parameters", {}),
            }

            # Optionally handle video preview if available
            video_data = result.get("video_object")
            if video_data:
                video_filename = f"{base_name}_preview.mp4"
                video_path = self.output_dir / video_filename

                try:
                    video_base64 = video_data
                    if isinstance(video_data, str) and "," in video_data:
                        video_base64 = video_data.split(",", 1)[1]

                    with open(video_path, "wb") as video_file:
                        video_file.write(base64.b64decode(video_base64))

                    metadata["has_video_preview"] = True
                    metadata["video_path"] = str(video_path)
                    logger.info(f"Video preview saved to {video_path}")
                except Exception as e:
                    logger.warning(f"Failed to save video preview: {e}")
                    metadata["has_video_preview"] = False
            else:
                metadata["has_video_preview"] = False

            with open(metadata_path, "w") as meta_file:
                json.dump(metadata, meta_file, indent=2)

            logger.info(f"3D model saved to {model_path}")
            logger.info(f"Metadata saved to {metadata_path}")

            return str(model_path), str(metadata_path)

        except Exception as e:
            logger.error(f"Failed to process 3D model result: {e}")
            raise
