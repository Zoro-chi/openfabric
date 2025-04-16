import os
import logging
import base64
from typing import Dict, Optional, Any, Tuple
import json
from pathlib import Path
import time
import uuid
import random
import requests
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

        # Maximum time to wait for job completion (in seconds)
        self.max_wait_time = 300  # 5 minutes

        # Polling interval for checking job status (in seconds)
        self.polling_interval = 5  # Check every 5 seconds

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
        rid = None
        try:
            start_time = time.time()

            # Make the API call - this will return immediately with a request ID
            response = self.stub.call(self.app_id, request_data)

            # Extract the request ID from logs
            rid = self._extract_rid_from_logs()
            if not rid:
                raise ValueError("Failed to extract request ID from logs")

            logger.info(f"Submitted image-to-3D job with request ID: {rid}")

            # Poll for job completion
            qid, result = self._poll_for_completion(rid)

            generation_time = time.time() - start_time
            logger.info(f"Image-to-3D generation completed in {generation_time:.2f}s")

            if not result:
                raise ValueError("Failed to get result data after job completion")

            # Process and save the result
            return self._process_result(result, image_path)

        except Exception as e:
            logger.error(f"Failed to generate 3D model: {e}")
            raise

    def _extract_rid_from_logs(self) -> str:
        """
        Extract the request ID (rid) from logs.
        The stub logs the rid when it creates a request in the format "Created rid{rid}"

        Returns:
            Request ID string or None if not found
        """
        import re

        # Try to scan the last few log lines for the request ID pattern
        log_handler = next(
            (
                h
                for h in logging.getLogger("root").handlers
                if isinstance(h, logging.StreamHandler)
            ),
            None,
        )

        if hasattr(log_handler, "stream") and hasattr(log_handler.stream, "getvalue"):
            # This is for StringIO in testing environments
            log_content = log_handler.stream.getvalue()
            matches = re.findall(r"Created rid([a-f0-9]+)", log_content)
            if matches:
                return matches[-1]  # Return the most recent match

        # Alternative approach: look for the most recently created request
        try:
            queue_url = f"https://{self.app_id}/queue/list"
            response = requests.get(queue_url)
            if response.status_code == 200:
                job_list = response.json()
                if job_list and isinstance(job_list, list) and len(job_list) > 0:
                    # Sort by creation time (newest first) and get the first one
                    sorted_jobs = sorted(
                        job_list, key=lambda x: x.get("created_at", ""), reverse=True
                    )
                    return sorted_jobs[0].get("rid")
        except Exception as e:
            logger.warning(f"Failed to get request ID from queue: {e}")

        return None

    def _poll_for_completion(self, rid: str) -> Tuple[str, Dict[str, Any]]:
        """
        Poll the queue list endpoint until the job is complete.

        Args:
            rid: Request ID to check

        Returns:
            Tuple of (queue_id, result_data)
        """
        start_time = time.time()
        qid = None
        result = None

        logger.info(f"Waiting for job completion (rid: {rid})...")

        while (time.time() - start_time) < self.max_wait_time:
            try:
                # Get the queue list
                queue_url = f"https://{self.app_id}/queue/list"
                response = requests.get(queue_url)

                if response.status_code != 200:
                    logger.error(f"Failed to get queue list: {response.status_code}")
                    time.sleep(self.polling_interval)
                    continue

                # Parse the response and find our job
                job_list = response.json()
                if not isinstance(job_list, list):
                    logger.error(f"Unexpected queue list format: {type(job_list)}")
                    time.sleep(self.polling_interval)
                    continue

                # Find our job by request ID
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
                if our_job.get("finished") and our_job.get("status") == "COMPLETED":
                    logger.info(f"Job completed successfully")

                    # Get the detailed result
                    result_url = f"https://{self.app_id}/queue/get?qid={qid}"
                    result_response = requests.get(result_url)

                    if result_response.status_code == 200:
                        result = result_response.json()
                        logger.info(f"Got result data: {result}")
                        return qid, result
                    else:
                        logger.error(
                            f"Failed to get result data: {result_response.status_code}"
                        )

                elif our_job.get("finished") and our_job.get("status") != "COMPLETED":
                    # Job failed
                    status = our_job.get("status")
                    messages = our_job.get("messages", [])
                    error_msgs = [
                        m.get("content") for m in messages if m.get("type") == "ERROR"
                    ]

                    error_msg = f"Job failed with status: {status}"
                    if error_msgs:
                        error_msg += f", errors: {'; '.join(error_msgs)}"

                    logger.error(error_msg)
                    raise ValueError(error_msg)

                # Job is still running
                status = our_job.get("status")
                progress = (
                    our_job.get("bars", {}).get("default", {}).get("percent", "0")
                )
                logger.info(f"Job status: {status}, progress: {progress}%")

            except Exception as e:
                logger.error(f"Error polling for job completion: {e}")

            # Wait before checking again
            time.sleep(self.polling_interval)

        # If we get here, we timed out
        raise TimeoutError(
            f"Timed out waiting for job completion after {self.max_wait_time} seconds"
        )

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
        # If result is None, raise an error - don't use mock data
        if result is None:
            raise ValueError("No result received from image-to-3D generation service")

        try:
            model_format = "glb"  # Default format for 3D models
            has_video_preview = False
            video_data = None
            model_base64 = None

            # Process Openfabric blob response format (most common case)
            # This will have 'generated_object' as a data_blob ID (without the base64 data)
            if "generated_object" in result:
                # Extract the model data or model blob reference
                model_data = result.get("generated_object")

                # Check if this is a blob reference (data URI format or plain string)
                if isinstance(model_data, str):
                    if "/" in model_data or model_data.startswith("data_"):
                        # This is a blob ID reference
                        blob_id = model_data
                        logger.info(f"3D model generation result ID: {blob_id}")

                        # Prepare the blob URL and try to download the actual model data
                        try:
                            # Construct URL for the blob
                            resource_url = (
                                f"https://{self.app_id}/resource?reid={blob_id}"
                            )
                            logger.info(
                                f"Fetching 3D model from blob URL: {resource_url}"
                            )

                            response = requests.get(resource_url)

                            if response.status_code == 200:
                                # We have the actual model data
                                model_binary = response.content
                                model_base64 = base64.b64encode(model_binary).decode(
                                    "utf-8"
                                )
                                logger.info(
                                    f"Successfully fetched 3D model from blob store"
                                )

                                # Set format based on content-type if available
                                content_type = response.headers.get("Content-Type", "")
                                if "gltf-binary" in content_type:
                                    model_format = "glb"
                                elif (
                                    "gltf+json" in content_type
                                    or "json" in content_type
                                ):
                                    model_format = "gltf"
                            else:
                                logger.error(
                                    f"Failed to fetch blob: {response.status_code} - {response.text}"
                                )
                                raise ValueError(
                                    f"Failed to fetch blob data: {response.status_code}"
                                )
                        except Exception as blob_error:
                            logger.error(f"Error accessing blob store: {blob_error}")
                            raise ValueError(
                                f"Failed to fetch 3D model from blob store: {blob_error}"
                            )
                    elif "," in model_data and "base64" in model_data:
                        # Extract base64 data if in data URI format
                        model_base64 = model_data.split(",", 1)[1]
                    else:
                        # Use as-is for plain base64 data
                        model_base64 = model_data
                else:
                    # If model_data is not a string, this is an unexpected format
                    raise ValueError(
                        f"Unexpected generated_object type: {type(model_data)}"
                    )

                # Also handle video preview if available
                video_data = result.get("video_object")
                has_video_preview = video_data is not None and video_data != ""

            # Handle result blob reference format (alternative format)
            elif "result" in result:
                blob_id = result.get("result")
                logger.info(f"3D model generation result ID: {blob_id}")

                # Try to fetch the actual model data from the blob store
                try:
                    # Construct URL for the blob
                    resource_url = f"https://{self.app_id}/resource?reid={blob_id}"
                    logger.info(f"Fetching 3D model from blob URL: {resource_url}")

                    response = requests.get(resource_url)

                    if response.status_code == 200:
                        # We have the actual model data
                        model_binary = response.content
                        model_base64 = base64.b64encode(model_binary).decode("utf-8")
                        logger.info(f"Successfully fetched 3D model from blob store")

                        # Set format based on content-type if available
                        content_type = response.headers.get("Content-Type", "")
                        if "gltf-binary" in content_type:
                            model_format = "glb"
                        elif "gltf+json" in content_type or "json" in content_type:
                            model_format = "gltf"
                    else:
                        logger.error(
                            f"Failed to fetch blob: {response.status_code} - {response.text}"
                        )
                        raise ValueError(
                            f"Failed to fetch blob data: {response.status_code}"
                        )
                except Exception as blob_error:
                    logger.error(f"Error accessing blob store: {blob_error}")
                    raise ValueError(
                        f"Failed to fetch 3D model from blob store: {blob_error}"
                    )

            # Handle direct model data format (which has 'model' field)
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

            if not model_base64:
                raise ValueError("No model data found in the result")

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
                "result_id": result.get("result", result.get("generated_object", "")),
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
