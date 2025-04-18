#!/usr/bin/env python
"""
AI Creative Application UI - Production-ready version
"""

import os
import sys
import logging
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv
import gradio as gr

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ui")

# Add parent directory to sys.path to import app modules
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

# Set environment variable to disable SSL verification for httpx
# Note: In production, you should properly configure certificates
os.environ["HTTPX_VERIFY"] = "0"

# Load environment variables first
load_dotenv()

# Set up paths
DATA_DIR = parent_dir / "data"
IMAGES_DIR = DATA_DIR / "images"
MODELS_DIR = DATA_DIR / "models"

# Get app IDs from environment with defaults
TEXT_TO_IMAGE_APP_ID = os.environ.get("TEXT_TO_IMAGE_APP_ID")
IMAGE_TO_3D_APP_ID = os.environ.get("IMAGE_TO_3D_APP_ID")

# Conditionally import with error handling
try:
    from core.pipeline import CreativePipeline
    from core.stub import Stub

    CORE_MODULES_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import core modules: {str(e)}")
    CORE_MODULES_AVAILABLE = False


def initialize_pipeline() -> Optional[CreativePipeline]:
    """Initialize the creative pipeline with proper error handling"""
    if not CORE_MODULES_AVAILABLE:
        logger.error("Cannot initialize pipeline - core modules unavailable")
        return None

    try:
        # Configure app IDs
        app_ids = []
        if TEXT_TO_IMAGE_APP_ID:
            app_ids.append(TEXT_TO_IMAGE_APP_ID)
        if IMAGE_TO_3D_APP_ID:
            app_ids.append(IMAGE_TO_3D_APP_ID)

        if not app_ids:
            logger.error(
                "No app IDs configured. Set TEXT_TO_IMAGE_APP_ID and/or IMAGE_TO_3D_APP_ID"
            )
            return None

        logger.info(f"Initializing pipeline with app IDs: {app_ids}")
        stub = Stub(app_ids=app_ids)
        pipeline = CreativePipeline(stub)
        logger.info("Pipeline initialized successfully")
        return pipeline
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {str(e)}", exc_info=True)
        return None


def generate_from_prompt(
    prompt: str, creative_strength: float = 0.7, pipeline=None
) -> Tuple:
    """
    Generate image from text prompt with proper error handling

    Returns: (status_message, image_path, model_path, image_info, model_info)
    """
    if not prompt or not prompt.strip():
        return "Please enter a prompt", None, None, "", ""

    if not pipeline:
        return (
            "Services not available. Check server status and API keys.",
            None,
            None,
            "",
            "",
        )

    try:
        # Parameters for generation
        params = {
            "image": {
                "creative_strength": creative_strength,
            },
            "model": {"quality": "standard"},
        }

        # Run the creative pipeline
        start_time = time.time()
        result = pipeline.create(prompt, params)
        elapsed = time.time() - start_time

        # Handle failed generation
        if not result.success and not getattr(result, "image_path", None):
            return (
                f"Failed to generate image: {getattr(result, 'error', 'unknown error')}",
                None,
                None,
                "",
                "",
            )

        # Process successful generation
        image_info = f"Original prompt: {result.original_prompt}\n"
        if (
            hasattr(result, "expanded_prompt")
            and result.expanded_prompt
            and result.expanded_prompt != result.original_prompt
        ):
            image_info += f"Enhanced prompt: {result.expanded_prompt}\n"

        image_info += f"Generation time: {elapsed:.1f}s"

        # Handle image path
        image_path = getattr(result, "image_path", None)

        # Handle 3D model
        model_path = getattr(result, "model_path", None)
        model_info = ""

        if model_path:
            model_info = f"3D model generated from image.\n"
            model_info += f"Model format: {Path(model_path).suffix[1:]}"
            status_msg = "Image and 3D model generated successfully!"
        else:
            status_msg = "Image generated successfully!"

        return status_msg, image_path, model_path, image_info, model_info

    except Exception as e:
        logger.error(f"Generation error: {str(e)}", exc_info=True)
        return f"Error: {str(e)}", None, None, "", ""


def main():
    """AI Creative application interface"""
    # Ensure necessary directories exist
    IMAGES_DIR.mkdir(exist_ok=True, parents=True)
    MODELS_DIR.mkdir(exist_ok=True, parents=True)

    # Initialize pipeline
    pipeline = initialize_pipeline()

    # Functions for gallery management
    def update_image_gallery():
        """Update the image gallery with latest files"""
        images = list(IMAGES_DIR.glob("*.png")) + list(IMAGES_DIR.glob("*.jpg"))
        return sorted(
            [str(img) for img in images],
            key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0,
            reverse=True,
        )[
            :20
        ]  # Only show the 20 most recent images

    def update_models_gallery():
        """Update the models gallery and return model data and paths"""
        models = list(MODELS_DIR.glob("*.glb")) + list(MODELS_DIR.glob("*.gltf"))
        model_data = []
        model_paths = []  # Store just the paths for easy access by index

        for model_path in sorted(
            models,
            key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0,
            reverse=True,
        )[
            :20
        ]:  # Only show the 20 most recent models
            # Try to load metadata file if available
            metadata_path = model_path.with_suffix(".json")
            creation_time = time.strftime(
                "%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(model_path))
            )

            source_image = "Unknown"
            format_type = model_path.suffix[1:]

            # Try to load metadata
            if metadata_path.exists():
                try:
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    source_image = metadata.get("source_image_filename", "Unknown")
                    format_type = metadata.get("format", format_type)
                except Exception as e:
                    logger.warning(f"Failed to read metadata for {model_path}: {e}")

            # Add to data table and path list
            model_paths.append(str(model_path))
            model_data.append(
                [
                    str(model_path),
                    source_image,
                    format_type,
                    creation_time,
                ]
            )

        return model_data, model_paths

    def view_model_by_index(evt: gr.SelectData):
        """View a model by its index in the table"""
        if (
            not hasattr(view_model_by_index, "model_paths")
            or not view_model_by_index.model_paths
        ):
            logger.warning("No model paths available")
            return None, None

        try:
            # Get the index from the selection event
            row_index = evt.index[0] if hasattr(evt, "index") and evt.index else 0
            if row_index < 0 or row_index >= len(view_model_by_index.model_paths):
                logger.warning(f"Invalid model index: {row_index}")
                return None, None

            # Get the model path
            model_path = view_model_by_index.model_paths[row_index]
            logger.info(f"Selected model at index {row_index}: {model_path}")

            if not model_path or not os.path.exists(model_path):
                logger.warning(f"Model file not found: {model_path}")
                return None, None

            # Handle the model path based on file extension
            model_path_obj = Path(model_path)
            file_format = model_path_obj.suffix.lower()

            # Check if it's a supported format (both GLB and GLTF should work)
            if file_format not in [".glb", ".gltf"]:
                logger.warning(f"Unsupported model format: {file_format}")
                return None, None

            metadata = {}
            # Get model metadata if available
            metadata_path = model_path_obj.with_suffix(".json")
            if metadata_path.exists():
                try:
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    logger.info(f"Loaded metadata for model: {model_path}")
                except Exception as e:
                    logger.warning(f"Failed to read metadata for {model_path}: {e}")

            # Return the model path directly - Gradio's Model3D component should handle both formats
            return model_path, metadata
        except Exception as e:
            logger.error(f"Error accessing selected model: {e}", exc_info=True)
            return None, None

    def store_model_paths(model_data, model_paths):
        """Store model paths for the view function"""
        view_model_by_index.model_paths = model_paths
        return model_data

    # Create the UI
    with gr.Blocks(title="AI Creative Studio") as demo:
        gr.Markdown("# AI Creative Studio")
        gr.Markdown("Generate images from text descriptions and convert to 3D models")

        with gr.Tab("Create"):
            with gr.Row():
                with gr.Column(scale=2):
                    # Input area
                    prompt_input = gr.Textbox(
                        label="Your creative prompt",
                        placeholder="Describe what you want to create...",
                        lines=3,
                    )

                    with gr.Row():
                        creative_strength = gr.Slider(
                            label="Creative Strength",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.7,
                            step=0.1,
                        )

                    generate_btn = gr.Button("Generate", variant="primary")
                    status = gr.Textbox(label="Status", interactive=False)

                with gr.Column(scale=3):
                    # Output area with tabs for different views
                    with gr.Tab("Image"):
                        with gr.Row():
                            image_output = gr.Image(
                                label="Generated Image", type="filepath"
                            )
                            image_info = gr.Textbox(
                                label="Image Details", interactive=False, lines=3
                            )

                    with gr.Tab("3D Model"):
                        with gr.Row():
                            model_viewer = gr.Model3D(label="3D Model")
                            model_info = gr.Textbox(
                                label="Model Details", interactive=False, lines=3
                            )

        with gr.Tab("Gallery"):
            with gr.Tabs() as gallery_tabs:
                with gr.Tab("Images"):
                    image_gallery = gr.Gallery(
                        label="Generated Images",
                        columns=4,
                        object_fit="contain",
                        height="auto",
                    )
                    refresh_img_btn = gr.Button("Refresh Images")

                with gr.Tab("3D Models"):
                    models_list = gr.Dataframe(
                        headers=["Model", "Source Image", "Format", "Created"],
                        label="Available 3D Models",
                        row_count=10,
                        col_count=(4, "fixed"),
                        interactive=False,
                    )
                    with gr.Row():
                        selected_model = gr.Model3D(label="Selected 3D Model")
                        model_details = gr.JSON(label="Model Details")

                    refresh_models_btn = gr.Button("Refresh Models")

                    # Make the dataframe selection trigger the model loading
                    models_list.select(
                        fn=view_model_by_index, outputs=[selected_model, model_details]
                    )

            # Wire up the gallery refresh buttons
            refresh_img_btn.click(fn=update_image_gallery, outputs=[image_gallery])
            refresh_models_btn.click(
                fn=lambda: store_model_paths(*update_models_gallery()),
                outputs=[models_list],
            )

            # Initial gallery loads
            demo.load(update_image_gallery, outputs=[image_gallery])
            demo.load(
                fn=lambda: store_model_paths(*update_models_gallery()),
                outputs=[models_list],
            )

        # Wire up the generate button with pipeline
        if pipeline:
            generate_btn.click(
                fn=lambda prompt, strength: generate_from_prompt(
                    prompt, strength, pipeline
                ),
                inputs=[prompt_input, creative_strength],
                outputs=[status, image_output, model_viewer, image_info, model_info],
            )
        else:
            # Handle the case where pipeline initialization failed
            def service_unavailable(*args):
                return "Service unavailable. Check server logs.", None, None, "", ""

            generate_btn.click(
                fn=service_unavailable,
                inputs=[prompt_input, creative_strength],
                outputs=[status, image_output, model_viewer, image_info, model_info],
            )

            # Add warning at startup
            demo.load(
                lambda: gr.Warning(
                    "Creative service initialization failed. Check logs for details."
                )
            )

    # Launch the UI
    port = int(os.environ.get("UI_PORT", 7860))
    logger.info(f"Launching UI on port {port}")

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=True,
        show_error=True,
    )


if __name__ == "__main__":
    main()
