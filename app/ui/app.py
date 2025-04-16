#!/usr/bin/env python
"""
AI Creative Application UI - Modified for stability and Pydantic compatibility
"""

import os
import sys
import logging
import time
import json  # Add the missing import for json
from pathlib import Path
from dotenv import load_dotenv
import gradio as gr

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ui")

# Add the parent directory to sys.path to import app modules
sys.path.append(str(Path(__file__).parent.parent))

# Set environment variable to disable SSL verification for httpx
os.environ["HTTPX_VERIFY"] = "0"

# Conditionally import app modules with error handling
try:
    from core.pipeline import CreativePipeline, PipelineResult
    from core.stub import Stub

    CORE_MODULES_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import core modules: {str(e)}")
    CORE_MODULES_AVAILABLE = False

# Load environment variables
load_dotenv()

# Get app IDs from environment with defaults
TEXT_TO_IMAGE_APP_ID = os.environ.get(
    "TEXT_TO_IMAGE_APP_ID", "c25dcd829d134ea98f5ae4dd311d13bc.node3.openfabric.network"
)
IMAGE_TO_3D_APP_ID = os.environ.get(
    "IMAGE_TO_3D_APP_ID", "f0b5f319156c4819b9827000b17e511a.node3.openfabric.network"
)


def main():
    """AI Creative application interface"""

    # Paths for saving generated content
    data_path = Path(__file__).parent.parent / "data"
    images_path = data_path / "images"
    models_path = data_path / "models"

    # Ensure necessary directories exist
    images_path.mkdir(exist_ok=True, parents=True)
    models_path.mkdir(exist_ok=True, parents=True)

    # Initialize pipeline only if modules are available
    pipeline = None
    if CORE_MODULES_AVAILABLE:
        try:
            # Configure app IDs - use both TEXT_TO_IMAGE and IMAGE_TO_3D
            app_ids = []
            if TEXT_TO_IMAGE_APP_ID:
                app_ids.append(TEXT_TO_IMAGE_APP_ID)
            if IMAGE_TO_3D_APP_ID:
                app_ids.append(IMAGE_TO_3D_APP_ID)

            logger.info(f"Using app IDs: {app_ids}")
            stub = Stub(app_ids=app_ids)
            pipeline = CreativePipeline(stub)
            logger.info("Pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {str(e)}")

    def generate_from_prompt(prompt, creative_strength=0.7):
        """
        Generate image from text prompt - Using a simpler return format to avoid Pydantic issues
        """
        if not prompt:
            return "Please enter a prompt", None, None, "", ""

        if not pipeline:
            return (
                "Services not available. Please check server status.",
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

            # Update status immediately
            status_msg = "Generating image from your prompt..."

            # Run the creative pipeline
            result = pipeline.create(prompt, params)

            # Handle failed generation
            if not result.success and not result.image_path:
                return "Failed to generate image from prompt", None, None, "", ""

            # Process successful generation
            image_info = f"Original prompt: {result.original_prompt}\n"
            if (
                hasattr(result, "expanded_prompt")
                and result.expanded_prompt
                and result.expanded_prompt != result.original_prompt
            ):
                image_info += f"Enhanced prompt: {result.expanded_prompt}\n"

            # Check for image path
            image_path = result.image_path if hasattr(result, "image_path") else None

            # Check for 3D model
            model_path = (
                result.model_path
                if hasattr(result, "model_path") and result.model_path
                else None
            )
            model_info = ""

            if model_path:
                model_info = f"3D model generated from image.\n"
                model_info += f"Model format: {Path(model_path).suffix[1:]}"
                status_msg = "Image and 3D model generated successfully!"
            else:
                status_msg = "Image generated successfully!"

            return status_msg, image_path, model_path, image_info, model_info

        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            return f"Error: {str(e)}", None, None, "", ""

    def list_gallery_items():
        """List available images in the gallery"""
        images = list(images_path.glob("*.png")) + list(images_path.glob("*.jpg"))
        return sorted(
            [(str(img), img.stem) for img in images], key=lambda x: x[1], reverse=True
        )

    with gr.Blocks(title="AI Creative Studio") as demo:
        gr.Markdown("# AI Creative Studio")
        gr.Markdown("Generate images from text descriptions")

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
            # Function to update the image gallery
            def update_image_gallery():
                images = list(images_path.glob("*.png")) + list(
                    images_path.glob("*.jpg")
                )
                return sorted(
                    [str(img) for img in images],
                    key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0,
                    reverse=True,
                )

            # Function to update the models gallery and return both the models list and model paths
            def update_models_gallery():
                models = list(models_path.glob("*.glb")) + list(
                    models_path.glob("*.gltf")
                )
                model_data = []
                model_paths = []  # Store just the paths for easy access by index

                for model_path in sorted(
                    models,
                    key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0,
                    reverse=True,
                ):
                    # Try to load metadata file if available
                    metadata_path = model_path.with_suffix(".json")
                    creation_time = time.strftime(
                        "%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(model_path))
                    )

                    if metadata_path.exists():
                        try:
                            with open(metadata_path, "r") as f:
                                metadata = json.load(f)
                            source_image = metadata.get(
                                "source_image_filename", "Unknown"
                            )
                            format_type = metadata.get("format", model_path.suffix[1:])
                        except Exception as e:
                            logger.error(
                                f"Failed to read metadata for {model_path}: {e}"
                            )
                            source_image = "Unknown"
                            format_type = model_path.suffix[1:]
                    else:
                        source_image = "Unknown"
                        format_type = model_path.suffix[1:]

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

            # Function to view model by index instead of relying on DataFrame selection
            def view_model_by_index(evt: gr.SelectData):
                if (
                    not hasattr(view_model_by_index, "model_paths")
                    or not view_model_by_index.model_paths
                ):
                    logger.warning("No model paths available")
                    return None, None

                try:
                    # Get the index from the selection event
                    row_index = evt.index[0] if hasattr(evt, "index") else 0
                    if row_index < 0 or row_index >= len(
                        view_model_by_index.model_paths
                    ):
                        logger.warning(f"Invalid model index: {row_index}")
                        return None, None

                    # Get the model path from our saved list
                    model_path = view_model_by_index.model_paths[row_index]
                    logger.info(f"Selected model at index {row_index}: {model_path}")

                    if not model_path or not os.path.exists(model_path):
                        logger.warning(f"Model file not found: {model_path}")
                        return None, None

                except (IndexError, AttributeError) as e:
                    logger.error(f"Error accessing selected model: {e}")
                    return None, None

                # Get model metadata if available
                metadata_path = Path(model_path).with_suffix(".json")
                metadata = {}

                if metadata_path.exists():
                    try:
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                        logger.info(f"Loaded metadata for model: {model_path}")
                    except Exception as e:
                        logger.error(f"Failed to read metadata for {model_path}: {e}")
                else:
                    logger.warning(f"No metadata file found for model: {metadata_path}")

                return model_path, metadata

            # Function to store model paths in the view function's namespace
            def store_model_paths(model_data, model_paths):
                view_model_by_index.model_paths = model_paths
                return model_data

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
                    # We'll keep this button for compatibility, but also add automatic selection on row click
                    view_model_btn = gr.Button("View Selected Model")

                    # Make the dataframe selection trigger the model loading automatically
                    models_list.select(
                        fn=view_model_by_index, outputs=[selected_model, model_details]
                    )

            # Wire up the gallery refresh buttons
            refresh_img_btn.click(fn=update_image_gallery, outputs=[image_gallery])
            refresh_models_btn.click(
                fn=lambda: store_model_paths(*update_models_gallery()),
                outputs=[models_list],
            )

            # Wire up the model viewer
            view_model_btn.click(
                fn=view_model_by_index,
                inputs=[models_list],
                outputs=[selected_model, model_details],
            )

            # Initial gallery loads
            demo.load(update_image_gallery, outputs=[image_gallery])
            demo.load(
                fn=lambda: store_model_paths(*update_models_gallery()),
                outputs=[models_list],
            )

        # Wire up the generate button - non-streaming mode to avoid Pydantic issues
        generate_btn.click(
            fn=generate_from_prompt,
            inputs=[prompt_input, creative_strength],
            outputs=[
                status,
                image_output,
                model_viewer,
                image_info,
                model_info,
            ],
        )

        # Initial gallery load
        demo.load(update_image_gallery, outputs=[image_gallery])
        demo.load(
            fn=lambda: store_model_paths(*update_models_gallery()),
            outputs=[models_list],
        )

    # Launch the UI with parameters compatible with Gradio 4.26.0
    port = int(os.environ.get("UI_PORT", 7860))
    logger.info(f"Launching UI on port {port}")
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=True,
        show_error=True,
        # Removed api_mode parameter that's not supported in 4.26.0
    )


if __name__ == "__main__":
    main()
