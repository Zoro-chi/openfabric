import logging
import os
from typing import Dict
from pathlib import Path
from dotenv import load_dotenv

from ontology_dc8f06af066e4a7880a5938933236037.config import ConfigClass
from ontology_dc8f06af066e4a7880a5938933236037.input import InputClass
from ontology_dc8f06af066e4a7880a5938933236037.output import OutputClass
from openfabric_pysdk.context import State
from core.stub import Stub
from core.pipeline import CreativePipeline


load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configurations for the app
configurations: Dict[str, ConfigClass] = dict()


############################################################
# Config callback function
############################################################
def config(configuration: Dict[str, ConfigClass], state: State) -> None:
    """
    Stores user-specific configuration data.

    Args:
        configuration (Dict[str, ConfigClass]): A mapping of user IDs to configuration objects.
        state (State): The current state of the application (not used in this implementation).
    """
    for uid, conf in configuration.items():
        logging.info(f"Saving new config for user with id:'{uid}'")
        configurations[uid] = conf


############################################################
# Execution callback function
############################################################
def execute(request_data) -> None:
    """
    Main execution entry point for handling a model pass.

    Args:
        request_data: The object containing request and response structures.
    """

    # Retrieve input
    request: InputClass = request_data.request
    user_prompt = request.prompt

    # Log the incoming request
    logger.info(f"Received request with prompt: '{user_prompt}'")

    # Retrieve user config
    user_config: ConfigClass = configurations.get("super-user", None)
    logger.info(f"Using configuration: {configurations}")

    # Initialize the Stub with app IDs
    app_ids = user_config.app_ids if user_config else []

    # Make sure app IDs are available
    if not app_ids:
        text_to_image_app_id = os.environ.get("TEXT_TO_IMAGE_APP_ID")
        image_to_3d_app_id = os.environ.get("IMAGE_TO_3D_APP_ID")
        app_ids = [text_to_image_app_id, image_to_3d_app_id]
        logger.info(
            f"No app_ids found in config, using environment defaults: {app_ids}"
        )

    stub = Stub(app_ids)

    # Create the creative pipeline
    pipeline = CreativePipeline(stub)

    # Execute the creative pipeline
    try:
        logger.info(f"Executing creative pipeline for prompt: '{user_prompt}'")
        result = pipeline.create(prompt=user_prompt)

        if result.success:
            response_message = (
                f"Created successfully! From your prompt '{user_prompt}', "
                f"I generated an image and a 3D model."
            )
            logger.info(f"Pipeline completed successfully: {result.to_dict()}")
        else:
            if result.image_path:
                response_message = (
                    f"Partially completed. I was able to generate an image from "
                    f"your prompt '{user_prompt}', but couldn't create the 3D model."
                )
                logger.warning(f"Pipeline partially completed: {result.to_dict()}")
            else:
                response_message = (
                    f"I'm sorry, I couldn't process your request '{user_prompt}'. "
                    f"Please try again with a different description."
                )
                logger.error(f"Pipeline failed: {result.to_dict()}")
    except Exception as e:
        logger.error(f"Error executing pipeline: {e}")
        response_message = f"An error occurred while processing your request: {str(e)}"

    # Prepare response
    response: OutputClass = request_data.response
    response.message = response_message
