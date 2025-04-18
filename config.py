import os
from typing import Dict, Any, Optional

# Default configurations
DEFAULT_CONFIG = {
    "llm": {
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "context_size": 4096,
        "max_tokens": 512,
        "temperature": 0.7,
    },
    "ui": {
        "host": "0.0.0.0",
        "port": 7860,
    },
    "openfabric": {
        "text_to_image_app_id": "f0997a01-d6d3-a5fe-53d8-561300318557",
        "image_to_3d_app_id": "69543f29-4d41-4afc-7f29-3d51591f11eb",
    },
}


# Environment-based configuration
def get_config() -> Dict[str, Any]:
    """
    Get configuration with environment variables overrides
    """
    config = DEFAULT_CONFIG.copy()

    # Override with environment variables if available
    if os.environ.get("LLM_MODEL_ID"):
        config["llm"]["model_id"] = os.environ.get("LLM_MODEL_ID")

    if os.environ.get("LLM_MODEL_PATH"):
        config["llm"]["model_path"] = os.environ.get("LLM_MODEL_PATH")

    # Add any other environment variable overrides as needed

    return config
