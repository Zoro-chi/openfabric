"""
Core module for AI Creative application.
Contains components for image generation, 3D model creation, and pipeline orchestration.
"""

import os
import sys
import logging
from pathlib import Path
import re

# Define log file path
log_file_path = os.path.join(os.path.dirname(__file__), "openfabric_service.log")


# Create a filter class to filter out repeated WebSocket "Already connected" errors
class WebSocketFilter(logging.Filter):
    """Filter to remove redundant WebSocket 'Already connected' error logs."""

    def filter(self, record):
        # Skip the "Already connected" WebSocket errors
        if (
            record.levelno == logging.ERROR
            and hasattr(record, "msg")
            and isinstance(record.msg, str)
            and "Could not connect to server. Retrying in" in record.msg
            and "Already connected" in str(record.exc_info)
        ):
            return False

        # Allow all other logs
        return True


# Configure root logger for general logging
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Remove any existing handlers to avoid duplicate logs
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Set up handlers
console_handler = logging.StreamHandler(sys.stdout)
file_handler = logging.FileHandler(log_file_path, mode="a")  # 'a' for append mode

# Create formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add WebSocket filter to both handlers
websocket_filter = WebSocketFilter()
console_handler.addFilter(websocket_filter)
file_handler.addFilter(websocket_filter)

# Add handlers to root logger
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

# Set up specific loggers for different components
# These will inherit handlers from the root logger
openfabric_logger = logging.getLogger("openfabric")
openfabric_logger.setLevel(logging.INFO)

# Filter out less important logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.WARNING)

# Add WebSocket filter to the root logger
root_logger.addFilter(websocket_filter)

# Also add filter to the root logger to catch 'Already connected' errors that might appear elsewhere
root_logger.addFilter(
    lambda record: not (
        record.levelno == logging.ERROR
        and hasattr(record, "exc_info")
        and record.exc_info
        and "Already connected" in str(record.exc_info[1])
    )
)

openfabric_logger.info(
    "Openfabric services logging initialized - writing to " + log_file_path
)
